"""A moving object (an asteroid): where it was at every SPHEREx exposure, and which images caught it.

JPL Horizons predicts the asteroid's position - as seen from the SPHEREx spacecraft itself, so the
parallax of SPHEREx's orbit around Earth (up to ~14" for a near-Earth asteroid like Eros) is included.
IRSA's image search then finds the images whose footprint contained the asteroid at the moment of the
exposure. Everything after that (cutouts, measurements, plots) reuses the star pipeline.
"""

from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from io import StringIO
import threading

import numpy as np
from matplotlib.path import Path as Polygon

from astropy.table import Table, vstack
from astropy.time import Time

from . import config as cfg
from . import cache
from .irsa import query_images_at, with_retries

SPHEREX_OBSERVER = '@-163182'       # JPL Horizons id of the SPHEREx spacecraft (observer = the telescope itself)
JD_MINUS_MJD = 2400000.5
EPOCHS_PER_REQUEST = 40             # exact-time positions are requested in batches of this many (Horizons takes the
                                    # epochs in the web address; 100 make it too long and its gateway answers 502)


class Ephemeris:
    """The asteroid's path over a time window (JPL Horizons, hourly), interpolated in between."""

    def __init__(self, table):
        self.table = table
        self.mjd = np.array(table['datetime_jd'], dtype=float) - JD_MINUS_MJD
        self.ra = np.unwrap(np.radians(np.array(table['RA'], dtype=float)))     # unwrapped across 360 -> 0
        self.dec = np.array(table['DEC'], dtype=float)

    @classmethod
    def from_arrays(cls, mjd, ra_deg, dec_deg):
        """A path given directly (e.g. fitted to an unknown moving object's detections) instead of from Horizons."""
        self = cls.__new__(cls)
        self.table = None
        self.mjd = np.asarray(mjd, dtype=float)
        self.ra = np.unwrap(np.radians(np.asarray(ra_deg, dtype=float)))
        self.dec = np.asarray(dec_deg, dtype=float)
        return self

    def position(self, mjd):
        """(RA, Dec) in degrees at the given time(s) (MJD)."""
        return np.degrees(np.interp(mjd, self.mjd, self.ra)) % 360, np.interp(mjd, self.mjd, self.dec)

    def path_length_deg(self):
        """Cumulative distance along the path (degrees) at every ephemeris row."""
        step = np.hypot(np.diff(np.degrees(self.ra)) * np.cos(np.radians(self.dec[:-1])), np.diff(self.dec))
        return np.r_[0.0, np.cumsum(step)]


# Serialize Horizons requests and reuse successful exact queries in memory and on disk.
# Earth-centred path searches and spacecraft-centred measurements have separate keys.
_HORIZONS_QUANTITIES = '1,3,9,19,20,23,24'
_horizons_lock = threading.RLock()
_horizons_tables = OrderedDict()
_spacecraft_rows = OrderedDict()


def fetch_ephemeris(asteroid, epochs, location='500'):
    """Fetch a Horizons observer table once per exact request, using Xarvis's cache."""
    target, location = str(asteroid).strip(), str(location)
    epoch_key = tuple(sorted(epochs.items())) if isinstance(epochs, dict) else tuple(map(float, epochs))
    key = cache.make_key('horizons-v1', target, location, _HORIZONS_QUANTITIES,
                         'range' if isinstance(epochs, dict) else 'dates', epoch_key)
    with _horizons_lock:
        if key in _horizons_tables:
            _horizons_tables.move_to_end(key)
            return _horizons_tables[key].copy()
        content = cache.get('horizons', key)
        table = None
        if content is not None:
            try:
                table = Table.read(content.decode().splitlines(), format='ascii.ecsv')
            except (ValueError, TypeError, OSError):
                pass
        if table is None:
            from astroquery.jplhorizons import Horizons  # only when an uncached request is needed
            def once():
                return Horizons(id=target, id_type='smallbody', location=location, epochs=epochs).ephemerides(
                    quantities=_HORIZONS_QUANTITIES, cache=False)
            table = with_retries(f'Horizons {target}', once)
            # Do not let a failed cache write discard a valid ephemeris.
            try:
                buffer = StringIO()
                table.write(buffer, format='ascii.ecsv')
                cache.put('horizons', key, buffer.getvalue().encode())
            except (OSError, ValueError, TypeError) as exc:
                print(f'  Horizons cache write skipped: {exc}')
        _horizons_tables[key] = table.copy()
        if len(_horizons_tables) > 32:
            _horizons_tables.popitem(last=False)
        return table.copy()


def spacecraft_ephemeris(asteroid, mjd):
    """Exact spacecraft observer rows, returned in input order, including duplicates.

    Shared by target tracking and sky-map refinement. Only missing epochs are
    requested; retain geometry columns for later spectral modelling. No reuse
    between different target IDs or observer locations.
    """
    target = str(asteroid).strip()
    times = np.asarray(mjd, dtype=float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
        raise ValueError('Provide a nonempty list of finite observation times.')
    epochs, back = np.unique(np.round(times, 6), return_inverse=True)
    with _horizons_lock:
        rows = {float(t): _spacecraft_rows[(target, float(t))] for t in epochs
                if (target, float(t)) in _spacecraft_rows}
        missing = np.array([t for t in epochs if float(t) not in rows])
        for start in range(0, len(missing), EPOCHS_PER_REQUEST):
            batch = missing[start:start + EPOCHS_PER_REQUEST]
            table = fetch_ephemeris(target, list(batch + JD_MINUS_MJD), SPHEREX_OBSERVER)
            returned = np.asarray(table['datetime_jd'], float) - JD_MINUS_MJD
            if (len(returned) != len(batch) or not np.isfinite(returned).all()
                    or not np.isfinite(np.ma.filled(np.ma.asarray(table['RA'], dtype=float), np.nan)).all()
                    or not np.isfinite(np.ma.filled(np.ma.asarray(table['DEC'], dtype=float), np.nan)).all()):
                raise RuntimeError('Horizons returned an incomplete spacecraft ephemeris.')
            order = np.argsort(returned)
            if not np.allclose(returned[order], batch, rtol=0, atol=2e-7):
                raise RuntimeError('Horizons returned unexpected observation times.')
            for t, i in zip(batch, order):
                row = table[int(i):int(i)+1].copy()
                rows[float(t)] = row
                _spacecraft_rows[(target, float(t))] = row
        while len(_spacecraft_rows) > 8192:
            _spacecraft_rows.popitem(last=False)
        unique = vstack([rows[float(t)] for t in epochs], metadata_conflicts='silent')
        return unique[back]


def find_asteroid(asteroid, window):
    """The asteroid's path over `window` (two dates), as seen from Earth's center, hourly."""
    start, stop = (Time(t).mjd for t in window)
    table = fetch_ephemeris(asteroid, {'start': Time(start, format='mjd').iso[:16],
                                       'stop': Time(stop, format='mjd').iso[:16], 'step': '60m'})
    ephemeris = Ephemeris(table)
    name = str(table['targetname'][0]).split('(')[0].strip()
    elong, rate = np.array(table['elong'], dtype=float), np.hypot(table['RA_rate'], table['DEC_rate']) * 24 / 3600
    print(f"Tracking asteroid {name}: moves {np.median(rate):.2f} deg/day ({np.median(rate) * 3600 / 6.15 / 24:.0f} SPHEREx "
          f"pixels per hour), V = {np.min(table['V']):.1f} - {np.max(table['V']):.1f}, "
          f"{np.median(table['delta']):.2f} AU away")
    if not ((elong > 80) & (elong < 100)).any():
        print(f"  WARNING: during this window the asteroid is {elong.min():.0f} - {elong.max():.0f} degrees from the Sun, "
              f"but SPHEREx only looks at 82 - 98 degrees: it will probably not be in any image")
    return {'name': name, 'asteroid': asteroid, 'ephemeris': ephemeris, 'window': (start, stop)}


def in_footprint(region, ra, dec):
    """True if (ra, dec) lies inside the footprint polygon of an image ('POLYGON ICRS ra dec ra dec ...')."""
    corners = np.array(str(region).split()[2:], dtype=float).reshape(-1, 2)
    flat = np.c_[((corners[:, 0] - ra + 180) % 360 - 180) * np.cos(np.radians(dec)), corners[:, 1] - dec]
    return Polygon(flat).contains_point((0.0, 0.0))


def find_images(tracked, spacing_deg=cfg.GRID_SPACING_DEG):
    """SPHEREx images whose footprint contained the asteroid at the time of the exposure, sorted by time.

    One position search every `spacing_deg` along the asteroid's path returns every image that ever covered
    that spot; the ones that covered it at a different time are discarded by checking the footprint against
    the asteroid's position at each image's own time."""
    ephemeris = tracked['ephemeris']
    length = ephemeris.path_length_deg()
    wanted = np.arange(0, length[-1] + spacing_deg, spacing_deg)
    samples = np.unique(np.clip(np.searchsorted(length, wanted), 0, len(length) - 1))
    samples = np.unique(np.r_[0, samples, len(length) - 1])
    ra, dec = ephemeris.position(ephemeris.mjd[samples])
    print(f'Searching for SPHEREx images along {length[-1]:.1f} degrees of path ({len(samples)} position searches)...')

    with ThreadPoolExecutor(max_workers=cfg.MAX_CONCURRENT_DOWNLOADS) as pool:
        tables = list(pool.map(query_images_at, ra, dec))
    candidates = vstack([t for t in tables if len(t)], metadata_conflicts='silent')
    # the same image can cover several samples: keep each once. (By file, not obs_id: the short- and long-
    # wavelength detectors of one exposure share an obs_id - de-duplicating by it threw away half the images.)
    _, first = np.unique(np.array(candidates['access_url'], dtype=str), return_index=True)
    candidates = candidates[first]

    t_mid = 0.5 * (np.array(candidates['t_min'], dtype=float) + np.array(candidates['t_max'], dtype=float))
    in_window = (t_mid >= tracked['window'][0]) & (t_mid <= tracked['window'][1])
    candidates, t_mid = candidates[in_window], t_mid[in_window]
    ra_t, dec_t = ephemeris.position(t_mid)
    caught = np.array([in_footprint(region, a, d) for region, a, d in zip(candidates['s_region'], ra_t, dec_t)], dtype=bool)

    images = candidates[caught]
    images['t_mid'] = t_mid[caught]
    images.sort('t_mid')
    print(f'{len(images)} SPHEREx images contained the asteroid at the time of the exposure '
          f'(of {len(candidates)} that passed over the path during the window)')
    return images


def add_predicted_positions(images, tracked):
    """Where the asteroid was at each exposure, as seen from the SPHEREx spacecraft: adds columns
    'target_ra' / 'target_dec' (degrees) to `images`. An object without a Horizons orbit (a path fitted to new
    detections, tracked['asteroid'] = None) uses its own path instead."""
    if not tracked.get('asteroid'):
        images['target_ra'], images['target_dec'] = tracked['ephemeris'].position(np.array(images['t_mid'], dtype=float))
        return images
    table = spacecraft_ephemeris(tracked['asteroid'], images['t_mid'])
    images['target_ra'] = np.array(table['RA'], dtype=float)
    images['target_dec'] = np.array(table['DEC'], dtype=float)
    # Keep the exact observer geometry already paid for, in image-row order.
    # For retained frames, match by access_url rather than assuming row order.
    tracked['spacecraft_ephemeris'] = table
    tracked['spacecraft_ephemeris_urls'] = np.array(images['access_url'], dtype=str)
    return images
