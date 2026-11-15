"""Entry points for a web app (or any other program): run an analysis and get plain, JSON-ready data back.

    from Jonah_Code_File import api
    result = api.analyze({'coords': '08h55m10.8s -07d14m42s'}, on_progress=print)    # a dict, json.dumps-able
    result, raw = api.analyze({'asteroid': '433'}, settings={'MAX_IMAGES': 40}, return_raw=True)
    png = api.frame_png(raw, 0)                 # the frame T1 as a PNG image (bytes), display-only colour
    objects = api.frame_objects(raw, 0)         # what the sky map found in T1, in that image's pixel coordinates

target: one of {'asteroid': id, 'window': [start, end]}, {'star': name}, {'coords': '...' or [ra, dec]}, or the
settings a results file stores ({'mode': ..., ...}; see results.known_objects).
settings: config.json values to use for this analysis only (e.g. {'MAX_IMAGES': 40, 'USE_PSF': False}).
on_progress(message): called with every progress message (the same lines the terminal shows).
isolated=True: run in a separate process - analyses with different settings can then run at the same time
(settings are module-wide in one process, so in-process analyses run one at a time).

Nothing here opens a window: images are made with visual.render (display-only) and returned as PNG bytes.

What was analysed before can be read back without running anything (the results folder, see results.py):
    api.known_targets()                         # every known target: analysed ones, then the standard test objects
    api.saved_summary('433 Eros')               # a saved analysis in the same form analyze() returns (no pixels)
    api.saved_candidates()                      # every NEW object in the saved results
    api.motion_fits()                           # algorithm.run_from_results(): the apparent path of every target
And for a web viewer of a finished analysis (`raw`):
    api.band_spectrum(raw['frames'])            # the spectrum in SPHEREx's 102 channels (spectrumanalyser.py)
    api.compare_frames(raw, 0)                  # T1 against the capture closest in wavelength: a real change?
    api.object_track(raw, 0, 3)                 # how to analyse object 3 of frame_objects(raw, 0) next (the T key)
The website's routes for all of this are in web.py; jobs.py runs the analyses in the background.
"""

import io
import json
import logging
import math
import threading
from contextlib import contextmanager
from datetime import date, datetime

import numpy as np

from . import config as cfg

log = logging.getLogger('Jonah_Code_File')
_run_lock = threading.Lock()                # in one process, one analysis at a time (settings are module-wide)


# ----------------------------------------------------------------------------
# Settings for one analysis
# ----------------------------------------------------------------------------

def _converted(name, value):
    """A JSON-style value in the Python type config.py uses for `name` (units, tuples, sets)."""
    current = getattr(cfg, name)
    if hasattr(current, 'unit') and not hasattr(value, 'unit'):
        return value * current.unit
    if isinstance(current, set) and isinstance(value, (list, tuple, set)):
        return set(value)
    if isinstance(current, tuple) and isinstance(value, list):
        return tuple(value)
    return value


@contextmanager
def settings_override(settings=None):
    """Use these config values (names as in config.json) for the duration of the block, then restore."""
    settings = settings or {}
    unknown = [name for name in settings if not (name.isupper() and hasattr(cfg, name))]
    if unknown:
        raise KeyError(f'Unknown setting(s): {", ".join(unknown)} (see config.json)')
    saved = {name: getattr(cfg, name) for name in settings}
    try:
        for name, value in settings.items():
            setattr(cfg, name, _converted(name, value))
        yield
    finally:
        for name, value in saved.items():
            setattr(cfg, name, value)


@contextmanager
def progress_events(on_progress):
    """Send every progress message to on_progress(message) while the block runs."""
    if on_progress is None:
        yield
        return

    class _Forward(logging.Handler):
        def emit(self, record):
            try:
                on_progress(record.getMessage())
            except Exception:                   # a broken listener must not stop the analysis
                pass

    handler = _Forward(level=logging.INFO)
    log.addHandler(handler)
    try:
        yield
    finally:
        log.removeHandler(handler)


# ----------------------------------------------------------------------------
# Running
# ----------------------------------------------------------------------------

def _run(target, show=False):
    from .main import run, run_settings
    target = dict(target)
    if target.get('max_images') is None and cfg.MAX_IMAGES:
        # run()'s own default was fixed when main.py was imported, so a MAX_IMAGES given in `settings` (as in the
        # example at the top) never reached it: pass the current value along
        target['max_images'] = cfg.MAX_IMAGES
    if 'mode' in target:
        return run_settings(target, show=show)
    allowed = {'asteroid', 'star', 'coords', 'window', 'max_images', 'ra_deg', 'dec_deg'}
    unknown = set(target) - allowed
    if unknown:
        raise KeyError(f'Unknown target field(s): {", ".join(sorted(unknown))}')
    if 'window' in target:
        target['window'] = tuple(target['window'])
    if isinstance(target.get('coords'), list):
        target['coords'] = tuple(target['coords'])
    return run(show=show, **target)


def analyze(target, settings=None, on_progress=None, isolated=False, return_raw=False):
    """Run an analysis without opening any window and return its JSON-ready summary (see to_json).
    return_raw=True: return (summary, raw results) - the raw results are needed for frame_png / frame_objects
    (not available with isolated=True: they stay in the other process)."""
    if isolated:
        if return_raw:
            raise ValueError('return_raw is not available with isolated=True')
        return _analyze_isolated(target, settings, on_progress)
    with _run_lock, settings_override(settings), progress_events(on_progress):
        raw = _run(target)
        summary = to_json(raw)
    return (summary, raw) if return_raw else summary


def _isolated_job(target, settings, queue):
    """(In the separate process.) Runs the analysis, sending progress messages through `queue`."""
    with settings_override(settings), progress_events(queue.put if queue is not None else None):
        return to_json(_run(target))


def _analyze_isolated(target, settings, on_progress):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    manager = multiprocessing.Manager() if on_progress else None
    queue = manager.Queue() if manager else None
    try:
        with ProcessPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_isolated_job, target, settings, queue)
            while True:
                try:
                    return future.result(timeout=0.2) if queue is None else _drain(future, queue, on_progress)
                except TimeoutError:
                    continue
    finally:
        if manager:
            manager.shutdown()


def _drain(future, queue, on_progress):
    """Forward progress messages until the job finishes; then return its result."""
    import queue as queue_module
    while True:
        try:
            on_progress(queue.get(timeout=0.2))
        except queue_module.Empty:
            if future.done():
                while not queue.empty():
                    on_progress(queue.get())
                return future.result()


# ----------------------------------------------------------------------------
# Plain data out
# ----------------------------------------------------------------------------

def jsonable(value):
    """Any value made of numpy / astropy / Python types -> plain JSON types (NaN / inf -> None)."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, 'ra') and hasattr(value, 'dec') and getattr(value, 'isscalar', True):   # SkyCoord
        try:
            return {'ra_deg': float(value.ra.deg), 'dec_deg': float(value.dec.deg)}
        except (TypeError, AttributeError):
            pass
    if hasattr(value, 'unit') and hasattr(value, 'value'):                                  # Quantity
        return {'value': jsonable(np.asarray(value.value).tolist()), 'unit': str(value.unit)}
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    return str(value)


FRAME_FIELDS = ('obs_id', 'date_obs', 'mjd', 'detector', 'wavelength_um', 'flux_mjy', 'flux_err_mjy', 'snr',
                'aperture_correction', 'n_bad_in_aperture', 'object_found', 'object_ra', 'object_dec',
                'object_err_arcsec', 'expected_ra', 'expected_dec', 'track', 'track_sep_px', 'position_method',
                'psf_chi2', 'saturated', 'glow', 'alignment_px', 'rotation_deg', 'astrometry_ok', 'moving_flags')
SUMMARY_ARRAYS = ('offset_arcsec', 'd_ra_arcsec', 'd_dec_arcsec', 'error_arcsec', 'residual_arcsec', 'flagged',
                  'detected', 'deviation', 'reference_mjy', 'used',
                  # what plots.show_time_series draws besides: each frame's own scatter (its flag threshold is
                  # MOVEMENT_SIGMA x that) and a named star's track as the catalog predicts it
                  'scatter_arcsec', 'expected_offset_arcsec', 'expected_d_ra_arcsec', 'expected_d_dec_arcsec')


def _positioned(frames):
    """The frames whose POSITION means something (measure.usable_frames) - as the plots choose them: if the
    object is detected in no single frame, every frame without a bad pixel in the circle."""
    from .measure import reliable_frames, usable_frames
    usable = usable_frames(frames)
    return usable if usable.any() else reliable_frames(frames)


def to_json(results):
    """The JSON-ready summary of main.run(...)'s results: target, per-frame measurements, movement and brightness
    per frame, the spectrum check, and the sky map's objects (NEW ones with every detection's time and place).
    Also, when the run has them: 'star' and 'position' (what a fixed target is, and where), 'detection' (how
    strongly it is detected per wavelength band, frames combined), 'contamination' (other sources inside the
    measuring circle) and 'field_candidates' (sources in no catalog, for a run without a sky map).
    Pixels are not included (see frame_png)."""
    tracked, frames = results['tracked'], results['frames']
    movement, brightness = results.get('movement') or {}, results.get('brightness') or {}
    out = {
        'target': tracked.get('name'),
        'kind': 'moving' if 'ephemeris' in tracked else 'fixed',
        'n_frames': len(frames),
        'frames': [{key: jsonable(f.get(key)) for key in FRAME_FIELDS} | {'T': i + 1} for i, f in enumerate(frames)],
        'movement': {key: jsonable(movement[key]) for key in SUMMARY_ARRAYS if key in movement}
                    | {key: jsonable(movement.get(key)) for key in ('reference_label', 'flag_label', 'notes')
                       if key in movement},
        'brightness': {key: jsonable(brightness[key]) for key in SUMMARY_ARRAYS + ('scatter', 'source')
                       if key in brightness},
    }
    if frames and 'detected' not in movement:               # (a moving object says so itself: 'detected')
        out['movement']['usable'] = jsonable(_positioned(frames))
    star = tracked.get('star')
    if star:                                                # a fixed target: what it is, and where
        out['star'] = jsonable({key: value for key, value in star.items() if key != 'magnitudes'})
        out['position'] = jsonable(tracked.get('coord'))
    if results.get('detection'):
        out['detection'] = jsonable(results['detection'])
    if results.get('contamination'):
        out['contamination'] = [{'name': name, 'separation_arcsec': separation, 'w1_mjy': w1, 'w2_mjy': w2}
                                for name, separation, w1, w2 in results['contamination']]
    if (results.get('field') or {}).get('candidates'):
        out['field_candidates'] = jsonable(results['field']['candidates'])
    spectrum = results.get('spectrum')
    if spectrum:
        out['spectrum'] = jsonable({k: v for k, v in spectrum.items() if k not in ('figure',)})
    if results.get('catalog_motion'):
        out['catalog_motion'] = jsonable(results['catalog_motion'])
    sky = results.get('skymap')
    if sky:
        from astropy.time import Time
        out['new_objects'] = [{
            'id': o['id'], 'class': o['kind'], 'description': o['description'], 'ra_deg': o['ra'], 'dec_deg': o['dec'],
            'track': jsonable(o.get('track')),
            'detections': [{'frame': f"T{m['frame'] + 1}", 'ra_deg': m['ra'], 'dec_deg': m['dec'], 'mjd': m['mjd'],
                            'time_utc': Time(m['mjd'], format='mjd').utc.isot[:23], 'significance': m['sigma']}
                           for m in sorted(o['detections'], key=lambda d: d['mjd'])],
        } for o in sky['new_objects']]
        out['known_small_bodies'] = [{'name': b['fullname'], 'kind': b['kind'], 'orbit_class': b['orbit_class'],
                                      'frames': sorted(f'T{i + 1}' for i in b['positions'])} for b in sky['bodies']]
    return jsonable(out)


# ----------------------------------------------------------------------------
# Images for a browser
# ----------------------------------------------------------------------------

def _display(raw):
    """What frame_png needs that is the same for every frame of a run: which frames are simultaneous partners
    (visual.find_partners) and the target's one colour (visual.anchor_colour). Worked out once and kept with the
    run, in raw['_display'] - it took most of the time of every call, and a web viewer asks for hundreds of frames."""
    from .visual import anchor_colour, find_partners
    zooms = raw['zooms']
    kept = raw.get('_display')
    if kept is None or kept[0] is not zooms:
        partners = find_partners(zooms)
        kept = raw['_display'] = (zooms, partners, anchor_colour(zooms, partners))
    return kept[1], kept[2]


def frame_png(raw, index, pretty=True, wide=False):
    """Frame T(index+1) as a PNG image (bytes): display-only colour, sharpening and upscaling (visual.render),
    with the target drawn at the same brightness and colour in every frame - or the plain original pixels in
    grayscale with pretty=False. wide=True: the wide view (downloaded on demand). North is not necessarily up:
    the image is in the frame's own pixel grid (frame_objects gives positions in the same grid)."""
    import matplotlib.image as mpimg
    from .plots import load_wide
    from .visual import render, target_xy
    zooms = raw['zooms']
    zoom = zooms[index]
    if pretty:
        partners, colour = _display(raw)
        partner = zooms[partners[index]] if partners[index] is not None else None
        factor = None
        if wide:
            zoom = load_wide(zoom)
            partner = load_wide(partner) if partner is not None else None
            factor = cfg.WIDE_VIEW_UPSCALE
        rgb, _ = render(zoom, partner, factor=factor, anchor={'xy': target_xy(zoom), 'colour': colour})
        image = np.flipud(rgb)                  # PNG rows run top to bottom; the frame's run bottom to top
        buffer = io.BytesIO()
        mpimg.imsave(buffer, image, format='png')
    else:
        from .measure import robust_scatter
        if wide:
            zoom = load_wide(zoom)
        pixels = zoom['image'][np.isfinite(zoom['image'])]
        sky, noise = np.median(pixels), robust_scatter(pixels)
        buffer = io.BytesIO()
        mpimg.imsave(buffer, np.flipud(zoom['image']), format='png', cmap='gray', vmin=sky - 3 * noise,
                     vmax=sky + 15 * noise)
    return buffer.getvalue()


def frame_objects(raw, index):
    """What the sky map found in frame T(index+1), with positions in that frame's (zoom) pixel grid - x to the
    right, y DOWN from the top, in original pixels (multiply by the upscale factor for frame_png's pixels).
    Plus the target: where it was expected and where it was found."""
    zoom = raw['zooms'][index]
    ny = zoom['image'].shape[0]
    wcs = zoom['wcs']

    def place(ra, dec):
        x, y = (float(v) for v in wcs.world_to_pixel_values(ra, dec))
        return {'x': x + 0.5, 'y': ny - (y + 0.5), 'ra_deg': float(ra), 'dec_deg': float(dec)}

    objects = [dict(place(e['ra'], e['dec']), kind=e['kind'], label=e['label'], detected=e.get('detected'),
                    info=e.get('info'), new_id=e.get('new_id')) for e in zoom.get('map', [])]
    target = {'expected': place(zoom['expected_ra'], zoom['expected_dec'])}
    if zoom.get('object_found'):
        target['found'] = place(zoom['object_ra'], zoom['object_dec'])
    return jsonable({'frame': f'T{index + 1}', 'image_size': [zoom['image'].shape[1], ny], 'target': target,
                     'objects': objects})


# ----------------------------------------------------------------------------
# More for a viewer of a finished run: the spectrum's channels, comparing captures, tracking another object
# ----------------------------------------------------------------------------

def band_spectrum(frames):
    """The target's spectrum in SPHEREx's 102 nominal channels (spectrumanalyser.py): the weighted mean brightness
    of the reliable frames in each channel, None for a channel without one.
    {'wavelength_um', 'flux_mjy', 'flux_err_mjy', 'count'}, 102 values each. `frames` = a run's results['frames']
    (any dicts with obs_id, detector, mjd, wavelength_um, flux_mjy, flux_err_mjy, n_bad_in_aperture)."""
    from .spectrumanalyser import SpectrumAnalyzer
    return jsonable(SpectrumAnalyzer(frames).band_spectrum())


def compare_frames(raw, index, other=None):
    """Frame T(index+1) against another capture (spectrumanalyser.SpectrumAnalyzer.compare): is a difference in
    brightness explained by the two wavelengths, or is it a real change - or a glitch? other=None: the capture
    closest in wavelength taken at another time (find_comparison_partner). Returns the comparison, plus
    'partner' (the other frame's index) - or None when there is no other frame to compare with."""
    from .spectrumanalyser import SpectrumAnalyzer
    analyzer = SpectrumAnalyzer.from_results(raw)
    if other is None:
        other = analyzer.find_comparison_partner(index)
    if other is None:
        return None
    return jsonable(dict(analyzer.compare(index, other), partner=int(other)))


def track_settings(spec):
    """A viewer's track request (plots.track_spec, or a new object's 'track' in to_json) as a target for
    analyze(): the same four kinds main.run_track runs. Raises ValueError with the reason for an object that
    can't be tracked (a detector artifact, a planet)."""
    kind = spec.get('kind')
    if kind == 'star':
        return {'mode': 'catalog star', 'star': spec['star']}
    if kind == 'asteroid':
        return {'mode': 'asteroid', 'asteroid': str(spec['id']), 'window': [str(day) for day in spec['window']]}
    if kind == 'coords':
        return {'mode': 'coordinates', 'coords': [float(value) for value in spec['coords']]}
    if kind == 'path':
        return {'mode': 'path', 'name': str(spec['name']), 'mjd': [float(v) for v in spec['mjd']],
                'ra': [float(v) for v in spec['ra']], 'dec': [float(v) for v in spec['dec']]}
    raise ValueError(f"Can't track {spec.get('name', 'this object')}: {spec.get('reason', 'not supported')}")


def object_track(raw, index, number):
    """How to analyse object `number` of frame_objects(raw, index)['objects'] next - what selecting it and
    pressing T does in the desktop viewer - as a target for analyze(). Raises ValueError (see track_settings)."""
    from .plots import track_spec
    zoom = raw['zooms'][index]
    entries = zoom.get('map', [])
    if not 0 <= number < len(entries):
        raise IndexError(f'T{index + 1} has no object {number}')
    return jsonable(track_settings(track_spec(entries[number], raw['tracked'], zoom['mjd'])))


# ----------------------------------------------------------------------------
# What was analysed before (the results folder): nothing is run, nothing is downloaded
# ----------------------------------------------------------------------------

def target_key(settings):
    """What identifies a target, however it was written - to find its saved results, or an analysis of it that is
    already under way. A sky position is identified by its name (= its results file, see main.coords_name)."""
    mode = settings.get('mode')
    text = lambda value: str(value).lower().replace(' ', '')
    if mode == 'coordinates':
        from .main import coords_name, parse_coords
        return mode, coords_name(parse_coords(settings['coords']))
    if mode == 'star':
        return mode, round(float(settings['ra_deg']), 6), round(float(settings['dec_deg']), 6)
    if mode == 'asteroid':
        return mode, text(settings['asteroid']), tuple(str(day) for day in settings.get('window') or ())
    if mode == 'named star':
        return mode, text(settings['star'])
    if mode == 'catalog star':
        return mode, text(settings['star'].get('name'))
    if mode == 'path':
        return mode, text(settings.get('name'))
    return mode, text(sorted(settings.items()))


def _same_target(a, b):
    """The same target, however it was written (the rule the start window uses: launcher._entries)."""
    key = lambda s: (s.get('mode'), str(s.get('star') or s.get('asteroid') or s.get('coords')).lower().replace(' ', ''))
    return key(a) == key(b)


def known_targets():
    """Every target the start window offers, as plain data: the analysed ones (results folder + registry, most
    recent first), then the standard test objects that haven't been analysed yet.
    [{'name', 'mode', 'settings', 'frames', 'saved', 'has_results', 'analysed'}]"""
    from .results import DEFAULT_OBJECTS, known_objects
    targets = [{'name': obj['name'], 'mode': obj['settings'].get('mode'), 'settings': obj['settings'],
                'frames': obj.get('frames'), 'saved': obj.get('saved') or None,
                'has_results': bool(obj.get('has_results')), 'analysed': True} for obj in known_objects()]
    for name, settings in DEFAULT_OBJECTS:
        if not any(t['name'] == name or _same_target(t['settings'], settings) for t in targets):
            targets.append({'name': name, 'mode': settings['mode'], 'settings': settings, 'frames': None,
                            'saved': None, 'has_results': False, 'analysed': False})
    return jsonable(targets)


def saved_target(settings, targets=None):
    """The known target (see known_targets) that holds the saved results of a run with these settings, or None.
    An asteroid is the same one whatever the time window was; `targets` = known_targets(), if already read."""
    key = target_key(settings)[:2]
    for target in known_targets() if targets is None else targets:
        if not target['has_results']:
            continue
        try:
            if target_key(target['settings'])[:2] == key:
                return target
        except (KeyError, TypeError, ValueError):       # settings written by hand that can't be read: not this one
            continue
    return None


def _plain_column(column):
    """A results-table column as a plain list, None for a missing value: NaN, an empty cell, or the text 'None' /
    'nan' (how results.save_results writes a missing value in a text column)."""
    data = np.asarray(column)
    if data.dtype.kind in 'US':
        values = [None if text in ('None', 'nan', '') else text for text in data.tolist()]
    else:
        values = jsonable(data)
    if np.ma.isMaskedArray(column) and np.ma.getmaskarray(column).any():
        values = [None if hidden else value for value, hidden in zip(values, np.ma.getmaskarray(column).tolist())]
    return values


def _saved_frames(table):
    """The rows of a saved results table as frame dicts holding what the checks read (measure.usable_frames,
    spectrumanalyser): numbers as numbers, NaN where there is none."""
    n = len(table)

    def numbers(name, default=np.nan):
        if name not in table.colnames:
            return np.full(n, default)
        return np.ma.filled(np.ma.asarray(table[name], dtype=float), np.nan)

    def texts(name, default):
        return [str(value) for value in table[name]] if name in table.colnames else [default] * n

    columns = zip(texts('obs_id', ''), numbers('detector', 0), numbers('mjd'), numbers('wavelength_um'),
                  numbers('flux_mjy'), numbers('flux_err_mjy'), numbers('snr'), numbers('n_bad_in_aperture', 0),
                  texts('track', 'ok'), texts('position_method', 'centroid'))
    return [{'obs_id': obs_id, 'detector': int(detector) if np.isfinite(detector) else 0, 'mjd': float(mjd),
             'wavelength_um': float(wavelength), 'flux_mjy': float(flux), 'flux_err_mjy': float(error),
             'snr': float(snr), 'n_bad_in_aperture': int(bad) if np.isfinite(bad) else 0, 'track': track,
             'position_method': method}
            for obs_id, detector, mjd, wavelength, flux, error, snr, bad, track, method in columns]


def _saved_new_objects(path):
    """The new objects of a saved analysis - <name>_new_objects.csv next to its results file `path`, one row per
    detection - as to_json lists them. None if that run made no sky map."""
    from astropy.table import Table
    csv = path.with_name(path.stem + '_new_objects.csv')
    if not csv.is_file():
        return None
    rows = Table.read(csv, format='ascii.csv')
    described = 'description' in rows.colnames
    objects = {}
    for row in rows:
        obj = objects.setdefault(str(row['object']), {
            'id': str(row['object']), 'class': str(row['class']),
            'description': str(row['description']) if described else '', 'detections': []})
        obj['detections'].append({'frame': str(row['frame']), 'ra_deg': float(row['ra_deg']),
                                  'dec_deg': float(row['dec_deg']), 'mjd': float(row['mjd']),
                                  'time_utc': str(row['time_utc']), 'significance': float(row['significance'])})
    for obj in objects.values():
        ra = np.array([d['ra_deg'] for d in obj['detections']])
        obj['ra_deg'] = float((ra[0] + np.median((ra - ra[0] + 180) % 360 - 180)) % 360)     # (safe across RA 0)
        obj['dec_deg'] = float(np.median([d['dec_deg'] for d in obj['detections']]))
        # a moving object's fitted path isn't in the file; anything else is tracked where it is
        obj['track'] = (None if obj['class'] == 'moving' else
                        {'kind': 'coords', 'coords': [obj['ra_deg'], obj['dec_deg']], 'name': obj['id']})
    return list(objects.values())


def saved_summary(name):
    """The summary of a SAVED analysis - results/<name>.ecsv, written by an earlier run - in the form to_json
    gives, without running anything: per-frame measurements, movement and brightness per frame, the spectrum's
    102 channels ('band_spectrum') and the new objects (from <name>_new_objects.csv). Plus 'settings' (how it was
    run), 'saved' (when) and 'file'. The pixels were never saved: frame_png / frame_objects need a run (quick
    while the images are still in the disk cache). Raises FileNotFoundError without saved results for `name`."""
    from astropy.table import Table
    from .measure import robust_scatter
    from .results import _settings_from_meta, results_path
    path = results_path(name)
    if not path.is_file():
        raise FileNotFoundError(f'No saved results for "{name}"')
    table = Table.read(path, format='ascii.ecsv')
    meta, n = table.meta, len(table)
    settings = json.loads(meta['settings_json']) if 'settings_json' in meta else _settings_from_meta(meta) or {}
    columns = {column: _plain_column(table[column]) for column in table.colnames}
    fields = [key for key in FRAME_FIELDS if key in columns]
    movement = {key: columns['movement_flagged' if key == 'flagged' else key] for key in SUMMARY_ARRAYS
                if ('movement_flagged' if key == 'flagged' else key) in columns}
    brightness = {key: columns[f'brightness_{key}'] for key in SUMMARY_ARRAYS if f'brightness_{key}' in columns}
    measured = _saved_frames(table)
    if n and 'detected' not in movement:
        movement['usable'] = jsonable(_positioned(measured))
    if 'deviation' in brightness and 'used' in brightness:      # (as measure.brightness_changes works it out)
        used = np.array(brightness['used'], dtype=bool)
        deviation = np.array([np.nan if value is None else value for value in brightness['deviation']], dtype=float)
        brightness['scatter'] = jsonable(robust_scatter(deviation[used]))
    out = {'target': str(meta.get('target', name)),
           'kind': 'moving' if settings.get('mode', meta.get('mode')) in ('asteroid', 'path') else 'fixed',
           'n_frames': n,
           'frames': [{key: columns[key][i] for key in fields} | {'T': i + 1} for i in range(n)],
           'movement': movement, 'brightness': brightness, 'band_spectrum': band_spectrum(measured),
           'settings': jsonable(settings), 'saved': meta.get('saved'), 'file': path.name}
    if out['kind'] == 'fixed' and n and columns.get('expected_ra') and columns['expected_ra'][0] is not None:
        out['position'] = {'ra_deg': columns['expected_ra'][0], 'dec_deg': columns['expected_dec'][0]}
    new_objects = _saved_new_objects(path)
    if new_objects is not None:
        out['new_objects'] = new_objects
    return jsonable(out)


def saved_candidates():
    """Every NEW object in the saved results - a source in no catalog, not a known asteroid, comet or planet, not
    a flagged artifact (see skymap.py) - over all targets: [{'target', 'id', 'class', 'description', 'ra_deg',
    'dec_deg', 'track', 'detections': [...]}]."""
    from .results import known_objects, results_path
    candidates = []
    for obj in known_objects():
        if obj.get('has_results'):
            for new in _saved_new_objects(results_path(obj['name'])) or []:
                candidates.append(dict(new, target=obj['name']))
    return jsonable(candidates)


MOTION_FIT_FIELDS = ('target', 'mode', 'model', 'motion_significance', 'is_moving_candidate',
                     'physical_orbit_available', 'proper_motion_mas_per_yr', 'proper_motion_err_mas_per_yr',
                     'parallax_mas', 'parallax_err_mas', 'parallax_fixed', 'parallax_reliable', 'reduced_chi2',
                     'n_visits', 'velocity_arcsec_per_day', 'acceleration_arcsec_per_day2', 'fit_degree')
MOTION_PATH_POINTS = 240            # a path is sent as about this many points (a star's fit has thousands)


def motion_fits(prediction_days=0.0):
    """The apparent path of every saved target: algorithm.run_from_results on the results folder (which also
    writes the paths to results/estimated_paths/estimated_paths.ecsv). JSON-ready:
    {'estimates': {target: {model, motion_significance, is_moving_candidate, ... 'path': {mjd, ra_deg, dec_deg,
    extrapolated}}}, 'moving_candidates': [targets], 'skipped': {file: why}, 'output_file'}.
    Raises FileNotFoundError if there are no saved results."""
    from .algorithm import run_from_results
    from .results import results_dir
    found = run_from_results(results_dir(), prediction_days)
    estimates = {}
    for estimate in found['estimates'].values():
        times = np.asarray(estimate['predicted_times_mjd'], dtype=float)
        observed = np.asarray(estimate['observation_times_mjd'], dtype=float)
        keep = np.unique(np.r_[np.arange(0, len(times), max(1, len(times) // MOTION_PATH_POINTS)), len(times) - 1])
        entry = {key: estimate[key] for key in MOTION_FIT_FIELDS if key in estimate}
        entry.update(reference=estimate['reference_coord'], n_observations=len(observed),
                     observed_mjd=[observed.min(), observed.max()],
                     path={'mjd': times[keep], 'ra_deg': np.asarray(estimate['predicted_ra_deg'])[keep],
                           'dec_deg': np.asarray(estimate['predicted_dec_deg'])[keep],
                           'extrapolated': times[keep] > observed.max()})
        estimates[estimate['target']] = entry
    return jsonable({'estimates': estimates,
                     'moving_candidates': [found['estimates'][stem]['target'] for stem in found['moving_candidates']],
                     'skipped': found['skipped'], 'prediction_days': float(prediction_days),
                     'output_file': str(found['output_path']) if found['output_path'] else None})
