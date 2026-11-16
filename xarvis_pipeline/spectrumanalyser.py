"""Pairwise diagnostics and interactive 102-channel SPHEREx visualization.

Standalone: python -m xarvis_pipeline.spectrumanalyser
Library: SpectrumAnalyzer.from_results(main.run(show=False)).show_bands()
Only this module needs replacing. No tracking or download logic is changed.
"""

import textwrap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from . import config as cfg
from .measure import reliable_frames, robust_scatter


class SpectrumAnalyzer:
    """Analyze and compare the spectral measurements of any two captures in a dataset.

    Initialized from `frames` (and optional `brightness` / `spectrum` dicts from `measure.py`),
    and optionally retained zooms and tracking results for the interactive viewer.
    """

    def __init__(self, frames, brightness=None, spectrum=None, *,
                 zooms=None, movement=None, tracked=None):
        self.frames = frames
        self.brightness = brightness
        self.spectrum = spectrum
        self.zooms = zooms
        self.movement = movement
        self.tracked = tracked

        self.reliable = reliable_frames(frames)
        self.wavelength = np.array([f['wavelength_um'] for f in frames], dtype=float)
        self.flux = np.array([f['flux_mjy'] for f in frames], dtype=float)
        self.flux_err = np.array([f['flux_err_mjy'] for f in frames], dtype=float)
        self.mjd = np.array([f['mjd'] for f in frames], dtype=float)
        # Both detector arms of one exposure share an obs_id, so an obs_id alone can be ambiguous:
        # '<obs_id>/D<detector>' always names one frame; a bare obs_id only when it is unique.
        self._obs_to_idx, ambiguous = {}, set()
        for i, f in enumerate(frames):
            self._obs_to_idx[f"{f['obs_id']}/D{f['detector']}"] = i
            key = str(f['obs_id'])
            if key in self._obs_to_idx:
                ambiguous.add(key)
            self._obs_to_idx[key] = i
        self._ambiguous_obs = ambiguous
        for key in ambiguous:
            del self._obs_to_idx[key]

        # Precompute sorted reliable spectrum points for interpolation
        ok = (self.reliable & np.isfinite(self.wavelength) & np.isfinite(self.flux)
              & np.isfinite(self.flux_err) & (self.flux_err > 0))
        order = np.argsort(self.wavelength[ok])
        self.spec_wl = self.wavelength[ok][order]
        self.spec_flux = self.flux[ok][order]
        self.spec_err = self.flux_err[ok][order]

    @classmethod
    def from_results(cls, results):
        """Convenience constructor accepting the dict returned by `main.run()`."""
        analyzer = cls(
            frames=results['frames'],
            brightness=results.get('brightness'),
            spectrum=results.get('spectrum'),
            zooms=results.get('zooms'),
            movement=results.get('movement'),
            tracked=results.get('tracked'),
        )
        analyzer.catalog_motion = results.get('catalog_motion')
        analyzer.combined_detection = results.get('detection')
        return analyzer

    def band_spectrum(self, indices=None):
        """A 102-channel target aperture spectrum, retaining empty channels as NaN.

        Optional indices: 0-based frame indices or a boolean mask. No new downloads.
        Returns wavelength_um, flux_mjy, flux_err_mjy and count arrays of length 102.
        """
        selected = np.ones(len(self.frames), bool)
        if indices is not None:
            selected[:] = False
            ix = np.asarray(indices)
            if ix.dtype == bool:
                if ix.shape != selected.shape:
                    raise ValueError("Boolean mask must match frame count")
                selected = ix.copy()
            elif ix.size:
                if ix.ndim != 1 or not np.issubdtype(ix.dtype, np.integer):
                    raise TypeError("Use integer indices or a boolean mask")
                if np.any(ix < 0) or np.any(ix >= len(selected)):
                    raise IndexError("Frame indices out of range")
                selected[ix] = True
        detector = np.array([f['detector'] for f in self.frames])
        mean, error, count = binned_spectrum(self.wavelength, self.flux,
            self.flux_err, detector, selected & self.reliable)
        return dict(wavelength_um=CENTERS.ravel().copy(), flux_mjy=mean,
                    flux_err_mjy=error, count=count)

    def observation_groups(self, tolerance_seconds=5.0):
        """Group near-simultaneous detector captures in chronological order.

        Same five-second convention as visual.find_partners; compare every time
        to the first capture in its group, not to the preceding capture. Missing
        timestamps remain singleton observations. Original frame indices survive.
        """
        if not np.isfinite(tolerance_seconds) or tolerance_seconds <= 0:
            raise ValueError('tolerance_seconds must be positive and finite')
        groups = []
        for i in np.argsort(self.mjd, kind='stable'):
            if (groups and np.isfinite(self.mjd[i]) and
                    (self.mjd[i] - self.mjd[groups[-1][0]]) * 86400 < tolerance_seconds):
                groups[-1].append(int(i))
            else:
                groups.append([int(i)])
        return [np.asarray(g, dtype=int) for g in groups]

    def reference_mask(self, idx, indices=None):
        """Usable reference frames, excluding the entire selected observation time.

        `indices` is an optional boolean mask (e.g. the chosen reference days).
        Exclude poor pointing and known bad aperture pixels. Do not require a
        detection for each reference frame: that would bias faint-source fluxes.
        """
        idx = self.resolve_index(idx)
        align = np.array([f.get('alignment_px', np.nan) for f in self.frames], float)
        ok = (self.reliable & np.isfinite(self.wavelength) & (self.wavelength > 0)
              & np.isfinite(self.flux) & np.isfinite(self.flux_err)
              & (self.flux_err > 0) & np.isfinite(self.mjd)
              & (self.mjd != self.mjd[idx])
              & ~(np.isfinite(align) & (align > cfg.ALIGNMENT_TOLERANCE_PIX)))
        if indices is not None:
            mask = np.asarray(indices, dtype=bool)
            if mask.shape != ok.shape:
                raise ValueError('Reference mask must match the number of captures.')
            ok &= mask
        return ok

    def capture_consistency(self, idx, reference_mask=None, *, epoch_ids=None):
        """Secondary, independent spectral brightness check for one capture.

        Compare with other observation times in the same detector and nominal
        channel, within half a local channel spacing. Multiple captures at one
        exact MJD (or optional grouped epoch ID) contribute only one median.
        At least three other times are
        required. The typical range is a heuristic: +/- 3 times the quadrature
        combination of selected uncertainty and the larger of reference scatter
        or typical reference measurement error. This is NOT an identity test or
        a calibrated probability. Real variability, blends and systematics can
        produce disagreements. A weak measurement cannot establish agreement.
        """
        idx = self.resolve_index(idx)
        f = self.frames[idx]
        out = dict(status='insufficient', title='Not enough information',
                   reason='', n_epochs=0, reference_mjy=np.nan, lower_mjy=np.nan,
                   upper_mjy=np.nan, deviation_sigma=np.nan)
        def finish(status, title, reason):
            out.update(status=status, title=title, reason=reason)
            return out
        if not self.reliable[idx]:
            return finish('quality', 'Check image quality',
                          'A bad pixel affects this brightness measurement.')
        if (not np.isfinite(self.flux[idx]) or not np.isfinite(self.flux_err[idx])
                or self.flux_err[idx] <= 0 or not np.isfinite(self.mjd[idx])):
            return finish('quality', 'Check image quality',
                          'The selected measurement or its uncertainty is unavailable.')
        alignment = f.get('alignment_px', np.nan)
        if np.isfinite(alignment) and alignment > cfg.ALIGNMENT_TOLERANCE_PIX:
            return finish('quality', 'Check image alignment',
                          'The telescope pointing may affect this measurement.')
        d = int(f['detector'])
        ch = int(channel_index(d, self.wavelength[idx]))
        if ch < 0:
            return finish('insufficient', 'Not enough information',
                          'No nominal wavelength channel is available for this capture.')
        local = ch % 17
        centers = CENTERS[d-1]
        spacing = min(np.diff(centers)[max(0,local-1):min(16,local+1)])
        det = np.array([f['detector'] for f in self.frames])
        ok = (self.reference_mask(idx, reference_mask) & (det == d)
              & (channel_index(d, self.wavelength) == ch)
              & (np.abs(self.wavelength-self.wavelength[idx]) <= spacing/2))
        times = self.mjd if epoch_ids is None else np.asarray(epoch_ids)
        if times.shape != self.mjd.shape:
            raise ValueError('epoch_ids must match frame count')
        epochs = np.unique(times[ok])
        out['n_epochs'] = len(epochs)
        if len(epochs) < 3:
            return finish('insufficient', 'Not enough information',
                          f'{len(epochs)} comparable observation times; need at least 3.')
        epoch_flux = np.array([np.median(self.flux[ok & (times == t)]) for t in epochs])
        epoch_err = np.array([np.median(self.flux_err[ok & (times == t)]) for t in epochs])
        typical = float(np.median(epoch_flux))
        spread = max(float(robust_scatter(epoch_flux)), float(np.median(epoch_err)))
        noise = float(np.hypot(self.flux_err[idx], spread))
        sig = float(abs(self.flux[idx]-typical)/noise)
        out.update(reference_mjy=typical, lower_mjy=typical-3*noise,
                   upper_mjy=typical+3*noise, deviation_sigma=sig)
        if typical < 3*spread:
            return finish('insufficient', 'Not enough information',
                          'Other observations are too faint or variable for a clear check.')
        if sig > 3:
            return finish('review', 'Worth another look',
                          f'Brightness differs from {len(epochs)} comparable observation times.')
        if self.flux[idx] < 3*self.flux_err[idx]:
            return finish('insufficient', 'Not enough information',
                          'This capture is too faint for a confident comparison.')
        return finish('consistent', 'Brightness looks consistent',
                      f'Agrees with {len(epochs)} comparable observation times.')

    def show_bands(self, zooms=None, movement=None, tracked=None, *, size=41):
        """Open linked channel gallery and capture/spectrum/tracking detail.

        Pass results['zooms'], results['movement'], results['tracked'] from run().
        Returns the viewer; call plt.show() to display its figures.
        """
        zooms = self.zooms if zooms is None else zooms
        movement = self.movement if movement is None else movement
        tracked = self.tracked if tracked is None else tracked
        if zooms is None or movement is None:
            raise ValueError('Use from_results(results), or pass zooms and movement explicitly.')
        return BandViewer(self, zooms, movement, tracked, size=size)

    def motion_fit(self, prediction_days=0.0, show_plot=False):
        """Optional apparent trajectory fit delegated to the new algorithm.py."""
        from .algorithm import orbital_prediction
        return orbital_prediction(self.frames, prediction_days=prediction_days,
                                  show_plot=show_plot)

    def spectrum_for_time(self, start_mjd, end_mjd):
        """Weighted target spectrum for an inclusive MJD interval."""
        if not np.isfinite([start_mjd, end_mjd]).all() or start_mjd > end_mjd:
            raise ValueError('Provide finite MJD bounds in increasing order.')
        return self.band_spectrum((self.mjd >= start_mjd) & (self.mjd <= end_mjd))

    def resolve_index(self, key):
        """Accept an integer frame index (0-based), an `obs_id` string, or '<obs_id>/D<detector>' (needed when
        both detector arms of that exposure are in the data - they share the obs_id)."""
        if isinstance(key, (int, np.integer)):
            if not (0 <= key < len(self.frames)):
                raise IndexError(f'Frame index {key} out of range (0..{len(self.frames) - 1})')
            return int(key)
        if str(key) in self._obs_to_idx:
            return self._obs_to_idx[str(key)]
        if str(key) in self._ambiguous_obs:
            options = [k for k in self._obs_to_idx if k.startswith(f'{key}/D')]
            raise KeyError(f'obs_id {key} names {len(options)} frames (both detector arms): use one of {options}')
        raise KeyError(f'Unknown obs_id: {key}')

    def expected_flux_at(self, wavelength_um, exclude_idx=None):
        """Estimate expected flux (mJy) at `wavelength_um` using:
        1. Local median of reliable SPHEREx neighbors within +/- SPECTRUM_WINDOW (matching `brightness_changes`),
        2. Fallback to scaled 2MASS/AllWISE catalog SED interpolation if `spectrum` is available.
        Returns (expected_mjy, source_label).
        """
        # 1. Local SPHEREx neighbors (same logic as measure.brightness_changes)
        good = np.flatnonzero(self.reliable & np.isfinite(self.wavelength)
                              & np.isfinite(self.flux) & np.isfinite(self.flux_err)
                              & (self.flux_err > 0))
        if exclude_idx is not None:
            good = good[good != exclude_idx]

        if len(good):
            dist = np.abs(self.wavelength[good] - wavelength_um)
            candidates = good[dist <= cfg.SPECTRUM_WINDOW * wavelength_um]
            candidates = candidates[np.argsort(np.abs(self.wavelength[candidates] - wavelength_um))[:cfg.SPECTRUM_NEIGHBORS]]
            if len(candidates) >= cfg.SPECTRUM_MIN_NEIGHBORS:
                return float(np.median(self.flux[candidates])), 'spherex_neighbors'

        # 2. Fallback: log-log interpolation of scaled 2MASS/AllWISE catalog SED
        if self.spectrum and self.spectrum.get('catalog'):
            cat = sorted(self.spectrum['catalog'], key=lambda p: p[1])
            cat_wl = np.array([p[1] for p in cat], dtype=float)
            cat_flux = np.array([p[2] for p in cat], dtype=float)
            if cat_wl.min() <= wavelength_um <= cat_wl.max():
                scale = float(self.spectrum.get('scale', 1.0))
                interp_mjy = float(np.exp(np.interp(np.log(wavelength_um), np.log(cat_wl), np.log(cat_flux))) * scale)
                return interp_mjy, 'scaled_catalog_sed'

        return np.nan, 'unavailable'

    def frame_spectral_profile(self, idx):
        """Return spectral metrics for a single frame `idx`."""
        idx = self.resolve_index(idx)
        f = self.frames[idx]
        wl = float(f['wavelength_um'])
        flux = float(f['flux_mjy'])
        err = float(f['flux_err_mjy'])

        # Use precomputed reference from `brightness_changes` if available, else compute on the fly
        if self.brightness is not None and np.isfinite(self.brightness['reference_mjy'][idx]):
            ref_mjy = float(self.brightness['reference_mjy'][idx])
            ref_source = 'spherex_neighbors'
        else:
            ref_mjy, ref_source = self.expected_flux_at(wl, exclude_idx=idx)

        if np.isfinite(ref_mjy) and ref_mjy > 0:
            deviation = (flux / ref_mjy) - 1.0
            scatter = float(self.brightness['scatter']) if self.brightness is not None else robust_scatter(
                self.flux[self.reliable]
            ) / max(np.nanmedian(self.flux[self.reliable]), 1e-9)
            total_noise = float(np.hypot(err / ref_mjy, scatter))
            sigma_dev = float(deviation / total_noise) if total_noise > 0 else np.nan
        else:
            deviation, sigma_dev = np.nan, np.nan

        is_flagged = bool(self.brightness['flagged'][idx]) if self.brightness is not None else (
            np.isfinite(sigma_dev) and abs(sigma_dev) > cfg.BRIGHTNESS_SIGMA
        )
        align_px = f.get('alignment_px', np.nan)
        aligned_ok = bool(np.isnan(align_px) or align_px <= cfg.ALIGNMENT_TOLERANCE_PIX)

        return {
            'index': idx,
            'label': f'T{idx + 1}',
            'obs_id': str(f['obs_id']),
            'date_obs': f['date_obs'],
            'mjd': float(f['mjd']),
            'detector': int(f['detector']),
            'wavelength_um': wl,
            'flux_mjy': flux,
            'flux_err_mjy': err,
            'reference_mjy': ref_mjy if np.isfinite(ref_mjy) else None,
            'reference_source': ref_source,
            'pct_deviation': round(float(deviation * 100), 2) if np.isfinite(deviation) else None,
            'sigma_deviation': round(float(sigma_dev), 2) if np.isfinite(sigma_dev) else None,
            'is_reliable': bool(self.reliable[idx]),
            'n_bad_in_aperture': int(f['n_bad_in_aperture']),
            'is_aligned': aligned_ok,
            'is_brightness_flagged': is_flagged,
        }

    def compare(self, idx_a, idx_b):
        """Compare two captures `idx_a` and `idx_b` (by integer index or `obs_id`).

        Separates apparent brightness changes caused by observing at two different wavelengths
        from genuine astrophysical variability or detector artifacts.
        """
        a = self.frame_spectral_profile(idx_a)
        b = self.frame_spectral_profile(idx_b)

        delta_days = b['mjd'] - a['mjd']
        delta_wl = b['wavelength_um'] - a['wavelength_um']
        rel_wl_diff = abs(delta_wl) / max(a['wavelength_um'], 1e-9)
        same_wavelength_band = bool(rel_wl_diff <= cfg.SPECTRUM_WINDOW)

        raw_ratio = (b['flux_mjy'] / a['flux_mjy']) if a['flux_mjy'] > 0 else np.nan
        raw_diff_mjy = b['flux_mjy'] - a['flux_mjy']
        raw_diff_err = float(np.hypot(a['flux_err_mjy'], b['flux_err_mjy']))

        # Expected flux ratio purely from the object's baseline spectrum shape
        if (a['reference_mjy'] is not None and a['reference_mjy'] > 0
                and b['reference_mjy'] is not None and b['reference_mjy'] > 0):
            expected_ratio = b['reference_mjy'] / a['reference_mjy']
            # Excess ratio = 1.0 means the brightness difference is 100% explained by wavelength
            spectral_excess_ratio = raw_ratio / expected_ratio
        else:
            expected_ratio = np.nan
            spectral_excess_ratio = np.nan

        # Direct statistical significance when in the same wavelength window
        same_band_sigma = (
            abs(raw_diff_mjy) / raw_diff_err if (same_wavelength_band and a['flux_err_mjy'] > 0 and b['flux_err_mjy'] > 0 and raw_diff_err > 0 and a['index'] != b['index']) else np.nan
        )

        status, summary = self._classify_pair(a, b, same_wavelength_band, same_band_sigma, spectral_excess_ratio)

        return {
            'capture_a': a,
            'capture_b': b,
            'comparison': {
                'delta_days': round(float(delta_days), 4),
                'delta_wavelength_um': round(float(delta_wl), 4),
                'same_wavelength_band': same_wavelength_band,
                'raw_flux_diff_mjy': round(float(raw_diff_mjy), 3),
                'raw_flux_ratio': round(float(raw_ratio), 3) if np.isfinite(raw_ratio) else None,
                'expected_spectral_ratio': round(float(expected_ratio), 3) if np.isfinite(expected_ratio) else None,
                'spectral_excess_ratio': round(float(spectral_excess_ratio), 3) if np.isfinite(spectral_excess_ratio) else None,
                'same_band_sigma': round(float(same_band_sigma), 2) if np.isfinite(same_band_sigma) else None,
                'status_code': status,
                'plain_english_summary': summary,
            },
        }

    def find_comparison_partner(self, idx, prefer_normal=True):
        """Find the best second capture to compare against `idx`:
        picks the capture closest in wavelength from a different timestamp (preferring unflagged, reliable frames).
        """
        idx = self.resolve_index(idx)
        target_wl = self.wavelength[idx]
        candidates = [i for i in range(len(self.frames)) if i != idx and self.reliable[i] and self.mjd[i] != self.mjd[idx]
                      and np.isfinite(self.wavelength[i]) and np.isfinite(self.flux[i])
                      and np.isfinite(self.flux_err[i]) and self.flux_err[i] > 0]
        if prefer_normal and self.brightness is not None:
            normal = [i for i in candidates if not self.brightness['flagged'][i]]
            if normal:
                candidates = normal
        if not candidates:
            return None
        return int(min(candidates, key=lambda i: abs(self.wavelength[i] - target_wl)))

    def _classify_pair(self, a, b, same_band, same_band_sigma, excess_ratio):
        if not a['is_reliable'] or not b['is_reliable']:
            bad_labels = [x['label'] for x in (a, b) if not x['is_reliable']]
            return (
                'detector_glitch',
                f"{' and '.join(bad_labels)} had a bad pixel or cosmic-ray spike inside the measurement circle.",
            )

        if not a['is_aligned'] or not b['is_aligned']:
            return (
                'alignment_warning',
                'Telescope pointing offset exceeded tolerance in one capture; brightness comparison may be affected.',
            )

        if a['is_brightness_flagged'] or b['is_brightness_flagged'] or (
            np.isfinite(same_band_sigma) and same_band_sigma >= cfg.BRIGHTNESS_SIGMA
        ):
            return (
                'real_variability',
                f"Candidate significant brightness variation involving {a['label']} ({a['flux_mjy']:.2f} mJy) and "
                f"{b['label']} ({b['flux_mjy']:.2f} mJy); inspect the observations before attributing a cause.",
            )

        if not same_band:
            dir_word = 'brighter' if b['flux_mjy'] > a['flux_mjy'] else 'dimmer'
            return (
                'wavelength_shift',
                f"{b['label']} appears {dir_word} than {a['label']} ({b['flux_mjy']:.2f} vs. {a['flux_mjy']:.2f} mJy) "
                f"while SPHEREx observed at different wavelengths ({b['wavelength_um']:.2f} µm vs. {a['wavelength_um']:.2f} µm), "
                f"the wavelength difference may contribute; variability is not ruled out.",
            )

        return (
            'consistent',
            f"{a['label']} and {b['label']} were observed at matching wavelengths (~{a['wavelength_um']:.2f} µm) "
            f"and show consistent brightness.",
        )


def compare_captures(results, idx_a, idx_b):
    """Functional shortcut to compare two frames directly from `main.run()` output."""
    return SpectrumAnalyzer.from_results(results).compare(idx_a, idx_b)


# SPHEREx Explanatory Supplement v2.1, Tables 29–34 (2026-11-10).
# Rounded median wavelengths, not the geometric detector channel map.
CENTERS = np.array([
[.753,.771,.790,.809,.828,.848,.868,.889,.910,.932,.955,.978,1.001,1.025,1.050,1.075,1.102],
[1.113,1.139,1.167,1.195,1.224,1.254,1.284,1.315,1.346,1.378,1.412,1.446,1.481,1.517,1.554,1.592,1.631],
[1.656,1.695,1.735,1.775,1.818,1.861,1.905,1.949,1.995,2.041,2.089,2.139,2.189,2.240,2.292,2.345,2.399],
[2.445,2.506,2.575,2.646,2.719,2.794,2.871,2.951,3.033,3.118,3.204,3.293,3.383,3.476,3.570,3.667,3.768],
[3.834,3.867,3.901,3.936,3.970,4.005,4.041,4.076,4.112,4.148,4.184,4.221,4.257,4.294,4.331,4.368,4.405],
[4.435,4.468,4.500,4.533,4.567,4.601,4.635,4.669,4.704,4.739,4.774,4.808,4.843,4.879,4.915,4.952,4.989]])
# Consistent false-color detector identifiers (not visible infrared colors).
DETECTOR_COLORS = ("#3366B0", "#C16A13", "#16805C", "#8752A3", "#007E94", "#B64868")
LIMITS = np.array([[.743,1.116],[1.099,1.651],[1.636,2.421],[2.422,3.827],[3.815,4.427],[4.418,5.009]])


def channel_index(detector, wavelength):
    """Nearest nominal wavelength within detector's nominal range; -1 otherwise.
    This is a wavelength-based approximation, not official geometric assignment."""
    wl = np.asarray(wavelength, dtype=float)
    if not 1 <= detector <= 6:
        return np.full(wl.shape, -1, dtype=int)
    c = CENTERS[detector-1]
    i = np.searchsorted((c[:-1]+c[1:])/2, wl) + 17*(detector-1)
    lo, hi = LIMITS[detector-1]
    return np.where(np.isfinite(wl) & (wl >= lo) & (wl <= hi), i, -1)


def binned_spectrum(wl, flux, err, detectors, selected):
    """102 weighted estimates; retain NaN for missing channels and negative fluxes."""
    wl, flux, err, detectors, selected = map(np.asarray, (wl, flux, err, detectors, selected))
    channel = np.full(wl.shape, -1, dtype=int)
    for d in range(1,7):
        m = detectors == d
        channel[m] = channel_index(d, wl[m])
    good = selected.astype(bool) & (channel >= 0) & np.isfinite(flux) & np.isfinite(err) & (err > 0)
    k = channel[good]
    w = 1/err[good]**2
    n = np.bincount(k, minlength=102)
    sw = np.bincount(k, weights=w, minlength=102)
    sf = np.bincount(k, weights=w*flux[good], minlength=102)
    avg = np.divide(sf, sw, out=np.full(102,np.nan), where=sw>0)
    var = np.divide(1, sw, out=np.full(102,np.nan), where=sw>0)
    return avg, np.sqrt(var), n


def tangent_sky(ra, dec, east, north):
    """Inverse TAN projection for offsets in arcsec; handles RA wrap and poles."""
    a, b = np.deg2rad([ra, dec])
    x, y = np.deg2rad(east/3600), np.deg2rad(north/3600)
    den = np.cos(b)-y*np.sin(b)
    return np.rad2deg(a+np.arctan2(x,den))%360, np.rad2deg(np.arctan2(np.sin(b)+y*np.cos(b),np.hypot(den,x)))


def sample_frame(frame, east, north):
    """Bilinear image/variance sampling on a target-centered TAN grid.
    Reject pixels with incomplete valid support; propagate squared weights.
    """
    wcs = frame['wcs']
    ra, dec = wcs.pixel_to_world_values(*frame['catalog_xy'])
    ra, dec = tangent_sky(ra, dec, east, north)
    x,y = map(np.asarray,wcs.world_to_pixel_values(ra,dec))
    finite = np.isfinite(x)&np.isfinite(y)
    x,y = np.where(finite,x,0),np.where(finite,y,0)
    ix,iy = np.floor(x).astype(int),np.floor(y).astype(int)
    dx,dy = x-ix,y-iy
    im, var = np.asarray(frame['image']),np.asarray(frame['variance'])
    valid = ~np.asarray(frame['bad'],bool)&np.isfinite(im)&np.isfinite(var)&(var>0)
    # Remove a scalar sky level for point-source detection, not diffuse photometry.
    yy,xx = np.indices(im.shape)
    cx,cy = frame['catalog_xy']
    bg = valid & (np.hypot(xx-cx,yy-cy)>8)
    sky = np.median(im[bg]) if bg.any() else 0.
    ny,nx = im.shape
    z,v,support = np.zeros(x.shape),np.zeros(x.shape),np.zeros(x.shape)
    for ox,oy,w in [(0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)]:
        j,k=ix+ox,iy+oy
        inside=finite&(j>=0)&(j<nx)&(k>=0)&(k<ny)
        j,k=np.clip(j,0,nx-1),np.clip(k,0,ny-1)
        good=inside&valid[k,j]
        weight=np.where(good,w,0.)
        z+=weight*np.where(good,im[k,j]-sky,0.)
        v+=weight**2*np.where(good,var[k,j],0.)
        support+=weight
    good=(support>1-1e-7)&(v>0)
    z,v=np.where(good,z,np.nan),np.where(good,v,np.nan)
    return z,v


def weighted_image(images, variances, selection=None):
    """Inverse-variance mean, diagonal variance, and coverage. No data => NaN."""
    images,variances=np.asarray(images),np.asarray(variances)
    good=np.isfinite(images)&np.isfinite(variances)&(variances>0)
    if selection is not None:
        good &= selection
    w=np.divide(1.,variances,out=np.zeros_like(variances,dtype=float),where=good)
    sw=w.sum(axis=0)
    num=(w*np.where(good,images,0)).sum(axis=0)
    mean=np.divide(num,sw,out=np.full(sw.shape,np.nan),where=sw>0)
    var=np.divide(1.,sw,out=np.full(sw.shape,np.nan),where=sw>0)
    return mean,var,good.sum(axis=0)




# Verification is deliberately separate from tracking and historical consistency.
# These are conservative screening rules, not calibrated identity probabilities.
class ObjectVerification:
    def __init__(self, analyzer, zooms, tracked, movement):
        self.a, self.zooms = analyzer, zooms
        self.tracked, self.movement = tracked or {}, movement or {}
        self.moving = 'ephemeris' in self.tracked or 'asteroid' in self.tracked
        self.known = self.moving or ('star' in self.tracked and
                     self.tracked['star'].get('object_type') != 'sky position') or (
                     'star' not in self.tracked and bool(self.tracked.get('catalog_sed')))
        self.groups = analyzer.observation_groups()
        self.epoch = np.empty(len(zooms), int)
        for j, g in enumerate(self.groups): self.epoch[g] = j
        self.controls = {}
        self.catalog = self.tracked.get('catalog_sed') or []
        self.templates = []
        self.template_note = ''
        self._load_templates()
        f = analyzer.frames
        self.quality = analyzer.reliable & np.array([
            x.get('astrometry_ok', True) and not (np.isfinite(x.get('alignment_px', np.nan)) and
            x['alignment_px'] > cfg.ALIGNMENT_TOLERANCE_PIX) for x in f])
        self.snr = np.divide(analyzer.flux, analyzer.flux_err, out=np.full(len(f), np.nan),
                             where=np.isfinite(analyzer.flux_err)&(analyzer.flux_err>0))
        self.separation = np.array([self._distance(x.get('object_ra',np.nan),x.get('object_dec',np.nan),
                          x.get('expected_ra',np.nan),x.get('expected_dec',np.nan)) for x in f])
        self.peak = np.array([x.get('object_found',False) and x.get('track','ok') in ('ok','off') for x in f])
        self.wings = np.array([x.get('object_found',False) and x.get('position_method')=='wings' for x in f])
        self.detected = self.quality & self.peak & (self.snr >= getattr(cfg,'DETECTION_SNR',5))
        self.position_usable = self.detected | (self.quality & self.peak & self.wings)
        self.limit = getattr(cfg,'TRACK_TOLERANCE_PIX',1.5)*cfg.PIXEL_ARCSEC
        self.on_position = self.position_usable & np.isfinite(self.separation) & (self.separation <= self.limit)
        self.catalog_check = self._catalog_comparison(self.catalog) if self.known and not self.moving else None
        self.competitors = self._field_candidates()
        self.motion = self._motion()

    @staticmethod
    def _distance(ra, dec, ra2, dec2):
        a,b,c,d = np.deg2rad([ra,dec,ra2,dec2])
        h = np.sin((b-d)/2)**2 + np.cos(b)*np.cos(d)*np.sin((a-c)/2)**2
        return float(np.rad2deg(2*np.arcsin(np.sqrt(np.clip(h,0,1))))*3600)

    @staticmethod
    def result(state, note): return dict(state=state,note=note)

    def background(self, i):
        """Same pipeline aperture and annulus, on unenhanced pixels. No source peaks in controls."""
        if i in self.controls: return self.controls[i]
        from .measure import aperture_flux, significant_peaks
        z = self.zooms[i]; x,y = z['catalog_xy']; ny,nx = z['image'].shape
        px,py,_ = significant_peaks(z,noise='pixel')
        radius = max(10., float(cfg.ANNULUS_PIX[1])+cfg.APERTURE_RADIUS_PIX+1)
        positions=[]; fluxes=[]
        for angle in np.linspace(0,2*np.pi,16,endpoint=False):
            cx,cy = x+radius*np.cos(angle),y+radius*np.sin(angle)
            outer = cfg.ANNULUS_PIX[1]
            if not (outer <= cx < nx-outer and outer <= cy < ny-outer): continue
            if np.any(np.hypot(px-cx,py-cy) <= cfg.APERTURE_RADIUS_PIX+1): continue
            if any(np.hypot(cx-u,cy-v)<2*cfg.APERTURE_RADIUS_PIX for u,v in positions): continue
            f,e,bad = aperture_flux(z,cx,cy)
            if bad==0 and np.isfinite(f) and np.isfinite(e) and e>0:
                positions.append((cx,cy)); fluxes.append(f)
        values=np.asarray(fluxes); excess=np.nan
        if len(values)>=5 and np.isfinite(self.a.flux[i]) and self.a.flux_err[i]>0:
            noise=max(float(robust_scatter(values)),float(self.a.flux_err[i]))
            excess=(self.a.flux[i]-np.median(values))/noise
        out=dict(positions=positions,fluxes=values,significance=excess)
        self.controls[i]=out
        return out

    def _catalog_comparison(self, catalog):
        """Approximate broad-filter comparison, not interpolated broadband photometry.
        Require three epochs and several wavelengths spanning each filter; do not fit a free scale.
        """
        ranges=[(1.235,1.11,1.36),(1.662,1.50,1.80),(2.159,1.99,2.31),
                (3.368,2.80,3.90),(4.618,4.00,5.10)]
        a=self.a; det=np.array([f['detector'] for f in a.frames]); ratios=[]; rows=[]
        valid=self.quality & np.isfinite(a.flux)&np.isfinite(a.flux_err)&(a.flux_err>0)
        # Include weak measurements in the median (no positive-flux selection bias).
        for name,w,f,e in catalog:
            if not (np.isfinite([w,f,e]).all() and f>0 and e>=0): continue
            center,lo,hi=min(ranges,key=lambda b:abs(b[0]-w))
            if abs(center-w)>.08: continue
            use=valid & (a.wavelength>=lo)&(a.wavelength<=hi)
            if len(np.unique(self.epoch[use]))<3: continue
            channels=np.unique(np.concatenate([np.atleast_1d(channel_index(d,a.wavelength[use&(det==d)])) for d in range(1,7)]))
            if (channels>=0).sum()<3 or np.ptp(a.wavelength[use]) < .5*(hi-lo): continue
            # Equal weight per populated channel, rather than repeated visits dominating a filter.
            sed=a.band_spectrum(use); k=(sed['count']>0)&(sed['wavelength_um']>=lo)&(sed['wavelength_um']<=hi)
            if k.sum()<3: continue
            measured=float(np.median(sed['flux_mjy'][k])); uncertainty=float(np.median(sed['flux_err_mjy'][k]))
            if measured < 3*uncertainty: continue
            ratio=measured/f; ratios.append(ratio); rows.append((name,w,f,e,measured))
        if len(rows)<3: return None
        scale=float(np.median(ratios)); scatter=float(np.sqrt(np.mean((np.array(ratios)/scale-1)**2)))
        return dict(rows=rows,scale=scale,scatter=scatter,compatible=bool(.4<=scale<=1.3 and scatter<=.25))

    def _field_candidates(self):
        """Use already retrieved catalogue positions and nearby 2MASS photometry, no new network calls."""
        table=self.tracked.get('field_stars'); out=[]
        if table is None: return out
        try:
            from .target import catalog_fluxes,TWOMASS_BANDS
            f=self.a.frames[0]; ra,dec=f['expected_ra'],f['expected_dec']
            for row in table:
                sep=self._distance(float(row['ra']),float(row['dec']),ra,dec)
                # The target itself (or catalog position within ~1 arcsec) is not a competitor.
                if not 1.0<sep<= (cfg.APERTURE_RADIUS_PIX+1)*cfg.PIXEL_ARCSEC: continue
                phot=catalog_fluxes(row,TWOMASS_BANDS)
                out.append(dict(name=str(row['designation']),separation=sep,phot=phot,
                                check=self._catalog_comparison(phot)))
        except (KeyError,TypeError,ValueError,ImportError):
            return out
        return out

    def _load_templates(self):
        """Optional independently measured, absolute-flux spectral library. No bundled fake templates.
        CSV next to module: name,role,wavelength_um,flux_mjy,flux_err_mjy,valid_from_mjd,valid_to_mjd.
        Moving objects require explicit geometry/time validity. Static spectra may omit dates.
        """
        import csv
        from pathlib import Path
        path=Path(getattr(cfg,'VERIFICATION_REFERENCES',Path(__file__).with_name('verification_references.csv')))
        if not path.is_file(): return
        try:
            groups={}
            with path.open(newline='',encoding='utf-8-sig') as f:
                for row in csv.DictReader(f):
                    name=row['name'].strip(); role=row['role'].strip().lower()
                    if role not in ('target','alternative') or not name: raise ValueError('invalid name or role')
                    lo=float(row.get('valid_from_mjd') or '-inf'); hi=float(row.get('valid_to_mjd') or 'inf')
                    if lo>hi or np.isnan([lo,hi]).any(): raise ValueError('invalid time range')
                    w,v,e=map(float,(row['wavelength_um'],row['flux_mjy'],row['flux_err_mjy']))
                    if not np.isfinite([w,v,e]).all() or w<=0 or v<=0 or e<=0: raise ValueError('invalid spectral point')
                    groups.setdefault((name,role,lo,hi),[]).append((w,v,e))
            for (name,role,lo,hi),points in groups.items():
                p=np.asarray(sorted(points));
                if len(p)<3 or np.any(np.diff(p[:,0])<=0): raise ValueError('need distinct wavelengths')
                self.templates.append(dict(name=name,role=role,lo=lo,hi=hi,points=p))
        except (OSError,ValueError,KeyError) as exc:
            self.templates=[]; self.template_note='External reference file could not be read: '+str(exc)
            print(self.template_note)

    def _template_comparison(self, members):
        """Compare candidates on exactly the same observed points, no extrapolation or scale fitting."""
        a=self.a; indices=np.asarray(members,int)
        indices=indices[self.on_position[indices]&self.detected[indices]&np.isfinite(a.wavelength[indices])]
        if not len(indices): return []
        times=a.mjd[indices]; wl=a.wavelength[indices]
        available=[]
        for t in self.templates:
            if self.moving and not np.isfinite([t['lo'],t['hi']]).all(): continue
            if t['role']=='target' and (not self.known or t['name'].casefold()!=str(self.tracked.get('name','')).casefold()): continue
            p=t['points']; j=np.searchsorted(p[:,0],wl); j=np.clip(j,1,len(p)-1)
            exact=np.isclose(wl,p[j,0],rtol=0,atol=1e-8)|np.isclose(wl,p[j-1,0],rtol=0,atol=1e-8)
            ok=(times>=t['lo'])&(times<=t['hi'])&(wl>=p[0,0])&(wl<=p[-1,0])&((p[j,0]-p[j-1,0]<=.05*wl)|exact)
            if len(np.unique([a.frames[i]['detector'] for i in indices[ok]]))>=3:
                available.append((t,ok,np.interp(wl,p[:,0],p[:,1]),np.interp(wl,p[:,0],p[:,2])))
        if not available: return []
        common=np.logical_and.reduce([x[1] for x in available])
        if len(np.unique([a.frames[i]['detector'] for i in indices[common]]))<3: return []
        out=[]
        for t,_,v,e in available:
            z=(a.flux[indices[common]]-v[common])/np.hypot(a.flux_err[indices[common]],e[common])
            out.append(dict(name=t['name'],role=t['role'],score=float(np.sqrt(np.mean(z*z))),compatible=bool(np.all(np.abs(z)<=3)),wavelength=wl[common],flux=v[common],error=e[common]))
        return sorted(out,key=lambda x:x['score'])

    def _motion(self):
        if not self.known: return self.result('unavailable','No named target or predicted path.')
        if not self.moving:
            # Existing named-star proper-motion fit, if available. No flags alone are not evidence.
            fit=getattr(self.a,'catalog_motion',None) or {}
            sig=np.asarray(fit.get('agreement_sigma',[]),float)
            if len(sig)==2 and np.isfinite(sig).all():
                return self.result('review' if np.any(sig>3) else 'pass','Measured motion vs catalog prediction (whole run).')
        groups=[g[self.on_position[g]] for g in self.groups if self.on_position[g].any()]
        if len(groups)<3: return self.result('unavailable','Need three separate well-located observations.')
        first,last=self.a.frames[groups[0][0]],self.a.frames[groups[-1][0]]
        travel=self._distance(first['expected_ra'],first['expected_dec'],last['expected_ra'],last['expected_dec'])
        if not np.isfinite(travel) or travel<2*self.limit:
            return self.result('unavailable','Expected motion is too small to distinguish confidently.')
        if np.any(self.position_usable & ~self.on_position): return self.result('review','Some measured locations differ from the predicted path.')
        return self.result('pass','Source follows a clearly moving predicted path (whole run).')

    def evaluate(self, members, history):
        r=self.result; members=np.asarray(members,int)
        if not len(members):
            return dict(title='No observation selected',tone='unavailable',rows=[r('unavailable','Choose a populated channel.')]*6)
        good=members[self.detected[members]]
        bg=[self.background(int(i)) for i in good]
        clears=[int(i) for i,b in zip(good,bg) if np.isfinite(b['significance']) and b['significance']>=5]
        if clears: detection=r('pass',f'Stands above nearby sky in {len(clears)} image(s).')
        elif len(good):
            detection=r('review' if any(np.isfinite(b['significance']) for b in bg) else 'unavailable','Source signal not confirmed by nearby sky controls.')
        elif np.any(self.wings[members]&self.position_usable[members]):
            detection=r('unavailable','Bright core saturated; position from outer glow only.')
        else:
            combined=getattr(self.a,'combined_detection',None) or []
            has_stack=any(b.get('significance',0)>=5 and b.get('formal_snr',0)>=5 for b in combined)
            detection=r('unavailable','Not confirmed here; a combined detection exists.' if has_stack else 'No reliable source detected in this observation.')
        loc=members[self.position_usable[members]]
        if len(loc) and np.any(~self.on_position[loc]): position=r('review','A measured source is away from the expected position.')
        elif len(loc): position=r('pass','Measured source is near the expected position.')
        else: position=r('unavailable','No reliable position measurement.')
        scores=self._template_comparison(members)
        # Gaia/CatWISE sources are alternative positions, not independent spectra.
        # Their positions are propagated to the observed date when the catalog is available.
        field_names=set()
        field=self.tracked.get('field')
        if field is not None and len(field):
            from .field import field_positions
            t=self.a.frames[int(members[0])]
            ra,dec=field_positions(field,t['mjd'])
            sep=np.array([self._distance(x,y,t['expected_ra'],t['expected_dec']) for x,y in zip(ra,dec)])
            near=np.isfinite(sep)&(sep<=(cfg.APERTURE_RADIUS_PIX+1)*cfg.PIXEL_ARCSEC)
            if self.known: near &= sep>1.0
            field_names={str(x) for x in field['name'][near]}

        target=next((x for x in scores if x['role']=='target'),None)
        alt=next((x for x in scores if x['role']=='alternative'),None)
        if target: external=r('pass' if target['compatible'] else 'review','Independent spectrum compared at this observation.')
        elif self.catalog_check is not None:
            x=self.catalog_check; external=r('pass' if x['compatible'] else 'review',f'{len(x["rows"])} catalog filters; approximate whole-run comparison.')
        else: external=r('unavailable','No usable independent spectrum for this target.')
        better=bool(target and alt and alt['compatible'] and alt['score']+1<target['score'])
        # Any nearby catalog source is a blending risk, not proof of a wrong identity.
        # Broad-filter candidate scores must use the same filters. A nearby source is a
        # possible contaminant even if its broadband spectrum does not fit particularly well.
        better_catalog = None
        if self.catalog_check is not None:
            target_rows={round(x[1],2):x for x in self.catalog_check['rows']}
            for candidate in self.competitors:
                other=candidate['check']
                if other is None: continue
                other_rows={round(x[1],2):x for x in other['rows']}
                shared=sorted(target_rows.keys() & other_rows.keys())
                if len(shared)<3: continue
                def mismatch(rows):
                    return float(np.sqrt(np.mean([np.log(rows[k][4]/rows[k][2])**2 for k in shared])))
                if other['compatible'] and mismatch(other_rows)+.25<mismatch(target_rows):
                    better_catalog=candidate['name']; break
        if better: alternatives=r('review',f'Alternative spectrum fits better: {alt["name"]}.')
        elif better_catalog: alternatives=r('review',f'Nearby 2MASS source {better_catalog} fits better.')
        elif self.competitors or (self.known and field_names): alternatives=r('review','Nearby catalog sources could blend with the target.')
        elif field_names: alternatives=r('unavailable',f'{len(field_names)} nearby catalog entries; identity not assigned.')
        elif scores and alt: alternatives=r('unavailable',f'Best library alternative: {alt["name"]}; identity not established.')
        else: alternatives=r('unavailable','No competing spectrum verified.')
        repeat=r('pass' if history['status']=='consistent' else 'review' if history['status']=='review' else 'unavailable',
                 f'{history["matched"]} of {history["available"]} bands agree with earlier/later images.')
        rows=[detection,position,self.motion,external,alternatives,repeat]
        if detection['state']!='pass': title,tone='Source not independently confirmed','review' if detection['state']=='review' else 'unavailable'
        elif position['state']=='review' or self.motion['state']=='review' or external['state']=='review' or better or better_catalog: title,tone='Possible mismatch — check the evidence','review'
        elif alternatives['state']=='review' or repeat['state']=='review': title,tone='Source detected — worth checking','review'
        elif not self.known: title,tone='Source detected — identity unknown','unavailable'
        elif position['state']=='pass' and (external['state']=='pass' or (self.moving and self.motion['state']=='pass')):
            title,tone='Evidence supports the target','pass'
        else: title,tone='Source detected — identity uncertain','unavailable'
        return dict(title=title,tone=tone,rows=rows,scores=scores)


class BandViewer:
    """Linked target-centered stacks and the pipeline's unchanged tracking results.

    The original loader retains one wavelength per exposure at the target. Thus
    channel membership is assigned to a whole cutout by its target wavelength;
    this is not a per-pixel calibrated spectral cube. Registration uses only WCS
    and the predicted/catalog center, never the detected brightness peak.
    """
    def __init__(self, analyzer, zooms, movement, tracked=None, size=41):
        from matplotlib.widgets import Slider, RadioButtons, RangeSlider, Button
        self.analyzer, self.zooms, self.movement = analyzer, zooms, movement
        self.name = (tracked or {}).get('name', 'SPHEREx target')
        self.n = len(analyzer.frames)
        if not self.n or len(zooms) != self.n:
            raise ValueError('Provide the nonempty results["zooms"] from the same run.')
        if any(str(f['obs_id']) != str(z['obs_id']) for f,z in zip(analyzer.frames, zooms)):
            raise ValueError('Frame/zoom order does not match.')
        if size < 3 or size % 2 != 1:
            raise ValueError('size must be odd and at least 3')
        self.size = size
        self.pretty = bool(getattr(cfg, 'PRETTY_IMAGES', True))
        if self.pretty:
            from .visual import find_partners, anchors
            self.partners = find_partners(zooms)
            self.frame_anchors = anchors(zooms, self.partners)
        else:
            self.partners = [None] * self.n
            self.frame_anchors = [None] * self.n
        self.detector = np.array([f['detector'] for f in analyzer.frames])
        if not np.isin(self.detector, np.arange(1, 7)).all():
            raise ValueError('SPHEREx detector IDs must be 1 through 6.')
        self.channels = np.full(self.n, -1, int)
        for d in range(1,7):
            m = self.detector == d
            self.channels[m] = channel_index(d, analyzer.wavelength[m])
        self.days = analyzer.mjd - analyzer.mjd[0]
        self.observations = analyzer.observation_groups()
        self.n_obs = len(self.observations)
        self.epoch_ids = np.empty(self.n, dtype=int)
        for j, members in enumerate(self.observations):
            self.epoch_ids[members] = j
        self.observation = 0
        self.obs_days = np.array([self.days[g[0]] for g in self.observations])
        self.good = (analyzer.reliable & np.isfinite(analyzer.flux) &
                     np.isfinite(analyzer.flux_err) & (analyzer.flux_err > 0))
        yy,xx = np.indices((size,size), dtype=float)
        east = -(xx-size//2)*cfg.PIXEL_ARCSEC
        north = (yy-size//2)*cfg.PIXEL_ARCSEC
        # Register once, reuse for all 108 views; native cutouts remain unchanged.
        sampled = [sample_frame(z,east,north) for z in zooms]
        self.images,self.variances = (np.asarray(v,dtype=np.float32) for v in zip(*sampled))
        self.keys = [(d,c) for d in range(1,7) for c in range(18)]
        self.groups = {key: self._members(*key) for key in self.keys}
        self.stacks = {key: weighted_image(self.images[idx], self.variances[idx])
                       for key,idx in self.groups.items()}
        self.sed = analyzer.band_spectrum()
        self.time_mask = np.ones(self.n, bool)
        self.verifier = ObjectVerification(analyzer, zooms, tracked, movement)
        self.spec_ylim = self._compute_spectrum_ylim()
        self.index = int(self.observations[0][0])
        self.key = (int(self.detector[self.index]), self._local_channel(self.index))
        self._syncing = False
        self.gallery = plt.figure(figsize=(19,8.5))
        gs = self.gallery.add_gridspec(6,18,left=.045,right=.99,top=.88,bottom=.07,
                                       wspace=.10,hspace=.34)
        self.tiles = {}
        cmap = plt.get_cmap('gray').copy(); cmap.set_bad('#353b45')
        for d,c in self.keys:
            key = (d,c)
            ax = self.gallery.add_subplot(gs[d-1,c]); self.tiles[ax] = key
            mean,var,n = self.stacks[key]
            snr = mean / np.sqrt(var)
            ax.imshow(snr,origin='lower',cmap=cmap,vmin=-3,vmax=12,interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            title = 'All' if c == 0 else f'{CENTERS[d-1,c-1]:.3f}'
            ax.set_title(f'{title}\nn={len(self.groups[key])}',fontsize=7,color=DETECTOR_COLORS[d-1])
            if c == 0: ax.set_ylabel(f'D{d}',fontsize=10,weight='bold',color=DETECTOR_COLORS[d-1])
            if not np.any(n): ax.text(.5,.5,'—',ha='center',va='center',transform=ax.transAxes,color='white')
        covered = int(np.count_nonzero(self.sed['count']))
        self.gallery.suptitle(f'{self.name} | 102-channel atlas + six detector stacks\n'
                              f'{covered}/102 channels with reliable target photometry | '
                              'approximate target-wavelength bins; click any tile',fontsize=14)
        self.gallery.text(.5,.025,'All = broad detector stack  •  n = retained captures  •  '
                          'grayscale = approximate per-pixel S/N (−3 to +12)  •  — = no valid image pixels',ha='center',fontsize=10)
        # One image, one spectrum. Original tracking graphs stay in their own windows.
        self.image_view = 'original'
        self.detail = plt.figure(figsize=(15,9.5), facecolor='white')
        self.ax_image = self.detail.add_axes([.055,.32,.39,.405])
        self.ax_spec = self.detail.add_axes([.54,.235,.415,.285])
        self.header = self.detail.text(.055,.95,'',fontsize=18,weight='bold',va='top')
        self.subtitle = self.detail.text(.055,.903,'',fontsize=10,color='#526174',va='top')
        self.status_text = self.detail.text(.54,.895,'',fontsize=14,weight='bold',va='top')
        self.reason_text = self.detail.text(.54,.565,'',fontsize=9,color='#475569',va='top')
        self.verify_rows=[]
        for j,label in enumerate(('Source detected','Expected position','Expected movement','External spectrum','Other possible objects','Repeat brightness')):
            y=.850-j*.044
            self.detail.text(.54,y,label,fontsize=10,weight='bold',va='top')
            state=self.detail.text(.735,y,'',fontsize=10,weight='bold',va='top')
            note=self.detail.text(.54,y-.019,'',fontsize=8.2,color='#475569',va='top')
            self.verify_rows.append((state,note))
        self.image_title = self.detail.text(.055,.842,'',fontsize=13,weight='bold',va='top')
        self.image_note = self.detail.text(.055,.258,'',fontsize=9,color='#475569',va='top')
        self.detail.text(.54,.177,'Small dots = other observations • Large dots = this observation\nBlack diamonds = independent reference (when available)',fontsize=8.5,color='#475569',va='top')
        self.detail.text(.54,.147,'Repeat brightness is not proof of identity. Catalog checks use all dates.',
                         fontsize=8.5,color='#64748b',va='top')
        self.show_controls=False
        self.control_button=Button(self.detail.add_axes([.055,.285,.19,.028]),'Show sky checks')
        self.control_button.on_clicked(lambda _: self._toggle_controls())
        self.detector_buttons = []
        for d, color in enumerate(DETECTOR_COLORS, 1):
            button = Button(self.detail.add_axes([.055 + (d-1)*.065,.735,.058,.027]), f'D{d}',
                            color='#f1f5f9', hovercolor='#e2e8f0')
            button.label.set_color(color)
            button.label.set_fontsize(9)
            button.on_clicked(lambda _, det=d: self.select_detector(det))
            self.detector_buttons.append(button)
        self.original_button = Button(self.detail.add_axes([.055,.765,.18,.047]),'This image')
        self.stack_button = Button(self.detail.add_axes([.245,.765,.20,.047]),'Combined view')
        self.original_button.on_clicked(lambda _: self.set_view('original'))
        self.stack_button.on_clicked(lambda _: self.set_view('stack'))
        self.mode = RadioButtons(self.detail.add_axes([.055,.135,.19,.087]),['Clear view','Brightness','Support'])
        self.mode.on_clicked(lambda _: self.set_view('stack'))
        self.mode_label = self.detail.text(.26,.18,'Combined image options\nClear view = visibility above noise',fontsize=8,color='#64748b',va='center')
        self.slider = Slider(self.detail.add_axes([.23,.074,.56,.032]),'Observation',1,max(2,self.n_obs),valinit=1,valstep=1,valfmt='%d')
        if self.n_obs == 1: self.slider.set_active(False)
        self.prev_button = Button(self.detail.add_axes([.055,.064,.075,.045]),'← Prev')
        self.next_button = Button(self.detail.add_axes([.875,.064,.08,.045]),'Next →')
        self.prev_button.on_clicked(lambda _: self._step(-1))
        self.next_button.on_clicked(lambda _: self._step(1))
        self.time_slider = None
        if np.isfinite(self.obs_days).all() and np.ptp(self.obs_days) > 0:
            self.time_slider = RangeSlider(self.detail.add_axes([.66,.112,.21,.018]),
                                           'Compare days',float(self.obs_days.min()),float(self.obs_days.max()),
                                           valinit=(float(self.obs_days.min()),float(self.obs_days.max())),valfmt='%.1f')
            self.time_slider.on_changed(self._on_time)
        self.detail.text(.055,.02,'← / → browse observation times   •   D1–D6 switch images   •   Click the separate gallery to explore a channel',
                         fontsize=9,color='#64748b')
        self.slider.on_changed(self._on_slider)
        self.gallery.canvas.mpl_connect('button_press_event',self._on_tile)
        self.detail.canvas.mpl_connect('key_press_event',self._on_key)
        # Matplotlib keeps weak refs to bound callbacks; keep the viewer alive.
        self.gallery._band_viewer = self.detail._band_viewer = self
        self._redraw()

    def _toggle_controls(self):
        self.show_controls=not self.show_controls
        self.control_button.label.set_text('Hide sky checks' if self.show_controls else 'Show sky checks')
        self._redraw()

    def _local_channel(self,i):
        return int(self.channels[i]%17+1) if self.channels[i]>=0 else 0

    def _members(self,d,c):
        mask = self.detector == d
        if c: mask &= self.channels == (d-1)*17+c-1
        return np.flatnonzero(mask)

    def _on_tile(self,event):
        if event.inaxes not in self.tiles: return
        self.select(*self.tiles[event.inaxes])

    def select(self,detector,channel=0):
        """Select a detector, and optionally a local nominal channel (1–17)."""
        key=(int(detector),int(channel))
        if key not in self.groups: raise ValueError('detector 1–6, channel 0–17')
        self.key=key
        self.image_view = 'stack'
        idx=self.groups[key]
        if len(idx):
            # Keep the current frame if it belongs to the chosen group; otherwise
            # select the closest time to the previous frame.
            closest=idx[np.argmin(np.abs(self.days[idx]-self.days[self.index]))]
            if self.index not in idx: self.index=int(closest)
            self.observation = int(self.epoch_ids[self.index])
            self._syncing=True
            try: self.slider.set_val(self.observation+1)
            finally: self._syncing=False
        self._redraw()

    def _on_time(self, bounds):
        lo, hi = bounds
        keep = (self.obs_days >= lo) & (self.obs_days <= hi)
        self.time_mask = keep[self.epoch_ids]
        self.sed = self.analyzer.band_spectrum(self.time_mask)
        self._redraw()

    def _on_slider(self,value):
        if self._syncing: return
        preferred = int(self.detector[self.index])
        self.observation = min(self.n_obs-1,max(0,int(value)-1))
        members = self.observations[self.observation]
        same = members[self.detector[members] == preferred]
        self.index = self._representative(same if len(same) else members)
        self.image_view = 'original'
        self.key=(int(self.detector[self.index]),self._local_channel(self.index))
        self._redraw()

    def _representative(self, members):
        """Pick a usable image for display only; no measurements discarded."""
        if self.index in members:
            return int(self.index)
        usable = members[self.good[members]]
        return int(usable[0] if len(usable) else members[0])

    def select_detector(self, detector):
        """Change the displayed image without changing the observation spectrum."""
        members = self.observations[self.observation]
        same = members[self.detector[members] == detector]
        if not len(same):
            return
        self.index = self._representative(same)
        self.key = (int(detector), self._local_channel(self.index))
        self.image_view = 'original'
        self._redraw()

    def set_view(self, view):
        """Switch the single image panel without changing the selected capture."""
        if view not in ('original', 'stack'):
            raise ValueError('view must be original or stack')
        self.image_view = view
        self._redraw()

    def _step(self, direction):
        self.slider.set_val(np.clip(self.observation+1+direction,1,self.n_obs))

    def _on_key(self,event):
        if event.key in ('left','right'):
            self._step(1 if event.key=='right' else -1)

    def _compute_spectrum_ylim(self):
        """Fixed scale including all finite measurements and their error bars.

        Weighted reference averages lie within the finite measurement range.
        No percentile clipping: an unusual observation must stay visible.
        """
        f, e = self.analyzer.flux, self.analyzer.flux_err
        ok = np.isfinite(f)
        good = ok & np.isfinite(e) & (e > 0)
        values = np.r_[f[ok], f[good]-e[good], f[good]+e[good]]
        if self.verifier.catalog_check is not None:
            values=np.r_[values,[(f-e) for _,w,f,e,m in self.verifier.catalog_check['rows']],[(f+e) for _,w,f,e,m in self.verifier.catalog_check['rows']]]
        for t in self.verifier.templates:
            if t['role']=='target' and t['name'].casefold()==self.name.casefold():
                p=t['points']; in_range=(p[:,0]>=.7)&(p[:,0]<=5.05)
                values=np.r_[values,p[in_range,1]-p[in_range,2],p[in_range,1]+p[in_range,2]]
        values = values[np.isfinite(values)]
        if not len(values):
            return (0., 1.)
        lo, hi = float(values.min()), float(values.max())
        pad = .10 * (hi-lo if hi > lo else max(abs(hi), 1.))
        return lo-pad, hi+pad

    def _show_pretty_stack(self, ax, data, mode, lim):
        """Sharpened display-only rendering for combined stacks."""
        from .visual import upsample
        finite = data[np.isfinite(data)]
        fill = float(np.nanmedian(finite)) if finite.size else 0.0
        filled = np.where(np.isfinite(data), data, fill)
        factor = max(1, int(getattr(cfg, 'DISPLAY_UPSCALE', 1)))
        big = upsample(filled, factor, sharp=True) if factor > 1 else filled
        extent = (-0.5, data.shape[1] - 0.5, -0.5, data.shape[0] - 0.5)
        cmap = 'viridis' if mode == 'Coverage' else 'gray'
        ax.imshow(big, origin='lower', cmap=cmap, vmin=lim[0], vmax=lim[1],
                  extent=extent, interpolation='nearest')
        if not finite.size:
            ax.text(.5, .5, 'No usable image pixels', transform=ax.transAxes,
                    ha='center', va='center', color='white',
                    bbox=dict(facecolor='#333333'))

    def _band_label(self, detector, channel):
        if channel == 0:
            return f'Band {detector} · all wavelengths'
        return f'Band {detector} · {CENTERS[detector-1,channel-1]:.3f} µm'

    def _draw_spectrum(self, active):
        """One observation, all available detectors, one independent reference."""
        from matplotlib.lines import Line2D
        ax, a = self.ax_spec, self.analyzer
        ax.clear()
        ax.set(xlim=(.7, 5.05), ylim=self.spec_ylim,
               xlabel='Infrared wavelength (µm)', ylabel='Brightness (mJy)')
        ax.set_xticks([1,2,3,4,5])
        ax.tick_params(labelsize=9)
        ax.grid(axis='y', alpha=.15)
        ax.spines[['top','right']].set_visible(False)
        handles = [Line2D([],[],marker='o',linestyle='none',markersize=5,
                          color=color,label=f'D{d}') for d,color in enumerate(DETECTOR_COLORS,1)]
        ax.legend(handles=handles,loc='lower left',bbox_to_anchor=(0,1.025,1,.04),
                  ncol=6,mode='expand',borderaxespad=0,frameon=False,
                  fontsize=9,handletextpad=.15,columnspacing=.6)
        members = self.observations[self.observation] if active else np.array([], dtype=int)
        # Remove the entire exposure group, including slight timestamp offsets.
        mask = self.time_mask.copy()
        mask[members] = False
        if active:
            mask = a.reference_mask(self.index, mask)
        else:
            align = np.array([f.get('alignment_px', np.nan) for f in a.frames], float)
            mask &= np.isfinite(a.mjd) & ~(np.isfinite(align) & (align > cfg.ALIGNMENT_TOLERANCE_PIX))
        reference = self.reference_sed = a.band_spectrum(mask)
        valid = reference['count'] > 0
        self.detector_checks = {}
        for d,color in enumerate(DETECTOR_COLORS,1):
            keep = valid & (np.arange(102)//17 == d-1)
            # Connect only consecutive populated wavelength channels, within a detector.
            y = np.where(keep, reference['flux_mjy'], np.nan)
            ax.plot(reference['wavelength_um'], y, color=color, alpha=.45,
                    lw=1.25, marker='o',markersize=3.8,zorder=2)
            checks = []
            for i in members[self.detector[members] == d]:
                check = a.capture_consistency(int(i), mask, epoch_ids=self.epoch_ids)
                checks.append(check)
                w,f,e = a.wavelength[i],a.flux[i],a.flux_err[i]
                if np.isfinite(w) and np.isfinite(f):
                    if np.isfinite(e) and e > 0:
                        ax.errorbar(w,f,yerr=e,fmt='none',color=color,alpha=.7,lw=1,capsize=2,zorder=3)
                    # Hollow marker for suspect measurements, rather than hiding them.
                    ax.scatter([w],[f],s=95,facecolor='white' if check['status']=='quality' else color,
                               edgecolor='#172033',linewidth=1.3,zorder=5)
                    if check['status']=='quality':
                        ax.scatter([w],[f],s=25,marker='x',color=color,zorder=6)
            if checks:
                # A discrepancy must not disappear in an average of detector checks.
                rank = {'consistent':0,'insufficient':1,'quality':2,'review':3}
                self.detector_checks[d] = max(checks,key=lambda c: rank[c['status']])
        statuses = {d:c['status'] for d,c in self.detector_checks.items()}
        matched = [d for d,v in statuses.items() if v=='consistent']
        review = [d for d,v in statuses.items() if v=='review']
        quality = [d for d,v in statuses.items() if v=='quality']
        total = len(statuses)
        if not active:
            status,title,reason = 'insufficient','Select an observation','This channel has no retained images.'
        elif review:
            status,title = 'review','Worth another look'
            reason = 'Brightness differs in '+', '.join(f'D{d}' for d in review)+f'. {len(matched)} of {total} available bands agree.'
        elif total and len(matched)==total:
            status,title = 'consistent','Brightness looks consistent'
            reason = f'All {total} available bands agree with comparable observations.'
            if total == 1: reason += ' Only one band was captured.'
        elif matched:
            status,title = 'insufficient','Partial agreement'
            reason = f'{len(matched)} of {total} available bands agree; the others cannot be checked confidently.'
        elif quality:
            status,title,reason = 'quality','Check image quality','No band could confirm agreement; some measurements have quality issues.'
        else:
            status,title,reason = 'insufficient','Not enough information','No available band has enough reliable comparable measurements.'
        self.check = dict(status=status,title=title,reason=reason,matched=len(matched),available=total)
        self.status_text.set_text('Quick check · '+title)
        self.status_text.set_color({'consistent':'#23744e','review':'#a35320','quality':'#a35320','insufficient':'#526174'}[status])
        self.reason_text.set_text('Spectrum history · '+title)
        self.verification = self.verifier.evaluate(members, self.check)
        self.status_text.set_text(self.verification['title'])
        palette={'pass':'#23744e','review':'#a35320','unavailable':'#526174'}
        self.status_text.set_color(palette[self.verification['tone']])
        for (state,note),row in zip(self.verify_rows,self.verification['rows']):
            state.set_text({'pass':'Supports','review':'Check','unavailable':'Not confirmed'}[row['state']])
            state.set_color(palette[row['state']]); note.set_text(row['note'][:105])
        target_ref=next((t for t in self.verification.get('scores',[]) if t['role']=='target'),None)
        if target_ref is not None:
            ax.errorbar(target_ref['wavelength'],target_ref['flux'],yerr=target_ref['error'],fmt='D',color='#222222',markersize=4,capsize=2,zorder=7)
        elif self.verifier.catalog_check is not None:
            for name,w,f,e,_ in self.verifier.catalog_check['rows']:
                ax.errorbar(w,f,yerr=e,fmt='D',color='#222222',markersize=4,capsize=2,zorder=7)
        if not valid.any():
            ax.text(.03,.05,'No usable reference points in these days.',transform=ax.transAxes,fontsize=8,color='#64748b')
        elif active and not any(np.isfinite(a.flux[members]) & np.isfinite(a.wavelength[members])):
            ax.text(.03,.05,'No measurable brightness in this observation.',transform=ax.transAxes,fontsize=8,color='#64748b')

    def _redraw(self):
        from .plots import draw_zoom
        a=self.analyzer; d,c=self.key; idx=self.groups[self.key]
        color=DETECTOR_COLORS[d-1]
        for ax,key in self.tiles.items():
            for spine in ax.spines.values():
                spine.set_color('#172033' if key==self.key else DETECTOR_COLORS[key[0]-1])
                spine.set_linewidth(2.6 if key==self.key else .8)
        self.gallery.canvas.draw_idle()
        ax=self.ax_image; ax.clear(); ax.set_axis_on()
        active=len(idx)>0 and self.index in idx
        stack=self.image_view=='stack'
        for button,selected in ((self.original_button,not stack),(self.stack_button,stack)):
            button.color='#dbe5f0' if selected else '#f1f5f9'
            button.ax.set_facecolor(button.color)
            button.label.set_fontweight('bold' if selected else 'normal')
        self.mode.ax.set_visible(stack); self.mode_label.set_visible(stack)
        self.control_button.ax.set_visible(not stack and active)
        label = self._band_label(d, c)
        self.header.set_text(self.name)
        members = self.observations[self.observation] if active else np.array([],dtype=int)
        present = set(self.detector[members])
        self.subtitle.set_text((f'Observation {self.observation+1} of {self.n_obs}  |  '
                                f'{len(present)}/6 detector bands available  |  {label}')
                               if active else f'{label}  |  No observation in this channel')
        for det, button in enumerate(self.detector_buttons,1):
            available = det in present
            selected = available and det == self.detector[self.index] and not stack
            button.set_active(available)
            button.color = '#dbe5f0' if selected else '#f1f5f9'
            button.ax.set_facecolor(button.color)
            button.label.set_color(DETECTOR_COLORS[det-1] if available else '#aab2bf')
            button.label.set_fontweight('bold' if selected else 'normal')
        self.subtitle.set_color(color)
        if stack:
            mean,var,n=self.stacks[self.key]
            mode=self.mode.value_selected
            display_mode = 'S/N' if mode == 'Clear view' else 'Brightness' if mode == 'Brightness' else 'Coverage'
            data=mean/np.sqrt(var) if display_mode=='S/N' else mean if display_mode=='Brightness' else np.where(n>0,n,np.nan)
            finite=data[np.isfinite(data)]
            lim=(-3,12) if display_mode=='S/N' else ((0,max(1,int(n.max()))) if display_mode=='Coverage' else
                 ((float(np.percentile(finite,2)),float(np.percentile(finite,99))) if finite.size else (0,1)))
            if lim[1]<=lim[0]: lim=(lim[0],lim[0]+1)
            if self.pretty:
                self._show_pretty_stack(ax, data, display_mode, lim)
            else:
                ax.imshow(data,origin='lower',cmap='gray' if display_mode!='Coverage' else 'viridis',
                          vmin=lim[0],vmax=lim[1],interpolation='nearest')
            center=self.size//2
            ax.add_patch(Circle((center,center),cfg.APERTURE_RADIUS_PIX,fill=False,ls='--',color='deepskyblue'))
            self.image_title.set_text(f'Combined image · {len(idx)} observations')
            self.image_note.set_text('Blue circle = expected position; images aligned to that position.\n'
                                     'Spectrum compares all bands from the selected observation.\n'
                                     'Combined image uses all dates.')
            if not np.any(n):
                ax.text(.5,.5,'No usable image pixels',transform=ax.transAxes,ha='center',color='white',bbox=dict(facecolor='#333333'))
        elif active:
            # find_partners() returns indices (or None), but draw_zoom/render
            # expects the matching zoom dictionary, including its pixels and WCS.
            partner_idx = self.partners[self.index]
            partner = self.zooms[partner_idx] if partner_idx is not None else None
            draw_zoom(ax, self.zooms[self.index], self.index, self.n, a.mjd[0], self.movement,
                      a.brightness, compact=True, pretty=self.pretty,
                      partner=partner, anchor=self.frame_anchors[self.index])
            ax.set_title('')
            i=self.index
            if self.show_controls:
                for x,y in self.verifier.background(i)['positions']:
                    ax.add_patch(Circle((x,y),cfg.APERTURE_RADIUS_PIX,fill=False,ls=':',lw=1,color='#47d7ee'))
            self.image_title.set_text(f'D{self.detector[i]} image · {self.zooms[i]["date_obs"][:16].replace("T", " ")}')
            detected=bool(self.movement.get('detected',np.ones(self.n,bool))[i])
            flagged=bool(self.movement['flagged'][i])
            if not detected or not a.reliable[i]: tracking='Tracking: low confidence in this observation.'
            elif flagged: tracking='Tracking: position flagged by the existing analysis.'
            else: tracking='Tracking: no movement warning from the existing analysis.'
            ref='expected position' if self.verifier.moving else 'first observed position'
            self.image_note.set_text(f'Red circle = measured position; blue = {ref}.\n'+tracking+('\nDotted cyan circles = nearby sky checks.' if self.show_controls else '')+('\nEnhanced colours and brightness are for display only.' if self.pretty else ''))
        else:
            self.image_title.set_text('This image')
            self.image_note.set_text('Choose a band with an observation, or browse with Prev / Next.')
            ax.text(.5,.5,'No observation available',ha='center',va='center',transform=ax.transAxes)
        # Hide native pixel coordinates in the beginner-facing view. They remain
        # available in the original tracking windows and unmodified zoom arrays.
        ax.set_xticks([]); ax.set_yticks([])
        self._draw_spectrum(active)
        self.detail.canvas.draw_idle()


def launch(show_original=True, show_fit=False, **run_kwargs):
    """Run the untouched pipeline, then open the atlas and existing tracking plots."""
    from .main import run
    from .plots import show_viewer, show_motion, show_time_series, show_spectrum
    run_kwargs.pop('show',None)
    results = run(show=False, **run_kwargs)
    # main.run() returns measurements, not an initialized SpectrumAnalyzer.
    # Construct it here; do not look up a possibly missing results['analyzer'].
    analyzer = SpectrumAnalyzer.from_results(results)
    results['analyzer'] = analyzer
    viewer = analyzer.show_bands()
    results['band_viewer'] = viewer
    if show_original:
        frames,tracked,movement=results['frames'],results['tracked'],results['movement']
        brightness=results.get('brightness')
        show_viewer(results['zooms'],tracked,movement,brightness)
        if brightness is None: show_motion(frames,tracked,movement)
        else:
            show_time_series(frames,tracked,movement,brightness)
            spectrum = results.get('spectrum')
            if spectrum is not None:
                show_spectrum(spectrum, tracked)
            else:
                print('Catalog spectrum unavailable; interactive SPHEREx comparison remains available.')
    if show_fit:
        results['motion_fit'] = analyzer.motion_fit()
    plt.show()
    return results


if __name__ == '__main__':
    launch()
