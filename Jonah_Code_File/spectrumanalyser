"""Pairwise spectrum analysis and capture comparison for SPHEREx frames."""

import numpy as np
from . import config as cfg
from .measure import reliable_frames, robust_scatter


class SpectrumAnalyzer:
    """Analyze and compare the spectral measurements of any two captures in a dataset.

    Initialized from `frames` (and optional `brightness` / `spectrum` dicts from `measure.py`),
    so pairwise comparisons for the UI happen in O(1) without recomputing baselines.
    """

    def __init__(self, frames, brightness=None, spectrum=None):
        self.frames = frames
        self.brightness = brightness
        self.spectrum = spectrum

        self.reliable = reliable_frames(frames)
        self.wavelength = np.array([f['wavelength_um'] for f in frames], dtype=float)
        self.flux = np.array([f['flux_mjy'] for f in frames], dtype=float)
        self.flux_err = np.array([f['flux_err_mjy'] for f in frames], dtype=float)
        self.mjd = np.array([f['mjd'] for f in frames], dtype=float)
        self._obs_to_idx = {str(f['obs_id']): i for i, f in enumerate(frames)}

        # Precompute sorted reliable spectrum points for interpolation
        ok = self.reliable & np.isfinite(self.wavelength) & np.isfinite(self.flux)
        order = np.argsort(self.wavelength[ok])
        self.spec_wl = self.wavelength[ok][order]
        self.spec_flux = self.flux[ok][order]
        self.spec_err = self.flux_err[ok][order]

    @classmethod
    def from_results(cls, results):
        """Convenience constructor accepting the dict returned by `main.run()`."""
        return cls(
            frames=results['frames'],
            brightness=results.get('brightness'),
            spectrum=results.get('spectrum'),
        )

    def resolve_index(self, key):
        """Accept either an integer frame index (0-based) or an `obs_id` string."""
        if isinstance(key, (int, np.integer)):
            if not (0 <= key < len(self.frames)):
                raise IndexError(f'Frame index {key} out of range (0..{len(self.frames) - 1})')
            return int(key)
        if str(key) in self._obs_to_idx:
            return self._obs_to_idx[str(key)]
        raise KeyError(f'Unknown obs_id: {key}')

    def expected_flux_at(self, wavelength_um, exclude_idx=None):
        """Estimate expected flux (mJy) at `wavelength_um` using:
        1. Local median of reliable SPHEREx neighbors within +/- SPECTRUM_WINDOW (matching `brightness_changes`),
        2. Fallback to scaled 2MASS/AllWISE catalog SED interpolation if `spectrum` is available.
        Returns (expected_mjy, source_label).
        """
        # 1. Local SPHEREx neighbors (same logic as measure.brightness_changes)
        good = np.flatnonzero(self.reliable)
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
        if a['reference_mjy'] and b['reference_mjy']:
            expected_ratio = b['reference_mjy'] / a['reference_mjy']
            # Excess ratio = 1.0 means the brightness difference is 100% explained by wavelength
            spectral_excess_ratio = raw_ratio / expected_ratio
        else:
            expected_ratio = np.nan
            spectral_excess_ratio = np.nan

        # Direct statistical significance when in the same wavelength window
        same_band_sigma = (
            abs(raw_diff_mjy) / raw_diff_err if (same_wavelength_band and raw_diff_err > 0) else np.nan
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
        candidates = [i for i in range(len(self.frames)) if i != idx and self.reliable[i]]
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
                f"Significant brightness change detected between {a['label']} ({a['flux_mjy']:.2f} mJy) and "
                f"{b['label']} ({b['flux_mjy']:.2f} mJy) beyond normal spectral variation.",
            )

        if not same_band:
            dir_word = 'brighter' if b['flux_mjy'] > a['flux_mjy'] else 'dimmer'
            return (
                'wavelength_shift',
                f"{b['label']} appears {dir_word} than {a['label']} ({b['flux_mjy']:.2f} vs. {a['flux_mjy']:.2f} mJy) "
                f"because SPHEREx observed at different wavelengths ({b['wavelength_um']:.2f} µm vs. {a['wavelength_um']:.2f} µm), "
                f"consistent with the object's normal spectrum.",
            )

        return (
            'consistent',
            f"{a['label']} and {b['label']} were observed at matching wavelengths (~{a['wavelength_um']:.2f} µm) "
            f"and show consistent brightness.",
        )


def compare_captures(results, idx_a, idx_b):
    """Functional shortcut to compare two frames directly from `main.run()` output."""
    return SpectrumAnalyzer.from_results(results).compare(idx_a, idx_b)