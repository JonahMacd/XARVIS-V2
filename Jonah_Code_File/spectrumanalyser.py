"""SPHEREx pairwise diagnostics and 102-channel target-wavelength explorer.

Run with python -m Jonah_Code_File.spectrumanalyser to open the new and original
tracking views without editing any other module.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from .import config as cfg
from .measure import reliable_frames, robust_scatter


class SpectrumAnalyzer:
    """Computes per-capture spectral baselines and pairwise comparisons for SPHEREx frames."""

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
                if not np.issubdtype(ix.dtype, np.integer):
                    raise TypeError("Use integer indices or a boolean mask")
                selected[ix] = True
        detector = np.array([f['detector'] for f in self.frames])
        mean, error, count = binned_spectrum(self.wavelength, self.flux,
            self.flux_err, detector, selected & self.reliable)
        return dict(wavelength_um=CENTERS.ravel().copy(), flux_mjy=mean,
                    flux_err_mjy=error, count=count)

    def show_bands(self, zooms, movement, tracked=None, *, size=41):
        """Open linked channel gallery and capture/spectrum/tracking detail.

        Pass results['zooms'], results['movement'], results['tracked'] from run().
        Returns the viewer; call plt.show() to display its figures.
        """
        return BandViewer(self, zooms, movement, tracked, size=size)

    def resolve_index(self, key):
        if isinstance(key, (int, np.integer)):
            if not (0 <= key < len(self.frames)):
                raise IndexError(f'Frame index {key} out of range (0..{len(self.frames) - 1})')
            return int(key)
        if str(key) in self._obs_to_idx:
            return self._obs_to_idx[str(key)]
        raise KeyError(f'Unknown obs_id: {key}')

    def expected_flux_at(self, wavelength_um, exclude_idx=None):
        """Estimate expected flux (mJy) from local SPHEREx neighbors or scaled 2MASS/AllWISE SED."""
        good = np.flatnonzero(self.reliable)
        if exclude_idx is not None:
            good = good[good != exclude_idx]

        if len(good):
            dist = np.abs(self.wavelength[good] - wavelength_um)
            candidates = good[dist <= cfg.SPECTRUM_WINDOW * wavelength_um]
            candidates = candidates[
                np.argsort(np.abs(self.wavelength[candidates] - wavelength_um))[:cfg.SPECTRUM_NEIGHBORS]
            ]
            if len(candidates) >= cfg.SPECTRUM_MIN_NEIGHBORS:
                return float(np.median(self.flux[candidates])), 'spherex_neighbors'

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
        """Return spectral metrics for a single capture."""
        idx = self.resolve_index(idx)
        f = self.frames[idx]
        wl = float(f['wavelength_um'])
        flux = float(f['flux_mjy'])
        err = float(f['flux_err_mjy'])

        if self.brightness is not None and np.isfinite(self.brightness['reference_mjy'][idx]):
            ref_mjy = float(self.brightness['reference_mjy'][idx])
            ref_source = 'spherex_neighbors'
        else:
            ref_mjy, ref_source = self.expected_flux_at(wl, exclude_idx=idx)

        if np.isfinite(ref_mjy) and ref_mjy > 0:
            deviation = (flux / ref_mjy) - 1.0
            scatter = float(self.brightness['scatter']) if self.brightness is not None else (
                robust_scatter(self.flux[self.reliable]) / max(np.nanmedian(self.flux[self.reliable]), 1e-9)
            )
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
            'wavelength_um': round(wl, 4),
            'flux_mjy': round(flux, 4),
            'flux_err_mjy': round(err, 4),
            'reference_mjy': round(ref_mjy, 4) if np.isfinite(ref_mjy) else None,
            'reference_source': ref_source,
            'pct_deviation': round(float(deviation * 100), 2) if np.isfinite(deviation) else None,
            'sigma_deviation': round(float(sigma_dev), 2) if np.isfinite(sigma_dev) else None,
            'is_reliable': bool(self.reliable[idx]),
            'n_bad_in_aperture': int(f['n_bad_in_aperture']),
            'is_aligned': aligned_ok,
            'is_brightness_flagged': is_flagged,
        }

    def find_comparison_partner(self, idx, prefer_normal=True):
        """Find the closest-wavelength capture from a different timestamp."""
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

    def compare(self, idx_a, idx_b):
        """Compare two captures and separate wavelength shifts from true brightness changes."""
        a = self.frame_spectral_profile(idx_a)
        b = self.frame_spectral_profile(idx_b)

        delta_days = round(float(b['mjd'] - a['mjd']), 4)
        delta_wl = round(float(b['wavelength_um'] - a['wavelength_um']), 4)
        same_band = bool(abs(delta_wl) / max(a['wavelength_um'], 1e-9) <= cfg.SPECTRUM_WINDOW)

        raw_diff_mjy = round(float(b['flux_mjy'] - a['flux_mjy']), 3)
        raw_ratio = round(float(b['flux_mjy'] / a['flux_mjy']), 3) if a['flux_mjy'] > 0 else None
        raw_diff_err = float(np.hypot(a['flux_err_mjy'], b['flux_err_mjy']))

        if a['reference_mjy'] and b['reference_mjy'] and raw_ratio is not None:
            expected_ratio = round(float(b['reference_mjy'] / a['reference_mjy']), 3)
            excess_ratio = round(float(raw_ratio / expected_ratio), 3) if expected_ratio > 0 else None
        else:
            expected_ratio, excess_ratio = None, None

        same_band_sigma = (
            round(float(abs(raw_diff_mjy) / raw_diff_err), 2)
            if (same_band and raw_diff_err > 0)
            else None
        )

        status, summary = self._classify(a, b, same_band, same_band_sigma)

        return {
            'capture_a': a,
            'capture_b': b,
            'comparison': {
                'delta_days': delta_days,
                'delta_wavelength_um': delta_wl,
                'same_wavelength_band': same_band,
                'raw_flux_diff_mjy': raw_diff_mjy,
                'raw_flux_ratio': raw_ratio,
                'expected_spectral_ratio': expected_ratio,
                'spectral_excess_ratio': excess_ratio,
                'same_band_sigma': same_band_sigma,
                'status_code': status,
                'summary': summary,
            },
        }

    def _classify(self, a, b, same_band, same_band_sigma):
        if not a['is_reliable'] or not b['is_reliable']:
            bad = [x['label'] for x in (a, b) if not x['is_reliable']]
            return 'detector_glitch', f"{' and '.join(bad)} had a bad pixel/cosmic ray inside the aperture."
        if not a['is_aligned'] or not b['is_aligned']:
            return 'alignment_warning', 'Telescope pointing offset exceeded tolerance in one capture.'
        if (
            a['is_brightness_flagged']
            or b['is_brightness_flagged']
            or (same_band_sigma is not None and same_band_sigma >= cfg.BRIGHTNESS_SIGMA)
        ):
            return (
                'real_variability',
                f"Real brightness change between {a['label']} ({a['flux_mjy']:.2f} mJy) "
                f"and {b['label']} ({b['flux_mjy']:.2f} mJy).",
            )
        if not same_band:
            direction = 'brighter' if b['flux_mjy'] > a['flux_mjy'] else 'dimmer'
            return (
                'wavelength_shift',
                f"{b['label']} is {direction} than {a['label']} ({b['flux_mjy']:.2f} vs {a['flux_mjy']:.2f} mJy) "
                f"due to wavelength change ({b['wavelength_um']:.2f} vs {a['wavelength_um']:.2f} µm).",
            )
        return (
            'consistent',
            f"{a['label']} and {b['label']} match in wavelength (~{a['wavelength_um']:.2f} µm) and brightness.",
        )

    def plot_diagnostic_comparison(self, zooms, idx_a=0, idx_b=None, tracked=None):
        """Temporary Matplotlib plot showing Cutout A, Cutout B, and their positions on the spectrum."""
        if idx_b is None:
            idx_b = self.find_comparison_partner(idx_a)
            if idx_b is None:
                idx_b = min(1, len(self.frames) - 1)

        report = self.compare(idx_a, idx_b)
        prof_a, prof_b, comp = report['capture_a'], report['capture_b'], report['comparison']

        fig = plt.figure(figsize=(13, 5.2))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.45])

        for panel, (idx, prof, color) in enumerate([(idx_a, prof_a, 'tab:blue'), (idx_b, prof_b, 'tab:orange')]):
            ax = fig.add_subplot(gs[0, panel])
            img = zooms[idx]['image']
            pixels = img[np.isfinite(img)]
            sky, noise = np.median(pixels), robust_scatter(pixels)
            ax.imshow(img, origin='lower', cmap='gray', vmin=sky - 3 * noise, vmax=sky + 15 * noise, interpolation='nearest')
            cx, cy = zooms[idx]['catalog_xy']
            ax.add_patch(Circle((cx, cy), cfg.APERTURE_RADIUS_PIX, edgecolor=color, facecolor='none', lw=2))
            ax.set_title(
                f"Capture {'A' if panel == 0 else 'B'}: {prof['label']} ({prof['date_obs'][:10]})\n"
                f"λ = {prof['wavelength_um']:.3f} µm | Flux = {prof['flux_mjy']:.2f} ± {prof['flux_err_mjy']:.2f} mJy\n"
                f"Baseline = {prof['reference_mjy']} mJy ({prof['pct_deviation']}%)",
                fontsize=9,
                color=color,
                fontweight='bold',
            )
            ax.set_xlabel('X (px)')
            if panel == 0:
                ax.set_ylabel('Y (px)')

        ax_spec = fig.add_subplot(gs[0, 2])
        ok = self.reliable
        ax_spec.errorbar(
            self.wavelength[ok], self.flux[ok], yerr=self.flux_err[ok],
            fmt='o', color='0.65', ms=3.5, lw=0.6, label='All reliable captures'
        )
        if np.any(~ok):
            ax_spec.scatter(self.wavelength[~ok], self.flux[~ok], marker='x', color='0.4', s=25, label='Bad pixel in aperture')

        for tag, prof, color in [('A', prof_a, 'tab:blue'), ('B', prof_b, 'tab:orange')]:
            ax_spec.scatter(
                [prof['wavelength_um']], [prof['flux_mjy']],
                color=color, s=95, zorder=5, edgecolor='black', label=f"Capture {tag} ({prof['label']})"
            )
            if prof['reference_mjy'] is not None:
                ax_spec.scatter(
                    [prof['wavelength_um']], [prof['reference_mjy']],
                    marker='D', facecolor='none', edgecolor=color, s=80, lw=1.8, zorder=5,
                    label=f"Expected {tag} baseline"
                )

        ax_spec.set_xlabel('Wavelength (µm)')
        ax_spec.set_ylabel('Flux Density (mJy)')
        ax_spec.set_title('Spectrum Comparison Verification', fontsize=10)
        ax_spec.legend(fontsize=7.5, loc='best')

        title = tracked['name'] if tracked else 'SPHEREx Target'
        fig.suptitle(f"[{comp['status_code'].upper()}] {title}: {comp['summary']}", fontsize=10.5)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        return fig

# SPHEREx Explanatory Supplement v2.1, Tables 29–34 (2026-11-10).
# Rounded median wavelengths, not the geometric detector channel map.
CENTERS = np.array([
[.753,.771,.790,.809,.828,.848,.868,.889,.910,.932,.955,.978,1.001,1.025,1.050,1.075,1.102],
[1.113,1.139,1.167,1.195,1.224,1.254,1.284,1.315,1.346,1.378,1.412,1.446,1.481,1.517,1.554,1.592,1.631],
[1.656,1.695,1.735,1.775,1.818,1.861,1.905,1.949,1.995,2.041,2.089,2.139,2.189,2.240,2.292,2.345,2.399],
[2.445,2.506,2.575,2.646,2.719,2.794,2.871,2.951,3.033,3.118,3.204,3.293,3.383,3.476,3.570,3.667,3.768],
[3.834,3.867,3.901,3.936,3.970,4.005,4.041,4.076,4.112,4.148,4.184,4.221,4.257,4.294,4.331,4.368,4.405],
[4.435,4.468,4.500,4.533,4.567,4.601,4.635,4.669,4.704,4.739,4.774,4.808,4.843,4.879,4.915,4.952,4.989]])
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




class BandViewer:
    """Linked target-centered stacks and the pipeline's unchanged tracking results.

    The original loader retains one wavelength per exposure at the target. Thus
    channel membership is assigned to a whole cutout by its target wavelength;
    this is not a per-pixel calibrated spectral cube. Registration uses only WCS
    and the predicted/catalog center, never the detected brightness peak.
    """
    def __init__(self, analyzer, zooms, movement, tracked=None, size=41):
        from matplotlib.widgets import Slider, RadioButtons
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
        self.detector = np.array([f['detector'] for f in analyzer.frames])
        self.channels = np.full(self.n, -1, int)
        for d in range(1,7):
            m = self.detector == d
            self.channels[m] = channel_index(d, analyzer.wavelength[m])
        self.days = analyzer.mjd - analyzer.mjd[0]
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
        self.key = (int(self.detector[0]), self._local_channel(0))
        self.index = 0
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
            ax.set_title(f'{title}\nn={len(self.groups[key])}',fontsize=7)
            if c == 0: ax.set_ylabel(f'D{d}',fontsize=10,weight='bold')
            if not np.any(n): ax.text(.5,.5,'—',ha='center',va='center',transform=ax.transAxes,color='white')
        covered = int(np.count_nonzero(self.sed['count']))
        self.gallery.suptitle(f'{self.name} | 102-channel atlas + six detector stacks\n'
                              f'{covered}/102 channels with reliable target photometry | '
                              'approximate target-wavelength bins; click any tile',fontsize=14)
        self.gallery.text(.5,.025,'All = broad detector stack  •  n = retained captures  •  '
                          'grayscale = nominal per-pixel S/N (−3 to +12)  •  — = no valid image pixels',ha='center',fontsize=10)
        self.detail = plt.figure(figsize=(14,8.5))
        grid = self.detail.add_gridspec(2,2,left=.075,right=.96,top=.83,bottom=.19,wspace=.31,hspace=.48)
        self.ax_stack = self.detail.add_subplot(grid[0,0])
        self.ax_native = self.detail.add_subplot(grid[0,1])
        self.ax_spec = self.detail.add_subplot(grid[1,0])
        self.ax_track = self.detail.add_subplot(grid[1,1])
        self.slider = Slider(self.detail.add_axes([.16,.065,.57,.035]),'Capture T',1,max(2,self.n),valinit=1,valstep=1,valfmt='%d')
        if self.n == 1: self.slider.set_active(False)
        self.mode = RadioButtons(self.detail.add_axes([.79,.018,.18,.09]),['S/N','Brightness','Coverage'])
        self.slider.on_changed(self._on_slider)
        self.mode.on_clicked(lambda _: self._redraw())
        self.gallery.canvas.mpl_connect('button_press_event',self._on_tile)
        self.detail.canvas.mpl_connect('key_press_event',self._on_key)
        self.detail.canvas.mpl_connect('pick_event',self._on_pick)
        # Matplotlib keeps weak refs to bound callbacks; keep the viewer alive.
        self.gallery._band_viewer = self.detail._band_viewer = self
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
        idx=self.groups[key]
        if len(idx):
            # Keep the current frame if it belongs to the chosen group; otherwise
            # select the closest time to the previous frame.
            closest=idx[np.argmin(np.abs(self.days[idx]-self.days[self.index]))]
            if self.index not in idx: self.index=int(closest)
            self._syncing=True
            try: self.slider.set_val(self.index+1)
            finally: self._syncing=False
        self._redraw()

    def _on_slider(self,value):
        if self._syncing: return
        self.index=min(self.n-1,max(0,int(value)-1))
        # Chronological browsing follows the actual detector and wavelength.
        self.key=(int(self.detector[self.index]),self._local_channel(self.index))
        self._redraw()

    def _on_key(self,event):
        if event.key in ('left','right'):
            self.slider.set_val(np.clip(self.index+1+(1 if event.key=='right' else -1),1,self.n))

    def _on_pick(self,event):
        idx=getattr(event.artist,'_frame_indices',None)
        if idx is not None and len(event.ind):
            self.slider.set_val(int(idx[event.ind[0]])+1)

    def _redraw(self):
        from .plots import draw_zoom, circle_legend
        a=self.analyzer; d,c=self.key; idx=self.groups[self.key]
        for ax,key in self.tiles.items():
            for spine in ax.spines.values():
                spine.set_color('#e39b36' if key==self.key else '#666666')
                spine.set_linewidth(2.2 if key==self.key else .4)
        self.gallery.canvas.draw_idle()
        ax=self.ax_stack; ax.clear()
        mean,var,n=self.stacks[self.key]
        mode=self.mode.value_selected
        data=mean/np.sqrt(var) if mode=='S/N' else mean if mode=='Brightness' else np.where(n>0,n,np.nan)
        finite=data[np.isfinite(data)]
        lim=(-3,12) if mode=='S/N' else ((0,max(1,int(n.max()))) if mode=='Coverage' else
             ((float(np.percentile(finite,2)),float(np.percentile(finite,99))) if finite.size else (0,1)))
        if lim[1]<=lim[0]: lim=(lim[0],lim[0]+1)
        ax.imshow(data,origin='lower',cmap='gray' if mode!='Coverage' else 'viridis',
                  vmin=lim[0],vmax=lim[1],interpolation='nearest')
        center=self.size//2
        ax.add_patch(Circle((center,center),cfg.APERTURE_RADIUS_PIX,fill=False,ls='--',color='deepskyblue'))
        label='all wavelengths' if c==0 else f'C{c:02d} ≈ {CENTERS[d-1,c-1]:.3f} µm'
        ax.set_title(f'D{d} | {label}\n{len(idx)} captures · {mode} · center coverage {n[center,center]}',fontsize=10)
        ax.set(xlabel='Target-relative x (east ←)',ylabel='Target-relative y (north ↑)')
        if not np.any(n): ax.text(.5,.5,'No valid image pixels',transform=ax.transAxes,ha='center',color='white',bbox=dict(facecolor='#333333'))
        ax=self.ax_native; ax.clear()
        active=len(idx)>0 and self.index in idx
        if active:
            draw_zoom(ax,self.zooms[self.index],self.index,self.n,a.mjd[0],self.movement,a.brightness)
            ax.set(xlabel='Original cutout x (px)',ylabel='Original cutout y (px)')
            ax.legend(handles=circle_legend(self.movement),loc='lower left',fontsize=7)
        else:
            ax.text(.5,.5,'No retained captures in this channel',ha='center',va='center',transform=ax.transAxes)
            ax.set_axis_off()
        if active: ax.set_axis_on()
        ax=self.ax_spec; ax.clear()
        colors=plt.get_cmap('tab10')
        for det in range(1,7):
            m=(self.detector==det)&self.good
            if m.any():
                ax.errorbar(a.wavelength[m],a.flux[m],yerr=a.flux_err[m],fmt='.',ms=4,alpha=.38,color=colors(det-1))
            sl=slice((det-1)*17,det*17); s=self.sed
            ok=s['count'][sl]>0
            ax.errorbar(s['wavelength_um'][sl][ok],s['flux_mjy'][sl][ok],yerr=s['flux_err_mjy'][sl][ok],
                        fmt='o',ms=4,color=colors(det-1),label=f'D{det}')
        bad=(~self.good)&np.isfinite(a.wavelength)&np.isfinite(a.flux)
        if bad.any(): ax.scatter(a.wavelength[bad],a.flux[bad],marker='x',color='0.5',s=10,label='Excluded')
        if c: ax.axvline(CENTERS[d-1,c-1],color='#e39b36',ls='--',lw=1)
        else: ax.axvspan(*LIMITS[d-1],color='#e39b36',alpha=.12)
        if active: ax.scatter(a.wavelength[self.index],a.flux[self.index],s=100,facecolors='none',edgecolors='red',zorder=5)
        ax.set(xlim=(.7,5.05),xlabel='Wavelength (µm)',ylabel='Aperture flux density (mJy)',
               title='Live spectrum | markers = weighted channel means')
        ax.legend(ncol=3,fontsize=7,loc='best'); ax.grid(alpha=.18)
        ax=self.ax_track; ax.clear()
        moving='residual_arcsec' in self.movement
        values=np.asarray(self.movement['residual_arcsec' if moving else 'offset_arcsec'])
        detected=np.asarray(self.movement.get('detected',np.ones(self.n,bool)),bool)
        flags=np.asarray(self.movement['flagged'],bool)
        for det in range(1,7):
            for ok,marker in [(True,'o'),(False,'x')]:
                keep=(self.detector==det)&(detected==ok)&np.isfinite(values)
                indices=np.flatnonzero(keep)
                if len(indices):
                    dots=ax.scatter(self.days[indices],values[indices],s=15,marker=marker,color=colors(det-1),alpha=.65,picker=5)
                    dots._frame_indices=indices
        if len(idx): ax.scatter(self.days[idx],values[idx],s=42,facecolors='none',edgecolors='#e39b36',label='Selected band')
        if flags.any(): ax.scatter(self.days[flags],values[flags],s=85,facecolors='none',edgecolors='red',label='Tracking flag')
        if active:
            ax.axvline(self.days[self.index],color='0.4',ls=':',lw=1)
            ax.scatter(self.days[self.index],values[self.index],marker='s',s=105,facecolors='none',edgecolors='black',label=f'T{self.index+1}')
        ax.set(xlabel='Days since T1',ylabel='Ephemeris residual (arcsec)' if moving else 'Offset from T1 (arcsec)',
               title='Existing tracking measurements | click a capture')
        ax.grid(alpha=.18)
        if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=7,loc='best')
        self.detail.suptitle(f'{self.name} | D{d} {label}\n'
                             'Local-sky-subtracted stack • native tracking • spectrum • time',fontsize=13)
        self.detail.canvas.draw_idle()


def launch(show_original=True, **run_kwargs):
    """Run the untouched pipeline, then open the atlas and existing tracking plots."""
    from .main import run
    from .plots import show_viewer, show_motion, show_time_series, show_spectrum
    run_kwargs.pop('show',None)
    results=run(show=False,**run_kwargs)
    results['band_viewer']=results['analyzer'].show_bands(results['zooms'],results['movement'],results['tracked'])
    if show_original:
        frames,tracked,movement=results['frames'],results['tracked'],results['movement']
        brightness=results.get('brightness')
        show_viewer(results['zooms'],tracked,movement,brightness)
        if brightness is None: show_motion(frames,tracked,movement)
        else:
            show_time_series(frames,tracked,movement,brightness)
            show_spectrum(results['spectrum'],tracked)
    plt.show()
    return results


if __name__ == '__main__':
    launch()
