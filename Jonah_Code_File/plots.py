"""Figures: a frame viewer for every image, movement and brightness over time, and the spectrum check."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.widgets import Slider

from . import config as cfg
from .measure import reliable_frames, robust_scatter

MAX_GRID_PANELS = 30        # show_frames draws one panel per frame; for more frames use show_viewer


def fmt_angle(arcsec):
    """An angle for a title: arcseconds below 1', arcminutes below 1 degree, then degrees."""
    if arcsec < 60:
        return f'{arcsec:.2f}"'
    return f"{arcsec / 60:.1f}'" if arcsec < 3600 else f'{arcsec / 3600:.2f}°'


def draw_zoom(ax, zoom, index, total, t0, movement, brightness=None, compact=False):
    """Draw one frame's zoom into `ax`: original pixels in grayscale, each pixel a sharp square.
    Red circle = where the object was found in this frame. Blue dashed circle = the reference position:
    where a star was at T1, or where the ephemeris predicted an asteroid (drawn only if it is in the zoom).
    `t0` = time of T1 (MJD). `brightness` = the star's brightness analysis (None for an asteroid).
    `compact` = shorter title, for the grid. Returns the image (for a color bar)."""
    # each frame is at its own wavelength, so each gets its own brightness scale (sky - 3 / + 15 noise)
    pixels = zoom['image'][np.isfinite(zoom['image'])]
    sky, noise = np.median(pixels), robust_scatter(pixels)
    im = ax.imshow(zoom['image'], origin='lower', cmap='gray', vmin=sky - 3 * noise, vmax=sky + 15 * noise,
                   interpolation='nearest')       # each original pixel as a sharp square
    # circles placed by sky position, converted to this frame's pixels (so they stay round)
    reference = movement['reference']
    reference = reference[index] if reference.shape else reference          # one position per frame, or one for all
    ref_x, ref_y = (float(v) for v in zoom['wcs'].world_to_pixel(reference))
    if 0 <= ref_x < zoom['image'].shape[1] and 0 <= ref_y < zoom['image'].shape[0]:
        ax.add_patch(Circle((ref_x, ref_y), cfg.APERTURE_RADIUS_PIX + 0.6,
                            edgecolor='deepskyblue', facecolor='none', lw=1.5, ls='--'))
    now_xy = zoom['wcs'].world_to_pixel_values(zoom['object_ra'], zoom['object_dec'])
    ax.add_patch(Circle((float(now_xy[0]), float(now_xy[1])), cfg.APERTURE_RADIUS_PIX,
                        edgecolor='red', facecolor='none', lw=1.3))

    moved = movement['flagged'][index]
    changed = bool(brightness['flagged'][index]) if brightness else False
    flag_label = movement.get('flag_label', 'MOVED')
    date = zoom['date_obs'][:16].replace('T', ' ')
    days = zoom['mjd'] - t0
    n_bad = zoom['n_bad_in_aperture']
    traveled = f"moved {fmt_angle(movement['offset_arcsec'][index])}"
    if compact:
        marks = (f'  {flag_label}' if moved else '') + ('  BRIGHTNESS' if changed else '') + ('  [bad px]' if n_bad else '')
        ax.set_title(f"T{index + 1}  {date}\n+{days:.2f} d   λ = {zoom['wavelength_um']:.3f} µm\n{traveled}{marks}",
                     fontsize=8)
    else:
        move_flag = f'  << {flag_label}' if moved else ''
        unreliable = f"   [{n_bad} bad pixel(s) in circle: not used]" if n_bad else ''
        if 'notes' in movement:                             # an asteroid: signal-to-noise and distance from ephemeris
            third = movement['notes'][index]
        else:                                               # a star: brightness against its own spectrum
            deviation = brightness['deviation'][index]      # NaN when too few frames at a similar wavelength
            versus = f'{deviation * 100:+.1f}%' if np.isfinite(deviation) else 'n/a (too few similar-wavelength frames)'
            third = f"vs spectrum {versus}{'  << BRIGHTNESS CHANGE' if changed else ''}"
        ax.set_title(f"T{index + 1} of {total}   {date}   +{days:.2f} d   detector {zoom['detector']}{unreliable}\n"
                     f"λ = {zoom['wavelength_um']:.3f} µm   {traveled}{move_flag}\n"
                     f"flux {zoom['flux_mjy']:.2f} ± {zoom['flux_err_mjy']:.2f} mJy   {third}", fontsize=9)
    return im


def circle_legend(movement):
    """Legend handles for the two circles (the blue one means different things for a star and an asteroid)."""
    return [Line2D([], [], color='deepskyblue', ls='--', label=movement['reference_label']),
            Line2D([], [], color='red', label='object found in this frame')]


def show_frames(zooms, tracked, movement, brightness=None, return_axes=False):
    """All zooms side by side in a grid, one panel per frame (for a handful of frames, e.g.
    run(max_images=10); up to MAX_GRID_PANELS). Each panel is scaled to its own sky level.

    return_axes=True returns (fig, zoom_axes) instead of just fig: zoom_axes[i] is the axes of frame T{i+1},
    so the caller can add its own overlays to the panels."""
    n = len(zooms)
    if n > MAX_GRID_PANELS:
        raise ValueError(f'{n} frames is too many for a grid (max {MAX_GRID_PANELS}) - use show_viewer instead')
    n_cols = min(5, n)
    n_rows = int(np.ceil(n / n_cols))
    fig = plt.figure(figsize=(3.8 * n_cols, 4.4 * n_rows))

    zoom_axes = []
    for i, zoom in enumerate(zooms):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection=zoom['wcs'])
        draw_zoom(ax, zoom, i, n, zooms[0]['mjd'], movement, brightness, compact=True)
        ax.coords[0].set_axislabel('RA' if i // n_cols == n_rows - 1 else ' ')
        ax.coords[1].set_axislabel('Dec' if i % n_cols == 0 else ' ')
        ax.coords[0].set_ticklabel(size=7)
        ax.coords[1].set_ticklabel(size=7)
        zoom_axes.append(ax)

    fig.legend(handles=circle_legend(movement), loc='upper right', fontsize=9)
    fig.suptitle(f"SPHEREx: {tracked['name']}   ({n} frames, original pixels)")
    return (fig, zoom_axes) if return_axes else fig


def show_viewer(zooms, tracked, movement, brightness=None):
    """Step through every frame (original pixels, grayscale). Red circle = where the object was found in this
    frame; blue dashed circle = the reference position (see draw_zoom).
    Controls: slider, left/right arrow keys, F = jump to the next flagged frame."""
    n = len(zooms)
    flagged = np.flatnonzero(movement['flagged'] | (brightness['flagged'] if brightness else False))
    fig = plt.figure(figsize=(8.5, 9))
    slider = Slider(fig.add_axes([0.14, 0.04, 0.66, 0.03]), 'Frame', 1, n, valinit=1, valstep=1, valfmt='T%d')
    cbar_ax = fig.add_axes([0.86, 0.2, 0.025, 0.6])
    state = {'ax': None}

    def draw(i):
        zoom = zooms[i]
        if state['ax'] is not None:
            state['ax'].remove()
        ax = state['ax'] = fig.add_axes([0.12, 0.15, 0.68, 0.7], projection=zoom['wcs'])
        im = draw_zoom(ax, zoom, i, n, zooms[0]['mjd'], movement, brightness)
        ax.coords[0].set_axislabel('RA')
        ax.coords[1].set_axislabel('Dec')
        cbar_ax.clear()
        fig.colorbar(im, cax=cbar_ax, label='MJy/sr (zodi subtracted)')
        fig.canvas.draw_idle()

    def on_key(event):
        i = int(slider.val) - 1
        if event.key == 'right':
            slider.set_val(min(i + 2, n))
        elif event.key == 'left':
            slider.set_val(max(i, 1))
        elif event.key in ('f', 'F') and len(flagged):
            later = flagged[flagged > i]
            slider.set_val((later[0] if len(later) else flagged[0]) + 1)

    slider.on_changed(lambda value: draw(int(value) - 1))
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.legend(handles=circle_legend(movement), loc='upper right', fontsize=9)
    fig.suptitle(f"SPHEREx: {tracked['name']}   (←/→ step, F = next flagged frame: {len(flagged)} flagged)",
                 fontsize=10, y=0.995)
    draw(0)
    fig._slider = slider                                   # keep the slider alive (matplotlib holds weak refs)
    return fig


def show_time_series(frames, tracked, movement, brightness):
    """Left top: offset from the T1 position vs time. Left bottom: brightness vs the object's own spectrum
    vs time. Right: the path on the sky. Points are colored by wavelength; flagged frames are ringed red."""
    days = np.array([f['mjd'] for f in frames]) - frames[0]['mjd']
    wavelength = np.array([f['wavelength_um'] for f in frames])

    fig = plt.figure(figsize=(15, 8))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.7, 1])
    ax_move = fig.add_subplot(grid[0, 0])
    ax_bright = fig.add_subplot(grid[1, 0], sharex=ax_move)
    ax_sky = fig.add_subplot(grid[:, 1])
    color = dict(cmap='turbo', vmin=wavelength.min(), vmax=wavelength.max(), s=14, zorder=3)

    # movement: each frame's flag threshold depends on its detector (gray dashes)
    threshold = cfg.MOVEMENT_SIGMA * movement['scatter_arcsec']
    limit = float(np.nanmedian(threshold))
    ax_move.scatter(days, threshold, marker='_', s=40, color='0.65', zorder=1,
                    label=f'flag threshold ({cfg.MOVEMENT_SIGMA}x its detector\'s scatter)')
    reliable = reliable_frames(frames)

    def dots(ax, x, y, yerr=None):
        """Reliable frames as dots colored by wavelength; frames with a bad pixel in the circle as gray x's
        (shown for completeness - they aren't used, so their odd values aren't events)."""
        if yerr is not None:
            ax.errorbar(x[reliable], y[reliable], yerr=yerr[reliable], fmt='none', ecolor='0.75', lw=0.6)
        artist = ax.scatter(x[reliable], y[reliable], c=wavelength[reliable], **color)
        ax.scatter(x[~reliable], y[~reliable], marker='x', s=14, color='0.6', lw=0.8, zorder=2,
                   label='bad pixel in circle (not used)')
        return artist

    offset = movement['offset_arcsec']
    sc = dots(ax_move, days, offset, movement['error_arcsec'])
    ax_move.axhline(cfg.PIXEL_ARCSEC, color='red', ls=':', lw=1, label=f'1 SPHEREx pixel ({cfg.PIXEL_ARCSEC}")')
    ax_move.set(ylabel='Offset from T1 (arcsec)', title='Movement over time',
                ylim=(0, max(cfg.PIXEL_ARCSEC * 1.15, np.nanmax(offset[reliable]) * 1.15)))
    ax_move.legend(fontsize=8, loc='upper left')

    # brightness
    deviation = brightness['deviation'] * 100
    flux_err_pct = np.array([f['flux_err_mjy'] for f in frames]) / brightness['reference_mjy'] * 100
    band = cfg.BRIGHTNESS_SIGMA * brightness['scatter'] * 100
    ax_bright.axhspan(-band, band, color='0.88', label=f'±{cfg.BRIGHTNESS_SIGMA}x typical deviation ({band:.1f}%)')
    ax_bright.axhline(0, color='0.5', lw=0.8)
    dots(ax_bright, days, deviation, flux_err_pct)
    span = max(3 * band, np.nanmax(np.abs(deviation[reliable])) * 1.15)
    ax_bright.set(xlabel='Days since T1', ylabel='Brightness vs own spectrum (%)', title='Brightness over time',
                  ylim=(-span, span))
    ax_bright.legend(fontsize=8, loc='upper left')

    for ax, values, flags in ((ax_move, offset, movement['flagged']), (ax_bright, deviation, brightness['flagged'])):
        if flags.any():
            ax.scatter(days[flags], values[flags], s=90, facecolor='none', edgecolor='red', lw=1.5, zorder=4)

    # path on the sky
    d_ra, d_dec = movement['d_ra_arcsec'], movement['d_dec_arcsec']
    ax_sky.add_patch(Circle((0, 0), limit, color='0.88', zorder=1))
    dots(ax_sky, d_ra, d_dec)
    ax_sky.scatter([0], [0], marker='+', s=150, color='deepskyblue', zorder=4, label='T1 position')
    extent = max(1.0, np.nanmax(np.abs(np.r_[d_ra[reliable], d_dec[reliable]])) * 1.2)
    ax_sky.set(xlim=(extent, -extent), ylim=(-extent, extent), aspect='equal',     # east (+RA) to the left
               xlabel='ΔRA (arcsec, east ←)', ylabel='ΔDec (arcsec, north ↑)', title='Path on the sky')
    ax_sky.legend(fontsize=8, loc='upper right')
    fig.colorbar(sc, ax=[ax_move, ax_bright, ax_sky], label='Wavelength (µm)', fraction=0.025, pad=0.02)
    fig.suptitle(f"{tracked['name']}: {len(frames)} frames over {days[-1]:.0f} days")
    return fig


def show_motion(frames, tracked, movement):
    """Where the asteroid went (detected frames only; undetected ones are gray x's).
    Top left: its path across the sky - JPL Horizons' path as a line, detections as dots colored by date.
    Top right: distance travelled since the first detection, over time (colored by wavelength).
    Bottom left: how far each detection is from where the ephemeris said it would be (1 pixel = 6.15").
    Bottom right: its spectrum - brightness against wavelength, colored by date."""
    detected = movement['detected']
    mjd = np.array([f['mjd'] for f in frames])
    days = mjd - mjd[0]
    wavelength = np.array([f['wavelength_um'] for f in frames])
    flux = np.array([f['flux_mjy'] for f in frames])
    ra, dec = np.array([f['object_ra'] for f in frames]), np.array([f['object_dec'] for f in frames])

    ephemeris = tracked['ephemeris']
    on_path =np.degrees(np.interp(mjd, ephemeris.mjd, ephemeris.ra))
    ra = on_path + (ra - on_path + 180) % 360 - 180           # same 360-degree branch as the path (no wrap-around)

    fig, ((ax_sky, ax_dist), (ax_res, ax_spec)) = plt.subplots(2, 2, figsize=(14, 9))

    ax_sky.plot(np.degrees(ephemeris.ra), ephemeris.dec, color='0.75', lw=2, zorder=1, label='JPL Horizons path')
    sc = ax_sky.scatter(ra[detected], dec[detected], c=days[detected], cmap='viridis', s=16, zorder=3)
    ax_sky.scatter(ra[~detected], dec[~detected], marker='x', color='0.5', s=16, zorder=2, label='not detected')
    ax_sky.invert_xaxis()                                      # east (larger RA) to the left, as on the sky
    ax_sky.set(xlabel='RA (degrees)', ylabel='Dec (degrees)', title='Path across the sky')
    ax_sky.legend(fontsize=8)
    fig.colorbar(sc, ax=ax_sky, label='Days since first frame')

    sc = ax_dist.scatter(days[detected], movement['offset_arcsec'][detected] / 3600, c=wavelength[detected],
                         cmap='turbo', s=16)
    ax_dist.set(xlabel='Days since first frame', ylabel='Distance travelled since first detection (degrees)',
                title='Movement over time')
    fig.colorbar(sc, ax=ax_dist, label='Wavelength (µm)')

    residual = movement['residual_arcsec']
    ax_res.scatter(days[detected], residual[detected], c='black', s=10)
    ax_res.axhline(cfg.PIXEL_ARCSEC, color='red', ls=':', lw=1, label=f'1 SPHEREx pixel ({cfg.PIXEL_ARCSEC}")')
    ax_res.axhline(cfg.RESIDUAL_LIMIT_PX * cfg.PIXEL_ARCSEC, color='red', lw=1,
                   label=f'flag limit ({cfg.RESIDUAL_LIMIT_PX} px)')
    ax_res.set(xlabel='Days since first frame', ylabel='Distance from ephemeris position (arcsec)',
               title=f'Found where predicted: median {np.median(residual[detected]):.1f}"',
               ylim=(0, max(1.3 * cfg.RESIDUAL_LIMIT_PX * cfg.PIXEL_ARCSEC, 1.1 * np.max(residual[detected]))))
    ax_res.legend(fontsize=8)

    shown = detected & (flux > 0)
    sc = ax_spec.scatter(wavelength[shown], flux[shown], c=days[shown], cmap='viridis', s=16)
    ax_spec.set(xlabel='Wavelength (µm)', ylabel='Flux density (mJy)', yscale='log', title='Its spectrum')
    fig.colorbar(sc, ax=ax_spec, label='Days since first frame')

    fig.suptitle(f"{tracked['name']}: {int(detected.sum())} detections over {days[detected].max() - days[detected].min():.0f} days")
    fig.tight_layout()
    return fig


def show_spectrum(check, tracked):
    """SPHEREx measurements (every frame) vs the object's IRSA catalog brightness."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    wl, flux, err = check['points'].T
    ax.errorbar(wl, flux, yerr=err, fmt='o', color='black', ms=2.5, lw=0.5, label=f'SPHEREx ({len(wl)} frames)')
    for band, band_wl, band_flux, band_err in check['catalog']:
        ax.errorbar(band_wl, band_flux, yerr=band_err, fmt='s', color='red', mfc='none', ms=9, mew=1.5)
        ax.annotate(band, (band_wl, band_flux), textcoords='offset points', xytext=(6, 6), fontsize=8, color='red')
    ax.plot([], [], 's', color='red', mfc='none', label='IRSA catalog (2MASS, AllWISE)')
    ax.set(xlabel='Wavelength (µm)', ylabel='Flux density (mJy)',
           title=f"Spectrum check: {tracked['name']} - {'same object' if check['same_object'] else 'MISMATCH'}\n"
                 f"(SPHEREx/catalog = {check['scale']:.2f}, shape within {check['scatter'] * 100:.0f}%)")
    ax.legend()
    fig.tight_layout()
    return fig
