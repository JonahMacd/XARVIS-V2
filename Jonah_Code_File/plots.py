"""Figures: the T1 ... Tn zooms, the movement over time, and the spectrum check."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from . import config as cfg


def show_frames(zooms, tracked, movement):
    """T1 ... Tn side by side in grayscale (original pixels), all on one shared brightness scale.
    Blue dashed circle = where the object was at T1. Red circle = where it is in this frame."""
    # scale from the typical sky level and its pixel-to-pixel noise (median and MAD), so neither a few
    # extreme pixels nor bright stars set the contrast: black = sky - 3 noise, white = sky + 15 noise
    all_pixels = np.concatenate([z['image'][np.isfinite(z['image'])] for z in zooms])
    sky = np.median(all_pixels)
    noise = 1.4826 * np.median(np.abs(all_pixels - sky))

    n_cols = min(5, len(zooms))
    n_rows = int(np.ceil(len(zooms) / n_cols))
    fig = plt.figure(figsize=(3.8 * n_cols + 1, 4.2 * n_rows))

    for i, zoom in enumerate(zooms):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection=zoom['wcs'])
        # interpolation='nearest': draw each original pixel as a sharp square (no display smoothing)
        im = ax.imshow(zoom['image'], origin='lower', cmap='gray', vmin=sky - 3 * noise, vmax=sky + 15 * noise,
                       interpolation='nearest')
        ax.add_patch(Circle(zoom['t1_xy'], cfg.APERTURE_RADIUS_PIX + 0.6, edgecolor='deepskyblue',
                            facecolor='none', lw=1.4, ls='--'))
        ax.add_patch(Circle(zoom['object_xy'], cfg.APERTURE_RADIUS_PIX, edgecolor='red', facecolor='none', lw=1.2))
        ax.set_title(f"T{i + 1}  {zoom['date_obs'][:16].replace('T', ' ')}\n"
                     f"+{movement['days'][i]:.2f} d   λ = {zoom['wavelength_um']:.3f} µm   "
                     f"moved {movement['offset_arcsec'][i]:.2f}\"", fontsize=8.5)
        ax.coords[0].set_axislabel('RA' if i // n_cols == n_rows - 1 else ' ')
        ax.coords[1].set_axislabel('Dec' if i % n_cols == 0 else ' ')
        ax.coords[0].set_ticklabel(size=7)
        ax.coords[1].set_ticklabel(size=7)

    fig.colorbar(im, ax=fig.axes, label='MJy/sr (zodi subtracted)', fraction=0.02, pad=0.02)
    fig.legend(handles=[Line2D([], [], color='deepskyblue', ls='--', label='object at T1'),
                        Line2D([], [], color='red', label='object in this frame')],
               loc='upper right', fontsize=9)
    fig.suptitle(f"SPHEREx: {tracked['name']}   ({len(zooms)} frames, original "
                 f"{cfg.ZOOM_SIZE_PIX}x{cfg.ZOOM_SIZE_PIX} pixels)")
    return fig


def show_movement(movement, tracked):
    """Left: offset from the T1 position vs time. Right: the path on the sky (east to the left, as on the sky).
    Gray band = how well frames line up (alignment check): offsets inside it are not real movement."""
    days, offset, error = movement['days'], movement['offset_arcsec'], movement['error_arcsec']
    floor = movement['floor_arcsec']
    labels = [f'T{i + 1}' for i in range(len(days))]

    fig, (ax_time, ax_sky) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={'width_ratios': [1.6, 1]})

    ax_time.axhspan(0, floor, color='0.85', label=f'alignment limit ({floor:.2f}")')
    ax_time.errorbar(days, offset, yerr=error, fmt='o-', color='black', ms=5, lw=1, capsize=3,
                     label='offset from T1 position')
    for d, o, label in zip(days, offset, labels):
        ax_time.annotate(label, (d, o), textcoords='offset points', xytext=(4, 6), fontsize=8)
    ax_time.axhline(cfg.PIXEL_ARCSEC, color='red', ls=':', lw=1, label=f'1 SPHEREx pixel ({cfg.PIXEL_ARCSEC}")')
    ax_time.set(xlabel='Days since T1', ylabel='Offset from T1 position (arcsec)',
                title='Movement over time', ylim=(0, max(cfg.PIXEL_ARCSEC * 1.2, np.nanmax(offset + error) * 1.2)))
    ax_time.legend(fontsize=8, loc='upper left')

    sc = ax_sky.scatter(movement['d_ra_arcsec'], movement['d_dec_arcsec'], c=days, cmap='viridis', s=40, zorder=3)
    ax_sky.plot(movement['d_ra_arcsec'], movement['d_dec_arcsec'], color='0.6', lw=0.8, zorder=2)
    ax_sky.add_patch(Circle((0, 0), floor, color='0.85', zorder=1))
    for x, y, label in zip(movement['d_ra_arcsec'], movement['d_dec_arcsec'], labels):
        ax_sky.annotate(label, (x, y), textcoords='offset points', xytext=(4, 4), fontsize=8)
    limit = max(1.0, np.nanmax(np.abs(np.r_[movement['d_ra_arcsec'], movement['d_dec_arcsec']])) * 1.4)
    ax_sky.set(xlim=(limit, -limit), ylim=(-limit, limit), aspect='equal',      # east (+RA) to the left
               xlabel='ΔRA (arcsec, east ←)', ylabel='ΔDec (arcsec, north ↑)', title='Path on the sky')
    fig.colorbar(sc, ax=ax_sky, label='Days since T1')

    fig.suptitle(f"{tracked['name']}: position relative to T1")
    fig.tight_layout()
    return fig


def show_spectrum(check, tracked):
    """SPHEREx measurements vs the object's IRSA catalog brightness."""
    fig, ax = plt.subplots(figsize=(9, 5))
    wl, flux, err = check['points'].T
    ax.errorbar(wl, flux, yerr=err, fmt='o', color='black', ms=4, lw=0.8, label='SPHEREx (measurement circle)')
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
