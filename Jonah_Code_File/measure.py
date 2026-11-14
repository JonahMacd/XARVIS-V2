"""Measurements: brightness, position, alignment check, movement, brightness changes, spectrum check."""

import numpy as np

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from astropy.wcs.utils import proj_plane_pixel_area

from . import config as cfg


def reliable_frames(frames):
    """True for frames with no bad pixel inside the measurement circle. Unreliable frames are still shown,
    but can't trigger a flag or influence the statistics."""
    return np.array([f['n_bad_in_aperture'] == 0 for f in frames])


def robust_scatter(values):
    """Typical spread (1 sigma equivalent) that ignores outliers: 1.4826 x median absolute deviation."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return 1.4826 * np.median(np.abs(values - np.median(values))) if values.size else np.nan


# ----------------------------------------------------------------------------
# Bad pixels
# ----------------------------------------------------------------------------

def bad_pixel_mask(image, flags, flags_header):
    """True where a pixel can't be trusted: flagged bad by SPHEREx, not a number, or a lone spike that the
    pipeline missed (one pixel far brighter than all 8 of its neighbors - real sources span several pixels)."""
    bad_bits = 0
    for key in flags_header['MSKN*']:
        if flags_header[key] in cfg.BAD_FLAG_NAMES:
            bad_bits |= 1 << int(key[4:])
    bad = ((flags & bad_bits) != 0) | ~np.isfinite(image)

    background = np.nanmedian(image)
    noise = robust_scatter(image)
    padded = np.pad(image - background, 1, constant_values=np.nan)
    ny, nx = image.shape
    neighbors = np.stack([padded[1 + dy:1 + dy + ny, 1 + dx:1 + dx + nx]
                          for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)])
    brightest_neighbor = np.nanmax(neighbors, axis=0).clip(min=noise)
    above = image - background
    spikes = (above > cfg.SPIKE_MIN_SIGMA * noise) & (above > cfg.SPIKE_NEIGHBOR_RATIO * brightest_neighbor)
    return bad | spikes


# ----------------------------------------------------------------------------
# Brightness and position in one frame (run while downloading, one call per image)
# ----------------------------------------------------------------------------

def measure_flux(frame, coord):
    """Brightness (mJy): sum inside the measurement circle minus the local background, measured in the
    ring around it (sigma-clipped so neighboring stars in the ring are ignored). Bad pixels are left out;
    returns (flux, error, number of bad pixels inside the circle)."""
    x, y = frame['wcs'].world_to_pixel(coord)
    yy, xx = np.indices(frame['image'].shape)
    r = np.hypot(xx - float(x), yy - float(y))
    image, variance, good = frame['image'], frame['variance'], ~frame['bad']
    circle = r <= cfg.APERTURE_RADIUS_PIX
    in_aperture = circle & good
    in_annulus = (r >= cfg.ANNULUS_PIX[0]) & (r <= cfg.ANNULUS_PIX[1]) & good
    n_bad = int((circle & ~good).sum())
    if in_annulus.sum() < 5 or circle.sum() < 9 or in_aperture.sum() == 0:
        return np.nan, np.nan, n_bad

    _, background, background_std = sigma_clipped_stats(image[in_annulus], sigma=3.0)
    n = in_aperture.sum()
    flux = np.sum(image[in_aperture] - background)
    flux_var = np.sum(variance[in_aperture]) + (n * 1.2533 * background_std / np.sqrt(in_annulus.sum())) ** 2
    to_mjy = (proj_plane_pixel_area(frame['wcs']) * u.deg**2).to_value(u.sr) * 1e9   # MJy/sr * pixels -> mJy
    return float(flux * to_mjy), float(np.sqrt(flux_var) * to_mjy), n_bad


def centroid(image, variance, x, y, half=1):
    """Brightness-weighted center of the (2*half+1)^2 pixels around (x, y), with its uncertainty from the
    pixel noise. Returns (cx, cy, err_x, err_y) in pixels, or None near the edge / with no signal."""
    cx, cy = int(np.round(float(x))), int(np.round(float(y)))
    if not (half <= cx < image.shape[1] - half and half <= cy < image.shape[0] - half):
        return None
    box = image[cy - half:cy + half + 1, cx - half:cx + half + 1]
    box_var = np.nan_to_num(variance[cy - half:cy + half + 1, cx - half:cx + half + 1])
    weights = np.nan_to_num(box - np.nanmedian(image)).clip(min=0)
    total = weights.sum()
    if total <= 0:
        return None
    dy, dx = np.mgrid[-half:half + 1, -half:half + 1]
    ox, oy = (weights * dx).sum() / total, (weights * dy).sum() / total
    err_x = np.sqrt((box_var * (dx - ox) ** 2).sum()) / total
    err_y = np.sqrt((box_var * (dy - oy) ** 2).sum()) / total
    return cx + ox, cy + oy, err_x, err_y


def locate_object(frame):
    """Where the object actually is in this frame: brightest pixel within SEARCH_RADIUS_PIX of the expected
    spot (so a moved object is still found), then the brightness-weighted center around it, converted to
    RA/Dec with this frame's own pointing."""
    x, y = frame['catalog_xy']
    clean = np.where(frame['bad'], np.nan, frame['image'])         # bad pixels can't pull the position
    yy, xx = np.indices(clean.shape)
    near = (np.hypot(xx - x, yy - y) <= cfg.SEARCH_RADIUS_PIX) & np.isfinite(clean)
    if not near.any():
        return {'object_found': False}
    peak = np.argmax(np.where(near, clean, -np.inf))
    c = centroid(clean, frame['variance'], xx.flat[peak], yy.flat[peak])
    if c is None:
        return {'object_found': False}
    sky = frame['wcs'].pixel_to_world(c[0], c[1])
    return {'object_found': True, 'object_ra': float(sky.ra.deg), 'object_dec': float(sky.dec.deg),
            'object_err_arcsec': float(np.hypot(c[2], c[3]) * cfg.PIXEL_ARCSEC)}


# ----------------------------------------------------------------------------
# Alignment check (verifies, never adjusts)
# ----------------------------------------------------------------------------

def check_alignment(frames, field_stars):
    """In every frame, compare where moderately bright catalog stars actually appear with where SPHEREx's
    pointing puts them (median over many stars, so one star with real motion or a saturated halo can't fake
    a misalignment). Stores each frame's offset as frame['alignment_px']; nothing is moved."""
    ks = np.ma.filled(field_stars['k_m'], 99.0)
    stars = SkyCoord(field_stars['ra'], field_stars['dec'], unit='deg')[(ks > 9) & (ks < 12)]  # not saturated

    for frame in frames:
        xs, ys = frame['wcs'].world_to_pixel(stars)
        clean = np.where(frame['bad'], np.nan, frame['image'])
        ny, nx = clean.shape
        offsets = []
        for x, y in zip(xs, ys):
            if 3 <= x < nx - 3 and 3 <= y < ny - 3:
                c = centroid(clean, frame['variance'], x, y)
                if c:
                    offsets.append((c[0] - x, c[1] - y))
        frame['alignment_px'] = float(np.hypot(*np.median(offsets, axis=0))) if len(offsets) >= 3 else np.nan

    alignment = np.array([f['alignment_px'] for f in frames])
    bad = np.flatnonzero(alignment > cfg.ALIGNMENT_TOLERANCE_PIX)
    print(f"Alignment check ({'OK' if not len(bad) else f'WARNING: {len(bad)} frames above tolerance'}): "
          f"{len(stars)} catalog stars appear where SPHEREx's pointing puts them, to within "
          f'{np.nanmedian(alignment):.2f} px typical, {np.nanmax(alignment):.2f} px worst '
          f'({np.nanmedian(alignment) * cfg.PIXEL_ARCSEC:.2f}" / {np.nanmax(alignment) * cfg.PIXEL_ARCSEC:.2f}")'
          + (f"; unchecked (too few stars): {int(np.isnan(alignment).sum())}" if np.isnan(alignment).any() else ''))
    return alignment


# ----------------------------------------------------------------------------
# Movement of the object relative to T1
# ----------------------------------------------------------------------------

def track_object(frames):
    """Offset of the object from its T1 position in every frame.

    Flagged as movement if the offset is bigger than MOVEMENT_SIGMA x the typical frame-to-frame scatter of
    the offsets AND 3x that frame's own measurement noise + alignment offset (so neither a noisy frame nor a
    poorly aligned one can fake it)."""
    reliable = reliable_frames(frames)
    t1 = SkyCoord(frames[0]['object_ra'], frames[0]['object_dec'], unit='deg')
    positions = SkyCoord([f['object_ra'] for f in frames], [f['object_dec'] for f in frames], unit='deg')
    east, north = t1.spherical_offsets_to(positions)              # east = RA offset on the sky (RA * cos Dec)
    d_ra, d_dec = east.to_value(u.arcsec), north.to_value(u.arcsec)
    offset = np.hypot(d_ra, d_dec)
    error = np.array([f['object_err_arcsec'] for f in frames])
    alignment = np.nan_to_num([f['alignment_px'] for f in frames]) * cfg.PIXEL_ARCSEC

    # Typical scatter per detector: each detector covers different wavelengths, where the star's image is
    # sharper or broader, so positions are measured more or less precisely (worst at ~5 um).
    detector = np.array([f['detector'] for f in frames])
    scatter = np.full(len(frames), np.nan)
    for d in np.unique(detector):
        same = (detector == d) & reliable
        scatter[detector == d] = robust_scatter(np.r_[d_ra[same], d_dec[same]])
    flagged = reliable & (offset > cfg.MOVEMENT_SIGMA * scatter) & (offset > 3 * np.hypot(error, alignment))
    largest = int(np.argmax(np.where(reliable, offset, -1)))

    per_detector = ', '.join(f'D{d} {scatter[detector == d][0]:.2f}"' for d in np.unique(detector))
    print(f'Movement vs T1: typical scatter per axis {per_detector}; largest offset {offset[largest]:.2f}" '
          f'({offset[largest] / cfg.PIXEL_ARCSEC:.2f} px) in T{largest + 1}. '
          + (f'FLAGGED: {", ".join(f"T{i + 1}" for i in np.flatnonzero(flagged))}' if flagged.any()
             else f'No significant movement (nothing above {cfg.MOVEMENT_SIGMA}x its detector\'s scatter)'))
    return {'t1': t1, 'd_ra_arcsec': d_ra, 'd_dec_arcsec': d_dec, 'offset_arcsec': offset,
            'error_arcsec': error, 'scatter_arcsec': scatter, 'flagged': flagged}


# ----------------------------------------------------------------------------
# Brightness changes: compare each frame with the object's own spectrum
# ----------------------------------------------------------------------------

def brightness_changes(frames):
    """SPHEREx sees the object at a different wavelength in almost every frame, and its brightness depends
    on wavelength. So each frame is compared with a reference: the median brightness of the
    SPECTRUM_NEIGHBORS frames closest in wavelength (from all visits, excluding itself).
    'deviation' = fractional difference from that reference; flagged if more than BRIGHTNESS_SIGMA x the
    typical deviation and x its own measurement error."""
    reliable = reliable_frames(frames)
    wavelength = np.array([f['wavelength_um'] for f in frames])
    flux = np.array([f['flux_mjy'] for f in frames])
    flux_err = np.array([f['flux_err_mjy'] for f in frames])

    # reference spectrum from reliable frames only, so a glitched frame can't distort its neighbors' reference
    # Neighbors must be within SPECTRUM_WINDOW of the frame's wavelength: a reference built from far-away
    # wavelengths (where the star is naturally brighter or fainter) would fake a change.
    good = np.flatnonzero(reliable)
    reference = np.full_like(flux, np.nan)
    for i in range(len(frames)):
        distance = np.abs(wavelength[good] - wavelength[i])
        candidates = good[(distance <= cfg.SPECTRUM_WINDOW * wavelength[i]) & (good != i)]
        candidates = candidates[np.argsort(np.abs(wavelength[candidates] - wavelength[i]))[:cfg.SPECTRUM_NEIGHBORS]]
        if len(candidates) >= cfg.SPECTRUM_MIN_NEIGHBORS:
            reference[i] = np.median(flux[candidates])

    deviation = flux / reference - 1                       # NaN where there's no reference: can't be judged
    checkable = reliable & np.isfinite(deviation)
    scatter = robust_scatter(deviation[checkable])
    noise = np.hypot(flux_err / reference, scatter)
    flagged = checkable & (np.abs(deviation) > cfg.BRIGHTNESS_SIGMA * noise)
    if (reliable & ~checkable).any():
        print(f'  {int((reliable & ~checkable).sum())} frames have too few other frames at a similar wavelength '
              f'to judge their brightness')

    print(f'Brightness vs the object\'s own spectrum: typical deviation {scatter * 100:.1f}%. '
          + (f'FLAGGED ({flagged.sum()}): ' + ', '.join(f'T{i + 1} ({deviation[i] * 100:+.0f}%)'
                                                      for i in np.flatnonzero(flagged)[:15])
             if flagged.any() else f'No significant brightness changes (nothing above {cfg.BRIGHTNESS_SIGMA}x)'))
    return {'reference_mjy': reference, 'deviation': deviation, 'scatter': scatter, 'flagged': flagged}


# ----------------------------------------------------------------------------
# Spectrum check: is the object in the circle the catalog star?
# ----------------------------------------------------------------------------

def spectrum_check(tracked, frames):
    """Compare the object's SPHEREx spectrum (every frame's brightness at its wavelength) with its IRSA
    catalog brightness (2MASS + AllWISE). No extra downloads: uses the frames already measured."""
    points = np.array([(f['wavelength_um'], f['flux_mjy'], f['flux_err_mjy'])
                       for f, ok in zip(frames, reliable_frames(frames)) if ok])
    points = points[np.argsort(points[:, 0])]

    # catalog brightness interpolated (log-log) to each SPHEREx wavelength inside the catalog's range
    catalog = sorted(tracked['catalog_sed'], key=lambda p: p[1])
    cat_wl = np.array([p[1] for p in catalog])
    cat_flux = np.array([p[2] for p in catalog])
    inside = (points[:, 0] >= cat_wl.min()) & (points[:, 0] <= cat_wl.max()) & (points[:, 1] > 0)
    expected = np.exp(np.interp(np.log(points[inside, 0]), np.log(cat_wl), np.log(cat_flux)))
    ratio = points[inside, 1] / expected
    scale = float(np.median(ratio))
    scatter = float(robust_scatter(ratio / scale))                # disagreement in SHAPE, after scaling

    # The circle misses a little of the star's light and catches a little of its neighbors', so the level
    # can be off somewhat - but the spectrum SHAPE should match the catalog.
    same_object = 0.4 <= scale <= 1.3 and scatter <= 0.25
    verdict = 'PASS - same object' if same_object else 'FAIL - may not be the catalog object'
    print(f'Spectrum check ({verdict}): SPHEREx / catalog = {scale:.2f} on average, shape agrees to '
          f'{scatter * 100:.0f}% ({inside.sum()} frames between {cat_wl.min():.2f} and {cat_wl.max():.2f} um)')
    return {'points': points, 'catalog': catalog, 'scale': scale, 'scatter': scatter, 'same_object': same_object}
