"""Measurements: brightness, position, alignment check, movement tracking, spectrum check."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from astropy.wcs.utils import proj_plane_pixel_area

from . import config as cfg
from .frames import make_frame
from .irsa import fetch_cutout


# ----------------------------------------------------------------------------
# Brightness and position of a source in one frame
# ----------------------------------------------------------------------------

def measure_flux(frame, coord):
    """Brightness (mJy): sum inside the measurement circle minus the local background, measured in the
    ring around it (sigma-clipped so neighboring stars in the ring are ignored)."""
    x, y = frame['wcs'].world_to_pixel(coord)
    yy, xx = np.indices(frame['image'].shape)
    r = np.hypot(xx - float(x), yy - float(y))
    image, variance = frame['image'], frame['variance']
    in_aperture = (r <= cfg.APERTURE_RADIUS_PIX) & np.isfinite(image)
    in_annulus = (r >= cfg.ANNULUS_PIX[0]) & (r <= cfg.ANNULUS_PIX[1]) & np.isfinite(image)
    if in_annulus.sum() < 5:
        return np.nan, np.nan

    _, background, background_std = sigma_clipped_stats(image[in_annulus], sigma=3.0)
    n = in_aperture.sum()
    flux = np.sum(image[in_aperture] - background)
    flux_var = np.sum(variance[in_aperture]) + (n * 1.2533 * background_std / np.sqrt(in_annulus.sum())) ** 2
    to_mjy = (proj_plane_pixel_area(frame['wcs']) * u.deg**2).to_value(u.sr) * 1e9   # MJy/sr * pixels -> mJy
    return float(flux * to_mjy), float(np.sqrt(flux_var) * to_mjy)


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


# ----------------------------------------------------------------------------
# Alignment check (verifies, never adjusts)
# ----------------------------------------------------------------------------

def check_alignment(zooms, field_stars):
    """In every zoom, compare where moderately bright catalog stars actually appear with where SPHEREx's
    pointing puts them. The median over many stars is robust: one star with a real motion or a saturated
    halo can't fake a misalignment. Returns the worst frame's offset in pixels (the 'noise floor' for
    movement measurements)."""
    ks = np.ma.filled(field_stars['k_m'], 99.0)
    stars = SkyCoord(field_stars['ra'], field_stars['dec'], unit='deg')[(ks > 9) & (ks < 12)]  # not saturated

    worst_frame, worst_offset = 0, 0.0
    for i, zoom in enumerate(zooms):
        offsets = []
        for (x, y) in zip(*zoom['wcs'].world_to_pixel(stars)):
            if not (3 <= x < zoom['image'].shape[1] - 3 and 3 <= y < zoom['image'].shape[0] - 3):
                continue
            c = centroid(zoom['image'], zoom['variance'], x, y)
            if c:
                offsets.append((c[0] - x, c[1] - y))
        if len(offsets) < 3:
            print(f'Alignment check: too few reference stars in T{i + 1}')
            continue
        offset = float(np.hypot(*np.median(offsets, axis=0)))
        if offset > worst_offset:
            worst_frame, worst_offset = i, offset

    status = 'OK' if worst_offset <= cfg.ALIGNMENT_TOLERANCE_PIX else 'WARNING: frames may not line up'
    print(f"Alignment check ({status}): {len(stars)} catalog stars appear where SPHEREx's pointing puts them, "
          f'to within {worst_offset:.2f} px ({worst_offset * cfg.PIXEL_ARCSEC:.2f}") in every zoom '
          f'(largest in T{worst_frame + 1})')
    return worst_offset


# ----------------------------------------------------------------------------
# Movement of the tracked object
# ----------------------------------------------------------------------------

def track_object(zooms, coord, alignment_floor_pix):
    """Measure the object's position in every zoom and its offset from where it was at T1.

    Position: brightest pixel within SEARCH_RADIUS_PIX of the expected spot (so a moved object is still
    found), then the brightness-weighted center around it, converted to RA/Dec with that frame's own
    pointing. Adds to each zoom: 'object_xy' (measured position) and 't1_xy' (T1's position, for the blue
    circle). Returns the movement table as a dict of arrays.
    """
    positions, errors = [], []
    for zoom in zooms:
        x, y = zoom['catalog_xy']
        yy, xx = np.indices(zoom['image'].shape)
        near = np.hypot(xx - x, yy - y) <= cfg.SEARCH_RADIUS_PIX
        peak = np.nanargmax(np.where(near, zoom['image'], -np.inf))
        c = centroid(zoom['image'], zoom['variance'], xx.flat[peak], yy.flat[peak])
        if c is None:                                       # no signal: fall back to the catalog position
            c = (x, y, np.nan, np.nan)
        zoom['object_xy'] = (c[0], c[1])
        positions.append(zoom['wcs'].pixel_to_world(c[0], c[1]))
        errors.append(np.hypot(c[2], c[3]) * cfg.PIXEL_ARCSEC)

    t1 = positions[0]
    d_ra, d_dec = [], []
    for zoom, position in zip(zooms, positions):
        zoom['t1_xy'] = tuple(float(v) for v in zoom['wcs'].world_to_pixel(t1))
        east, north = t1.spherical_offsets_to(position)    # east = RA offset on the sky (RA * cos Dec)
        d_ra.append(east.to_value(u.arcsec))
        d_dec.append(north.to_value(u.arcsec))

    movement = {
        'days': np.array([z['mjd'] - zooms[0]['mjd'] for z in zooms]),
        'd_ra_arcsec': np.array(d_ra),
        'd_dec_arcsec': np.array(d_dec),
        'offset_arcsec': np.hypot(d_ra, d_dec),
        'error_arcsec': np.array(errors),
        'floor_arcsec': alignment_floor_pix * cfg.PIXEL_ARCSEC,   # how well frames line up (from the check)
    }

    # significant = bigger than 3x (measurement noise + alignment floor)
    limit = 3 * np.hypot(np.nan_to_num(movement['error_arcsec']), movement['floor_arcsec'])
    moved = movement['offset_arcsec'] > limit
    largest = int(np.argmax(movement['offset_arcsec']))
    print(f"Movement vs T1: largest offset {movement['offset_arcsec'][largest]:.2f}\" in T{largest + 1} "
          f"({movement['offset_arcsec'][largest] / cfg.PIXEL_ARCSEC:.2f} px). "
          + (f"SIGNIFICANT in {', '.join(f'T{i + 1}' for i in np.flatnonzero(moved))}"
             if moved.any() else 'No significant movement (all offsets within 3x the noise + alignment floor)'))
    return movement


# ----------------------------------------------------------------------------
# Spectrum check: is the object in the circle the catalog star?
# ----------------------------------------------------------------------------

def spectrum_check(tracked, images):
    """Measure the object in images from every SPHEREx detector (0.75 - 5 um) and compare that spectrum
    with its IRSA catalog brightness (2MASS + AllWISE)."""
    coord = tracked['coord']
    picks = []
    for band in sorted(set(images['energy_bandpassname'])):
        rows = [row for row in images if row['energy_bandpassname'] == band]
        for i in np.unique(np.linspace(0, len(rows) - 1, cfg.SPECTRUM_IMAGES_PER_DETECTOR).astype(int)):
            picks.append(rows[i])

    def measure(row):
        frame = make_frame(fetch_cutout(str(row['access_url']), coord, cfg.SPECTRUM_CUTOUT_SIZE), coord, row)
        return (frame['wavelength_um'], *measure_flux(frame, coord))

    with ThreadPoolExecutor(max_workers=cfg.MAX_CONCURRENT_DOWNLOADS) as pool:
        points = np.array([p for p in pool.map(measure, picks) if np.isfinite(p[1])])
    points = points[np.argsort(points[:, 0])]

    # catalog brightness interpolated (log-log) to each SPHEREx wavelength inside the catalog's range
    catalog = sorted(tracked['catalog_sed'], key=lambda p: p[1])
    cat_wl = np.array([p[1] for p in catalog])
    cat_flux = np.array([p[2] for p in catalog])
    inside = (points[:, 0] >= cat_wl.min()) & (points[:, 0] <= cat_wl.max())
    expected = np.exp(np.interp(np.log(points[inside, 0]), np.log(cat_wl), np.log(cat_flux)))
    ratio = points[inside, 1] / expected
    scale = float(np.median(ratio))
    scatter = float(1.4826 * np.median(np.abs(ratio / scale - 1)))   # disagreement in SHAPE, after scaling

    # The circle misses a little of the star's light and catches a little of its neighbors', so the level
    # can be off somewhat - but the spectrum SHAPE should match the catalog.
    same_object = 0.4 <= scale <= 1.3 and scatter <= 0.25
    verdict = 'PASS - same object' if same_object else 'FAIL - may not be the catalog object'
    print(f'Spectrum check ({verdict}): {len(points)} SPHEREx images, SPHEREx / catalog = {scale:.2f} on average, '
          f'shape agrees to {scatter * 100:.0f}% ({inside.sum()} points between {cat_wl.min():.2f} and {cat_wl.max():.2f} um)')
    return {'points': points, 'catalog': catalog, 'scale': scale, 'scatter': scatter, 'same_object': same_object}
