"""Download and measure every SPHEREx image of the object, and zoom on the ORIGINAL pixels for display.

Each image is downloaded once (a small cutout) and everything is measured from it right away: wavelength,
brightness, position. Frames are placed using ONLY SPHEREx's pointing: IRSA's coordinate solution (WCS) for
each image, fitted to thousands of catalog stars over the whole 2040 x 2040 detector. Nothing inside our
small cutout is used to line frames up, so real motion or brightness changes in the sky are preserved.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from astropy import units as u
from astropy.wcs import WCS

from . import config as cfg
from .irsa import fetch_cutout
from .measure import bad_pixel_mask, locate_object, measure_flux


def make_frame(hdulist, coord, row):
    """One frame as a plain dict: the cutout image plus everything later steps need."""
    header = hdulist['IMAGE'].header
    wcs = WCS(header)       # SPHEREx's pointing for this image

    # wavelength at the object: SPHEREx's wavelength changes across the detector.
    # The spectral WCS is alternate WCS "W" and uses the WCS-WAVE lookup table.
    spectral_wcs = WCS(header=header, fobj=hdulist, key='W')
    spectral_wcs.sip = None
    x, y = wcs.world_to_pixel(coord)
    wavelength, _ = spectral_wcs.pixel_to_world(x, y)

    # where the object landed on the full detector (alternate WCS "A" = 0-based detector pixels)
    # and how the detector was rotated on the sky: together these describe SPHEREx's pointing
    detector_x, detector_y = WCS(header, key='A').pixel_to_world_values(x, y)
    rotation_deg = wcs.pixel_to_world(x, y).position_angle(wcs.pixel_to_world(x, y + 1)).deg

    image = hdulist['IMAGE'].data.astype(float) - hdulist['ZODI'].data          # MJy/sr, zodi subtracted
    flags = hdulist['FLAGS'].data.astype(np.int64)
    return {
        'image': image,
        'variance': hdulist['VARIANCE'].data.astype(float),
        'flags': flags,
        'bad': bad_pixel_mask(image, flags, hdulist['FLAGS'].header),          # ignored in all measurements
        'wcs': wcs,
        'catalog_xy': (float(x), float(y)),                # the object's catalog position in this cutout
        'mjd': float(header['MJD-AVG']),
        'date_obs': header['DATE-OBS'],
        'wavelength_um': float(wavelength.to_value(u.um)),
        'detector': int(header['DETECTOR']),
        'detector_xy': (float(detector_x), float(detector_y)),
        'rotation_deg': float(rotation_deg),
        'obs_id': str(row['obs_id']),
    }


def load_frame(row, coord):
    """Download one image's cutout and measure the object in it (runs in parallel, one per image)."""
    frame = make_frame(fetch_cutout(str(row['access_url']), coord, cfg.CUTOUT_SIZE), coord, row)
    frame['flux_mjy'], frame['flux_err_mjy'], frame['n_bad_in_aperture'] = measure_flux(frame, coord)
    frame.update(locate_object(frame))
    return frame


def load_all_frames(coord, images, max_images=cfg.MAX_IMAGES):
    """Download and measure every image (or the first `max_images` in time), in parallel. Sorted by time."""
    rows = list(images if max_images is None else images[:max_images])
    frames, failures = [], []
    start = time.time()
    print(f'Downloading and measuring {len(rows)} images ({cfg.MAX_CONCURRENT_DOWNLOADS} at a time)...')

    with ThreadPoolExecutor(max_workers=cfg.MAX_CONCURRENT_DOWNLOADS) as pool:
        futures = [pool.submit(load_frame, row, coord) for row in rows]
        for done, future in enumerate(as_completed(futures), start=1):
            try:
                frames.append(future.result())
            except Exception as e:                          # one bad image shouldn't stop the whole run
                failures.append(str(e).splitlines()[0] if str(e) else type(e).__name__)
            if done % 50 == 0 or done == len(rows):
                elapsed = time.time() - start
                print(f'  {done}/{len(rows)} images  ({done / elapsed:.1f}/s, '
                      f'{(len(rows) - done) * elapsed / done:.0f} s left)')

    if failures:
        print(f'  skipped {len(failures)} images that could not be used, e.g.: {failures[0][:100]}')
    usable = [f for f in frames if np.isfinite(f['flux_mjy']) and f['object_found']]
    if len(usable) < len(frames):
        print(f'  {len(frames) - len(usable)} images had the object too close to the edge to measure')
    unreliable = sum(f['n_bad_in_aperture'] > 0 for f in usable)
    if unreliable:
        print(f'  {unreliable} frames have a bad pixel (glitch/hot pixel) inside the measurement circle: '
              f'shown, but not used for flags or statistics')
    # time order; the 6 detectors expose simultaneously (same timestamp), so break ties by detector -
    # otherwise T-numbers would depend on which download happened to finish first
    usable.sort(key=lambda f: (f['mjd'], f['detector']))
    return usable


PIXEL_ARRAYS = ('image', 'variance', 'flags', 'bad')     # the per-pixel arrays of a frame


def zoom_original(frame, size_pix=cfg.ZOOM_SIZE_PIX, release_original=False):
    """Crop the display zoom from the original pixels (nothing resampled), centered on the object.
    Near the edge of the detector the zoom is cut off rather than padded.

    Returns a new dict: the frame's measurements (time, wavelength, flux, position ...) plus the cropped
    pixel arrays, the cropped coordinates (`wcs`) and the object's catalog position inside the zoom.

    release_original=True: the full-size pixel arrays are deleted from `frame` once the crop is made, so
    only the zoom stays in memory (the frame keeps its measurements). The crop is copied for this, because
    a plain slice is only a view that keeps the whole original array alive. Do it once per frame, and only
    when nothing needs the full-size pixels any more."""
    x, y = frame['catalog_xy']
    cx, cy = int(np.round(x)), int(np.round(y))
    ny, nx = frame['image'].shape
    x0, y0 = max(cx - size_pix // 2, 0), max(cy - size_pix // 2, 0)
    x1, y1 = min(cx - size_pix // 2 + size_pix, nx), min(cy - size_pix // 2 + size_pix, ny)

    crop = {key: frame[key][y0:y1, x0:x1] for key in PIXEL_ARRAYS}
    if release_original:
        crop = {key: pixels.copy() for key, pixels in crop.items()}

    zoom = dict(frame)                                      # measurements carry over unchanged
    zoom.update(crop, wcs=frame['wcs'][y0:y1, x0:x1], catalog_xy=(x - x0, y - y0))
    if release_original:
        for key in PIXEL_ARRAYS:
            del frame[key]
    return zoom


def pixel_sky_positions(frame):
    """RA/Dec (deg) of every pixel center, from that frame's own pointing - for pixel-by-pixel comparisons
    between frames (which can be rotated relative to each other)."""
    yy, xx = np.indices(frame['image'].shape)
    sky = frame['wcs'].pixel_to_world(xx, yy)
    return sky.ra.deg, sky.dec.deg


def pointing_summary(frames):
    """How SPHEREx's pointing varied over the dataset."""
    days = np.array([f['mjd'] for f in frames]) - frames[0]['mjd']
    visits = 1 + int(np.sum(np.diff(days) > 30))           # survey passes, separated by months
    x1, y1 = frames[0]['detector_xy']
    shifts = [np.hypot(f['detector_xy'][0] - x1, f['detector_xy'][1] - y1) for f in frames]
    rotations = sorted({int(np.round(f['rotation_deg'])) % 360 for f in frames})
    detectors = len({f['detector'] for f in frames})
    print(f'\n{len(frames)} frames over {days[-1]:.0f} days in {visits} survey visits, {detectors} detectors '
          f'({min(f["wavelength_um"] for f in frames):.2f} - {max(f["wavelength_um"] for f in frames):.2f} um)')
    print(f'SPHEREx pointing: the object fell up to {max(shifts):.0f} px ({max(shifts) * cfg.PIXEL_ARCSEC / 60:.0f}\') '
          f'from its T1 detector position; detector orientations used: {rotations} deg')
