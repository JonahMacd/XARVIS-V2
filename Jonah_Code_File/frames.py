"""Steps 1-2: pick T1 ... Tn at a similar wavelength, and zoom on the ORIGINAL pixels around the object.

The zoom is placed using ONLY SPHEREx's pointing: IRSA's coordinate solution (WCS) for each image, which
is fitted to thousands of catalog stars over the whole 2040 x 2040 detector. Nothing inside our small
zoom is used to line frames up, so real motion or brightness changes in the sky are preserved.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from astropy import units as u
from astropy.wcs import WCS

from . import config as cfg
from .irsa import fetch_cutout


def make_frame(hdulist, coord, row):
    """One frame as a plain dict: the image plus everything later steps need."""
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

    return {
        'image': hdulist['IMAGE'].data.astype(float) - hdulist['ZODI'].data,   # MJy/sr, zodi subtracted
        'variance': hdulist['VARIANCE'].data.astype(float),
        'flags': hdulist['FLAGS'].data.astype(np.int64),
        'wcs': wcs,
        'mjd': float(header['MJD-AVG']),
        'date_obs': header['DATE-OBS'],
        'wavelength_um': float(wavelength.to_value(u.um)),
        'detector': int(header['DETECTOR']),
        'detector_xy': (float(detector_x), float(detector_y)),
        'rotation_deg': float(rotation_deg),
        'obs_id': str(row['obs_id']),
    }


def zoom_bounds(frame, coord, size_pix=cfg.ZOOM_SIZE_PIX):
    """(x0, y0, x1, y1) of the zoom in the frame's pixels, or None if it doesn't fit inside the cutout."""
    x, y = frame['wcs'].world_to_pixel(coord)
    cx, cy = int(np.round(float(x))), int(np.round(float(y)))             # pixel containing the object
    x0, y0 = cx - size_pix // 2, cy - size_pix // 2
    x1, y1 = x0 + size_pix, y0 + size_pix
    ny, nx = frame['image'].shape
    return (x0, y0, x1, y1) if x0 >= 0 and y0 >= 0 and x1 <= nx and y1 <= ny else None


def select_frames(coord, images):
    """T1 = earliest image where the full zoom fits; then up to MAX_FRAMES images (in time order)
    at a similar wavelength to T1."""
    def load(row):
        frame = make_frame(fetch_cutout(str(row['access_url']), coord, cfg.CUTOUT_SIZE), coord, row)
        return frame if zoom_bounds(frame, coord) else None

    for first_index, row in enumerate(images):
        first = load(row)
        if first is not None:
            break
    else:
        raise RuntimeError(f'No SPHEREx image has room for a {cfg.ZOOM_SIZE_PIX}-pixel zoom around the object')
    print(f"T1: {first['date_obs']}  wavelength {first['wavelength_um']:.3f} um  (detector {first['detector']})")

    # only the same detector can see the same wavelength, so skip the others without downloading them
    band = images[first_index]['energy_bandpassname']
    candidates = [row for row in images[first_index + 1:] if row['energy_bandpassname'] == band]
    frames = [first]

    batch_size = cfg.MAX_CONCURRENT_DOWNLOADS
    with ThreadPoolExecutor(max_workers=batch_size) as pool:
        # download in batches, in time order, and stop as soon as there are enough frames
        for start in range(0, len(candidates), batch_size):
            for frame in pool.map(load, candidates[start:start + batch_size]):
                if frame and abs(frame['wavelength_um'] - first['wavelength_um']) <= cfg.WAVELENGTH_TOLERANCE_UM:
                    frames.append(frame)
                    print(f"T{len(frames)}: {frame['date_obs']}  wavelength {frame['wavelength_um']:.3f} um")
                if len(frames) == cfg.MAX_FRAMES:
                    return frames
    return frames


def zoom_original(frame, coord):
    """Crop the zoom from the original pixels - values exactly as delivered by SPHEREx, nothing resampled."""
    x0, y0, x1, y1 = zoom_bounds(frame, coord)
    x, y = frame['wcs'].world_to_pixel(coord)
    zoom = dict(frame)
    zoom.update(
        image=frame['image'][y0:y1, x0:x1],
        variance=frame['variance'][y0:y1, x0:x1],
        flags=frame['flags'][y0:y1, x0:x1],
        wcs=frame['wcs'][y0:y1, x0:x1],                     # coordinates for the cropped pixels
        catalog_xy=(float(x) - x0, float(y) - y0),          # the object's catalog position in the zoom
    )
    return zoom


def pixel_sky_positions(zoom):
    """RA/Dec (deg) of every pixel center in a zoom, from that frame's own pointing. For pixel-by-pixel
    comparisons between frames (which can be rotated relative to each other)."""
    yy, xx = np.indices(zoom['image'].shape)
    sky = zoom['wcs'].pixel_to_world(xx, yy)
    return sky.ra.deg, sky.dec.deg


def pointing_report(frames):
    """How far SPHEREx's pointing moved between frames (where the object fell on the detector)."""
    print("\nSPHEREx pointing vs T1 (where the object fell on the detector):")
    x1, y1 = frames[0]['detector_xy']
    for i, frame in enumerate(frames):
        shift = np.hypot(frame['detector_xy'][0] - x1, frame['detector_xy'][1] - y1)
        rotation = (frame['rotation_deg'] - frames[0]['rotation_deg'] + 180) % 360 - 180
        print(f"  T{i + 1:<2} moved {shift:6.1f} px ({shift * cfg.PIXEL_ARCSEC / 60:5.1f}'), rotated {rotation:+7.2f} deg")
