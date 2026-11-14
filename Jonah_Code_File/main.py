"""Run the whole analysis: python -m Jonah_Code_File   (or run Test_Jonah_code.py)."""

import time
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord

from . import config as cfg
from .frames import pointing_report, select_frames, zoom_original
from .irsa import query_images
from .measure import check_alignment, measure_flux, spectrum_check, track_object
from .plots import show_frames, show_movement, show_spectrum
from .target import find_tracked_object


def run(ra_deg=cfg.RA_DEG, dec_deg=cfg.DEC_DEG, show=True):
    start = time.time()
    tracked = find_tracked_object(SkyCoord(ra_deg, dec_deg, unit='deg'))      # step 0: a real star to track
    coord = tracked['coord']
    images = query_images(coord)
    print(f"{len(images)} SPHEREx images cover {tracked['name']}")

    with ThreadPoolExecutor(max_workers=1) as background:
        spectrum = background.submit(spectrum_check, tracked, images)          # runs while frames download

        frames = select_frames(coord, images)                                 # step 1: T1 ... Tn
        zooms = [zoom_original(frame, coord) for frame in frames]             # step 2: original-pixel zooms
        pointing_report(frames)
        alignment_floor = check_alignment(zooms, tracked['field_stars'])
        movement = track_object(zooms, coord, alignment_floor)
        for i, zoom in enumerate(zooms):
            flux, flux_err = measure_flux(zoom, coord)
            print(f"  T{i + 1:<2} flux {flux:7.2f} +/- {flux_err:.2f} mJy at {zoom['wavelength_um']:.3f} um")

        check = spectrum.result()                                             # step 3: same object?
    print(f'Done in {time.time() - start:.0f} s')

    results = {'tracked': tracked, 'zooms': zooms, 'movement': movement, 'spectrum': check}
    if show:
        show_frames(zooms, tracked, movement)
        show_movement(movement, tracked)
        show_spectrum(check, tracked)
        plt.show()
    return results


if __name__ == '__main__':
    run()
