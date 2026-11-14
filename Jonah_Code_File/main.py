"""Run the whole analysis: python -m Jonah_Code_File   (or run Test_Jonah_code.py)."""

import time

import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord

from . import config as cfg
from .frames import load_all_frames, pointing_summary, zoom_original
from .irsa import query_images
from .measure import brightness_changes, check_alignment, spectrum_check, track_object
from .plots import show_spectrum, show_time_series, show_viewer
from .target import find_tracked_object


def run(ra_deg=cfg.RA_DEG, dec_deg=cfg.DEC_DEG, max_images=cfg.MAX_IMAGES, show=True, zoom_only=False):
    """Run the whole analysis. Returns a dict with 'tracked', 'frames', 'zooms', 'movement', 'brightness',
    'spectrum'.

    zoom_only=True: after the analysis, keep only the zoomed pixels in memory - each frame's full-size
    cutout arrays are released as its zoom is cut (results['frames'] then hold measurements only, and
    results['zooms'] hold the pixels). The default keeps both."""
    start = time.time()
    tracked = find_tracked_object(SkyCoord(ra_deg, dec_deg, unit='deg'))      # a real star to track
    coord = tracked['coord']
    images = query_images(coord)
    print(f"{len(images)} SPHEREx images cover {tracked['name']}")

    frames = load_all_frames(coord, images, max_images)       # one download per image; measured on arrival
    pointing_summary(frames)
    check_alignment(frames, tracked['field_stars'])            # verifies the pointing, never adjusts it
    movement = track_object(frames)                            # position vs T1
    brightness = brightness_changes(frames)                    # brightness vs the object's own spectrum
    spectrum = spectrum_check(tracked, frames)                 # is it the catalog star?

    zooms = []                                                 # original-pixel zoom of every frame, in order
    for frame in frames:
        zooms.append(zoom_original(frame, release_original=zoom_only))
    print(f'Done in {time.time() - start:.0f} s')

    results = {'tracked': tracked, 'frames': frames, 'zooms': zooms, 'movement': movement,
               'brightness': brightness, 'spectrum': spectrum}
    if show:
        show_viewer(zooms, tracked, movement, brightness)
        show_time_series(frames, tracked, movement, brightness)
        show_spectrum(spectrum, tracked)
        plt.show()
    return results


if __name__ == '__main__':
    run()
