"""Run the whole analysis: run __main__.py, or python -m Jonah_Code_File (settings: config.py)."""

import time

import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord

from . import cache
from . import config as cfg
from .frames import load_all_frames, pointing_summary, zoom_original
from .irsa import query_images
from .measure import (brightness_changes, check_alignment, spectrum_check, track_moving,
                      track_object)
from .moving import add_predicted_positions, find_asteroid, find_images
from .plots import show_motion, show_spectrum, show_time_series, show_viewer
from .results import save_results
from .target import find_tracked_object
from .spectrumanalyser import SpectrumAnalyzer

def make_zooms(frames, zoom_only):
    """Original-pixel zoom of every frame, in order. zoom_only=True releases each frame's full-size pixel
    arrays as its zoom is cut, so only the zooms stay in memory."""
    zooms = []
    for frame in frames:
        zooms.append(zoom_original(frame, release_original=zoom_only))
    return zooms


def run(ra_deg=cfg.RA_DEG, dec_deg=cfg.DEC_DEG, max_images=cfg.MAX_IMAGES, show=True, zoom_only=cfg.ZOOM_ONLY,
        asteroid=cfg.ASTEROID):
    """Run the whole analysis. Tracks the asteroid set in config.ASTEROID (or the `asteroid` argument); with
    asteroid=None it tracks the star at ra_deg / dec_deg instead.

    Returns a dict with 'tracked', 'frames', 'zooms', 'movement' (+ 'brightness' and 'spectrum' for a star).
    The measurements are also saved to results/<target>.ecsv (config.SAVE_RESULTS).

    zoom_only=True (the default, config.ZOOM_ONLY): after the analysis, keep only the zoomed pixels in memory -
    each frame's full-size cutout arrays are released as its zoom is cut (results['frames'] then hold
    measurements only, and results['zooms'] hold the pixels). False keeps both."""
    if asteroid:
        return run_asteroid(asteroid, cfg.ASTEROID_WINDOW, max_images, show, zoom_only)
    return run_star(ra_deg, dec_deg, max_images, show, zoom_only)


def run_asteroid(asteroid=cfg.ASTEROID, window=cfg.ASTEROID_WINDOW, max_images=cfg.MAX_IMAGES, show=True,
                 zoom_only=cfg.ZOOM_ONLY):
    """A moving object: JPL Horizons predicts where the asteroid was at every SPHEREx exposure, IRSA finds
    the images that caught it, and each image is cut around the predicted position and measured there."""
    start = time.time()
    try:
        tracked = find_asteroid(asteroid, window)              # its path, from JPL Horizons
        images = find_images(tracked)                          # the SPHEREx images that caught it
        if not len(images):
            raise RuntimeError('No SPHEREx image caught the asteroid in this window - try another window (config.py)')
        if max_images:
            images = images[:max_images]
        images = add_predicted_positions(images, tracked)      # its exact position, as seen from SPHEREx

        frames = load_all_frames(None, images)                 # one cutout per image, centered on the prediction
        pointing_summary(frames)
        movement = track_moving(frames)                        # distance travelled, found where predicted?
        zooms = make_zooms(frames, zoom_only)
    finally:
        cache.prune()                                          # clean up the disk cache, even if the run failed
    print(f'Done in {time.time() - start:.0f} s')

    results = {'tracked': tracked, 'frames': frames, 'zooms': zooms, 'movement': movement}
    if cfg.SAVE_RESULTS:
        save_results(results, {'mode': 'asteroid', 'asteroid': asteroid, 'window': window, 'max_images': max_images})

    analyzer = SpectrumAnalyzer(frames)
    results = {'tracked': tracked, 'frames': frames, 'zooms': zooms, 'movement': movement, 'analyzer': analyzer}
    
    if show:
        show_viewer(zooms, tracked, movement)
        show_motion(frames, tracked, movement)
        plt.show()
    return results


def run_star(ra_deg=cfg.RA_DEG, dec_deg=cfg.DEC_DEG, max_images=cfg.MAX_IMAGES, show=True, zoom_only=cfg.ZOOM_ONLY):
    """A star: stays put on the sky, so its brightness and position are compared with its own history."""
    start = time.time()
    try:
        tracked = find_tracked_object(SkyCoord(ra_deg, dec_deg, unit='deg'))      # a real star to track
        coord = tracked['coord']
        images = query_images(coord)
        print(f"{len(images)} SPHEREx images cover {tracked['name']}")

        frames = load_all_frames(coord, images, max_images)   # one download per image; measured on arrival
        pointing_summary(frames)
        check_alignment(frames, tracked['field_stars'])        # verifies the pointing, never adjusts it
        movement = track_object(frames)                        # position vs T1
        brightness = brightness_changes(frames)                # brightness vs the object's own spectrum
        spectrum = spectrum_check(tracked, frames)             # is it the catalog star?
        zooms = make_zooms(frames, zoom_only)
    finally:
        cache.prune()                                          # clean up the disk cache, even if the run failed
    print(f'Done in {time.time() - start:.0f} s')

    results = {'tracked': tracked, 'frames': frames, 'zooms': zooms, 'movement': movement,
               'brightness': brightness, 'spectrum': spectrum}
    if cfg.SAVE_RESULTS:
        save_results(results, {'mode': 'star', 'ra_deg': ra_deg, 'dec_deg': dec_deg, 'max_images': max_images})
    analyzer = SpectrumAnalyzer(frames, brightness=brightness, spectrum=spectrum)
    results = {'tracked': tracked, 'frames': frames, 'zooms': zooms, 'movement': movement, 'brightness': brightness, 'spectrum': spectrum, 'analyzer': analyzer}
    
    if show:
        show_viewer(zooms, tracked, movement, brightness)
        show_time_series(frames, tracked, movement, brightness)
        show_spectrum(spectrum, tracked)
        plt.show()
    return results


if __name__ == '__main__':
    run()
