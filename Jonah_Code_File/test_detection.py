"""Offline tests of how the sky map decides what is a source and what is NEW (measure.saturated_cores,
field.compare_frame, skymap.stacked_significance / known_star_nearby / classify_new), on synthetic frames shaped
like the failures found in real SPHEREx fields. No network.

Run:  python -m unittest discover -s Jonah_Code_File -p "test_*.py" -t .
"""

import unittest
from unittest import mock

import numpy as np
from astropy.io import fits
from astropy.table import Table

from Jonah_Code_File import config as cfg
from Jonah_Code_File import skymap
from Jonah_Code_File.field import compare_frame
from Jonah_Code_File.measure import saturated_cores, saturated_spill
from Jonah_Code_File.test_pipeline import frame_with_stars, gaussian_psf

FLAG_NAMES = {0: 'TRANSIENT', 1: 'OVERFLOW', 2: 'BLOOM', 3: 'OUTLIER', 4: 'SUR_ERROR'}
BIT = {name: 1 << number for number, name in FLAG_NAMES.items()}
WIDE_PSF = gaussian_psf(sigma_px=1.6, size=91)   # a bright star's broad wings, out to +/-9 px (what surrounds a
                                                 # saturated core; the default 33-sample stamp ends at 3.3 px)


def flagged(frame):
    """A synthetic frame with a FLAGS plane and its header (bit numbers as SPHEREx writes them: MSKNnn = name)."""
    header = fits.Header({f'MSKN{number:02d}': name for number, name in FLAG_NAMES.items()})
    return dict(frame, flags=np.zeros(frame['image'].shape, np.int64), flags_header=header, mjd=61000.0)


def saturated_star(x, y, others=()):
    """A frame with a saturated star at (x, y) - broad wings, its core masked and flagged OVERFLOW - and others."""
    frame = flagged(frame_with_stars([(x, y, 60000.0)], psf=WIDE_PSF))
    if others:
        frame['image'] = frame['image'] + frame_with_stars(list(others), noise=0.0)['image']
    return mask(frame, x, y, 2.2, 'OVERFLOW')


def mask(frame, x, y, radius, flag):
    """Mask the pixels within `radius` of (x, y) as SPHEREx would (bad, NaN, flagged)."""
    yy, xx = np.indices(frame['image'].shape)
    core = np.hypot(xx - x, yy - y) <= radius
    frame['bad'] = frame['bad'] | core
    frame['image'] = np.where(core, np.nan, frame['image'])
    frame['flags'] = frame['flags'] | np.where(core, BIT[flag], 0)
    return frame


def catalog_at(frame, positions, sources=None, gaia_match=None):
    """A field catalog (field.query_field's columns, plus catalog_layer's label / hip) at pixel positions."""
    ra, dec = frame['wcs'].pixel_to_world_values(np.array([p[0] for p in positions]), np.array([p[1] for p in positions]))
    n = len(positions)
    return Table({'source': sources or ['Gaia'] * n, 'name': [f'star {k}' for k in range(n)],
                  'label': [f'star {k}' for k in range(n)], 'ra': np.atleast_1d(ra), 'dec': np.atleast_1d(dec),
                  'pmra': np.zeros(n), 'pmdec': np.zeros(n), 'epoch_mjd': np.full(n, 61000.0), 'mag': np.full(n, 12.0),
                  'parallax': np.zeros(n), 'gaia_match': np.array(gaia_match or [False] * n), 'hip': np.zeros(n, int)})


def series(n, stars=(), detectors=(2, 5), seed=0):
    """n frames of the same sky (the frames' WCS are identical), alternating detector arms, two survey visits."""
    frames = []
    for i in range(n):
        frame = flagged(frame_with_stars(list(stars), seed=seed + i))
        frame.update(detector=detectors[i % 2], mjd=61000.0 + 0.1 * i + (200 if i >= n // 2 else 0))
        frames.append(frame)
    return frames


class SaturatedStarTests(unittest.TestCase):
    def test_a_saturated_core_is_one_source(self):
        cores = saturated_cores(saturated_star(25.0, 25.0))
        self.assertEqual(len(cores), 1)
        x, y, radius, significance, halo = cores[0]
        self.assertLess(np.hypot(x - 25, y - 25), 0.3)
        self.assertGreater(radius, 1.5)
        self.assertGreater(significance, 5)
        self.assertGreaterEqual(halo, radius + 1)                        # its glow reaches past the mask

    def test_spill_beside_a_saturated_star_is_that_star(self):
        frame = flagged(frame_with_stars([(25.0, 25.0, 400000.0)], psf=WIDE_PSF))  # very bright: a long glow
        frame = mask(mask(frame, 25.0, 25.0, 2.2, 'OVERFLOW'), 28.6, 22.0, 1.0, 'BLOOM')   # + a separate spilled patch
        cores = sorted(saturated_cores(frame), key=lambda c: -c[2])
        self.assertEqual(len(cores), 2)                                             # (two masked patches)
        self.assertTrue(saturated_spill(frame, cores[1], cores[0]))
        result = compare_frame(frame, catalog_at(frame, [(25.0, 25.0)]))
        self.assertEqual(int(result['peak_saturated'].sum()), 2)
        self.assertTrue(result['matched'][result['peak_saturated']].all())          # both: the one catalog star
        self.assertEqual(set(result['peak_cat'][result['peak_saturated']].tolist()), {0})

    def test_a_saturated_source_beside_a_saturated_star_stays_its_own(self):
        # (the measured case: a 2000-sigma transient 4.3 px from a bright star, both saturated)
        frame = flagged(frame_with_stars([(20.0, 25.0, 60000.0), (24.5, 25.0, 60000.0)], psf=WIDE_PSF))
        frame = mask(mask(frame, 20.0, 25.0, 2.2, 'OVERFLOW'), 24.5, 25.0, 1.0, 'OVERFLOW')
        cores = sorted(saturated_cores(frame), key=lambda c: -c[2])
        self.assertEqual(len(cores), 2)
        self.assertFalse(saturated_spill(frame, cores[1], cores[0]))
        result = compare_frame(frame, catalog_at(frame, [(20.0, 25.0)]))
        self.assertEqual(int((result['peak_saturated'] & ~result['matched']).sum()), 1)   # unexplained: a candidate

    def test_a_real_source_on_a_bright_glow_is_kept(self):
        frame = saturated_star(25.0, 25.0)
        yy, xx = np.indices(frame['image'].shape)
        frame['image'] = frame['image'] + 40.0 * np.exp(-np.hypot(xx - 25, yy - 25) / 3.0)    # a broad glow
        frame['image'] = frame['image'] + frame_with_stars([(32.0, 25.0, 2000.0)], noise=0.0)['image']  # 7 px out
        frame['image'][frame['bad']] = np.nan
        result = compare_frame(frame, catalog_at(frame, [(25.0, 25.0)]))
        self.assertLess(np.min(np.hypot(result['peak_x'] - 32, result['peak_y'] - 25)), 1.0)

    def test_cosmic_rays_and_empty_overflow_are_not_stars(self):
        frame = mask(flagged(frame_with_stars([])), 12.0, 30.0, 1.5, 'TRANSIENT')     # a cosmic-ray hit
        frame = mask(frame, 35.0, 12.0, 1.5, 'OVERFLOW')                               # flagged, but no star around
        self.assertEqual(saturated_cores(frame), [])
        self.assertEqual(saturated_cores(frame_with_stars([])), [])                   # (no FLAGS plane at all)

    def test_compare_frame_replaces_the_rim_peaks_by_the_star(self):
        frame = saturated_star(25.0, 25.0, others=[(40.0, 10.0, 600.0)])
        catalog = catalog_at(frame, [(26.0, 26.5), (40.0, 10.0)])       # the bright star 1.8 px from the core centre
        result = compare_frame(frame, catalog)
        near = np.hypot(result['peak_x'] - 25, result['peak_y'] - 25) < 6
        self.assertEqual(int(near.sum()), 1)                             # one source, not the rim of the mask
        self.assertTrue(result['peak_saturated'][near][0])
        self.assertTrue(result['matched'][near][0])                      # (matched within the core's radius)
        self.assertTrue(result['peak_clean'][near][0])                   # a star, not an artifact
        self.assertTrue(result['cat_seen'].all())                        # the faint star too

    def test_a_catwise_twin_never_takes_the_match_from_gaia(self):
        frame = flagged(frame_with_stars([(25.0, 25.0, 900.0)]))
        catalog = catalog_at(frame, [(25.3, 25.0), (25.05, 25.0)], sources=['Gaia', 'CatWISE'], gaia_match=[False, True])
        result = compare_frame(frame, catalog)
        k = int(np.argmin(np.hypot(result['peak_x'] - 25, result['peak_y'] - 25)))
        self.assertEqual(result['peak_cat'][k], 0)                       # the Gaia entry (the sky map draws that one)
        self.assertEqual(list(result['cat_seen']), [True, False])


class ConfirmationTests(unittest.TestCase):
    def test_a_faint_real_source_is_confirmed(self):
        frames = series(40, stars=[(25.0, 25.0, 90.0)])                  # ~3-4 sigma per frame: rarely a peak alone
        ra, dec = frames[0]['wcs'].pixel_to_world_values(25.0, 25.0)
        self.assertGreater(skymap.stacked_significance(frames, range(40), float(ra), float(dec)), 10)

    def test_a_chance_alignment_of_noise_is_not(self):
        frames = series(60)
        ra, dec = (float(v) for v in frames[0]['wcs'].pixel_to_world_values(30.0, 20.0))
        for i in (5, 41):                                                # two frames happen to have a 6-sigma blip
            frames[i]['image'] = frames[i]['image'] + 6.0 * np.exp(
                -((np.indices((50, 50))[1] - 30.0) ** 2 + (np.indices((50, 50))[0] - 20.0) ** 2) / (2 * 0.6 ** 2))
        self.assertLess(skymap.stacked_significance(frames, range(60), ra, dec), 3)
        self.assertLess(skymap.stacked_significance(frames, [5, 41, 6, 40], ra, dec), cfg.NEW_CONFIRM_SIGMA)

    def test_frames_where_the_place_is_near_the_edge_are_not_used(self):
        frames = series(10, stars=[(2.5, 25.0, 3000.0)])
        ra, dec = (float(v) for v in frames[0]['wcs'].pixel_to_world_values(2.5, 25.0))
        self.assertIsNone(skymap.stacked_significance(frames, range(10), ra, dec))


class KnownStarTests(unittest.TestCase):
    def frames_with_star(self, detected_in, n=20, offset_px=2.0):
        """n frames; a catalog star offset_px from the object at (25, 25), detected (in the sky map) in `detected_in`."""
        frames = series(n)
        catalog = catalog_at(frames[0], [(25.0 + offset_px, 25.0)])
        for i, frame in enumerate(frames):
            frame['map'] = [{'kind': 'star', 'catalog_row': 0, 'detected': i in detected_in, 'label': 'star 0',
                             'ra': float(catalog['ra'][0]), 'dec': float(catalog['dec'][0])}]
        ra, dec = (float(v) for v in frames[0]['wcs'].pixel_to_world_values(25.0, 25.0))
        return frames, catalog, ra, dec

    def test_a_blend_next_to_a_star_seen_in_most_frames(self):
        frames, catalog, ra, dec = self.frames_with_star(set(range(4, 20)))
        own = [{'frame': i} for i in range(4)]
        self.assertEqual(skymap.known_star_nearby(frames, list(range(20)), ra, dec, catalog, own), 0)

    def test_a_star_pushed_further_off_only_when_it_is_missing_exactly_there(self):
        frames, catalog, ra, dec = self.frames_with_star(set(range(2, 20)), offset_px=4.0)
        displaced = [{'frame': 0}, {'frame': 1}]                         # the star is missing in exactly these
        self.assertEqual(skymap.known_star_nearby(frames, list(range(20)), ra, dec, catalog, displaced), 0)
        beside = [{'frame': 5}, {'frame': 6}]                            # the star is seen there too: something else
        self.assertIsNone(skymap.known_star_nearby(frames, list(range(20)), ra, dec, catalog, beside))

    def test_a_faint_star_seen_now_and_then_proves_nothing(self):
        frames, catalog, ra, dec = self.frames_with_star(set(range(8, 20)), offset_px=4.0)    # in 60% of frames
        self.assertIsNone(skymap.known_star_nearby(frames, list(range(20)), ra, dec, catalog, [{'frame': 0}]))
        frames, catalog, ra, dec = self.frames_with_star(set(range(15, 20)), offset_px=2.0)   # in 25%
        self.assertIsNone(skymap.known_star_nearby(frames, list(range(20)), ra, dec, catalog, [{'frame': 0}]))


class ClassifyTests(unittest.TestCase):
    def detections(self, frames, x, y, which, sigma=6.0):
        ra, dec = (float(v) for v in frames[0]['wcs'].pixel_to_world_values(x, y))
        return [(i, ra, dec, sigma, False) for i in which]

    def test_noise_coincidences_are_dropped_and_real_sources_kept(self):
        frames = series(40, stars=[(30.0, 30.0, 150.0)])                # a faint real source at (30, 30)
        unexplained = (self.detections(frames, 30.0, 30.0, [3, 8, 25, 31])          # its few single detections
                       + self.detections(frames, 15.0, 12.0, [6, 33]))               # noise at the same spot, twice
        with mock.patch.object(skymap, 'log'):
            new = skymap.classify_new(frames, unexplained, catalog=catalog_at(frames[0], [(5.0, 45.0)]))
        self.assertEqual(len(new), 1)
        self.assertEqual(new[0]['kind'], 'static')
        self.assertGreater(new[0]['confirmed_sigma'], cfg.NEW_CONFIRM_SIGMA)
        self.assertIn('confirmed over all frames', new[0]['description'])
        self.assertLess(np.hypot(*(np.array(frames[0]['wcs'].world_to_pixel_values(new[0]['ra'], new[0]['dec'])) - 30)), 0.5)

    def test_movers_are_checked_along_their_path(self):
        frames = series(40)                                             # 20 frames per visit, 0.1 day apart
        start = np.array([12.0, 12.0])
        step = np.array([0.6, 0.45])                                    # px per frame: a straight line, steady rate

        def place(i):
            return start + step * i

        def mover_detections(which, sigma=6.0):
            out = []
            for i in which:
                ra, dec = (float(v) for v in frames[0]['wcs'].pixel_to_world_values(*place(i)))
                out.append((i, ra, dec, sigma, False))
            return out

        with mock.patch.object(skymap, 'log'):                      # three noise peaks that line up: nothing between
            self.assertEqual(skymap.classify_new(frames, mover_detections([2, 9, 16]), catalog=None), [])
        for i in range(20):                                         # a real mover: there in every frame of the visit
            x, y = place(i)
            frames[i]['image'] = frames[i]['image'] + frame_with_stars([(x, y, 120.0)], noise=0.0)['image']
        with mock.patch.object(skymap, 'log'):
            new = skymap.classify_new(frames, mover_detections([2, 9, 16]), catalog=None)
        self.assertEqual([o['kind'] for o in new], ['moving'])
        self.assertGreater(new[0]['confirmed_sigma'], cfg.NEW_CONFIRM_SIGMA)

    def test_fragments_of_one_source_become_one_object(self):
        frames = series(40, stars=[(30.0, 30.0, 150.0)])
        unexplained = (self.detections(frames, 30.0, 30.0, [3, 8, 25, 31, 36])
                       + self.detections(frames, 31.7, 30.0, [11, 28]))              # noisier ones, 1.7 px off
        with mock.patch.object(skymap, 'log'):
            new = skymap.classify_new(frames, unexplained, catalog=catalog_at(frames[0], [(5.0, 45.0)]))
        self.assertEqual(len(new), 1)
        self.assertEqual(sorted(m['frame'] for m in new[0]['detections']), [3, 8, 11, 25, 28, 31, 36])

    def test_a_faint_catalog_star_seen_only_in_the_frames_combined(self):
        frames = series(40, stars=[(30.0, 30.0, 150.0)])                # a faint star, rarely a peak on its own
        catalog = catalog_at(frames[0], [(31.6, 30.0)])                 # its catalog entry, 1.6 px away
        for frame in frames:
            frame['map'] = [{'kind': 'star', 'catalog_row': 0, 'detected': False, 'label': 'star 0',
                             'ra': float(catalog['ra'][0]), 'dec': float(catalog['dec'][0])}]
        with mock.patch.object(skymap, 'log'):
            new = skymap.classify_new(frames, self.detections(frames, 30.0, 30.0, [3, 8, 25, 31]), catalog=catalog)
        self.assertEqual(new, [])

    def test_a_bright_source_beside_a_faint_catalog_star_is_kept(self):
        frames = series(40, stars=[(30.0, 30.0, 600.0)])                # bright, at (30, 30)
        catalog = catalog_at(frames[0], [(32.4, 30.0)])                 # a faint catalog star SPHEREx doesn't see
        for frame in frames:
            frame['map'] = [{'kind': 'star', 'catalog_row': 0, 'detected': False, 'label': 'star 0',
                             'ra': float(catalog['ra'][0]), 'dec': float(catalog['dec'][0])}]
        with mock.patch.object(skymap, 'log'):
            new = skymap.classify_new(frames, self.detections(frames, 30.0, 30.0, [3, 8, 25, 31]), catalog=catalog)
        self.assertEqual(len(new), 1)

    def test_a_blended_catalog_star_is_marked_as_that_star(self):
        frames = series(20, stars=[(25.0, 25.0, 900.0)])
        catalog = catalog_at(frames[0], [(26.8, 25.0)])                # 1.8 px off: the peak of a crowded blend
        for frame in frames:
            frame['map'] = [{'kind': 'star', 'catalog_row': 0, 'detected': True, 'label': 'star 0',
                             'ra': float(catalog['ra'][0]), 'dec': float(catalog['dec'][0])}]
        for i in (2, 12):
            frames[i]['map'] = []                                       # unmatched there: looked unexplained
        with mock.patch.object(skymap, 'log'):
            new = skymap.classify_new(frames, self.detections(frames, 25.0, 25.0, [2, 12], sigma=40.0), catalog=catalog)
        self.assertEqual(new, [])
        for i in (2, 12):                                               # ... and drawn as the star it is
            self.assertEqual([(e['kind'], e['catalog_row'], e['detected']) for e in frames[i]['map']], [('star', 0, True)])


if __name__ == '__main__':
    unittest.main()
