"""Offline tests of the website's bridge to the pipeline (api.py's saved-results functions, jobs.py, web.py) and
of the website with the bridge connected: synthetic data, no network, nothing written outside temporary folders.

Run:  python -m unittest discover -s Jonah_Code_File -p "test_*.py" -t .
"""

import importlib.util
import io
import json
import sys
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import matplotlib
matplotlib.use('Agg')
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time

from Jonah_Code_File import api
from Jonah_Code_File import config as cfg
from Jonah_Code_File.jobs import JobManager, QueueFull
from Jonah_Code_File.main import coords_name, parse_coords
from Jonah_Code_File.measure import brightness_changes, track_object
from Jonah_Code_File.results import save_results
from Jonah_Code_File.test_pipeline import simple_wcs

try:
    import flask
except ImportError:                             # the pipeline alone doesn't need Flask; the website does
    flask = None

SITE = Path(__file__).resolve().parent.parent / 'Test_Robert_Code_Folder'
PNG = b'\x89PNG\r\n\x1a\n'


def synthetic_results(n=12, name='Synthetic field'):
    """A small finished analysis, like main.run_coords returns, made up: a steady star in the middle of every
    frame (both detector arms of each exposure, two survey visits), and a sky map with a catalog star, a flagged
    artifact and one NEW object."""
    rng = np.random.default_rng(3)
    wcs = simple_wcs()
    ra0, dec0 = (float(v) for v in wcs.pixel_to_world_values(25.0, 25.0))
    star_ra, star_dec = (float(v) for v in wcs.pixel_to_world_values(38.0, 12.0))
    new_ra, new_dec = (float(v) for v in wcs.pixel_to_world_values(10.0, 40.0))
    yy, xx = np.indices((50, 50))
    spot = lambda x, y, peak: peak * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 0.9 ** 2))
    frames, zooms = [], []
    for i in range(n):
        exposure, arm = divmod(i, 2)
        mjd = 60800.0 + 0.5 * exposure + (200.0 if exposure >= n // 4 else 0.0)        # two survey visits
        frame = {
            'obs_id': f'2025W19_1B_{exposure:04d}_1', 'date_obs': Time(mjd, format='mjd').isot, 'mjd': mjd,
            'detector': 2 if arm == 0 else 5, 'wavelength_um': (1.20 if arm == 0 else 4.00) + 0.004 * exposure,
            'flux_mjy': (30.0 if arm == 0 else 12.0) * (1 + 0.01 * rng.normal()), 'flux_err_mjy': 0.3, 'snr': 60.0,
            'aperture_correction': 0.9, 'n_bad_in_aperture': 0, 'object_found': True,
            'object_ra': ra0 + 1e-5 * rng.normal(), 'object_dec': dec0 + 1e-5 * rng.normal(),
            'object_err_arcsec': 0.05, 'expected_ra': ra0, 'expected_dec': dec0, 'track': 'ok', 'track_sep_px': 0.1,
            'position_method': 'psf', 'psf_chi2': 1.0, 'saturated': False, 'alignment_px': 0.05,
            'rotation_deg': 10.0, 'astrometry_ok': True, 'psf_fwhm_arcsec': 8.0, 'field_unmatched': 0,
            'access_url': 'https://example.invalid/image.fits'}
        frames.append(frame)
        image = rng.normal(0, 0.02, (50, 50)) + spot(25, 25, 3.0 if arm == 0 else 1.5) + spot(38, 12, 1.0)
        sky_map = [{'kind': 'star', 'source': 'Gaia', 'label': 'Gaia DR3 1', 'ra': star_ra, 'dec': star_dec,
                    'detected': True, 'named': False, 'mag': 12.0, 'catalog_row': 0},
                   {'kind': 'artifact', 'label': 'artifact: GHOST', 'ra': new_ra, 'dec': dec0, 'detected': True},
                   {'kind': 'new', 'label': 'NEW-1 (static)', 'ra': new_ra, 'dec': new_dec, 'detected': True,
                    'new_id': 'NEW-1', 'info': 'same place in all 2 visits'}]
        zooms.append(dict(frame, image=image, variance=np.full((50, 50), 0.02 ** 2), bad=np.zeros((50, 50), bool),
                          flags=np.zeros((50, 50), np.int64), wcs=wcs, catalog_xy=(25.0, 25.0), map=sky_map))
    new_object = {'id': 'NEW-1', 'kind': 'static', 'description': 'same place in all 2 visits (3 detections), "quoted"',
                  'ra': new_ra, 'dec': new_dec, 'track': {'kind': 'coords', 'coords': (new_ra, new_dec), 'name': 'NEW-1'},
                  'detections': [{'frame': k, 'ra': new_ra, 'dec': new_dec, 'sigma': 6.0 + k, 'mjd': frames[k]['mjd']}
                                 for k in (0, 5, 9)]}
    catalog = Table(rows=[{'source': 'Gaia', 'name': 'Gaia DR3 1', 'label': 'Gaia DR3 1', 'ra': star_ra,
                           'dec': star_dec, 'pmra': 10.0, 'pmdec': -5.0, 'epoch_mjd': 57388.0, 'parallax': 2.0}])
    sky = {'catalog': catalog, 'bodies': [], 'planets': {}, 'new_objects': [new_object], 'checked': set()}
    tracked = {'name': name, 'coord': SkyCoord(ra0, dec0, unit='deg'), 'skymap': sky, 'catalog_sed': [],
               'star': {'name': name, 'object_type': 'sky position', 'query': 'made up', 'ra': ra0, 'dec': dec0,
                        'pmra': 0.0, 'pmdec': 0.0, 'parallax': None, 'rv': None, 'magnitudes': {}, 'spectral_type': '?'}}
    quiet = mock.patch('Jonah_Code_File.measure.log')           # (the summaries these two print)
    with quiet:
        movement, brightness = track_object(frames), brightness_changes(frames)
    return {'tracked': tracked, 'frames': frames, 'zooms': zooms, 'movement': movement, 'brightness': brightness,
            'spectrum': None, 'catalog_motion': {'usable': n}, 'field': None, 'skymap': sky,
            'detection': [{'band_um': (1.1, 1.6), 'frames': n // 2, 'single_detections': n // 2, 'flux_mjy': 30.0,
                           'noise_mjy': 0.1, 'significance': 300.0, 'formal_snr': 250.0}],
            'contamination': [('CatWISE J1', 6.6, 0.03, 0.02)]}


@contextmanager
def temporary_folders():
    """A results folder and a cache folder (the known-objects registry lives next to it) nobody else uses."""
    with tempfile.TemporaryDirectory() as folder:
        with api.settings_override({'RESULTS_DIR': str(Path(folder) / 'results'),
                                    'CACHE_DIR': str(Path(folder) / 'cache' / 'cache')}):
            with mock.patch('Jonah_Code_File.results.log'), mock.patch('Jonah_Code_File.algorithm.log'):
                yield Path(folder)


SETTINGS = {'mode': 'coordinates', 'coords': '150.0 2.0'}


def fake_analyze(target, settings=None, on_progress=None, isolated=False, return_raw=False):
    """Stands in for api.analyze: the made-up analysis, with the progress lines a real one sends."""
    if target.get('star') == 'nothing there':
        raise RuntimeError('No SPHEREx image covers nothing there yet')
    raw = synthetic_results()
    for line in ('Downloading and measuring 12 images (16 at a time)...', '  12/12 images  (40.0/s, 0 s left)',
                 '\n12 frames over 203 days in 2 survey visits', 'Done in 0 s'):
        if on_progress:
            on_progress(line)
    summary = api.to_json(raw)
    return (summary, raw) if return_raw else summary


def wait_for(job, state='done', seconds=20):
    end = time.time() + seconds
    while job.state != state and time.time() < end:
        time.sleep(0.01)
    return job.state


# --------------------------------------------------------------------------------------------------------------
class CoordinateTests(unittest.TestCase):
    def test_a_position_stored_as_a_pair_can_be_read_back(self):
        # results files store a position given as numbers as '(48.192, -12.635)': it must run again
        for text in ('(48.192, -12.635)', '[48.192, -12.635]', '48.192 -12.635', (48.192, -12.635)):
            coord = parse_coords(text)
            self.assertAlmostEqual(coord.ra.deg, 48.192, places=9)
            self.assertAlmostEqual(coord.dec.deg, -12.635, places=9)
        self.assertEqual(coords_name(parse_coords('(48.192, -12.635)')), 'RA 03:12:46.1 Dec -12:38:06')

    def test_sexagesimal_forms_are_unchanged(self):
        for text in ('08h55m10.831s -07d14m42.53s', '08:55:10.831 -07:14:42.53', '08 55 10.831 -07 14 42.53'):
            coord = parse_coords(text)
            self.assertAlmostEqual(coord.ra.deg, 133.795129, places=5)
            self.assertAlmostEqual(coord.dec.deg, -7.245147, places=5)

    def test_max_images_setting_reaches_the_run(self):
        # settings={'MAX_IMAGES': N} (the example in api.py) used to be ignored
        with mock.patch('Jonah_Code_File.main.run_settings', return_value='by settings') as by_settings, \
                mock.patch('Jonah_Code_File.main.run', return_value='by target') as by_target:
            with api.settings_override({'MAX_IMAGES': 7}):
                api._run({'mode': 'named star', 'star': 'x'})
                api._run({'star': 'x'})
                api._run({'star': 'x', 'max_images': 3})                    # the target's own limit wins
            api._run({'star': 'x'})                                          # no setting: as before
        self.assertEqual(by_settings.call_args.args[0]['max_images'], 7)
        self.assertEqual([call.kwargs.get('max_images') for call in by_target.call_args_list], [7, 3, None])


class SummaryTests(unittest.TestCase):
    def test_summary_is_strict_json_with_the_extras(self):
        raw = synthetic_results()
        summary = api.to_json(raw)
        json.dumps(summary, allow_nan=False)
        self.assertEqual(summary['n_frames'], 12)
        self.assertEqual(len(summary['movement']['usable']), 12)
        self.assertTrue(all(summary['movement']['usable']))
        self.assertEqual(len(summary['movement']['scatter_arcsec']), 12)
        self.assertEqual(summary['star']['object_type'], 'sky position')
        self.assertNotIn('magnitudes', summary['star'])
        self.assertAlmostEqual(summary['position']['ra_deg'], raw['frames'][0]['expected_ra'])
        self.assertEqual(summary['detection'][0]['band_um'], [1.1, 1.6])
        self.assertEqual(summary['contamination'][0], {'name': 'CatWISE J1', 'separation_arcsec': 6.6,
                                                       'w1_mjy': 0.03, 'w2_mjy': 0.02})
        self.assertEqual(summary['new_objects'][0]['track']['kind'], 'coords')

    def test_band_spectrum_has_102_channels(self):
        bands = api.band_spectrum(synthetic_results()['frames'])
        self.assertEqual({len(values) for values in bands.values()}, {102})
        self.assertEqual(sum(bands['count']), 12)
        filled = [flux for flux, count in zip(bands['flux_mjy'], bands['count']) if count]
        self.assertTrue(all(10 < flux < 32 for flux in filled))
        self.assertIsNone(bands['flux_mjy'][0])                              # an empty channel: null, not NaN

    def test_compare_frames_picks_a_partner_at_another_time(self):
        raw = synthetic_results()
        result = api.compare_frames(raw, 0)
        self.assertNotEqual(raw['frames'][result['partner']]['mjd'], raw['frames'][0]['mjd'])
        self.assertEqual(result['comparison']['status_code'], 'consistent')
        self.assertEqual(api.compare_frames(raw, 0, 1)['partner'], 1)
        json.dumps(result, allow_nan=False)

    def test_frame_images_are_unchanged_by_keeping_the_colour(self):
        from Jonah_Code_File.visual import anchors, find_partners, render, target_xy
        raw = synthetic_results()
        first = api.frame_png(raw, 3)
        self.assertTrue(first.startswith(PNG))
        self.assertIn('_display', raw)
        self.assertEqual(api.frame_png(raw, 3), first)                       # the kept colour: the same picture
        # ... and the same picture as working it out for this frame alone, as frame_png did before
        zooms = raw['zooms']
        partners = find_partners(zooms)
        colour = anchors(zooms, partners)[0]['colour']
        rgb, _ = render(zooms[3], zooms[partners[3]], anchor={'xy': target_xy(zooms[3]), 'colour': colour})
        import matplotlib.image as mpimg
        buffer = io.BytesIO()
        mpimg.imsave(buffer, np.flipud(rgb), format='png')
        self.assertEqual(first, buffer.getvalue())
        self.assertTrue(api.frame_png(raw, 3, pretty=False).startswith(PNG))

    def test_track_requests_become_targets(self):
        raw = synthetic_results()
        star = api.object_track(raw, 0, 0)
        self.assertEqual(star['mode'], 'catalog star')
        self.assertEqual(star['star']['name'], 'Gaia DR3 1')
        new = api.object_track(raw, 0, 2)
        self.assertEqual(new['mode'], 'coordinates')
        self.assertEqual(len(new['coords']), 2)
        with self.assertRaisesRegex(ValueError, 'artifact'):
            api.object_track(raw, 0, 1)
        with self.assertRaises(IndexError):
            api.object_track(raw, 0, 99)
        self.assertEqual(api.track_settings({'kind': 'asteroid', 'id': 433, 'window': ['2026-01-01', '2026-02-01']}),
                         {'mode': 'asteroid', 'asteroid': '433', 'window': ['2026-01-01', '2026-02-01']})
        path = api.track_settings({'kind': 'path', 'name': 'NEW-2', 'mjd': [1, 2], 'ra': [10, 11], 'dec': [0, 1]})
        self.assertEqual((path['mode'], path['mjd']), ('path', [1.0, 2.0]))


class SavedResultsTests(unittest.TestCase):
    def test_a_saved_analysis_reads_back_as_it_was_summarised(self):
        raw = synthetic_results()
        summary = api.to_json(raw)
        with temporary_folders() as folder:
            save_results(raw, dict(SETTINGS, max_images=None))
            self.assertTrue((folder / 'results' / 'Synthetic_field.ecsv').is_file())
            saved = api.saved_summary('Synthetic field')
            json.dumps(saved, allow_nan=False)
            self.assertEqual((saved['target'], saved['kind'], saved['n_frames']), ('Synthetic field', 'fixed', 12))
            self.assertEqual(saved['settings'], dict(SETTINGS, max_images=None))
            for key in ('obs_id', 'date_obs', 'detector', 'track', 'position_method', 'saturated', 'astrometry_ok'):
                self.assertEqual([f[key] for f in saved['frames']], [f[key] for f in summary['frames']], key)
            for key in ('mjd', 'wavelength_um', 'flux_mjy', 'flux_err_mjy', 'snr', 'object_ra', 'expected_dec'):
                np.testing.assert_allclose([f[key] for f in saved['frames']], [f[key] for f in summary['frames']],
                                           rtol=1e-12, err_msg=key)
            for key in ('offset_arcsec', 'd_ra_arcsec', 'd_dec_arcsec', 'error_arcsec', 'scatter_arcsec'):
                np.testing.assert_allclose(saved['movement'][key], summary['movement'][key], rtol=1e-9, atol=1e-12)
            self.assertEqual(saved['movement']['flagged'], summary['movement']['flagged'])
            self.assertEqual(saved['movement']['usable'], summary['movement']['usable'])
            self.assertEqual(saved['brightness']['used'], summary['brightness']['used'])
            np.testing.assert_allclose(saved['brightness']['deviation'], summary['brightness']['deviation'], rtol=1e-9)
            self.assertAlmostEqual(saved['brightness']['scatter'], summary['brightness']['scatter'], places=12)
            self.assertEqual(saved['band_spectrum'], api.band_spectrum(raw['frames']))
            self.assertEqual(saved['position'], {'ra_deg': raw['frames'][0]['expected_ra'],
                                                 'dec_deg': raw['frames'][0]['expected_dec']})
            new, = saved['new_objects']
            self.assertEqual((new['id'], new['class'], new['description']),
                             ('NEW-1', 'static', summary['new_objects'][0]['description']))
            self.assertEqual([d['frame'] for d in new['detections']], ['T1', 'T6', 'T10'])
            self.assertAlmostEqual(new['ra_deg'], summary['new_objects'][0]['ra_deg'], places=5)
            self.assertEqual(new['track']['kind'], 'coords')

    def test_known_targets_and_finding_the_saved_one(self):
        with temporary_folders():
            before = api.known_targets()
            self.assertEqual([t['name'] for t in before][:2], ['433 Eros', "Barnard's Star"])      # the test objects
            self.assertFalse(any(t['analysed'] for t in before))
            self.assertIsNone(api.saved_target(SETTINGS))
            save_results(synthetic_results(name=coords_name(parse_coords('150.0 2.0'))), SETTINGS)
            save_results(synthetic_results(name="Barnard's star"), {'mode': 'named star', 'star': "Barnard's Star"})
            targets = api.known_targets()
            self.assertEqual(sum(t['analysed'] for t in targets), 2)
            self.assertEqual(len(targets), len(before) + 1)          # Barnard's Star is listed once, as analysed
            self.assertEqual(api.saved_target({'mode': 'coordinates', 'coords': [150.0, 2.0]})['name'],
                             'RA 10:00:00.0 Dec +02:00:00')
            self.assertEqual(api.saved_target({'mode': 'named star', 'star': "barnard's star"})['frames'], 12)
            self.assertIsNone(api.saved_target({'mode': 'coordinates', 'coords': '151.0 2.0'}))
            candidates = api.saved_candidates()
            self.assertEqual(sorted(c['target'] for c in candidates), ["Barnard's star", 'RA 10:00:00.0 Dec +02:00:00'])
            self.assertEqual({c['id'] for c in candidates}, {'NEW-1'})

    def test_no_file_is_read_for_an_unknown_name(self):
        with temporary_folders():
            for name in ('nothing', '../../config', 'C:\\Windows\\win.ini', 'results.ecsv'):
                with self.assertRaises(FileNotFoundError):
                    api.saved_summary(name)

    def test_motion_fits_of_the_saved_results(self):
        with temporary_folders() as folder:
            save_results(synthetic_results(), SETTINGS)
            fits = api.motion_fits()
            json.dumps(fits, allow_nan=False)
            fit = fits['estimates']['Synthetic field']
            self.assertEqual((fit['mode'], fit['model'], fit['is_moving_candidate']),
                             ('coordinates', 'weighted_polynomial', False))
            self.assertEqual(len(fit['path']['mjd']), len(fit['path']['ra_deg']))
            self.assertFalse(any(fit['path']['extrapolated']))
            self.assertTrue((folder / 'results' / 'estimated_paths' / 'estimated_paths.ecsv').is_file())
            self.assertTrue(any(api.motion_fits(30.0)['estimates']['Synthetic field']['path']['extrapolated']))


class JobTests(unittest.TestCase):
    def test_a_job_runs_and_reports_progress(self):
        manager = JobManager(analyze=fake_analyze)
        job = manager.submit(SETTINGS)
        self.assertEqual(wait_for(job), 'done')
        snapshot = job.snapshot()
        self.assertEqual(snapshot['progress'], {'done': 12, 'total': 12})
        self.assertEqual(snapshot['messages'][-1], 'Done in 0 s')
        self.assertNotIn('', snapshot['messages'])
        self.assertEqual(job.snapshot(since=snapshot['next'])['messages'], [])
        self.assertEqual(job.snapshot(since=snapshot['next'] - 1)['messages'], ['Done in 0 s'])
        self.assertEqual((snapshot['name'], snapshot['n_frames'], snapshot['error']), ('Synthetic field', 12, None))
        self.assertEqual(len(job.summary['band_spectrum']['count']), 102)
        self.assertTrue(job.image(0).startswith(PNG))
        self.assertEqual(manager.status(), {'busy': False, 'running': None, 'waiting': 0, 'finished': [job.id]})

    def test_the_same_request_is_not_run_twice(self):
        manager = JobManager(analyze=fake_analyze)
        job = manager.submit(SETTINGS)
        self.assertIs(manager.submit(dict(SETTINGS)), job)
        self.assertIs(manager.submit({'mode': 'coordinates', 'coords': [150.0, 2.0]}), job)     # written differently
        self.assertIs(manager.latest({'mode': 'coordinates', 'coords': '10h00m00s +02d00m00s'}), job)
        self.assertIsNot(manager.submit(dict(SETTINGS, max_images=5)), job)                      # a quick look
        self.assertIsNot(manager.submit(SETTINGS, {'SAVE_RESULTS': False}), job)
        self.assertIsNot(manager.submit(SETTINGS, force=True), job)
        self.assertIsNone(manager.latest({'mode': 'coordinates', 'coords': '10.0 10.0'}))

    def test_a_failed_analysis_does_not_stop_the_queue(self):
        manager = JobManager(analyze=fake_analyze)
        with mock.patch('Jonah_Code_File.jobs.log'):
            bad = manager.submit({'mode': 'named star', 'star': 'nothing there'})
            good = manager.submit(SETTINGS)
            self.assertEqual(wait_for(good), 'done')
        self.assertEqual((bad.state, bad.error), ('error', 'RuntimeError: No SPHEREx image covers nothing there yet'))
        self.assertIsNone(manager.latest({'mode': 'named star', 'star': 'nothing there'}))
        self.assertIsNot(manager.submit({'mode': 'named star', 'star': 'nothing there'}), bad)     # can be tried again

    def test_queue_limit_and_forgetting_old_results(self):
        release = threading.Event()

        def slow(target, **kwargs):
            release.wait(20)
            return fake_analyze(target, **kwargs)

        manager = JobManager(analyze=slow, keep_finished=2, max_waiting=2)
        jobs = [manager.submit({'mode': 'named star', 'star': f'star {k}'}) for k in range(3)]   # 1 running + 2 waiting
        self.assertEqual(wait_for(jobs[0], 'running'), 'running')
        self.assertEqual((manager.status()['busy'], manager.status()['waiting']), (True, 2))
        with self.assertRaises(QueueFull):
            manager.submit({'mode': 'named star', 'star': 'one too many'})
        release.set()
        self.assertEqual(wait_for(jobs[2]), 'done')
        time.sleep(0.05)
        self.assertIsNone(manager.get(jobs[0].id))                           # only the newest two are kept
        self.assertEqual(manager.status()['finished'], [jobs[1].id, jobs[2].id])


POPUP = {'request': 'sky_point_analysis', 'version': 1, 'target': {'ra_deg': 150.0, 'dec_deg': 2.0},
         'patch': {'half_width_deg': 0.0427}, 'named_target': None}
GAIA_TARGET = {'name': 'Some Gaia star', 'gaia_dr3': '123', 'ra_deg_at_epoch': 269.4485, 'dec_deg_at_epoch': 4.7394,
               'epoch': 2016.0, 'pmra_mas_yr': -801.55, 'pmdec_mas_yr': 10362.39, 'parallax_mas': 546.98}


@unittest.skipUnless(flask, 'Flask is not installed (the website needs it; the pipeline alone does not)')
class WebTests(unittest.TestCase):
    def setUp(self):
        from Jonah_Code_File.web import create_blueprint
        self.manager = JobManager(analyze=fake_analyze)
        app = flask.Flask(__name__)
        app.register_blueprint(create_blueprint(self.manager))
        self.client = app.test_client()
        self.folders = temporary_folders()
        self.folder = self.folders.__enter__()
        self.addCleanup(self.folders.__exit__, None, None, None)

    def start(self, body, expect=202):
        response = self.client.post('/api/analysis', json=body)
        self.assertEqual(response.status_code, expect, response.get_data(as_text=True))
        return response.get_json()

    def finish(self, body):
        job = self.start(body)['job']
        self.assertEqual(wait_for(self.manager.get(job['id'])), 'done')
        return job['id']

    def test_status_describes_the_pipeline(self):
        status = self.client.get('/api/analysis/status').get_json()
        self.assertTrue(status['available'])
        self.assertEqual(status['window']['zoom_size_pix'], cfg.ZOOM_SIZE_PIX)
        self.assertAlmostEqual(status['window']['size_arcmin'], cfg.ZOOM_SIZE_PIX * cfg.PIXEL_ARCSEC / 60)
        self.assertEqual((status['busy'], status['waiting']), (False, 0))
        self.assertEqual(status['results_dir'], str(self.folder / 'results'))

    def test_the_popup_payload_runs_an_analysis_to_the_end(self):
        job_id = self.finish({**POPUP, 'options': {'save': False}})
        state = self.client.get(f'/api/analysis/jobs/{job_id}').get_json()
        self.assertEqual((state['state'], state['target'], state['saves']),
                         ('done', {'mode': 'coordinates', 'coords': '150 2'}, False))
        self.assertEqual(state['progress'], {'done': 12, 'total': 12})
        self.assertEqual(self.client.get(f'/api/analysis/jobs/{job_id}?since={state["next"]}').get_json()['messages'], [])
        result = self.client.get(f'/api/analysis/jobs/{job_id}/result').get_json()
        self.assertEqual((result['target'], result['n_frames'], result['source'], result['saved']),
                         ('Synthetic field', 12, 'job', False))
        self.assertEqual(len(result['band_spectrum']['count']), 102)
        self.assertEqual(result['new_objects'][0]['id'], 'NEW-1')

    def test_frame_images_objects_and_comparison(self):
        from PIL import Image
        job_id = self.finish({**POPUP, 'options': {'save': False}})
        pretty = self.client.get(f'/api/analysis/jobs/{job_id}/frames/0.png')
        self.assertEqual((pretty.status_code, pretty.mimetype), (200, 'image/png'))
        self.assertIn('immutable', pretty.headers['Cache-Control'])
        self.assertEqual(Image.open(io.BytesIO(pretty.data)).size, (50 * cfg.DISPLAY_UPSCALE, 50 * cfg.DISPLAY_UPSCALE))
        original = self.client.get(f'/api/analysis/jobs/{job_id}/frames/11.png?pretty=0')
        self.assertEqual(Image.open(io.BytesIO(original.data)).size, (50, 50))
        self.assertEqual(self.client.get(f'/api/analysis/jobs/{job_id}/frames/12.png').status_code, 404)
        objects = self.client.get(f'/api/analysis/jobs/{job_id}/frames/0/objects').get_json()
        self.assertEqual((objects['frame'], objects['image_size']), ('T1', [50, 50]))
        self.assertEqual([o['kind'] for o in objects['objects']], ['star', 'artifact', 'new'])
        self.assertAlmostEqual(objects['target']['expected']['x'], 25.5, places=6)       # pixel centre, from the top left
        self.assertAlmostEqual(objects['target']['expected']['y'], 24.5, places=6)
        compare = self.client.get(f'/api/analysis/jobs/{job_id}/frames/0/compare').get_json()['comparison']
        self.assertEqual(compare['comparison']['status_code'], 'consistent')
        self.assertEqual(self.client.get(f'/api/analysis/jobs/{job_id}/frames/0/compare?with=1').get_json()
                         ['comparison']['partner'], 1)
        self.assertEqual(self.client.get(f'/api/analysis/jobs/{job_id}/frames/0/compare?with=50').status_code, 404)

    def test_an_object_picked_in_a_frame_is_analysed_next(self):
        job_id = self.finish({**POPUP, 'options': {'save': False, 'max_images': 12}})
        response = self.client.post(f'/api/analysis/jobs/{job_id}/track', json={'frame': 0, 'object': 0})
        self.assertEqual(response.status_code, 202)
        target = response.get_json()['job']['target']
        self.assertEqual((target['mode'], target['star']['name'], target['max_images']), ('catalog star', 'Gaia DR3 1', 12))
        self.assertFalse(response.get_json()['job']['saves'])                  # as the analysis it was picked in
        new = self.client.post(f'/api/analysis/jobs/{job_id}/track', json={'frame': 0, 'object': 2}).get_json()['job']
        self.assertEqual(new['target']['mode'], 'coordinates')
        self.assertRegex(new['target']['coords'], r'^[\d.]+ -?[\d.]+$')
        artifact = self.client.post(f'/api/analysis/jobs/{job_id}/track', json={'frame': 0, 'object': 1})
        self.assertEqual(artifact.status_code, 400)
        self.assertIn('artifact', artifact.get_json()['error'])
        self.assertEqual(self.client.post(f'/api/analysis/jobs/{job_id}/track', json={'frame': 0, 'object': 9}).status_code, 404)
        self.assertEqual(self.client.post(f'/api/analysis/jobs/{job_id}/track', json={'frame': 50, 'object': 0}).status_code, 404)

    def test_lookup_offers_the_right_choices_and_starts_nothing(self):
        save_results(synthetic_results(name=coords_name(parse_coords('150 2'))), SETTINGS)
        save_results(synthetic_results(name="Barnard's star"), {'mode': 'named star', 'star': "Barnard's Star"})
        plain = self.client.post('/api/analysis/lookup', json=POPUP).get_json()
        self.assertEqual([c['mode'] for c in plain['choices']], ['coordinates'])
        self.assertEqual(plain['choices'][0]['saved']['name'], 'RA 10:00:00.0 Dec +02:00:00')
        self.assertIsNone(plain['choices'][0]['job'])
        standard = self.client.post('/api/analysis/lookup', json={**POPUP, 'named_target': {'name': "Barnard's Star"}}).get_json()
        self.assertEqual([c['mode'] for c in standard['choices']], ['named star', 'coordinates'])
        self.assertEqual(standard['choices'][0]['target'], {'mode': 'named star', 'star': "Barnard's Star"})
        self.assertEqual(standard['choices'][0]['saved']['frames'], 12)
        gaia = self.client.post('/api/analysis/lookup', json={**POPUP, 'named_target': GAIA_TARGET}).get_json()
        self.assertEqual([c['mode'] for c in gaia['choices']], ['catalog star', 'coordinates'])
        star = gaia['choices'][0]['target']['star']
        self.assertAlmostEqual(star['dec'], 4.7394 - 16 * 10.36239 / 3600, places=6)      # moved back to J2000
        self.assertEqual((star['pmdec'], star['parallax'], star['query']), (10362.39, 546.98, 'Gaia DR3 123'))
        self.assertEqual(self.manager.status()['finished'] + [self.manager.status()['running']], [None])
        job_id = self.finish({**POPUP, 'options': {'save': False}})
        self.assertEqual(self.client.post('/api/analysis/lookup', json=POPUP).get_json()['choices'][0]['job']['id'], job_id)

    def test_which_choice_and_how_many_images(self):
        named = {**POPUP, 'named_target': {'name': "Barnard's Star"}}
        self.assertEqual(self.start(named)['job']['target'], {'mode': 'named star', 'star': "Barnard's Star"})
        job = self.start({**named, 'options': {'mode': 'coordinates', 'max_images': '40'}})['job']
        self.assertEqual(job['target'], {'mode': 'coordinates', 'coords': '150 2', 'max_images': 40})
        self.assertFalse(job['saves'])                                         # a quick look is not saved ...
        self.assertTrue(self.start({**named, 'options': {'mode': 'coordinates', 'max_images': 40, 'save': True}})['job']['saves'])
        self.assertTrue(self.start(POPUP)['job']['saves'])                     # ... a full analysis is
        self.assertIn('can be analysed as', self.start({**POPUP, 'options': {'mode': 'asteroid'}}, 400)['error'])

    def test_targets_in_the_pipelines_own_form(self):
        asteroid = {'target': {'mode': 'asteroid', 'asteroid': '433', 'window': ['2026-02-15', '2026-06-15']}}
        self.assertEqual(self.start(asteroid)['job']['target'], asteroid['target'])
        self.assertEqual(self.start({'target': {'star': 'R Dra'}})['job']['target'], {'mode': 'named star', 'star': 'R Dra'})
        kept = self.start({'target': {'mode': 'coordinates', 'coords': '(48.192, -12.635)'}})['job']['target']
        self.assertEqual(kept['coords'], '(48.192, -12.635)')                  # a saved position, as it was written
        self.assertEqual(self.start({'target': {'mode': 'star', 'ra_deg': 304.69, 'dec_deg': 42.44}})['job']['target']['mode'], 'star')

    def test_bad_requests_are_refused_with_a_reason(self):
        bad = [({'target': {'ra_deg': 400, 'dec_deg': 0}}, 'right ascension'),
               ({'target': {'ra_deg': 10, 'dec_deg': 'north'}}, 'declination'),
               ({'target': {'mode': 'coordinates', 'coords': 'not a place'}}, 'not understood'),
               ({'target': {'mode': 'named star', 'star': '<script>'}}, 'star name'),
               ({'target': {'mode': 'named star', 'star': 'x' * 81}}, 'star name'),
               ({'target': {'mode': 'asteroid', 'asteroid': '433'}}, 'time window'),
               ({'target': {'mode': 'asteroid', 'asteroid': '433', 'window': ['2026-06-15', '2026-02-15']}}, 'forward'),
               ({'target': {'mode': 'asteroid', 'asteroid': '433', 'window': ['2020-01-01', '2026-01-01']}}, 'at most'),
               ({'target': {'mode': 'asteroid', 'asteroid': '433; drop', 'window': ['2026-02-15', '2026-06-15']}}, 'id is needed'),
               ({'target': {'mode': 'path', 'mjd': [1], 'ra': [1], 'dec': [1]}}, 'path needs'),
               ({'target': {'mode': 'teleport'}}, 'unknown kind'),
               ({'target': 'Eros'}, 'must be an object'),
               ({**POPUP, 'options': {'max_images': -3}}, 'max images'),
               ({}, 'must be an object')]
        for body, reason in bad:
            self.assertIn(reason, self.start(body, 400)['error'], body)
        self.assertEqual(self.client.post('/api/analysis', data='ra=1', content_type='application/x-www-form-urlencoded').status_code, 400)
        self.assertEqual(self.manager.status()['waiting'] + len(self.manager.status()['finished']), 0)     # nothing started

    def test_jobs_that_are_unknown_unfinished_or_failed(self):
        self.assertEqual(self.client.get('/api/analysis/jobs/nope').status_code, 404)
        self.assertEqual(self.client.get('/api/analysis/jobs/nope/frames/0.png').status_code, 404)
        release = threading.Event()
        self.manager._analyze = lambda target, **kwargs: (release.wait(20), fake_analyze(target, **kwargs))[1]
        job_id = self.start(POPUP)['job']['id']
        waiting = self.client.get(f'/api/analysis/jobs/{job_id}/result')
        self.assertEqual(waiting.status_code, 409)
        self.assertIn('not finished', waiting.get_json()['error'])
        self.assertEqual(self.client.get(f'/api/analysis/jobs/{job_id}/frames/0.png').status_code, 409)
        self.assertTrue(self.client.get('/api/analysis/status').get_json()['busy'])
        release.set()
        self.assertEqual(wait_for(self.manager.get(job_id)), 'done')
        with mock.patch('Jonah_Code_File.jobs.log'):
            failed = self.start({'target': {'mode': 'named star', 'star': 'nothing there'}})['job']['id']
            self.assertEqual(wait_for(self.manager.get(failed), 'error'), 'error')
        answer = self.client.get(f'/api/analysis/jobs/{failed}/result')
        self.assertEqual((answer.status_code, answer.get_json()['error']),
                         (409, 'RuntimeError: No SPHEREx image covers nothing there yet'))
        self.assertEqual(self.client.get(f'/api/analysis/jobs/{failed}').get_json()['state'], 'error')

    def test_saved_results_targets_candidates_and_paths(self):
        save_results(synthetic_results(), SETTINGS)
        targets = self.client.get('/api/analysis/targets').get_json()['targets']
        self.assertEqual((targets[0]['name'], targets[0]['frames'], targets[0]['has_results'], targets[0]['job']),
                         ('Synthetic field', 12, True, None))
        self.assertEqual(sum(not t['analysed'] for t in targets), 6)
        saved = self.client.get('/api/analysis/results?name=Synthetic%20field').get_json()
        self.assertEqual((saved['source'], saved['n_frames'], saved['file']), ('saved', 12, 'Synthetic_field.ecsv'))
        for name in ('nothing', '../../config', 'Synthetic_field.ecsv', ''):
            self.assertIn(self.client.get('/api/analysis/results', query_string={'name': name}).status_code, (400, 404), name)
        table = self.client.get('/api/analysis/results/file?name=Synthetic%20field')
        self.assertEqual(table.status_code, 200)
        self.assertIn('Synthetic_field.ecsv', table.headers['Content-Disposition'])
        self.assertTrue(table.data.startswith(b'# %ECSV'))
        table.close()
        new = self.client.get('/api/analysis/results/file?name=Synthetic%20field&kind=new_objects')
        self.assertTrue(new.data.startswith(b'object,class,ra_deg'))
        new.close()
        self.assertEqual(self.client.get('/api/analysis/results/file?name=nothing').status_code, 404)
        candidates = self.client.get('/api/analysis/candidates').get_json()['candidates']
        self.assertEqual([(c['target'], c['id'], c['class']) for c in candidates], [('Synthetic field', 'NEW-1', 'static')])
        paths = self.client.get('/api/analysis/paths').get_json()
        self.assertEqual(list(paths['estimates']), ['Synthetic field'])
        self.assertEqual(self.client.get('/api/analysis/paths?days=-1').status_code, 400)

    def test_no_saved_results_at_all(self):
        self.assertEqual(self.client.get('/api/analysis/candidates').get_json(), {'candidates': []})
        self.assertEqual(self.client.get('/api/analysis/paths').get_json()['estimates'], {})
        self.assertFalse(any(t['analysed'] for t in self.client.get('/api/analysis/targets').get_json()['targets']))


@unittest.skipUnless(flask and (SITE / 'app.py').is_file(), 'the website (Test_Robert_Code_Folder) or Flask is missing')
class WebsiteTests(unittest.TestCase):
    """The website itself, with the bridge connected: its own pages and routes must be as they were."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('xarvis_site', SITE / 'app.py')
        cls.site = importlib.util.module_from_spec(spec)
        sys.modules['xarvis_site'] = cls.site
        spec.loader.exec_module(cls.site)
        cls.client = cls.site.app.test_client()

    def test_the_pipeline_is_connected(self):
        self.assertIsNone(self.site.PIPELINE_ERROR)
        status = self.client.get('/api/analysis/status').get_json()
        self.assertTrue(status['available'])
        self.assertEqual(Path(status['results_dir']), Path(__file__).resolve().parent.parent / 'results')

    def test_every_route_is_there(self):
        rules = {rule.rule for rule in self.site.app.url_map.iter_rules()}
        self.assertLessEqual({'/', '/about', '/about.html', '/api/meta', '/api/track', '/api/spherex', '/api/orbits',
                              '/api/sbdb/lookup', '/api/sbident'}, rules)                     # the website's own
        self.assertLessEqual({'/api/analysis', '/api/analysis/status', '/api/analysis/targets', '/api/analysis/lookup',
                              '/api/analysis/jobs/<job_id>', '/api/analysis/jobs/<job_id>/result',
                              '/api/analysis/jobs/<job_id>/frames/<int:index>.png',
                              '/api/analysis/jobs/<job_id>/frames/<int:index>/objects',
                              '/api/analysis/jobs/<job_id>/frames/<int:index>/compare',
                              '/api/analysis/jobs/<job_id>/track', '/api/analysis/results', '/api/analysis/results/file',
                              '/api/analysis/candidates', '/api/analysis/paths'}, rules)

    def test_the_pages_and_the_404s_are_as_before(self):
        page = self.client.get('/')
        self.assertEqual(page.status_code, 200)
        self.assertIn('js/pipeline.js', page.get_data(as_text=True))
        self.assertEqual(self.client.get('/about').status_code, 200)
        missing = self.client.get('/no-such-page')
        self.assertEqual((missing.status_code, missing.mimetype), (404, 'text/html'))
        self.assertEqual(self.client.get('/api/no-such-route').get_json(), {'error': 'not found'})
        self.assertEqual(self.client.get('/api/analysis/no-such-route').get_json(), {'error': 'not found'})
        self.assertEqual(self.client.get('/api/sbdb/lookup?q=').status_code, 400)

    def test_the_scripts_the_page_loads_exist(self):
        import re
        html = self.client.get('/').get_data(as_text=True)
        scripts = re.findall(r'<script src="/static/(js/[\w.]+)\?v=(\d+)"', html)
        self.assertEqual([name for name, _ in scripts], ['js/core.js', 'js/scene.js', 'js/draw.js', 'js/tracker.js',
                                                         'js/view3d.js', 'js/analysis.js', 'js/pipeline.js', 'js/app.js'])
        self.assertTrue(all(int(version) > 0 for _, version in scripts))      # (0 = the file is missing)


if __name__ == '__main__':
    unittest.main()
