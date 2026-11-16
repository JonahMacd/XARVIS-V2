"""The website's bridge to the pipeline: every /api/analysis/... route, as a Flask blueprint.

    from xarvis_pipeline.web_blueprint import create_blueprint
    app.register_blueprint(create_blueprint())

The XARVIS website (website/app.py) registers it under /api/pipeline instead
(app.register_blueprint(create_blueprint(name='pipeline'), url_prefix='/api/pipeline')): its own /api/analysis routes
run the page's image time-lapse through web.py's analyse(). The routes below are then /api/pipeline/...

Routes (JSON unless noted; an error is {"error": "..."} with a 4xx / 5xx status)
    GET  /api/analysis/status                         is the pipeline there, is it busy, the window it analyses
    GET  /api/analysis/targets                        known targets: analysed ones, then the standard test objects
    POST /api/analysis/lookup                         what can be analysed for a request, and what already
                                                      exists for it - saved results, a job (starts nothing)
    POST /api/analysis                                start an analysis -> 202 {"job": {...}}
    GET  /api/analysis/jobs/<id>?since=N              its state, progress, and the progress lines from number N
    GET  /api/analysis/jobs/<id>/result               its summary once finished (api.to_json + 'band_spectrum')
    GET  /api/analysis/jobs/<id>/frames/<i>.png       frame T(i+1), a PNG image (?pretty=0: the original pixels,
                                                      ?wide=1: the wide view, downloaded on demand)
    GET  /api/analysis/jobs/<id>/frames/<i>/objects   what the sky map found in it, in its pixel grid
    GET  /api/analysis/jobs/<id>/frames/<i>/compare   it against the capture closest in wavelength (?with=j: frame j)
    POST /api/analysis/jobs/<id>/track                {"frame": i, "object": k}: analyse that object next -> 202
    GET  /api/analysis/results?name=                  a saved analysis (results/<name>.ecsv), in a job result's form
    GET  /api/analysis/results/file?name=             that file itself (&kind=new_objects: its new-objects table)
    GET  /api/analysis/candidates                     every NEW object in the saved results
    GET  /api/analysis/paths?days=0                   algorithm.run_from_results: every saved target's apparent path

What to analyse - the body of POST /api/analysis and /lookup - is either
  - the sky-point popup's payload exactly as the page builds it (static/js/analysis.js):
      {"request": "sky_point_analysis", "target": {"ra_deg": .., "dec_deg": ..}, "named_target": {...} | null, ...}
    A point can be analysed as that exact position; a named target also as the star itself (see choices_for), or
  - {"target": {...}} in the pipeline's own form - what a results file stores (results.known_objects):
      {"mode": "coordinates", "coords": "133.795 -7.245"}     {"mode": "named star", "star": "Barnard's Star"}
      {"mode": "asteroid", "asteroid": "433", "window": ["2026-02-15", "2026-06-15"]}    (also 'star', 'catalog
      star' and 'path').
with "options": {"mode": which of the choices (default: the first), "max_images": only the first N images,
"save": false = don't write results/<target>.ecsv (the default when max_images is given, so a quick look never
replaces a full analysis), "force": true = run again even if that was analysed already}.

Nothing from a request is used as a file path: saved results are only read for names in the known-objects list.
"""

import logging
import re
import threading

import matplotlib

matplotlib.use('Agg')                   # a web server never opens a window (and Tk is not safe in its threads)

import numpy as np                                                      # noqa: E402  (after choosing the backend)
from astropy.time import Time                                           # noqa: E402

from . import api                                                       # noqa: E402
from . import config as cfg                                             # noqa: E402
from .jobs import JobManager, QueueFull                                 # noqa: E402
from .main import parse_coords                                          # noqa: E402
from .results import DEFAULT_OBJECTS, results_dir, results_path         # noqa: E402

log = logging.getLogger(__name__)

MAX_IMAGES_LIMIT = 5000                 # (IRSA's image search returns at most this many)
MAX_WINDOW_DAYS = 400                   # an asteroid's search window: the survey's first year and a bit
MAX_PATH_POINTS = 500
STAR_NAME = re.compile(r"^[\w .,+\-*/'()\[\]:]{1,80}$")                 # a SIMBAD identifier
ASTEROID_ID = re.compile(r"^[\w .\-/'()]{1,40}$")                       # a JPL Horizons id or designation
MODES = ('coordinates', 'named star', 'asteroid', 'star', 'catalog star', 'path')


class Problem(Exception):
    """A request that can't be answered: the message goes to the browser as {"error": ...} with this status."""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.message, self.status = message, status


# ----------------------------------------------------------------------------
# What the browser sent -> a target the pipeline runs
# ----------------------------------------------------------------------------

def _number(value, low, high, what):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise Problem(f'{what}: a number is needed') from None
    if not (np.isfinite(number) and low <= number <= high):
        raise Problem(f'{what}: must be between {low} and {high}')
    return number


def _decimal(number):
    """A number as plain decimal text (never 1e-05), without trailing zeros."""
    return np.format_float_positional(number, trim='-')


def _text(value, limit, default=''):
    return str(value if value is not None else default).strip()[:limit]


def clean_target(raw):
    """A target as the browser sent it -> the pipeline's own form (what main.run_settings runs), every value
    checked. Raises Problem. A position given as numbers becomes text ('133.795129 -7.245147'): that is what a
    results file can store and read back; one given as text is kept as written, so re-running a saved target
    asks for exactly the same position."""
    if not isinstance(raw, dict):
        raise Problem('the target must be an object, e.g. {"mode": "coordinates", "coords": "133.795 -7.245"}')
    mode = raw.get('mode')
    if mode is None:                                    # the short forms api.analyze takes
        mode = ('asteroid' if raw.get('asteroid') else 'named star' if raw.get('star') else
                'coordinates' if raw.get('coords') is not None else None)
    if mode == 'coordinates':
        coords = raw.get('coords')
        if isinstance(coords, (list, tuple)) and len(coords) == 2:
            ra = _number(coords[0], 0, 360, 'right ascension (degrees)')
            dec = _number(coords[1], -90, 90, 'declination (degrees)')
            return {'mode': mode, 'coords': f'{_decimal(ra)} {_decimal(dec)}'}
        if isinstance(coords, str) and 0 < len(coords.strip()) <= 60:
            try:
                parse_coords(coords)
            except Exception:
                raise Problem('coordinates not understood - e.g. "08h55m10.8s -07d14m42s" or "133.795 -7.245"') from None
            return {'mode': mode, 'coords': coords.strip()}
        raise Problem('coordinates are needed: "RA Dec" as text, or [ra, dec] in degrees')
    if mode == 'named star':
        star = _text(raw.get('star'), 200)
        if not STAR_NAME.match(star):
            raise Problem('a star name is needed (letters, digits and . , + - * / \' ( ) [ ] :, at most 80 characters)')
        return {'mode': mode, 'star': star}
    if mode == 'asteroid':
        asteroid = _text(raw.get('asteroid'), 200)
        if not ASTEROID_ID.match(asteroid):
            raise Problem('an asteroid or comet id is needed (a JPL Horizons number or designation, e.g. 433)')
        window = raw.get('window')
        if not (isinstance(window, (list, tuple)) and len(window) == 2):
            raise Problem('an asteroid needs a time window: two dates, e.g. ["2026-02-15", "2026-06-15"]')
        window = [_text(day, 30) for day in window]
        try:
            start, stop = (Time(day) for day in window)
        except ValueError:
            raise Problem('the time window needs two dates like 2026-02-15') from None
        if not 0 < (stop - start).jd <= MAX_WINDOW_DAYS:
            raise Problem(f'the time window must run forward, for at most {MAX_WINDOW_DAYS} days')
        return {'mode': mode, 'asteroid': asteroid, 'window': window}
    if mode == 'star':
        return {'mode': mode, 'ra_deg': _number(raw.get('ra_deg'), 0, 360, 'right ascension (degrees)'),
                'dec_deg': _number(raw.get('dec_deg'), -90, 90, 'declination (degrees)')}
    if mode == 'catalog star':
        star = raw.get('star')
        if not isinstance(star, dict):
            raise Problem('a catalog star needs its position and motion')
        optional = lambda key, low, high: None if star.get(key) is None else _number(star[key], low, high, key)
        name = _text(star.get('name'), 80) or 'catalog star'
        return {'mode': mode, 'star': {
            'name': name, 'object_type': _text(star.get('object_type'), 40, 'catalog star'),
            'query': _text(star.get('query'), 80, name),
            'ra': _number(star.get('ra'), 0, 360, 'right ascension (degrees)'),
            'dec': _number(star.get('dec'), -90, 90, 'declination (degrees)'),
            'pmra': _number(star.get('pmra') or 0, -1e5, 1e5, 'proper motion in RA (mas/yr)'),
            'pmdec': _number(star.get('pmdec') or 0, -1e5, 1e5, 'proper motion in Dec (mas/yr)'),
            'parallax': optional('parallax', -1e4, 1e4), 'rv': optional('rv', -1e4, 1e4),
            'magnitudes': {}, 'spectral_type': _text(star.get('spectral_type'), 20, '?')}}
    if mode == 'path':
        mjd, ra, dec = (raw.get(key) for key in ('mjd', 'ra', 'dec'))
        if not (all(isinstance(values, (list, tuple)) for values in (mjd, ra, dec))
                and 2 <= len(mjd) == len(ra) == len(dec) <= MAX_PATH_POINTS):
            raise Problem(f'a path needs matching lists of times (mjd), ra and dec: 2 to {MAX_PATH_POINTS} points')
        return {'mode': mode, 'name': _text(raw.get('name'), 80) or 'moving object',
                'mjd': [_number(v, 50000, 80000, 'time (MJD)') for v in mjd],
                'ra': [_number(v, 0, 360, 'right ascension (degrees)') for v in ra],
                'dec': [_number(v, -90, 90, 'declination (degrees)') for v in dec]}
    raise Problem(f'unknown kind of target - one of: {", ".join(MODES)}')


def clean_options(raw):
    """The "options" of a request, checked: {'mode', 'max_images', 'save', 'force'}."""
    raw = raw if isinstance(raw, dict) else {}
    max_images = raw.get('max_images')
    if max_images in (None, '', 0, '0'):
        max_images = None
    else:
        max_images = int(_number(max_images, 1, MAX_IMAGES_LIMIT, 'max images'))
    save = raw.get('save')
    return {'mode': _text(raw.get('mode'), 20) or None, 'max_images': max_images,
            'save': (max_images is None) if save is None else bool(save), 'force': bool(raw.get('force'))}


def gaia_star(named):
    """A named target of the page that carries catalog motion (ESA Gaia: its position at `epoch`, proper motion,
    parallax - see static/targets.json) as a star the pipeline can track: the position moved back to J2000
    along the proper motion, exactly as skymap.star_to_track does for a catalog row."""
    years = 2000.0 - float(named['epoch'])
    pmra, pmdec = float(named['pmra_mas_yr']), float(named['pmdec_mas_yr'])
    dec = float(named['dec_deg_at_epoch']) + pmdec * years / 3.6e6
    ra = float(named['ra_deg_at_epoch']) + pmra * years / 3.6e6 / np.cos(np.radians(dec))
    parallax = named.get('parallax_mas')
    return {'name': named['name'], 'object_type': 'Gaia DR3 star',
            'query': f"Gaia DR3 {named['gaia_dr3']}" if named.get('gaia_dr3') else named['name'],
            'ra': ra % 360, 'dec': dec, 'pmra': pmra, 'pmdec': pmdec,
            'parallax': float(parallax) if parallax and float(parallax) > 0 else None, 'rv': None}


def is_popup_payload(body):
    point = body.get('target')
    return body.get('request') == 'sky_point_analysis' or (isinstance(point, dict) and 'ra_deg' in point
                                                           and 'mode' not in point)


def choices_for(body):
    """What can be analysed for a request, the best choice first: [{'mode', 'label', 'about', 'target'}].
    The popup's payload: the named target itself when it has one - a standard test object by its SIMBAD entry,
    any other one with catalog motion by that - and always the exact position. A target in the pipeline's own
    form: itself."""
    if not is_popup_payload(body):
        target = clean_target(body.get('target'))
        return [{'mode': target['mode'], 'label': TARGET_LABEL[target['mode']], 'about': '', 'target': target}]
    point = body.get('target') if isinstance(body.get('target'), dict) else {}
    ra = _number(point.get('ra_deg'), 0, 360, 'right ascension (degrees)')
    dec = _number(point.get('dec_deg'), -90, 90, 'declination (degrees)')
    choices = []
    named = body.get('named_target') if isinstance(body.get('named_target'), dict) else {}
    name = _text(named.get('name'), 80)
    standard = {label.lower(): settings for label, settings in DEFAULT_OBJECTS}
    if name.lower() in standard:
        settings = standard[name.lower()]
        moving = settings['mode'] == 'asteroid'
        choices.append({'mode': settings['mode'], 'label': f'Track {name}', 'target': clean_target(settings),
                        'about': ('JPL Horizons predicts where it was at every exposure; each image is cut around '
                                  'that position and measured there.' if moving else
                                  'Its catalog position, proper motion and parallax (SIMBAD) give where it was at '
                                  'every exposure; SPHEREx\'s own measurement of its motion and brightness is '
                                  'compared with the catalog.')})
    elif name and all(named.get(key) is not None for key in ('epoch', 'pmra_mas_yr', 'pmdec_mas_yr',
                                                             'ra_deg_at_epoch', 'dec_deg_at_epoch')):
        try:
            target = clean_target({'mode': 'catalog star', 'star': gaia_star(named)})
        except (Problem, TypeError, ValueError):
            target = None                               # motion that can't be read: only the position is offered
        if target:
            choices.append({'mode': 'catalog star', 'label': f'Track {name}', 'target': target,
                            'about': 'Its catalog position, proper motion and parallax (ESA Gaia, as shown above) '
                                     'give where it was at every exposure.'})
    choices.append({'mode': 'coordinates', 'label': 'Map this exact position',
                    'target': clean_target({'mode': 'coordinates', 'coords': [ra, dec]}),
                    'about': 'Fixed on the sky: every object in the images is named (Hipparcos, Gaia DR3, CatWISE, '
                             'planets, known asteroids and comets) and anything in none of them is marked NEW.'})
    return choices


TARGET_LABEL = {'coordinates': 'Map this exact position', 'named star': 'Track this star',
                'star': 'Track the nearest bright star', 'catalog star': 'Track this star',
                'asteroid': 'Track this asteroid or comet', 'path': 'Follow this path'}


# ----------------------------------------------------------------------------
# The routes
# ----------------------------------------------------------------------------

def create_blueprint(manager=None, name='analysis'):
    """The Flask blueprint with every /api/analysis/... route. manager: the JobManager to use (a new one by
    default; tests pass one with their own analyze function)."""
    from flask import Blueprint, Response, jsonify, request, send_file
    from werkzeug.exceptions import HTTPException

    manager = manager or JobManager()
    bp = Blueprint(name, __name__, url_prefix='/api/analysis')
    bp.manager = manager                                # (so the app, or a test, can reach the jobs)
    fits = {'key': None, 'value': None, 'lock': threading.Lock()}

    @bp.errorhandler(Problem)
    def problem(e):
        return jsonify({'error': e.message}), e.status

    @bp.errorhandler(Exception)
    def unexpected(e):                                  # never an HTML error page for an API call
        if isinstance(e, HTTPException):
            return jsonify({'error': e.description}), e.code
        log.exception('The request %s failed', request.path)
        return jsonify({'error': f'{type(e).__name__}: {e}'}), 500

    def body_of_request():
        body = request.get_json(silent=True)            # (only JSON: a form posted by another site is refused)
        if not isinstance(body, dict):
            raise Problem('a JSON object is needed (Content-Type: application/json)')
        return body

    def job_or_404(job_id):
        job = manager.get(job_id)
        if job is None:
            raise Problem('no such analysis - it may have been replaced by newer ones: run it again', 404)
        return job

    def finished(job_id, index=None):
        """The job, which must have finished well (and have frame `index`, if given)."""
        job = job_or_404(job_id)
        if job.state == 'error':
            raise Problem(job.error, 409)
        if job.state != 'done':
            raise Problem(f'that analysis has not finished yet ({job.state})', 409)
        if index is not None and not 0 <= index < len(job.raw['zooms']):
            raise Problem(f"that analysis has frames T1 to T{len(job.raw['zooms'])}", 404)
        return job

    def brief(job):
        """A job without its progress lines."""
        snapshot = job.snapshot(since=10 ** 9)
        del snapshot['messages']
        return snapshot

    def saves(job):
        """Was this job allowed to write its results file?"""
        return job.settings.get('SAVE_RESULTS', True) is not False

    def known_name(name):
        """`name`, if it is a known target with saved results (the only names a file is ever read for)."""
        name = (name or '').strip()
        if not name:
            raise Problem('which target? (?name=...)')
        if not any(t['name'] == name and t['has_results'] for t in api.known_targets()):
            raise Problem(f'no saved results for "{name}"', 404)
        return name

    @bp.route('/status')
    def status():
        size = cfg.ZOOM_SIZE_PIX * cfg.PIXEL_ARCSEC / 60
        return jsonify({
            'available': True, **manager.status(),
            'window': {'zoom_size_pix': cfg.ZOOM_SIZE_PIX, 'pixel_arcsec': cfg.PIXEL_ARCSEC, 'size_arcmin': size,
                       'aperture_radius_pix': cfg.APERTURE_RADIUS_PIX, 'display_upscale': cfg.DISPLAY_UPSCALE,
                       'wide_view_arcmin': cfg.WIDE_VIEW_SIZE, 'wide_view_upscale': cfg.WIDE_VIEW_UPSCALE},
            'limits': {'detection_snr': cfg.DETECTION_SNR, 'movement_sigma': cfg.MOVEMENT_SIGMA,
                       'brightness_sigma': cfg.BRIGHTNESS_SIGMA, 'residual_limit_px': cfg.RESIDUAL_LIMIT_PX,
                       'track_tolerance_pix': cfg.TRACK_TOLERANCE_PIX},
            'save_results': bool(cfg.SAVE_RESULTS), 'results_dir': str(results_dir())})

    @bp.route('/targets')
    def targets():
        found = api.known_targets()
        for target in found:
            job = manager.latest(target['settings']) if target['settings'].get('mode') in MODES else None
            target['job'] = brief(job) if job else None
        return jsonify({'targets': found, 'results_dir': str(results_dir())})

    @bp.route('/lookup', methods=['POST'])
    def lookup():
        choices = choices_for(body_of_request())
        known = api.known_targets()
        for choice in choices:
            saved = api.saved_target(choice['target'], known)
            choice['saved'] = {key: saved[key] for key in ('name', 'frames', 'saved')} if saved else None
            job = manager.latest(choice['target'])
            choice['job'] = brief(job) if job else None
        return jsonify({'choices': choices, **manager.status()})

    @bp.route('', methods=['POST'])
    def start():
        body = body_of_request()
        options = clean_options(body.get('options'))
        choices = choices_for(body)
        choice = choices[0]
        if options['mode']:
            choice = next((c for c in choices if c['mode'] == options['mode']), None)
            if choice is None:
                raise Problem(f'this can be analysed as: {", ".join(c["mode"] for c in choices)}')
        return submit(choice['target'], options)

    def submit(target, options):
        target = dict(target)
        if options['max_images']:
            target['max_images'] = options['max_images']
        try:
            job = manager.submit(target, {} if options['save'] else {'SAVE_RESULTS': False}, options['force'])
        except QueueFull as e:
            raise Problem(str(e), 429) from None
        return jsonify({'job': job.snapshot()}), 202

    @bp.route('/jobs/<job_id>')
    def job_state(job_id):
        return jsonify(job_or_404(job_id).snapshot(max(request.args.get('since', 0, type=int) or 0, 0)))

    @bp.route('/jobs/<job_id>/result')
    def job_result(job_id):
        job = finished(job_id)
        return jsonify(dict(job.summary, job=job.id, settings=job.target, source='job',
                            saved=saves(job) and bool(cfg.SAVE_RESULTS)))

    @bp.route('/jobs/<job_id>/frames/<int:index>.png')
    def frame_image(job_id, index):
        job = finished(job_id, index)
        wide = request.args.get('wide') == '1'
        try:
            png = job.image(index, pretty=request.args.get('pretty') != '0', wide=wide)
        except Exception as e:                          # (the wide view is a download, which can fail)
            raise Problem(f'that image could not be made ({type(e).__name__}: {e})', 502 if wide else 500) from None
        response = Response(png, mimetype='image/png')
        response.headers['Cache-Control'] = 'private, max-age=86400, immutable'     # a job's frames never change
        return response

    @bp.route('/jobs/<job_id>/frames/<int:index>/objects')
    def frame_objects(job_id, index):
        return jsonify(api.frame_objects(finished(job_id, index).raw, index))

    @bp.route('/jobs/<job_id>/frames/<int:index>/compare')
    def frame_compare(job_id, index):
        job = finished(job_id, index)
        other = request.args.get('with', type=int)
        if other is not None and not 0 <= other < len(job.raw['zooms']):
            raise Problem(f"that analysis has frames T1 to T{len(job.raw['zooms'])}", 404)
        return jsonify({'comparison': api.compare_frames(job.raw, index, other)})

    @bp.route('/jobs/<job_id>/track', methods=['POST'])
    def track(job_id):
        body = body_of_request()
        index = int(_number(body.get('frame'), 0, 10 ** 6, 'frame'))
        job = finished(job_id, index)
        try:
            target = api.object_track(job.raw, index, int(_number(body.get('object'), 0, 10 ** 6, 'object')))
        except IndexError as e:
            raise Problem(str(e), 404) from None
        except ValueError as e:                         # an artifact, a planet: the reason is the message
            raise Problem(str(e)) from None
        asked = body.get('options') if isinstance(body.get('options'), dict) else {}
        # as the viewer's T key: the same image limit as the analysis it was picked in, saved the same way
        options = clean_options({'max_images': job.target.get('max_images'), 'save': saves(job), **asked})
        return submit(clean_target(target), options)

    @bp.route('/results')
    def saved_results():
        return jsonify(dict(api.saved_summary(known_name(request.args.get('name'))), source='saved'))

    @bp.route('/results/file')
    def saved_file():
        path = results_path(known_name(request.args.get('name')))
        if request.args.get('kind') == 'new_objects':
            path = path.with_name(path.stem + '_new_objects.csv')
        if path.resolve().parent != results_dir().resolve() or not path.is_file():
            raise Problem('no such file for that target', 404)
        return send_file(path, mimetype='text/csv' if path.suffix == '.csv' else 'text/plain', as_attachment=True,
                         download_name=path.name, max_age=0)

    @bp.route('/candidates')
    def candidates():
        return jsonify({'candidates': api.saved_candidates()})

    @bp.route('/paths')
    def paths():
        days = _number(request.args.get('days', 0), 0, 3650, 'days')
        folder = results_dir()
        files = sorted(folder.glob('*.ecsv')) if folder.is_dir() else []
        if not files:
            return jsonify({'estimates': {}, 'moving_candidates': [], 'skipped': {}, 'output_file': None,
                            'prediction_days': days})
        # the fits of ~15 targets take a few seconds: kept until a results file changes
        key = (days, tuple((f.name, f.stat().st_mtime_ns, f.stat().st_size) for f in files))
        with fits['lock']:
            if fits['key'] != key:
                fits['value'], fits['key'] = api.motion_fits(days), key
            return jsonify(fits['value'])

    return bp
