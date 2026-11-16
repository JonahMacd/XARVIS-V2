"""Analyses in the background, for a web app: start one, follow its progress, read its results.

    from xarvis_pipeline.jobs import JobManager
    manager = JobManager()
    job = manager.submit({'mode': 'coordinates', 'coords': '133.795129 -7.245147'})
    job.snapshot()              # {'id', 'state': 'queued' | 'running' | 'done' | 'error', 'progress', 'messages', ...}
    job.summary, job.raw        # once 'done': api.to_json(...) plus 'band_spectrum', and the raw results that
    job.image(0)                # api.frame_png / frame_objects need (frame T1 as PNG bytes)

One analysis runs at a time - settings are module-wide (see api.py) - and the others wait their turn. Asking again
for a target that is already waiting, running or finished returns that job instead of starting another one.
Finished jobs stay in memory - their zoomed pixels are what the frame images are made from, ~0.1 MB per frame -
until KEEP_FINISHED newer ones have finished. Nothing here outlives the program: what lasts is the results folder
(results.py) and the disk cache (cache.py), which makes running a target again quick.
"""

import json
import logging
import queue
import re
import threading
import time
import traceback
import uuid
from collections import OrderedDict

from . import api

log = logging.getLogger(__name__)

KEEP_FINISHED = 4               # finished analyses kept in memory (with their pixels)
MAX_WAITING = 8                 # analyses that may wait in the queue
MAX_MESSAGES = 3000             # progress lines kept per job (a run prints ~50)

_COUNT = re.compile(r'^\s*(\d+)/(\d+) images')                      # '  50/258 images  (15.2/s, 14 s left)'
_TOTAL = re.compile(r'Downloading and measuring (\d+) images')


class QueueFull(RuntimeError):
    """Too many analyses are waiting already."""


class Job:
    """One analysis: what was asked for, how far it is, and its results."""

    def __init__(self, target, settings=None, key=None):
        self.id = uuid.uuid4().hex[:12]                 # random: a browser's cached frame images are per job
        self.target, self.settings, self.key = dict(target), dict(settings or {}), key
        self.state = 'queued'
        self.created, self.started, self.finished = time.time(), None, None
        self.messages, self.dropped = [], 0             # progress lines; how many old ones were dropped
        self.done = self.total = 0                      # images measured so far / in all
        self.error = None
        self.summary = self.raw = None
        self._lock = threading.Lock()
        self._render = threading.Lock()

    def log(self, message):
        """A progress message from the pipeline (the lines the terminal shows)."""
        with self._lock:
            for line in str(message).splitlines():
                if not line.strip():
                    continue
                self.messages.append(line.rstrip())
                count = _COUNT.match(line)
                if count:
                    self.done, self.total = int(count.group(1)), int(count.group(2))
                elif _TOTAL.search(line):
                    self.done, self.total = 0, int(_TOTAL.search(line).group(1))
            extra = len(self.messages) - MAX_MESSAGES
            if extra > 0:
                del self.messages[:extra]
                self.dropped += extra

    def snapshot(self, since=0):
        """Where the job is, as plain data. 'messages' = the progress lines from number `since` on; ask again
        with since = 'next' for only the new ones."""
        with self._lock:
            start = max(int(since), self.dropped)
            end = self.finished or time.time()
            return {'id': self.id, 'state': self.state, 'target': self.target,
                    'name': (self.summary or {}).get('target'), 'n_frames': (self.summary or {}).get('n_frames'),
                    'saves': self.settings.get('SAVE_RESULTS', True) is not False,
                    'created': self.created, 'started': self.started, 'finished': self.finished,
                    'elapsed': round(end - (self.started or self.created), 1),
                    'progress': {'done': self.done, 'total': self.total},
                    'messages': self.messages[start - self.dropped:], 'next': self.dropped + len(self.messages),
                    'error': self.error}

    def image(self, index, pretty=True, wide=False):
        """Frame T(index+1) as PNG bytes (api.frame_png). The images of one job are made one at a time - it is
        numpy work, several at once would only slow each other down - except the wide view, which mostly waits
        for a download."""
        if wide:
            return api.frame_png(self.raw, index, pretty=pretty, wide=True)
        with self._render:
            return api.frame_png(self.raw, index, pretty=pretty)


class JobManager:
    """The queue of analyses and the finished ones still in memory.
    analyze: what runs one (api.analyze; tests pass their own: analyze(target, settings=, on_progress=,
    return_raw=True) -> (summary, raw))."""

    def __init__(self, analyze=None, keep_finished=KEEP_FINISHED, max_waiting=MAX_WAITING):
        self._analyze = analyze or api.analyze
        self.keep_finished, self.max_waiting = keep_finished, max_waiting
        self._jobs = OrderedDict()                      # id -> Job, oldest first
        self._waiting = queue.Queue()
        self._lock = threading.RLock()
        self._worker = None

    @staticmethod
    def _target_key(target):
        try:
            return api.target_key(target)
        except Exception:                               # not a target the pipeline knows: itself
            return json.dumps(target, sort_keys=True, default=str)

    def _key(self, target, settings):
        return json.dumps([self._target_key(target), target.get('max_images'), sorted((settings or {}).items())],
                          default=str)

    def submit(self, target, settings=None, force=False):
        """Queue an analysis of `target` (api.analyze's target; `settings` = config values for it only) and
        return its Job - or, unless force=True, the job already waiting, running or finished for the same
        request. Raises QueueFull when MAX_WAITING analyses are waiting."""
        key = self._key(target, settings)
        with self._lock:
            if not force:
                for job in reversed(self._jobs.values()):
                    if job.key == key and job.state != 'error':
                        return job
            if sum(job.state == 'queued' for job in self._jobs.values()) >= self.max_waiting:
                raise QueueFull(f'{self.max_waiting} analyses are waiting already - try again when one has finished')
            job = Job(target, settings, key)
            self._jobs[job.id] = job
            self._waiting.put(job)
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._work, name='analysis-jobs', daemon=True)
                self._worker.start()
            return job

    def get(self, job_id):
        with self._lock:
            return self._jobs.get(job_id)

    def latest(self, target):
        """The newest job for this target that didn't fail (however many images it was limited to), or None."""
        key = self._target_key(target)
        with self._lock:
            for job in reversed(self._jobs.values()):
                if job.state != 'error' and self._target_key(job.target) == key:
                    return job
        return None

    def status(self):
        with self._lock:
            jobs = list(self._jobs.values())
        running = next((job for job in jobs if job.state == 'running'), None)
        return {'busy': running is not None, 'running': running.id if running else None,
                'waiting': sum(job.state == 'queued' for job in jobs),
                'finished': [job.id for job in jobs if job.state == 'done']}

    def _work(self):
        while True:
            self._run(self._waiting.get())

    def _run(self, job):
        with job._lock:
            job.state, job.started = 'running', time.time()
        try:
            summary, raw = self._analyze(job.target, settings=job.settings, on_progress=job.log, return_raw=True)
            try:
                summary['band_spectrum'] = api.band_spectrum(raw['frames'])
            except Exception as e:                      # an extra: the analysis itself is fine without it
                log.info(f'(102-channel spectrum not available: {type(e).__name__}: {e})')
            with job._lock:
                job.summary, job.raw, job.state = summary, raw, 'done'
        except Exception as e:                          # the job failed; the queue goes on
            log.info('That analysis stopped:\n' + traceback.format_exc())
            with job._lock:
                job.error, job.state = f'{type(e).__name__}: {e}', 'error'
        finally:
            with job._lock:
                job.finished = time.time()
            self._forget_old()

    def _forget_old(self):
        """Keep only the newest keep_finished finished jobs (the others' pixels are released)."""
        with self._lock:
            finished = [job for job in self._jobs.values() if job.state in ('done', 'error')]
            for job in finished[:max(len(finished) - self.keep_finished, 0)]:
                del self._jobs[job.id]
