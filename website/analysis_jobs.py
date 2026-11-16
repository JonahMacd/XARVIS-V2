"""Background jobs for the SPHEREx image time-lapse (the bridge between this website and xarvis_pipeline).

A run takes seconds to minutes (IRSA downloads), far longer than one web request should, so it runs on a worker thread
and the browser polls for progress. ONE analysis at a time: the pipeline's settings are module-wide, and the page
makes it clear that another point can only be analysed once the current run has finished or been stopped.
Each job writes into <results>/jobs/<id>/ (summary.json + frame_<n>.png); the id is a hash of the request, so the
same request finds its finished results again - also after a server restart.

    submit(request)  -> job id                 (starts the run, or reuses a running / finished one; raises Busy while
                                                 ANOTHER analysis is running)
    status(job_id)   -> dict, or None           {'status': queued|running|done|error|cancelled, 'done', 'total',
                                                 'message', 'summary' (when done), 'error' (when failed)}
    active()         -> the running analysis    ({'job', 'status', 'done', 'total', 'message', 'request'}) or None
    stop()           -> stops the running one   (its partial files are removed)
    reset()          -> stops every run and deletes every saved analysis (the page's "Stop & reset" button, and the
                        first page load of a new browser session)
    folder(job_id)   -> where its files are

The pipeline is imported on first use, in the worker - the viewer itself starts without it (and still works if the
pipeline's packages are missing: the popup then shows the error).
"""
import hashlib
import json
import os
import shutil
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent              # the Xarvis folder: xarvis_pipeline is imported from here
os.environ.setdefault("MPLBACKEND", "Agg")                # before anything imports matplotlib: never open a window

_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="spherex-analysis")
_jobs = {}
_lock = threading.Lock()
RUNNING = ("queued", "running", "stopping")


class Busy(Exception):
    """Another analysis is running: only one at a time. `job` is its state (as active() gives it)."""

    def __init__(self, job):
        super().__init__("Another SPHEREx analysis is still running. Wait for it to finish, or stop it, "
                         "before starting a new one.")
        self.job = job


def pipeline():
    """The pipeline's web module (xarvis_pipeline.web), wherever the server was started from."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from xarvis_pipeline import web
    return web


def job_id(request):
    return hashlib.sha1(json.dumps(request, sort_keys=True).encode()).hexdigest()[:16]


def folder(jid):
    return pipeline().results_folder() / jid


def _saved_summary(jid):
    try:
        return json.loads((folder(jid) / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _public(jid, job):
    """A job's state as the browser gets it (without the internal stop flag)."""
    return {"job": jid, **{k: v for k, v in job.items() if k != "cancel"}}


def status(jid):
    with _lock:
        job = {k: v for k, v in _jobs[jid].items() if k != "cancel"} if jid in _jobs else None
    if job is not None and job["status"] in RUNNING + ("error", "cancelled"):
        return job
    summary = _saved_summary(jid)
    if summary is None:
        return None if job is None else {**job, "status": "error", "error": "The saved results are missing."}
    return {**(job or {"done": summary["n_frames"], "total": summary["n_frames"], "message": "Saved results"}),
            "status": "done", "summary": summary}


def active():
    """The analysis that is queued or running right now (with its request, so the page can reopen it), or None."""
    with _lock:
        for jid, job in _jobs.items():
            if job["status"] in RUNNING:
                return _public(jid, job)
    return None


def finished():
    """Every finished analysis (newest first), as the viewer draws it on the sky: where it is, what it was, which
    SPHEREx images it holds and the request that made it (to open it again). A tracked object that moves (an asteroid,
    a NEW mover) has the path its images followed; anything else one centre."""
    try:
        folders = [f for f in pipeline().results_folder().iterdir() if JOB_FOLDER(f.name)]
    except OSError:
        return []
    out = []
    for path in folders:
        try:
            summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
            frames, request, target = summary["frames"], summary["request"], summary["target"]
            points = [[f["expected_ra"], f["expected_dec"]] for f in frames
                      if f.get("expected_ra") is not None and f.get("expected_dec") is not None]
            if not points:                                  # (saved before positions were kept: the patch centre)
                points = [[target["ra_deg"], target["dec_deg"]]]
            moving = bool(target.get("moving"))
            ra, dec = points[len(points) // 2] if moving else points[0]
            database = summary.get("database") or {}
            out.append({
                "job": path.name, "name": target.get("name") or "analysis", "tracking": summary.get("tracking"),
                "moving": moving, "ra_deg": ra, "dec_deg": dec, "path": points if moving else None,
                "patch_px": summary["patch_px"], "pixel_arcsec": summary["pixel_arcsec"],
                "n_frames": summary["n_frames"], "images_available": summary.get("images_available"),
                "first_utc": frames[0]["time_utc"] if frames else None, "last_utc": frames[-1]["time_utc"] if frames else None,
                "new_objects": len(database.get("new_objects") or []), "request": request,
                "saved": (path / "summary.json").stat().st_mtime})
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            continue                                        # an unreadable folder is skipped, not fatal
    return sorted(out, key=lambda a: -a["saved"])


def JOB_FOLDER(name):
    return len(name) == 16 and all(c in "0123456789abcdef" for c in name)


def friendly(error):
    """One sentence for the popup from whatever stopped the run."""
    text = " ".join(str(error).split())[:600] or type(error).__name__
    if isinstance(error, ImportError):
        return f"The analysis pipeline could not be loaded on the server ({text}). Check that requirements.txt is installed."
    if "attempts failed" in text or "HTTP 5" in text or "unreachable" in text:
        return f"NASA/IPAC IRSA did not answer ({text}). The archive is often briefly overloaded: try again in a minute."
    return text


def _remove(jid):
    """Delete a job's folder (a stopped run's partial files, or a saved analysis being reset)."""
    try:
        shutil.rmtree(folder(jid), ignore_errors=True)
    except Exception:                                       # (the pipeline can't be loaded: nothing was written)
        pass


class _Stopped(Exception):
    """Stand-in for xarvis_pipeline.api.Cancelled when the pipeline can't be imported (only test runs reach it)."""


def _cancelled_error():
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from xarvis_pipeline.api import Cancelled
        return Cancelled
    except Exception:
        return _Stopped


def submit(request, run=None):
    """Start the job for `request` (see xarvis_pipeline.web.web_request) unless it is already running or finished.
    Raises Busy while a DIFFERENT analysis is queued or running. run(request, folder, on_progress): the work - the
    pipeline's analyse by default (tests pass a stand-in)."""
    jid = job_id(request)
    with _lock:
        known = _jobs.get(jid)
        if known and known["status"] in ("queued", "running"):
            return jid
        for other, job in _jobs.items():
            if job["status"] in RUNNING:
                raise Busy(_public(other, job))
        if _saved_summary(jid) is not None:
            return jid
        cancel = threading.Event()
        entry = _jobs[jid] = {"status": "queued", "done": 0, "total": 0, "message": "Starting...",
                              "request": request, "cancel": cancel}     # (updated through `entry`: never looked up again)

    def check_cancelled():
        if cancel.is_set():
            raise _cancelled_error()()

    def progress(done, total, message):
        check_cancelled()
        with _lock:
            entry.update(status="running", done=int(done), total=int(total), message=str(message))

    progress.check_cancelled = check_cancelled              # web.analyse calls it at every step it can stop at

    def work():
        try:
            progress(0, 0, "Starting...")
            (run or pipeline().analyse)(request, folder(jid), progress)
            check_cancelled()                               # stopped while the images were being written
            with _lock:
                entry.update(status="done", message="Done")
        except BaseException as e:                         # every failure reaches the browser as a message
            if cancel.is_set():
                _remove(jid)
                with _lock:
                    entry.update(status="cancelled", message="Stopped", error="The analysis was stopped.")
                return
            traceback.print_exc()
            with _lock:
                entry.update(status="error", error=friendly(e))

    _pool.submit(work)
    return jid


def stop():
    """Stop the running analysis (it ends at its next step: within a second or two while images download). Returns
    its job id, or None if nothing was running."""
    with _lock:
        for jid, job in _jobs.items():
            if job["status"] in ("queued", "running"):
                job["cancel"].set()
                job.update(status="stopping", message="Stopping...")
                return jid
    return None


def reset(stop_running=True):
    """Delete every saved analysis; with stop_running (a new browser session) also stop the one running, otherwise
    (the page's Reset button: Stop is a button of its own) leave it running. Returns {'stopped': n, 'deleted': n}."""
    stopped, keep = 0, set()
    with _lock:
        for jid, job in _jobs.items():
            if job["status"] in ("queued", "running"):
                if not stop_running:
                    keep.add(jid)
                    continue
                job["cancel"].set()
                job.update(status="stopping", message="Stopping...")
                stopped += 1
            elif job["status"] == "stopping" and not stop_running:
                keep.add(jid)                               # (it removes its own files when it ends)
        for jid in [jid for jid, job in _jobs.items() if job["status"] not in RUNNING]:
            del _jobs[jid]                                  # finished / failed / stopped: forgotten
    deleted = 0
    try:
        folders = [f for f in pipeline().results_folder().iterdir()
                   if f.is_dir() and JOB_FOLDER(f.name) and f.name not in keep]
    except Exception:                                       # no results folder yet (or no pipeline): nothing saved
        folders = []
    for path in folders:
        shutil.rmtree(path, ignore_errors=True)
        deleted += 1
    return {"stopped": stopped, "deleted": deleted}
