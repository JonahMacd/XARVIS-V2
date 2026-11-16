# XARVIS V2

**eXplorable Astronomical Record Viewer for Infrared Surveys.** A website that lets anyone explore NASA's SPHEREx
all-sky infrared survey: pick a point on the sky, and XARVIS fetches every SPHEREx image of it, plays them as a
time-lapse, names every source from the catalogs, flags what no catalog explains, and lets you track any object.

This repository continues the project after the NASA Space Apps Challenge 2026. The hackathon-era code is kept in
[`legacy_xarvis/`](legacy_xarvis/README.md); the original repository is https://github.com/nfortier7972/Xarvis.git.

## Run it

```
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell   (macOS / Linux: source .venv/bin/activate)
pip install -r requirements.txt
python run_web.py
```

Then open http://127.0.0.1:5000. Double-click the sky, or use "Go to sky coordinates".

## Where things are

| Folder / file | What it is |
|---|---|
| `run_web.py` | Starts the website. The one command you need |
| `website/` | The site: `app.py` (Flask server and routes), `analysis_jobs.py` (background analysis jobs), `templates/` (pages), `static/` (JavaScript, CSS, data files) |
| `xarvis_pipeline/` | The SPHEREx analysis, a Python package: download and measure images, compare with catalogs, map the sky, track objects. `web.py` is the website's way in; `python -m xarvis_pipeline` runs the desktop app |
| `config.json` | Pipeline settings |
| `tests/` | Every test, for both parts |
| `tools/` | Scripts that build the site's data files, and `verify_motion.py` (45 accuracy checks) |
| `docs/` | Audience definition, website notes, presentation, screenshots |
| `legacy_xarvis/` | The hackathon-era folder layout, kept for reference. Not used by anything above |
| `results/` | Created when you run an analysis (not in git) |

## Test it

```
python -m unittest discover -s tests -t .
python tools/verify_motion.py
```

The first runs every offline test (no internet needed). The second needs the internet. To also run one real analysis
against NASA/IPAC IRSA, set `XARVIS_ONLINE=1` first (PowerShell: `$env:XARVIS_ONLINE=1`).

## Working on it

- Turn on the commit check once per clone: `git config core.hooksPath .githooks`. It refuses commits with
  merge-conflict markers, Python that doesn't compile, or invalid JSON.
- Line endings are LF everywhere (`.gitattributes`), so Windows and Mac don't produce whole-file diffs.
- More detail: [docs/website.md](docs/website.md) (site files, data sources, accuracy checks) and
  [docs/AUDIENCE.md](docs/AUDIENCE.md) (who XARVIS is for).

## Team

Robert Laes, William Wright, Jonah MacDonald, Nickolas Fortier.
