"""Start the XARVIS website with the SPHEREx image pipeline connected:

    python run_web.py            then open http://127.0.0.1:5000

Works from any folder: the paths are taken from this file's location. The website is website/app.py;
the pipeline (xarvis_pipeline) is imported as a package from this folder. The desktop app is unchanged:
python -m xarvis_pipeline
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLBACKEND", "Agg")                # the web server never opens matplotlib windows
for folder in (ROOT, ROOT / "website"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from app import main                                       # noqa: E402  (website/app.py)

if __name__ == "__main__":
    main()
