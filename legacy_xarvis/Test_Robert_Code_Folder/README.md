# XARVIS

XARVIS is a Python + Flask web app that visualizes the solar system, the orbit of NASA's SPHEREx infrared survey telescope, and the sky it can survey. It is designed for astronomy exploration and data browsing, with official NASA, ESA, and IPAC data sources, and a browser-side 3D viewer built with HTML canvas and JavaScript.

This project is a science-driven visualization tool for understanding:
- where the planets and Moon are in time,
- where SPHEREx is and what parts of the sky it can view,
- what known asteroids/comets are in a patch of sky,
- how proper motion and orbital geometry affect star and object positions,
- which stars and constellations are visible in a viewer of the sky.

The app is not a general-purpose astronomy engine; it is a viewer and data browser focused on the SPHEREx mission and nearby moving objects.

## Quick start

```bash
cd /Users/nicko/Desktop/Xarvis/Test_Robert_Code_Folder
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open:
- http://127.0.0.1:5000/
- http://127.0.0.1:5000/about

The first run may trigger a background download of the DE440 ephemeris and related astronomy data. The server also serves static data bundles used by the browser.

## Repository layout

```text
.
├── app.py
├── TEST_Robert_code.py
├── requirements.txt
├── dwarf_cache.json
├── static/
│   ├── css/
│   │   └── viewer.css
│   ├── js/
│   │   ├── analysis.js
│   │   ├── app.js
│   │   ├── core.js
│   │   ├── draw.js
│   │   ├── scene.js
│   │   ├── tracker.js
│   │   └── view3d.js
│   ├── site.css
│   ├── sky.json
│   ├── small_bodies.json
│   └── targets.json
├── templates/
│   ├── 404.html
│   ├── about.html
│   └── index.html
├── tools/
│   ├── build_small_bodies.py
│   ├── build_sky.py
│   ├── build_targets.py
│   └── verify_motion.py
└── README.md
```

## File-by-file overview

### `app.py`
This is the main backend. It is a Flask application that:
- serves the main viewer page and about page,
- provides JSON API endpoints for astronomy data,
- downloads/loads NASA/JPL ephemerides and orbital state data,
- converts between coordinate frames and coordinate systems,
- queries JPL Horizons, SBDB, and SB Ident APIs,
- computes SPHEREx visibility geometry and solar-system state.

Key parts of `app.py`:
- `app = Flask(__name__)` creates the server.
- `load_ephemeris()` loads NASA/JPL DE440s via Astropy when available.
- `utc(jd)` converts Julian Date to an Astropy time object.
- `eq_to_ecl()` rotates equatorial coordinates into the ecliptic frame.
- `close_loop()` smooths orbit paths so they close without visible seam drift.
- `horizons_spherex()` and `horizons_spherex_track()` fetch real SPHEREx positions from JPL Horizons when available.
- `sso_snapshot()` builds a fallback sun-synchronous orbit model using official SPHEREx orbital parameters.
- `/api/meta`, `/api/track`, `/api/orbits`, `/api/spherex`, `/api/sbdb/lookup`, and `/api/sbident` are the main API endpoints used by the browser.
- `main()` launches the Flask app in debug mode with the ephemeris loader running in a background thread.

This is the heart of the project.

### `TEST_Robert_code.py`
This is a standalone script used to test and inspect SPHEREx spectral-image access via IRSA. It is not used by the web app itself, but it demonstrates how to:
- query the IRSA archive,
- locate SPHEREx spectral-image FITS files,
- download a FITS file,
- inspect FITS headers and WCS metadata,
- optionally open Firefly viewer.

This is a good reference if you want to understand how SPHEREx spectral image data can be processed after the viewer identifies a patch of sky.

### `requirements.txt`
Lists the Python dependencies:
- Flask
- NumPy
- Astropy
- jplephem
- requests

This is the bare minimum to run the viewer and access the ephemeris logic.

### `templates/index.html`
This is the main HTML shell for the app. It defines:
- the canvas for the 3D viewer,
- the side panels, time controls, zoom controls, and object tracker UI,
- the startup intro panel,
- the analysis popup shell for sky-point inspection,
- the JavaScript files loaded in order.

It is not the logic itself; it is the structure that the browser scripts draw into.

### `templates/about.html`
A static about page describing the purpose of the project and the mission context. It explains the project as a NASA Space Apps Challenge effort.

### `templates/404.html`
A custom 404 page. It is used when a user hits a route that is not found. API endpoints return JSON errors instead, while normal browser requests show the page.

### `static/css/viewer.css`
Styles for the viewer layout and interface. This file shapes the appearance of the main page: cards, dock panels, text, toolbars, buttons, and intro overlays.

### `static/site.css`
General CSS for the non-viewer pages such as the about page.

### `static/js/core.js`
The shared math and time foundation for the browser app. It contains:
- canvas sizing and screen helpers,
- the simulation clock and playback controls,
- coordinate conversions,
- 3D view matrices,
- orbital formulas for JPL mean elements,
- utility functions for formatting RA/Dec and Julian dates.

This file is the mathematical backbone for the rest of the client scripts.

### `static/js/scene.js`
This is the orchestration layer between the backend and the drawing system. It:
- fetches `/api/track`, `/api/spherex`, and `/api/orbits`,
- prepares `state`, which is the data structure consumed by the renderer,
- tracks whether data is up to date,
- models star motion and proper motion,
- computes orbit rings and Moon ring paths.

This is where the browser turns backend JSON into a 3D frame description.

### `static/js/draw.js`
This file contains a large number of rendering helpers:
- canvas resizing/projection,
- stereographic sky projection,
- clipping for paths,
- hover hit testing,
- star painting,
- constellation labels,
- asteroid/Kuiper-belt rendering,
- tooltip generation.

This is the core of the visual renderer.

### `static/js/tracker.js`
This handles the object tracker panel and tracking logic. It defines:
- how catalog entries are created,
- how known objects are filtered,
- how orbit paths are drawn for tracked items,
- how JPL search results are merged into the catalog,
- a hook `window.XARVIS_addCandidates(rows)` for unknown objects from the backend.

This is the user tool for exploring planets, Moon, asteroids, trojans, and Kuiper-belt objects.

### `static/js/view3d.js`
This file addresses visibility geometry and 3D rendering of the sky. It computes:
- SPHEREx Sun-angle constraints,
- Earth-occlusion geometry,
- sky fractions visible now or overall,
- the actual 3D scene and cursor readout.

It is the part that answers: “What parts of the sky can SPHEREx see right now?”

### `static/js/analysis.js`
This is the sky-point analysis workflow. It handles:
- picking a point in the sky,
- converting that point into coordinates,
- deciding visibility and data availability,
- displaying patch information to the user,
- preparing payloads that the backend could use to run deeper analysis.

It is a “place-holder and hook” area for expanding the analysis pipeline.

### `static/js/app.js`
This is the main browser control loop. It contains:
- render loop (`requestAnimationFrame`),
- time progression logic,
- UI updates,
- keyboard and mouse handlers,
- the intro popup,
- object tracking interactions,
- startup boot logic.

This file is the browser equivalent of the application shell.

### `static/sky.json`
A packaged star catalogue generated by the build scripts. It includes:
- star positions,
- magnitudes and colors,
- proper motions,
- constellation figure definitions,
- star names and label anchors.

It is the data that drives the background star field and label system.

### `static/small_bodies.json`
A compressed catalog of real asteroid and small-body orbits. It is built from NASA/JPL SBDB data and used by the browser to draw representative known small bodies such as:
- NEOs,
- main-belt asteroids,
- Jupiter Trojans,
- Kuiper-belt objects.

Each row contains the orbital elements needed for two-body motion and a name or designation.

### `static/targets.json`
A curated list of challenge or demonstration objects such as Barnard’s Star or Gaia target stars. These are used to test or illustrate coordinate selections and patch analysis in the viewer.

### `tools/build_sky.py`
Builds `static/sky.json`. It fetches star data from ESA or a fallback star catalog, creates proper-motion-aware coordinates, and emits region/constellation data. It also validates the output against independent constellation boundaries.

### `tools/build_small_bodies.py`
Builds `static/small_bodies.json` from NASA/JPL SBDB. It keeps only representative brighter objects per class so the browser is fast and the JSON remains manageable.

### `tools/build_targets.py`
Builds `static/targets.json` by querying Gaia and adding any challenge targets that are not in Gaia.

### `tools/verify_motion.py`
This script validates the viewer against independent astronomy data: eclipses, transits, equinoxes/solstices, orbital periods, Moon phases, and proper motion checks. It is essentially a truth-check framework for the orbital calculations.

### `dwarf_cache.json`
A legacy cache file containing older precomputed positions. It appears to be a leftover artifact and is not central to the running app. It may be ignored unless you are debugging older generated data.

## How the app works

The project has a simple architecture:

1. The Flask backend (`app.py`) loads astronomy data and exposes JSON endpoints.
2. The browser loads the page and JavaScript assets.
3. `core.js` and `scene.js` fetch the astronomy state and convert it into a scene representation.
4. `draw.js` draws the stars, planets, Moon, SPHEREx orbit, and overlays.
5. `tracker.js` and `view3d.js` handle interactivity, tracking, and field-of-view calculations.
6. `app.js` runs the render loop and keeps the UI in sync with time.

This design makes it easy to separate:
- scientific calculations,
- data access,
- rendering,
- interface logic.

## Basic astronomy concepts you need to know

### 1. Julian Date and J2000
The app uses Julian Dates for time and J2000 as the reference epoch.
- Julian Date is a continuous count of days from a fixed origin.
- J2000 = 2451545.0 is the standard epoch used for many astronomy calculations.
- Positions are often described in the J2000 frame, so they are consistent across the app.

This is why many functions in the code convert to/from JD and use `J2000` as a reference.

### 2. Equatorial vs ecliptic coordinates
Astronomy commonly uses two key sky planes:
- Equatorial coordinates: RA/Dec, tied to the Earth’s rotation and equator.
- Ecliptic coordinates: longitude/latitude relative to the Earth–Sun plane.

The app frequently converts between these systems using the obliquity of the ecliptic (`OBLIQ` in `app.py` and `EPS` in JS).

This matters because:
- the viewer looks at the solar system in the ecliptic frame,
- stellar catalog coordinates are usually stored equatorially,
- SPHEREx mission and solar-system geometry are easier to reason about in the ecliptic plane.

### 3. Orbital elements
For planets and small bodies, the code uses Keplerian or osculating orbital elements such as:
- semi-major axis `a`
- eccentricity `e`
- inclination `i`
- longitude of ascending node `Ω`
- argument of perihelion `ω`
- mean anomaly `M`
- epoch

These elements describe the shape and orientation of an orbit. The browser and backend use them to predict where objects are at a given instant, and to draw orbit paths.

### 4. Proper motion
Stars are not fixed. Their positions slowly drift because of:
- proper motion in RA and Dec,
- small parallax effects,
- differences between observation epoch and reference frame.

The star catalog (`static/sky.json`) stores positions at J2000 and proper-motion vectors, then the JavaScript updates positions on the fly using those velocities.

### 5. Visibility geometry
SPHEREx can only point away from the Sun by a sufficient angle; the project uses the official rule:
- Sun limit: at least 91° from the Sun.

The app also excludes directions behind Earth, because Earth blocks the line of sight. The visible region is therefore a geometrical field-of-regard in the sky rather than a simple hemisphere.

### 6. Small-body and asteroid motion
The app keeps a database of known asteroid and comet orbits from JPL. These objects move relative to the Sun and Earth. For the browser, the code uses the stored orbital elements and advances them forward in time with two-body motion, which is a simplified but useful approximation.

This is why the viewer can show main-belt asteroids, NEOs, Trojans, and Kuiper-belt objects without requiring a heavy full-n-body simulation.

## Data sources

The project is deliberately built around official data sources:

- Planets and Moon: NASA/JPL DE440s ephemerides
- SPHEREx orbit: NASA/JPL Horizons
- Small bodies: NASA/JPL SBDB and SB Ident
- Stars: ESA Hipparcos / Gaia
- Constellation boundaries and names: Stellarium sky culture data (not an agency data source but a widely used astronomical reference)

The README, app text, and UI all label these data sources so users know what is official and what is interpretive.

## Rebuilding the generated data files

If you want to refresh the static JSON data files:

```bash
python tools/build_sky.py --source esa --stellarium index.json --out static/sky.json --check
python tools/build_small_bodies.py --out static/small_bodies.json
python tools/build_targets.py --out static/targets.json
```

Note: these scripts require internet access and may take some time.

## Validation and checks

The project includes a motion-validation script:

```bash
python tools/verify_motion.py
```

This checks that the viewer’s calculations match astrophysical reality in a number of ways, including:
- Moon and orbital events,
- planetary motion,
- eclipses and transits,
- SPHEREx orbit geometry,
- star motion consistency.

## Known limitations

This project is intentionally practical, not perfect. Some limitations include:
- star motion is based on proper motion; it does not include radial velocity effects,
- the SPHEREx orbit is drawn as a circular approximation when exact real-time orbital data are unavailable,
- small-body orbits are approximated using osculating elements and two-body motion,
- the Moon and flight-direction avoidance angles are not drawn because the mission publication does not provide them explicitly,
- the whole project is centered around SPHEREx mission dates and mission geometry rather than all astronomy use cases.

## Conclusion

XARVIS is a browser-based astronomy visualization tool that turns public ephemerides and mission geometry into an interactive, visually explorable system. The code is organized around a central Flask API and a large JavaScript rendering engine. The project is best understood by combining:
- backend orbital math,
- star and small-body catalogs,
- mission geometry,
- browser rendering logic.

If you are reading this repository for the first time, start at:
1. `app.py`
2. `templates/index.html`
3. `static/js/core.js`
4. `static/js/scene.js`
5. `static/js/draw.js`
6. `static/js/app.js`

From there, follow the data sources and build scripts to see how the astronomy data are materialized.
