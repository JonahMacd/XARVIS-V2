# XARVIS sky viewer (test build)

Run:  `python app.py`  then open http://127.0.0.1:5000

Python packages: `flask  numpy  astropy  jplephem  requests`  (`pip install flask numpy astropy jplephem requests`).
The first start downloads NASA/JPL's DE440s ephemeris (32 MB, cached by astropy) from naif.jpl.nasa.gov.

## What it shows
* A **landing panel** explains the controls (tick "Don't show this again" to skip it; the **Controls** link brings it back).
* An **About** page (`/about`, `templates/about.html`) explains the challenge, features, data sources, accuracy checks and design decisions; unknown pages get a 404 page (`templates/404.html`). Both share `static/site.css`.
* Above the bottom-left panel: links to About, Controls and each dataset's official access site.
* The **solar system** in 3D (drag to rotate, scroll to zoom) with SPHEREx's real orbit around Earth, the planets, the Moon,
  real asteroids/Trojans/Kuiper-belt objects and the stars behind them. A plain sentence says where SPHEREx is right now.
* A **time machine** in the bottom bar: play forward/backward from real time up to 1 year per second, or type any UTC date.
* **Advanced orbital parameters** (top-left, collapsed by default): semi-major axis, eccentricity, inclination, RAAN,
  argument of latitude, period, speed, beta angle, ascending-node local time, heliocentric position, data sources.
* **Double-click** anywhere on the sky to open the analysis popup. It packages the sky direction (geocentric J2000 RA/Dec,
  ecliptic, galactic, angle from the Sun) plus time and SPHEREx's position. It is a PLACEHOLDER: the backend call is the
  `requestAnalysis()` stub in `templates/index.html` (search for `TODO(backend)`); nothing is sent yet.
* **What SPHEREx can see** (Layers, on by default), painted on the star backdrop:
  *Visible now* = at least 91 deg from the Sun (NASA/JPL Horizons) and not behind Earth's disc from SPHEREx's real position;
  *Visible overall* = the Sun limit alone (Earth moves out of the way every orbit). Advanced shows the solid angles
  (sr and % of sky) and one exposure's field of view, 3.5 x 11.3 deg (IPAC Explanatory Supplement). Moon and
  flight-direction avoidance are not drawn because their angles are not published.
* **Go to sky coordinates** (top right): type RA/Dec (sexagesimal or decimal degrees, J2000) or pick a challenge test target
  (Barnard's Star, Epsilon Indi B, WISE 0855-0714, RW Aurigae, R Draconis). The patch size is an Advanced setting in SPHEREx pixels (default 50 x 50 px at 6.15 arcsec = 5.1 arcmin). The view turns to the patch, a
  marker shows it, and the analysis popup opens (same as double-clicking the sky). The popup shows:
  * the target from **ESA Gaia DR3** (`static/targets.json`, built by `tools/build_targets.py`), moved with its proper motion
    to the date shown, and how far it moves during the SPHEREx survey in SPHEREx pixels (6.15 arcsec);
  * the known asteroids and comets inside the patch on that date from **NASA/JPL SB Ident** (`/api/sbident`, first pass,
    about 1 arcmin), with their motion;
  * a map of the patch (north up, east left) with both, and a placeholder where the SPHEREx image time-lapse will play once
    the IRSA image backend is connected (`requestAnalysis()`, search `TODO(backend)`).
* **SPHEREx data dates**: time is limited to the span SPHEREx has data for (1 May 2025 to the latest observation, shown in the time panel); the "Jump to..." menu jumps to the first science data (1 May 2025) or the latest data. The analysis popup only shows the image box when the point is in SPHEREx's view and the date is inside that span (IRSA releases images within 60 days).
* **Track objects** (top right): search, then two filters: Show (known object classes, or unknown SPHEREx candidates) and Only (visible to SPHEREx now, near Earth, hazardous, over 100 km, measured size). Search and filter every object the viewer knows (planets, Moon, SPHEREx, ~12,700 JPL small
  bodies: near-Earth objects, main belt, Jupiter Trojans, Kuiper belt) by type, current location, size, known/unknown
  (numbered or provisional orbit, measured or estimated size, potentially hazardous, SPHEREx candidates) and whether SPHEREx
  can see it now. Anything else can be looked up live in NASA/JPL's SBDB (`/api/sbdb/lookup`); its orbit is then taken from
  NASA/JPL Horizons at today's date. Tracked objects are highlighted with their orbit; "Centre" follows one.
  SPHEREx candidates: placeholder. The backend calls `XARVIS_addCandidates(rows)` (search `TODO(backend)`).
* The bottom-left panel shows the position under the cursor (AU on the ecliptic plane, distance from the Sun, sky position).

## Data sources
| What | Source | Official? |
|---|---|---|
| Stars: positions, proper motions | ESA Hipparcos new reduction, from ESA's archive (`public.hipparcos_newreduction`) -> `static/sky.json` | yes (ESA) |
| Planets and Moon | NASA/JPL **DE440s** ephemeris (NAIF), 1849-12-26 .. 2150-01-22. Outside that range: JPL's published mean orbital elements (about 0.2 deg) | yes (NASA/JPL) |
| SPHEREx trajectory | NASA/JPL **Horizons**, spacecraft -163182 (built from US Space Force Space-Track TLEs): 2025-03-11 .. now + 30 days; positions every 2 minutes, interpolated (matches Horizons to ~2 arcsec in direction). Otherwise a sun-synchronous model from the official orbital parameters (position along the orbit unknown) | yes (NASA/JPL) |
| SPHEREx facts (launch 12 Mar 2025 03:10:12 UTC, science from 1 May 2025, 25-month mission, orbit table, 91 deg Sun limit) | IPAC SPHEREx Explanatory Supplement v2.1 and the Horizons entry | yes (IPAC / NASA/JPL) |
| Asteroids, near-Earth objects, Jupiter Trojans, Kuiper belt | NASA/JPL **Small-Body Database** -> `static/small_bodies.json` (real orbits, names, H, measured diameters; brightest objects of each class; two-body motion from each object's elements). Live lookups: SBDB API + Horizons osculating elements | yes (NASA/JPL) |
| Test targets (Barnard's Star, Epsilon Indi B, RW Aur, R Dra) | **ESA Gaia DR3** (gea.esac.esa.int) -> `static/targets.json`; WISE 0855-0714 from the challenge text (not in Gaia) | yes (ESA / NASA) |
| Asteroids and comets inside a patch | NASA/JPL **SB Ident** API (live, via `/api/sbident`) | yes (NASA/JPL) |
| Constellation figures, borders, labels, star names | Stellarium "modern" sky culture (CC BY-SA 4.0), built on the IAU borders | **no**: open source, not an agency. The IAU defines the 88 constellations and borders but publishes no stick figures |

Removed on purpose: dwarf planets (extra dependency), CelesTrak (not an agency), SGP4, and the SPHEREx Deep Fields (the
official documents give no size or exact centre, and IRSA's image service could not be queried to measure them).

## Rebuilding the data files (not needed to run the app)
* Stars:         `python tools/build_sky.py --source esa --stellarium index.json --out static/sky.json --check`
  (`index.json` = Stellarium's `skycultures/modern/index.json`; needs access to gea.esac.esa.int)
* Small bodies:  `python tools/build_small_bodies.py --out static/small_bodies.json`  (needs ssd-api.jpl.nasa.gov)

## Accuracy: `python tools/verify_motion.py`
Checks the motion against real events and independent code: transits of Venus (2012) and Mercury (2019), solar and lunar
eclipses, equinoxes/solstices, perihelion/aphelion, orbital periods, the Moon's months, astropy's own star-motion code,
the real SPHEREx orbit against the official parameters, and the small-body motion against JPL Horizons.

Known limits: star motion is proper motion only (Hipparcos has no radial velocity; exact to <0.1" within +-10,000 yr, a few
nearby stars drift beyond that); planets cannot be predicted beyond about +-5,000 years; small bodies drift from their
element epoch (about 0.001 AU after a year, 0.01 AU after five); SPHEREx's eccentricity is not measured (the orbit is drawn as a circle).
