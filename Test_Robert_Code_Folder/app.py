"""XARVIS proof of concept: live solar-system model with SPHEREx's position.

Run:  python app.py   then open http://127.0.0.1:5000

Frame for everything sent to the browser: heliocentric ecliptic J2000, AU.

Planets, Sun, Moon
    astropy's ephemeris (default 'builtin': ERFA epv00/plan94/moon98).
    Planet *orbit paths* are traced from ephemeris positions over one full period,
    not drawn as ideal ellipses. Set XARVIS_EPHEMERIS=de440s (needs `pip install jplephem`
    and a one-off ~32 MB kernel download) for JPL DE440 accuracy.
    If astropy is missing, falls back to Keplerian elements (Standish, ~arcminute).

SPHEREx
    Live TLE from CelesTrak, propagated with SGP4 (needs `pip install sgp4`), then rotated
    TEME -> GCRS -> ecliptic with astropy (precession/nutation included).
    Without a TLE it falls back to a physically correct sun-synchronous orbit
    (inclination from the J2 sun-sync condition, 650 km, dawn-dusk plane that tracks the Sun)
    but the position *along* the orbit is NOT known in that mode.
"""
import datetime as dt
import math
import os
import threading
import time
import urllib.request
import warnings

import numpy as np
from flask import Flask, jsonify, render_template

try:
    from sgp4.api import Satrec
except ImportError:
    Satrec = None

ASTRO = False
try:
    import astropy.units as u
    from astropy.time import Time
    from astropy.utils import iers
    from astropy.coordinates import (CartesianRepresentation, GCRS, TEME,
                                     get_body_barycentric, solar_system_ephemeris)
    iers.conf.auto_download = False          # never block on a network fetch
    iers.conf.iers_degraded_accuracy = "warn"
    warnings.filterwarnings("ignore", message=".*dubious year.*")
    ASTRO = True
except ImportError:
    pass

app = Flask(__name__)

AU_KM = 149_597_870.7
EARTH_R_KM = 6371.0
EARTH_EQ_R_KM = 6378.137
MU_EARTH = 398_600.4418
OBLIQ = math.radians(84381.406 / 3600)       # IAU 2006 mean obliquity at J2000

PLANETS = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
PERIOD_DAYS = {"Mercury": 87.969, "Venus": 224.701, "Earth": 365.256, "Mars": 686.980,
               "Jupiter": 4332.59, "Saturn": 10759.22, "Uranus": 30688.5, "Neptune": 60182.0}
RADIUS_KM = {"Mercury": 2440, "Venus": 6052, "Earth": 6371, "Mars": 3390, "Jupiter": 69911,
             "Saturn": 58232, "Uranus": 25362, "Neptune": 24622, "Moon": 1737}

# SPHEREx orbit (fallback model only; the TLE path ignores these)
SPHEREX_ALT_KM = 650.0
LTAN_HOURS = 18.0   # local time of ascending node; dawn-dusk orbit (6 pm). Unverified; only flips the side.

EPH_NAME = "Keplerian approximation (astropy not installed)"
if ASTRO:
    want = os.environ.get("XARVIS_EPHEMERIS", "builtin")
    try:
        solar_system_ephemeris.set(want)
        get_body_barycentric("earth", Time(dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)))
        EPH_NAME = "astropy built-in (ERFA)" if want == "builtin" else f"JPL {want.upper()}"
    except Exception as exc:
        print(f"[ephemeris] '{want}' unavailable ({exc!s:.80}); using astropy built-in")
        solar_system_ephemeris.set("builtin")
        EPH_NAME = "astropy built-in (ERFA)"


# -------------------------------------------------------------------- maths ---
def eq_to_ecl(v):
    """Equatorial (ICRS/J2000) -> ecliptic J2000 for arrays shaped (..., 3)."""
    v = np.asarray(v, float)
    c, s = math.cos(OBLIQ), math.sin(OBLIQ)
    return np.stack([v[..., 0], v[..., 1] * c + v[..., 2] * s, -v[..., 1] * s + v[..., 2] * c], axis=-1)


def normalize(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def close_loop(pts):
    """Make a path sampled over one period, centred on "now" (middle sample), into a closed curve.

    Real orbits don't repeat exactly (perturbations, the Sun's barycentric wobble), so the first and
    last samples differ by up to millions of km. The drift is removed with a correction that is zero
    at the middle sample, so the curve still passes exactly through the body's current position and
    the seam ends up on the far side of the orbit, where it can't be seen.
    """
    pts = np.array(pts, float)
    n = len(pts) - 1
    w = (np.arange(n + 1) - n / 2) / (n / 2)              # -1 .. 0 .. +1
    pts = pts - 0.5 * (pts[-1] - pts[0]) * w[:, None]
    pts[-1] = pts[0]
    return pts


def centred_times(t0, period, n, unit):
    return t0 + np.linspace(-0.5, 0.5, n + 1) * period * unit


def to_time(now):
    return Time(now.replace(tzinfo=None), scale="utc")


# --------------------------------------------------- ephemeris (astropy path) ---
def helio_ecl(body, t):
    """Heliocentric ecliptic J2000 position in AU of `body` at astropy Time `t` (scalar or array)."""
    d = get_body_barycentric(body.lower(), t) - get_body_barycentric("sun", t)
    return eq_to_ecl(np.moveaxis(d.xyz.to_value(u.au), 0, -1))


def geocentric_moon_ecl(t):
    d = get_body_barycentric("moon", t) - get_body_barycentric("earth", t)
    return eq_to_ecl(np.moveaxis(d.xyz.to_value(u.au), 0, -1))


# ------------------------------------------- ephemeris (Keplerian fallback) ---
_KEP = {  # a, da, e, de, I, dI, L, dL, long.peri, dlp, long.node, dln  (Standish, 1800-2050)
    "Mercury": (0.38709927, 0.00000037, 0.20563593, 0.00001906, 7.00497902, -0.00594749, 252.25032350, 149472.67411175, 77.45779628, 0.16047689, 48.33076593, -0.12534081),
    "Venus": (0.72333566, 0.00000390, 0.00677672, -0.00004107, 3.39467605, -0.00078890, 181.97909950, 58517.81538729, 131.60246718, 0.00268329, 76.67984255, -0.27769418),
    "Earth": (1.00000261, 0.00000562, 0.01671123, -0.00004392, -0.00001531, -0.01294668, 100.46457166, 35999.37244981, 102.93768193, 0.32327364, 0.0, 0.0),
    "Mars": (1.52371034, 0.00001847, 0.09339410, 0.00007882, 1.84969142, -0.00813131, -4.55343205, 19140.30268499, -23.94362959, 0.44441088, 49.55953891, -0.29257343),
    "Jupiter": (5.20288700, -0.00011607, 0.04838624, -0.00013253, 1.30439695, -0.00183714, 34.39644051, 3034.74612775, 14.72847983, 0.21252668, 100.47390909, 0.20469106),
    "Saturn": (9.53667594, -0.00125060, 0.05386179, -0.00050991, 2.48599187, 0.00193609, 49.95424423, 1222.49362201, 92.59887831, -0.41897216, 113.66242448, -0.28867794),
    "Uranus": (19.18916464, -0.00196176, 0.04725744, -0.00004397, 0.77263783, -0.00242939, 313.23810451, 428.48202785, 170.95427630, 0.40805281, 74.01692503, 0.04240589),
    "Neptune": (30.06992276, 0.00026291, 0.00859048, 0.00005105, 1.77004347, 0.00035372, -55.12002969, 218.45945325, 44.96476227, -0.32241464, 131.78422574, -0.00508664),
}


def _kep_pos(name, T, E=None):
    a0, da, e0, de, i0, di, l0, dl, w0, dw, o0, do = _KEP[name]
    a, e, inc = a0 + da * T, e0 + de * T, math.radians(i0 + di * T)
    L, lp, node = l0 + dl * T, w0 + dw * T, o0 + do * T
    w, node_r = math.radians(lp - node), math.radians(node)
    if E is None:
        M = math.radians(((L - lp) + 180) % 360 - 180)
        E = M + e * math.sin(M)
        for _ in range(12):
            E -= (E - e * math.sin(E) - M) / (1 - e * math.cos(E))
    xp, yp = a * (math.cos(E) - e), a * math.sqrt(1 - e * e) * math.sin(E)
    cw, sw, co, so, ci, si = math.cos(w), math.sin(w), math.cos(node_r), math.sin(node_r), math.cos(inc), math.sin(inc)
    return [(cw * co - sw * so * ci) * xp + (-sw * co - cw * so * ci) * yp,
            (cw * so + sw * co * ci) * xp + (-sw * so + cw * co * ci) * yp,
            (sw * si) * xp + (cw * si) * yp]


def julian_date(now):
    return now.timestamp() / 86400.0 + 2440587.5


# ----------------------------------------------------------------- SPHEREx ---
_tle = {"sat": None, "fetched": 0.0}
_tle_lock = threading.Lock()


def get_satrec():
    """SPHEREx TLE from CelesTrak (cached 6 h; failures retried after 10 min)."""
    if Satrec is None:
        return None
    with _tle_lock:
        age = time.time() - _tle["fetched"]
        if _tle["sat"] is not None and age < 6 * 3600:
            return _tle["sat"]
        if _tle["sat"] is None and _tle["fetched"] and age < 600:
            return None
        _tle["fetched"] = time.time()
        try:
            url = "https://celestrak.org/NORAD/elements/gp.php?NAME=SPHEREX&FORMAT=TLE"
            with urllib.request.urlopen(url, timeout=4) as r:
                lines = [l.strip() for l in r.read().decode().splitlines()]
            l1 = next(l for l in lines if l.startswith("1 "))
            l2 = next(l for l in lines if l.startswith("2 "))
            _tle["sat"] = Satrec.twoline2rv(l1, l2)
        except Exception as exc:
            print(f"[SPHEREx] TLE unavailable ({exc!s:.80}); using sun-synchronous orbit model")
            _tle["sat"] = None
        return _tle["sat"]


_teme_cache = {"hour": None, "M": None}


def teme_to_ecl(r_km, t):
    """TEME (km, shape (N,3)) -> ecliptic J2000 axes. TEME->GCRS is a slow precession/nutation
    rotation, so it is computed once per hour and cached (the full transform costs ~0.4 s)."""
    first = t if np.ndim(t.utc.jd1) == 0 else t[0]
    hour = int((float(first.utc.jd1) + float(first.utc.jd2)) * 24)
    if _teme_cache["M"] is None or hour != _teme_cache["hour"]:
        try:
            basis = CartesianRepresentation(np.eye(3) * 7000 * u.km)     # column k = unit vector k
            g = TEME(basis, obstime=first).transform_to(GCRS(obstime=first))
            _teme_cache["M"] = g.cartesian.xyz.to_value(u.km) / 7000     # M @ r_teme = r_gcrs
        except Exception as exc:
            print(f"[SPHEREx] TEME->GCRS failed ({exc!s:.80}); ignoring precession (~0.4 deg)")
            _teme_cache["M"] = np.eye(3)
        _teme_cache["hour"] = hour
    return eq_to_ecl(np.asarray(r_km) @ _teme_cache["M"].T)


def sso_offset_km(when, earth_helio, minutes=0.0):
    """Fallback: SPHEREx relative to Earth on an ideal sun-synchronous dawn-dusk orbit (ecliptic, km)."""
    a = EARTH_R_KM + SPHEREX_ALT_KM
    inc = math.acos(-((a / 12352.0) ** 3.5))                 # J2 sun-synchronous condition
    period_s = 2 * math.pi * math.sqrt(a ** 3 / MU_EARTH)
    # Sun's right ascension from Earth (equatorial frame)
    sun_ecl = -np.asarray(earth_helio)
    c, s = math.cos(OBLIQ), math.sin(OBLIQ)
    sun_eq = np.array([sun_ecl[0], sun_ecl[1] * c - sun_ecl[2] * s, sun_ecl[1] * s + sun_ecl[2] * c])
    ra_sun = math.atan2(sun_eq[1], sun_eq[0])
    raan = ra_sun + math.radians((LTAN_HOURS - 12.0) * 15.0)
    u_arg = 2 * math.pi * (((when.timestamp() + minutes * 60) % period_s) / period_s)   # phase is arbitrary
    cu, su, co, so, ci, si = math.cos(u_arg), math.sin(u_arg), math.cos(raan), math.sin(raan), math.cos(inc), math.sin(inc)
    eq = a * np.array([cu * co - su * ci * so, cu * so + su * ci * co, su * si])
    return eq_to_ecl(eq), period_s / 60.0


_ring_cache = {"t": 0.0, "ring": None, "period": None, "key": None}


def spherex_geometry(now, earth_helio, sat):
    """-> offset from Earth centre (km, ecliptic), orbit ring (AU, relative to Earth), period (min)."""
    if sat is not None and ASTRO:
        t = to_time(now)
        _e, r, _v = sat.sgp4_array(np.atleast_1d(t.utc.jd1), np.atleast_1d(t.utc.jd2))
        period = 2 * math.pi / sat.no_kozai
        if _e[0] == 0:
            off = teme_to_ecl(r[0], t)
            if _ring_cache["key"] != "tle" or time.time() - _ring_cache["t"] > 30:
                ts = centred_times(t, period, 120, u.min)
                err, rr, _ = sat.sgp4_array(ts.utc.jd1, ts.utc.jd2)
                ring = close_loop(teme_to_ecl(rr, ts) / AU_KM)
                _ring_cache.update(t=time.time(), ring=ring.tolist(), period=period, key="tle")
            return off, _ring_cache["ring"], period
    off, period = sso_offset_km(now, earth_helio)
    if _ring_cache["key"] != "sso" or time.time() - _ring_cache["t"] > 30:
        ring = [(sso_offset_km(now, earth_helio, period * (k / 120 - 0.5))[0] / AU_KM).tolist() for k in range(121)]
        _ring_cache.update(t=time.time(), ring=ring, period=period, key="sso")
    return off, _ring_cache["ring"], period


# ------------------------------------------------------------ dwarf planets ---
# name: (JPL Horizons id, id_type, orbital period in days, radius km)
DWARFS = {
    "Ceres": ("1", "smallbody", 1680.0, 473),
    "Pluto": ("999", "majorbody", 90560.0, 1188),
    "Haumea": ("136108", "smallbody", 103400.0, 816),
    "Makemake": ("136472", "smallbody", 111800.0, 715),
    "Eris": ("136199", "smallbody", 204200.0, 1163),
}
DWARF_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dwarf_cache.json")
_dwarf = {"paths": {}, "jd": {}, "status": "starting", "source": {}}
_dwarf_lock = threading.Lock()


def _load_dwarf_cache():
    try:
        import json
        with open(DWARF_CACHE) as f:
            d = json.load(f)
        jd_now = julian_date(dt.datetime.now(dt.timezone.utc))
        for name, v in d.items():
            jd = np.array(v["jd"])
            span = jd[-1] - jd[0]
            if jd[0] + 0.1 * span < jd_now < jd[-1] - 0.1 * span:      # still centred well enough
                _dwarf["paths"][name] = np.array(v["pts"]); _dwarf["jd"][name] = jd
                _dwarf["source"][name] = "JPL Horizons (cached)"
    except Exception:
        pass


def _fetch_dwarfs():
    """Background thread: trace each dwarf planet's orbit from JPL Horizons (one call per body)."""
    try:
        from astroquery.jplhorizons import Horizons
    except ImportError:
        _dwarf["status"] = "astroquery not installed"
        return
    import json
    now = dt.datetime.now(dt.timezone.utc)
    cache, failed = {}, []
    for name, (hid, idt, period, _r) in DWARFS.items():
        if name in _dwarf["paths"]:
            continue
        try:
            step = max(1, round(period / 360))
            ep = {"start": (now - dt.timedelta(days=period / 2)).strftime("%Y-%m-%d"),
                  "stop": (now + dt.timedelta(days=period / 2)).strftime("%Y-%m-%d"), "step": f"{step}d"}
            vec = Horizons(id=hid, id_type=idt, location="@sun", epochs=ep).vectors(refplane="ecliptic")
            pts = np.column_stack([np.array(vec["x"], float), np.array(vec["y"], float), np.array(vec["z"], float)])
            jd = np.array(vec["datetime_jd"], float)
            with _dwarf_lock:
                _dwarf["paths"][name] = pts; _dwarf["jd"][name] = jd; _dwarf["source"][name] = "JPL Horizons"
            cache[name] = {"pts": pts.tolist(), "jd": jd.tolist()}
        except Exception as exc:
            failed.append(name)
            print(f"[dwarfs] {name}: Horizons unavailable ({exc!s:.90})")
    if cache:
        try:
            old = {}
            if os.path.exists(DWARF_CACHE):
                with open(DWARF_CACHE) as f:
                    old = json.load(f)
            old.update(cache)
            with open(DWARF_CACHE, "w") as f:
                json.dump(old, f)
        except Exception:
            pass
    _dwarf["status"] = "ready" if not failed else f"partial (offline: {', '.join(failed)})"


def start_dwarf_fetch():
    _load_dwarf_cache()
    threading.Thread(target=_fetch_dwarfs, daemon=True).start()


# Offline fallback: Pluto from Standish's approximate elements (the only dwarf I can do reliably without Horizons)
_KEP["Pluto"] = (39.48211675, -0.00031596, 0.24882730, 0.00005170, 17.14001206, 0.00004818,
                 238.92903833, 145.20780515, 224.06891629, -0.04062942, 110.30393684, -0.01183482)


def dwarf_positions(now):
    """[{name, pos, radius_km, source}] for every dwarf planet we can place."""
    jd_now = julian_date(now)
    out = []
    for name, (_h, _i, period, radius) in DWARFS.items():
        with _dwarf_lock:
            pts, jd = _dwarf["paths"].get(name), _dwarf["jd"].get(name)
        if pts is not None and jd[0] <= jd_now <= jd[-1]:
            pos = [float(np.interp(jd_now, jd, pts[:, k])) for k in range(3)]
            out.append({"name": name, "pos": pos, "radius_km": radius, "source": _dwarf["source"][name]})
        elif name == "Pluto":
            T = (jd_now - 2451545.0) / 36525.0
            out.append({"name": name, "pos": _kep_pos("Pluto", T), "radius_km": radius,
                        "source": "approximate (Standish elements; Horizons offline)"})
    return out


def dwarf_paths():
    """{name: closed path (AU)} plus fallback ellipse for Pluto when Horizons data is missing."""
    out = {}
    for name in DWARFS:
        with _dwarf_lock:
            pts = _dwarf["paths"].get(name)
        if pts is not None:
            out[name] = np.round(close_loop(pts), 5).tolist()
        elif name == "Pluto":
            T = (julian_date(dt.datetime.now(dt.timezone.utc)) - 2451545.0) / 36525.0
            out[name] = [[round(c, 5) for c in _kep_pos("Pluto", T, 2 * math.pi * k / 360)] for k in range(361)]
    return out


# ------------------------------------------------------------------- routes ---
@app.route("/")
def index():
    return render_template("index.html")


_orbit_cache = {"t": 0.0, "data": None}


@app.route("/api/orbits")
def orbits():
    """Each planet's path over one full period, starting now (from ephemeris positions)."""
    if _orbit_cache["data"] is None or time.time() - _orbit_cache["t"] > 3600:
        now = dt.datetime.now(dt.timezone.utc)
        out = {}
        if ASTRO:
            t0 = to_time(now)
            for name in PLANETS:
                ts = centred_times(t0, PERIOD_DAYS[name], 360, u.day)
                out[name] = np.round(close_loop(helio_ecl(name, ts)), 7).tolist()
        else:
            T = (julian_date(now) - 2451545.0) / 36525.0
            for name in PLANETS:
                out[name] = [[round(c, 6) for c in _kep_pos(name, T, 2 * math.pi * k / 360)] for k in range(361)]
        _orbit_cache.update(t=time.time(), data=out)
    return jsonify(_orbit_cache["data"])


@app.route("/api/dwarf_orbits")
def dwarf_orbits():
    return jsonify({"status": _dwarf["status"], "paths": dwarf_paths(),
                    "complete": all(n in _dwarf["paths"] for n in DWARFS)})


@app.route("/api/state")
def state():
    now = dt.datetime.now(dt.timezone.utc)
    moon_ring = None
    if ASTRO:
        t = to_time(now)
        planets = [{"name": n, "pos": helio_ecl(n, t).tolist(), "radius_km": RADIUS_KM[n]} for n in PLANETS]
        earth = np.array(next(p["pos"] for p in planets if p["name"] == "Earth"))
        moon_off = geocentric_moon_ecl(t)
        planets.append({"name": "Moon", "pos": (earth + moon_off).tolist(), "radius_km": RADIUS_KM["Moon"]})
        moon_ring = np.round(close_loop(geocentric_moon_ecl(centred_times(t, 27.32, 240, u.day))), 8).tolist()
    else:
        T = (julian_date(now) - 2451545.0) / 36525.0
        planets = [{"name": n, "pos": _kep_pos(n, T), "radius_km": RADIUS_KM[n]} for n in PLANETS]
        earth = np.array(next(p["pos"] for p in planets if p["name"] == "Earth"))

    sat = get_satrec()
    off_km, ring, period = spherex_geometry(now, earth, sat)
    pos = earth + off_km / AU_KM
    r = float(np.linalg.norm(pos))
    dist_earth = float(np.linalg.norm(off_km))
    if sat is not None and ASTRO:
        tle_age = julian_date(now) - (sat.jdsatepoch + sat.jdsatepochF)
        source = f"CelesTrak TLE, SGP4 (epoch {tle_age:.1f} d old)"
    else:
        tle_age = None
        source = "APPROXIMATE: sun-sync model, phase unknown (pip install sgp4)" if Satrec is None \
            else "APPROXIMATE: sun-sync model, phase unknown (TLE fetch failed)"
    return jsonify({
        "utc": now.strftime("%Y-%m-%d %H:%M:%S"),
        "ephemeris": EPH_NAME,
        "planets": planets,
        "dwarfs": dwarf_positions(now),
        "moon_ring": moon_ring,
        "sun_dir": normalize(-earth).tolist(),
        "spherex": {
            "pos": pos.tolist(), "ring": ring,
            "helio_dist_au": r,
            "ecl_lon_deg": math.degrees(math.atan2(pos[1], pos[0])) % 360,
            "ecl_lat_deg": math.degrees(math.asin(pos[2] / r)),
            "earth_dist_km": dist_earth,
            "altitude_km": dist_earth - EARTH_R_KM,
            "period_min": period,
            "live": bool(sat is not None and ASTRO),
            "source": source,
        },
    })


if __name__ == "__main__":
    start_dwarf_fetch()                      # background: dwarf-planet orbits from JPL Horizons
    app.run(debug=True, use_reloader=False, port=5000)
