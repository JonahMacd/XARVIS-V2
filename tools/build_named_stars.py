#!/usr/bin/env python3
"""Build xarvis_pipeline/data/named_stars.json: every star with a common name, with its Gaia DR3 id.

The sky map (xarvis_pipeline/skymap.py, catalog_layer) names a star from this file FIRST, by its Gaia DR3 id - so it
needs no network for the names, and nothing depends on Hipparcos's 1991 positions (which put a fast star such as
Barnard's Star ~6' from where it is today). Stars too bright for Gaia (Sirius, Vega...) have no Gaia id: they are
placed with their SIMBAD J2000 position and proper motion instead.

Names    Stellarium's "modern" sky culture (CC BY-SA 4.0): {"HIP n": [{"english": ...}]}, the same list the sky map used
         before (skymap.star_names).
Ids      SIMBAD (CDS) TAP: the Gaia DR3 identifier of each HIP star, plus its J2000 (ICRS) position, proper motion
         and V magnitude.
         The Harvard mirror is tried first, then Strasbourg.

Usage:  python tools/build_named_stars.py --out xarvis_pipeline/data/named_stars.json
Needs:  requests, and access to raw.githubusercontent.com and simbad.harvard.edu (or simbad.cds.unistra.fr)
"""
import argparse
import csv
import io
import json
import re

import requests

STELLARIUM = 'https://raw.githubusercontent.com/Stellarium/stellarium/master/skycultures/modern/index.json'
SIMBAD_TAP = ('https://simbad.harvard.edu/simbad/sim-tap/sync', 'https://simbad.cds.unistra.fr/simbad/sim-tap/sync')
CHUNK = 150


def stellarium_names():
    data = requests.get(STELLARIUM, timeout=120).json()
    names = {}
    for key, entries in data.get('common_names', {}).items():
        match = re.match(r'HIP (\d+)$', key)
        if match and entries:
            names[int(match.group(1))] = entries[0].get('english') or entries[0].get('native')
    return names


def tap(query):
    last = None
    for url in SIMBAD_TAP:
        try:
            r = requests.post(url, data={'REQUEST': 'doQuery', 'LANG': 'ADQL', 'FORMAT': 'csv', 'QUERY': query}, timeout=180)
            if r.ok and r.text.startswith('hip,'):
                return list(csv.DictReader(io.StringIO(r.text)))
            last = f'{url}: HTTP {r.status_code} {r.text[:80]!r}'
        except requests.RequestException as e:
            last = f'{url}: {e}'
    raise RuntimeError(f'SIMBAD TAP: no server answered ({last})')


def number(value):
    try:
        return round(float(value), 8)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    names = stellarium_names()
    hips = sorted(names)
    found = {}
    for k in range(0, len(hips), CHUNK):
        ids = ', '.join(f"'HIP {h}'" for h in hips[k:k + CHUNK])
        rows = tap(f"""SELECT i1.id AS hip, g.id AS gaia, b.ra, b.dec, b.pmra, b.pmdec, f.V AS vmag
                       FROM ident AS i1 JOIN basic AS b ON b.oid = i1.oidref
                       LEFT OUTER JOIN allfluxes AS f ON f.oidref = b.oid
                       LEFT OUTER JOIN (SELECT oidref, id FROM ident WHERE id LIKE 'Gaia DR3 %') AS g ON g.oidref = i1.oidref
                       WHERE i1.id IN ({ids})""")
        for row in rows:
            hip = int(row['hip'].split()[1])
            gaia = (row['gaia'] or '').replace('Gaia DR3', '').strip() or None
            found[hip] = {'hip': hip, 'name': names[hip], 'gaia_dr3': gaia, 'ra': number(row['ra']),
                          'dec': number(row['dec']), 'pmra': number(row['pmra']) or 0.0, 'pmdec': number(row['pmdec']) or 0.0,
                          'vmag': number(row['vmag'])}
        print(f'{min(k + CHUNK, len(hips))}/{len(hips)} HIP stars looked up')
    stars = [found[h] for h in hips if h in found and found[h]['ra'] is not None]
    missing = [h for h in hips if h not in found]
    out = {'source': 'Names: Stellarium modern sky culture (CC BY-SA 4.0). Gaia DR3 ids, J2000 ICRS positions and '
                     'proper motions (mas/yr, RA*cos(Dec)), V magnitudes: SIMBAD (CDS).',
           'stars': stars}
    with open(a.out, 'w', newline='\n') as f:
        json.dump(out, f, indent=0, ensure_ascii=False)
        f.write('\n')
    print(f'wrote {a.out}: {len(stars)} named stars, {sum(s["gaia_dr3"] is not None for s in stars)} with a Gaia DR3 id'
          + (f'; not in SIMBAD: {missing}' if missing else ''))


if __name__ == '__main__':
    main()
