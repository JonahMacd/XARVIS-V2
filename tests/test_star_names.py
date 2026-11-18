"""Offline tests of how the sky map names catalog stars (skymap.catalog_layer): a real name (Stellarium common name,
the analysed target's own name, HIP number) always comes before a Gaia DR3 / CatWISE catalog id. No network: the
Hipparcos, Stellarium and Gaia / CatWISE queries are replaced with fixed answers, and the built-in named-star list
(skymap.named_stars, data/named_stars.json) is used as shipped or replaced.

Run:  python -m unittest discover -s tests -t .
"""

import unittest
from contextlib import ExitStack
from unittest import mock

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from xarvis_pipeline import skymap
from xarvis_pipeline.field import GAIA_EPOCH_MJD

# Barnard's Star: Hipparcos (new reduction, J1991.25) and Gaia DR3 (J2016.0) values
BARNARD_HIP = {'hip': 87937, 'ra': 269.45402305, 'dec': 4.66828815, 'pmra': -797.84, 'pmdec': 10326.93,
               'parallax': 548.31, 'hp': 9.5}
BARNARD_GAIA = {'source': 'Gaia', 'name': 'Gaia DR3 4472832130942575872', 'ra': 269.44850252543836,
                'dec': 4.739420051112412, 'pmra': -801.55, 'pmdec': 10362.39, 'epoch_mjd': GAIA_EPOCH_MJD,
                'mag': 7.0, 'parallax': 546.98, 'gaia_match': False}
OTHER_GAIA = dict(BARNARD_GAIA, name='Gaia DR3 4472832130942575001', ra=269.46, dec=4.75, pmra=1.0, pmdec=-2.0, mag=14.0)
# where Barnard's Star is in 2025 (the field is centred there): ~6' from its Hipparcos (1991) position
NOW = SkyCoord(269.44636 * u.deg, 4.76675 * u.deg)


def hipparcos_cone(stars):
    """skymap.hipparcos as VizieR answers it: the stars whose CATALOG (J1991.25) position is in the cone."""
    def query(coord, radius):
        inside = [s for s in stars if coord.separation(SkyCoord(s['ra'] * u.deg, s['dec'] * u.deg)) <= radius]
        return Table(rows=[[s[c] for c in ('hip', 'ra', 'dec', 'pmra', 'pmdec', 'parallax', 'hp')] for s in inside] or None,
                     names=('hip', 'ra', 'dec', 'pmra', 'pmdec', 'parallax', 'hp'))
    return query


def vizier_down(coord, radius):
    raise RuntimeError('Hipparcos: no VizieR server answered')


def layer(names=None, target=None, hip_stars=(BARNARD_HIP,), builtin=(), hipparcos=None, field_rows=None):
    """catalog_layer around Barnard's Star. builtin: the built-in named stars (None = the real list)."""
    field = Table(rows=field_rows or [BARNARD_GAIA, OTHER_GAIA])
    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(skymap, 'hipparcos', hipparcos or hipparcos_cone(list(hip_stars))))
        stack.enter_context(mock.patch.object(skymap, 'star_names', lambda: dict(names or {})))
        if builtin is not None:
            stack.enter_context(mock.patch.object(skymap, 'named_stars', lambda: tuple(builtin)))
        table = skymap.catalog_layer(NOW, 4 * u.arcmin, field=field, target=target)
    return {str(r['name']): r for r in table}


class StarNameTests(unittest.TestCase):

    def test_a_fast_star_is_found_in_hipparcos_and_named(self):
        # its 1991 Hipparcos position is outside the 4' field: the search must still find it
        rows = layer(names={87937: "Barnard's Star"})
        barnard = rows[BARNARD_GAIA['name']]
        self.assertEqual(barnard['label'], "Barnard's Star (HIP 87937)")
        self.assertTrue(barnard['named'])
        self.assertEqual(barnard['hip'], 87937)
        self.assertEqual(rows[OTHER_GAIA['name']]['label'], OTHER_GAIA['name'])     # unnamed: its catalog id
        self.assertFalse(rows[OTHER_GAIA['name']]['named'])

    def test_the_analysed_target_is_named_without_a_common_name(self):
        target = {'name': "Barnard's Star", 'query': '4472832130942575872'}
        rows = layer(target=target, hip_stars=())                 # not in Hipparcos at all (e.g. Epsilon Indi B)
        self.assertEqual(rows[BARNARD_GAIA['name']]['label'], "Barnard's Star")
        self.assertTrue(rows[BARNARD_GAIA['name']]['named'])
        rows = layer(target=target)                               # in Hipparcos without a common name: both
        self.assertEqual(rows[BARNARD_GAIA['name']]['label'], "Barnard's Star (HIP 87937)")

    def test_a_common_name_comes_before_the_target_name(self):
        rows = layer(names={87937: 'Velox Barnardi'}, target={'name': 'my star', 'query': 'Gaia DR3 4472832130942575872'})
        self.assertEqual(rows[BARNARD_GAIA['name']]['label'], 'Velox Barnardi (HIP 87937)')

    def test_a_plain_sky_position_names_nothing(self):
        self.assertEqual(skymap.target_label({'name': 'RA 269.44636 Dec +4.76675', 'query': '269.44636 4.76675'}), {})
        self.assertEqual(skymap.target_label(None), {})

    def test_the_built_in_list_names_a_star_when_vizier_is_down(self):
        rows = layer(hipparcos=vizier_down, builtin=None)        # the real data/named_stars.json
        self.assertEqual(rows[BARNARD_GAIA['name']]['label'], "Barnard's Star (HIP 87937)")
        self.assertTrue(rows[BARNARD_GAIA['name']]['named'])
        self.assertFalse(rows[OTHER_GAIA['name']]['named'])

    def test_a_named_star_too_bright_for_gaia_is_placed_by_position(self):
        bright = {'hip': 99999, 'name': 'Testar', 'gaia_dr3': None, 'ra': 269.45, 'dec': 4.76, 'pmra': 0.0,
                  'pmdec': 0.0, 'vmag': 1.0}
        rows = layer(hipparcos=vizier_down, builtin=[bright])
        self.assertEqual(rows['HIP 99999']['label'], 'Testar (HIP 99999)')
        self.assertEqual(rows['HIP 99999']['source'], 'Hipparcos')
        far = dict(bright, hip=99998, ra=270.0)                    # 33' away: not in the field
        self.assertNotIn('HIP 99998', layer(hipparcos=vizier_down, builtin=[far]))

    def test_the_built_in_names_cover_the_test_stars(self):
        by_hip = {s['hip']: s for s in skymap.named_stars()}
        self.assertGreater(len(by_hip), 600)
        for hip, name, gaia in ((87937, "Barnard's Star", '4472832130942575872'), (70890, 'Proxima', '5853498713190525696'),
                                (24186, "Kapteyn's Star", '4810594479418041856'), (32349, 'Sirius', None)):
            self.assertEqual((by_hip[hip]['name'], by_hip[hip]['gaia_dr3']), (name, gaia))

    def test_a_vizier_error_answer_is_not_a_table(self):
        error = (b'#INFO\tError=The database is not currently reachable.\t\n#INFO\tQUERY_STATUS=ERROR\t\n')
        with self.assertRaises(skymap.net.BadResponse):
            skymap._check_vizier(error)
        skymap._check_vizier(b'#Column\tHIP\t(I6)\nHIP\tRArad\n 87937\t269.45\n')      # a real answer passes

    def test_hipparcos_stars_far_outside_the_field_are_left_out(self):
        far = dict(BARNARD_HIP, hip=1, ra=269.55, dec=4.72, pmra=0.0, pmdec=0.0)       # ~7' away, not moving
        rows = layer(hip_stars=(BARNARD_HIP, far))
        self.assertNotIn('HIP 1', rows)
        self.assertEqual(int(np.sum([r['hip'] > 0 for r in rows.values()])), 1)


if __name__ == '__main__':
    unittest.main()
