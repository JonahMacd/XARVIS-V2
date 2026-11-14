import http.client
import glob
import re
import time
import urllib.error
import copy
from packaging.version import Version

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import quantity_support
from astropy.wcs import WCS
from astroquery.ipac.irsa import Irsa

from photutils.psf import ImagePSF

# The time it takes to read SPHEREx files can exceed
# astropy's default timeout limit. Increase it.
from astropy.utils.data import conf
conf.remote_timeout = 120

# Let matplotlib recognize astropy Quantities and auto-label axes with their units.
quantity_support()