"""SPHEREx transient analysis: every SPHEREx image of a real star, T1 ... Tn in time order.

Modules
  config.py   all settings
  irsa.py     talking to IRSA: timeouts, retries, image search, catalogs, cutouts
  target.py   pick the star to track and its catalog brightness
  frames.py   download + measure every image (placed by SPHEREx's pointing only); original-pixel zooms
  measure.py  brightness, position, alignment check, movement, brightness changes, spectrum check
  plots.py    frame viewer (slider), frame grid, time series, spectrum
  main.py     runs everything (python -m Jonah_Code_File, or run Test_Jonah_code.py)
"""

import warnings

from astropy import log
from astropy.wcs import FITSFixedWarning

# astropy prints a long, harmless note about SIP distortion for every SPHEREx header; hide it
log.setLevel('WARNING')
warnings.simplefilter('ignore', FITSFixedWarning)
