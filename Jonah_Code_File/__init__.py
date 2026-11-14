"""SPHEREx transient analysis: the same patch of sky at T1 ... Tn, centered on a real star.

Modules
  config.py   all settings
  irsa.py     talking to IRSA: timeouts, retries, image search, catalogs, cutouts
  target.py   step 0 - pick the star to track and its catalog brightness
  frames.py   steps 1-2 - pick T1 ... Tn and zoom on the original pixels (placed by SPHEREx's pointing only)
  measure.py  brightness, position, alignment check, movement tracking, spectrum check
  plots.py    figures
  main.py     runs everything (python -m Jonah_Code_File, or run Test_Jonah_code.py)
"""

import warnings

from astropy import log
from astropy.wcs import FITSFixedWarning

# astropy prints a long, harmless note about SIP distortion for every SPHEREx header; hide it
log.setLevel('WARNING')
warnings.simplefilter('ignore', FITSFixedWarning)
