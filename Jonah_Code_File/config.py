"""All settings in one place."""

from astropy import units as u

## WHAT TO TRACK. An asteroid (moves a lot - the transient case), or a star (stays put).
# JPL Horizons id of the asteroid: a number ('433' = Eros) or a name. Set to None to track the star below instead.
ASTEROID = '433'
# Time window to search, as dates (SPHEREx data runs 2025-04-29 to about 2026-06-14). SPHEREx always looks ~90 degrees
# away from the Sun, so an asteroid is only in its images while it is near that solar elongation (82-98 degrees).
ASTEROID_WINDOW = ('2026-02-15', '2026-06-15')      # Eros: ~125 images over 4 months, crossing ~100 degrees of sky
GRID_SPACING_DEG = 1.5              # image search: one position search every this many degrees along the path
DETECTION_SNR = 5                   # the asteroid counts as detected in a frame above this signal-to-noise
RESIDUAL_LIMIT_PX = 1.5             # flag a detection this far (pixels) from where the ephemeris says it should be

## the star (used when ASTEROID is None): snapped to the nearest bright, isolated 2MASS star (see target.py)
RA_DEG = 304.693508808
DEC_DEG = 42.4436872991

SNAP_RADIUS = 60 * u.arcsec         # look for a catalog star within this distance of RA/DEC
SNAP_MAX_KS_MAG = 13.0              # only stars at least this bright (2MASS Ks magnitude)
ISOLATION_RADIUS = 18 * u.arcsec    # ~3 SPHEREx pixels: neighbors this close blend into the red circle...
ISOLATION_MAX_FRACTION = 0.15       # ...and together must be under 15% of the star's brightness

## dataset
MAX_IMAGES = None                   # None = every SPHEREx image of the object; e.g. 40 for a quick test
ZOOM_ONLY = True                    # after the analysis keep only the zoomed pixels in memory (frees the rest)

## disk cache: makes re-runs fast (no re-downloading), and cleans itself up after every run (see cache.py)
CACHE_ENABLED = True                # False = never write anything to disk
CACHE_DIR = None                    # None = a local folder OneDrive doesn't sync (%LOCALAPPDATA%\Jonah_Code_File\cache)
CACHE_MAX_AGE_DAYS = 7              # files not used for this long are deleted...
CACHE_MAX_MB = 500                  # ...and the least recently used ones above this size (~95 KB per image)

## results: the measurements of every frame, saved as a small table (one file per target, replaced each run)
SAVE_RESULTS = True
RESULTS_DIR = None                  # None = a 'results' folder next to Jonah_Code_File
ZOOM_SIZE_PIX = 50                  # zoom: 50 x 50 ORIGINAL pixels centered on the object
CUTOUT_SIZE = 5.5 * u.arcmin        # downloaded cutout: ~54 x 54 pixels, a little bigger than the zoom

## measurements
APERTURE_RADIUS_PIX = 2.0           # brightness is summed inside this circle...
ANNULUS_PIX = (4.0, 6.5)            # ...minus the background measured in this ring
SEARCH_RADIUS_PIX = 3               # look this far from the expected position for the object's peak
ALIGNMENT_TOLERANCE_PIX = 0.25      # alignment check warns about frames above this
MOVEMENT_SIGMA = 5                  # flag movement bigger than this many times the typical scatter
BRIGHTNESS_SIGMA = 5                # flag brightness this many times off the object's own spectrum
SPECTRUM_NEIGHBORS = 10             # brightness reference: median of up to this many nearest-wavelength frames...
SPECTRUM_WINDOW = 0.03              # ...within +/-3% of the frame's wavelength...
SPECTRUM_MIN_NEIGHBORS = 4          # ...and at least this many, otherwise the frame's brightness isn't judged

## bad pixels (ignored in all measurements)
# SPHEREx FLAGS that make a pixel unreliable. Not included: SOURCE ("a known star is here"), FULLSAMPLE,
# DICHROIC, PHANMISS, and HALO (it marks the surroundings of ordinary stars too, so it isn't a defect).
# TRANSIENT here means a jump during detector readout (usually a cosmic ray).
BAD_FLAG_NAMES = {'TRANSIENT', 'OVERFLOW', 'SUR_ERROR', 'NONFUNC', 'MISSING_DATA', 'HOT', 'COLD', 'NONLINEAR',
                  'PERSIST', 'OUTLIER', 'CROSSTALK', 'GHOST', 'GHOST_FPA', 'GHOST_EXT', 'STREAK', 'BLOOM',
                  'SNOWBALL'}
# For a MOVING object these two are not defects: SPHEREx's own pipeline marks a moving source's brightest pixels
# as OUTLIER / TRANSIENT, because it changes during the exposure the way a cosmic-ray hit does. (Measured on Eros:
# 103 of its pixels in 16 of 28 frames were flagged OUTLIER.) Throwing them away throws away the asteroid itself.
BAD_FLAG_NAMES_MOVING = BAD_FLAG_NAMES - {'OUTLIER', 'TRANSIENT'}
# Spikes the pipeline missed (e.g. an unflagged hot pixel): a real source always spreads over several
# pixels, so a single pixel this many times brighter than ALL its neighbors, and far above the noise, isn't one.
# Measured: real star centers reach ~7x their brightest neighbor; the hot pixels found were 48x and 95x.
SPIKE_NEIGHBOR_RATIO = 20
SPIKE_MIN_SIGMA = 20

## IRSA access (IRSA often answers "502 Proxy Error" and then succeeds on a retry)
MAX_ATTEMPTS = 50
WAIT_SECONDS = 0.001                # wait between the first FAST_ATTEMPTS retries...
FAST_ATTEMPTS = 10
MAX_WAIT_SECONDS = 5                # ...then back off (0.1 s, 0.2 s, 0.4 s ... up to this) for longer outages
REQUEST_TIMEOUT = 30                # seconds to wait for IRSA to start answering before retrying
MAX_CONCURRENT_DOWNLOADS = 16       # parallel cutout downloads (measured: 16 is fastest; 24 makes IRSA refuse more)
COLLECTIONS = ['spherex_dr1', 'spherex_qr3']
GATOR_URL = 'https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query'
SIA_URL = 'https://irsa.ipac.caltech.edu/SIA'

PIXEL_ARCSEC = 6.15                 # SPHEREx pixel size
