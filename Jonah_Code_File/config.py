"""All settings in one place."""

from astropy import units as u

## target: snapped to the nearest bright, isolated 2MASS star (see target.py)
RA_DEG = 304.693508808
DEC_DEG = 42.4436872991

SNAP_RADIUS = 60 * u.arcsec         # look for a catalog star within this distance of RA/DEC
SNAP_MAX_KS_MAG = 13.0              # only stars at least this bright (2MASS Ks magnitude)
ISOLATION_RADIUS = 18 * u.arcsec    # ~3 SPHEREx pixels: neighbors this close blend into the red circle...
ISOLATION_MAX_FRACTION = 0.15       # ...and together must be under 15% of the star's brightness

## dataset
MAX_IMAGES = None                   # None = every SPHEREx image of the object; e.g. 40 for a quick test
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

PIXEL_ARCSEC = 6.15                 # SPHEREx pixel size
