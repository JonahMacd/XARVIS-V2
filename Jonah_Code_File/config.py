"""All settings in one place."""

from astropy import units as u

## target: snapped to the nearest bright, isolated 2MASS star (see target.py)
RA_DEG = 304.693508808
DEC_DEG = 42.4436872991

SNAP_RADIUS = 60 * u.arcsec         # look for a catalog star within this distance of RA/DEC
SNAP_MAX_KS_MAG = 13.0              # only stars at least this bright (2MASS Ks magnitude)
ISOLATION_RADIUS = 18 * u.arcsec    # ~3 SPHEREx pixels: neighbors this close blend into the red circle...
ISOLATION_MAX_FRACTION = 0.15       # ...and together must be under 15% of the star's brightness

## time sequence
MAX_FRAMES = 10                     # T1 ... T10
WAVELENGTH_TOLERANCE_UM = 0.05      # how close in wavelength a frame must be to T1
ZOOM_SIZE_PIX = 50                  # zoom: 50 x 50 ORIGINAL pixels centered on the object
CUTOUT_SIZE = 5.5 * u.arcmin        # downloaded cutout: ~54 x 54 pixels, a little bigger than the zoom

## measurements
APERTURE_RADIUS_PIX = 2.0           # brightness is summed inside this circle...
ANNULUS_PIX = (4.0, 6.5)            # ...minus the background measured in this ring
SEARCH_RADIUS_PIX = 3               # look this far from the expected position for the object's peak
ALIGNMENT_TOLERANCE_PIX = 0.25      # alignment check warns above this
SPECTRUM_IMAGES_PER_DETECTOR = 8
SPECTRUM_CUTOUT_SIZE = 90 * u.arcsec

## IRSA access (IRSA often answers "502 Proxy Error" and then succeeds on a retry)
MAX_ATTEMPTS = 50
WAIT_SECONDS = 0.001
REQUEST_TIMEOUT = 30                # seconds to wait for IRSA to start answering before retrying
MAX_CONCURRENT_DOWNLOADS = 8        # total parallel cutout downloads, shared by everything
COLLECTIONS = ['spherex_dr1', 'spherex_qr3']
GATOR_URL = 'https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query'

PIXEL_ARCSEC = 6.15                 # SPHEREx pixel size
