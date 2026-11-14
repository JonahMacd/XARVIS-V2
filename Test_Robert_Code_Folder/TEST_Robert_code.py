import http.client
import time
import urllib.error

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.utils.data import conf

from astroquery.ipac.irsa import Irsa
from pyvo.dal.exceptions import DALServiceError
import requests

from firefly_client import FireflyClient

# Reading SPHEREx files can exceed astropy's default timeout.
conf.remote_timeout = 120

ra_deg = 304.693508808
dec_deg = 42.4436872991
coord = SkyCoord(ra_deg, dec_deg, unit="deg")
search_radius = 1 * u.arcsec

# Errors that mean "server not ready", not "your code is wrong".
RETRYABLE = (
    DALServiceError,
    requests.exceptions.RequestException,
    urllib.error.URLError,
    http.client.HTTPException,
    TimeoutError,
    ConnectionError,
)


def query_with_retry(interval=0.1, give_up_after=None):
    """Retry the IRSA SIA query every `interval` seconds until it succeeds."""
    start = time.time()
    attempt = 0
    while True:
        attempt += 1
        try:
            print(f"[attempt {attempt}] querying IRSA SIA...")
            return Irsa.query_sia(
                pos=(coord, search_radius),
                collection=["spherex_dr1", "spherex_qr3"],
            )
        except RETRYABLE as e:
            elapsed = time.time() - start
            print(f"  failed after {elapsed:.0f}s: {str(e)[:100]!r}")
            if give_up_after is not None and elapsed + interval > give_up_after:
                raise
            time.sleep(interval)

results = query_with_retry()
print(f"Got {len(results)} spectral images")

spectral_image_url = results['access_url'][0]
print(spectral_image_url)

fc = FireflyClient.make_client(url="https://irsa.ipac.caltech.edu/irsaviewer")

fc.reinit_viewer()

fc.show_fits_image(file_input=spectral_image_url,
             plot_id="spectral_image",
             Title="Spectral Image"
             )

spectral_image_header = hdulist[1].header

keywords_to_print = ['EXTNAME', 'NAXIS1', 'NAXIS2', 'BUNIT', 'DETECTOR', 'OBSID', 'DATE', 'PSF_FWHM']

for keyword in keywords_to_print:
    value = spectral_image_header.get(keyword, 'Keyword not found')
    print(f"{keyword}: {value}")

spectral_image_header['WCSNAME*']

wcs = WCS(spectral_image_header)
wcs

ra, dec = wcs.wcs.crval
x, y = wcs.world_to_pixel(SkyCoord(ra=ra, dec=dec, unit="deg"))
print(ra, dec, x, y)

# Load the Spectral WCS from the header.
# Note that we need to provide a reference to HDU List, which contains a lookup table.
spectral_wcs = WCS(header=hdulist["IMAGE"].header, fobj=hdulist, key="W")

# SIP distortions must be turned off for the spectral WCS.
spectral_wcs.sip = None

# The standard Astropy methods for converting pixel coordinates to world coordinates can also be used to obtain spectral coordinates.
# Take the pixel coordinates that we determined for the image center and resolve them to the wavelength and bandwidth for that pixel
wl, bw = spectral_wcs.pixel_to_world(x, y)
wl, bw

# Read in the spectral image data.
spectral_image = hdulist["IMAGE"]
spectral_image_data = spectral_image.data

# Get arrays of pixel coordinates for the image.
(y, x) = np.indices(spectral_image.shape)

# Use the spectral WCS to convert these pixel coordinates to spectral coordinates.
spectral_coords = spectral_wcs.pixel_to_world(x, y)

# Break out the two spectral coordinates (wavelength and bandwidth) from the spectral coordinates, and print the results.
wavelength, bandwidth = spectral_coords
print("Wavelength: \n", wavelength)
print("Bandwidth: \n", bandwidth)

# Plot the wavelength as a greyscale color map across the image pixels.
plt.imshow(wavelength.value, origin='lower', cmap='gray')
plt.colorbar(label=f"Wavelength [{wavelength.unit}]")

detector = hdulist["IMAGE"].header["DETECTOR"]
plt.title(f"Wavelength - Detector {detector}")
plt.show()

spectral_image_header['L2 N_*']