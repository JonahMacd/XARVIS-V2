import http.client
import os
import tempfile
import time
import urllib.error

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import requests

from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.visualization import ImageNormalize, ZScaleInterval, AsinhStretch

from astroquery.ipac.irsa import Irsa

from firefly_client import FireflyClient

from pyvo.dal.exceptions import DALServiceError

## the time to access SPHEREx files sometimes can excees astropy's default timeout limit. Incerase it by:
from astropy.utils.data import conf
conf.remote_timeout = 120

## settings
MAX_ATTEMPTS = 20
WAIT_SECONDS = 0.001

SHOW_IN_FIREFLY = False       # True = also open the image in IRSA's web viewer
SUBTRACT_ZODI = True          # the ZODI background model is NOT subtracted from IMAGE in the files
CUTOUT_SIZE = 3 * u.arcmin    # size of the zoomed-in panel around the target (SPHEREx pixels are ~6.2")


# ----------------------------------------------------------------------------
# Loading and plotting (kept free of Firefly so they can be reused in a web app)
# ----------------------------------------------------------------------------

def download_spherex_image(url):
    """Download a SPHEREx Level 2 file and return its HDUList with every extension loaded into memory.

    Files are ~70 MB; astropy caches them, so the same URL only downloads once.
    """
    for attempt in range(MAX_ATTEMPTS):
        try:
            # memmap=False + touching hdu.data reads every extension now, so the HDUList stays usable after closing
            with fits.open(url, memmap=False) as hdulist:
                for hdu in hdulist:
                    hdu.data
            return hdulist
        except (TimeoutError, urllib.error.URLError, http.client.HTTPException) as e:
            short_error = str(e).splitlines()[0] if str(e) else type(e).__name__
            print(f"Download: attempt {attempt + 1}/{MAX_ATTEMPTS} failed: {short_error} Retrying in {WAIT_SECONDS} seconds...")
            time.sleep(WAIT_SECONDS)
    raise RuntimeError("Download: all attempts failed.")


def plot_spherex_image(hdulist, target, cutout_size=CUTOUT_SIZE, subtract_zodi=SUBTRACT_ZODI, title=None):
    """Plot a SPHEREx spectral image: full detector on the left, zoom on the target on the right.

    Returns the matplotlib Figure (a web app can save it to PNG instead of calling plt.show()).
    """
    header = hdulist['IMAGE'].header
    image = hdulist['IMAGE'].data.astype(float)
    if subtract_zodi:
        image = image - hdulist['ZODI'].data

    # sky coordinates (RA/Dec) for each pixel
    wcs = WCS(header)

    # wavelength for each pixel: SPHEREx is a spectral imager, so wavelength changes across the detector.
    # The spectral WCS is stored as alternate WCS "W" and needs the WCS-WAVE lookup table from the HDUList.
    spectral_wcs = WCS(header=header, fobj=hdulist, key='W')
    spectral_wcs.sip = None  # optical distortion doesn't apply to wavelength
    target_x, target_y = wcs.world_to_pixel(target)
    wavelength, bandwidth = spectral_wcs.pixel_to_world(target_x, target_y)

    cutout = Cutout2D(image, target, cutout_size, wcs=wcs)

    unit = header.get('BUNIT', 'MJy/sr')
    fig = plt.figure(figsize=(14, 6.5))

    # left: full detector
    ax_full = fig.add_subplot(1, 2, 1, projection=wcs)
    norm_full = ImageNormalize(image[np.isfinite(image)], interval=ZScaleInterval(), stretch=AsinhStretch())
    im_full = ax_full.imshow(image, origin='lower', cmap='gray', norm=norm_full)
    (x0, x1), (y0, y1) = cutout.bbox_original[1], cutout.bbox_original[0]
    ax_full.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor='red', facecolor='none', lw=1.5))
    ax_full.set_title(f"Detector {header.get('DETECTOR', '?')}")
    ax_full.set_xlabel('RA')
    ax_full.set_ylabel('Dec')
    fig.colorbar(im_full, ax=ax_full, label=unit, fraction=0.046, pad=0.04)

    # right: zoom on the target
    ax_zoom = fig.add_subplot(1, 2, 2, projection=cutout.wcs)
    norm_zoom = ImageNormalize(cutout.data[np.isfinite(cutout.data)], interval=ZScaleInterval(), stretch=AsinhStretch())
    im_zoom = ax_zoom.imshow(cutout.data, origin='lower', cmap='gray', norm=norm_zoom)
    ax_zoom.scatter(target.ra.deg, target.dec.deg, transform=ax_zoom.get_transform('world'),
                    s=300, marker='o', facecolor='none', edgecolor='red', lw=1.5)
    ax_zoom.set_title(f"Target  |  λ = {wavelength.to(u.um).value:.3f} µm  (Δλ = {bandwidth.to(u.um).value:.3f} µm)")
    ax_zoom.set_xlabel('RA')
    ax_zoom.set_ylabel('Dec')
    fig.colorbar(im_zoom, ax=ax_zoom, label=unit, fraction=0.046, pad=0.04)

    zodi_note = 'zodi subtracted' if subtract_zodi else 'zodi NOT subtracted'
    fig.suptitle(f"{title or 'SPHEREx spectral image'}  ({zodi_note})")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Firefly (IRSA web viewer) helpers
# ----------------------------------------------------------------------------

FIREFLY_URL = 'https://irsa.ipac.caltech.edu/irsaviewer'
PAGE_CONNECT_TIMEOUT = 30  # seconds to wait for the browser tab to load and connect


def with_retries(step_name, func, *args, **kwargs):
    """Run one Firefly step, retrying when IRSA's server errors out (e.g. 502 Proxy Error)."""
    for attempt in range(MAX_ATTEMPTS):
        try:
            result = func(*args, **kwargs)
            # Firefly commands report failure as {'success': False} instead of raising
            if isinstance(result, dict) and result.get('success') is False:
                raise ValueError(f"server returned {result}")
            return result
        except (ValueError, requests.RequestException) as e:
            short_error = str(e).splitlines()[0] if str(e) else type(e).__name__
            print(f"{step_name}: attempt {attempt + 1}/{MAX_ATTEMPTS} failed: {short_error} Retrying in {WAIT_SECONDS} seconds...")
            time.sleep(WAIT_SECONDS)
    raise RuntimeError(f"{step_name}: all attempts failed.")


def page_connected(fc):
    """True once the Firefly browser tab has loaded and connected back to the server."""
    try:
        return fc._is_page_connected()
    except (ValueError, requests.RequestException):
        return False


def open_firefly_viewer():
    """Connect to the IRSA Firefly server and make sure a browser tab is open and listening."""
    # launch_browser=False: we open the tab ourselves below so we can check that it really connected
    fc = with_retries("Connect to Firefly", FireflyClient.make_client, url=FIREFLY_URL, launch_browser=False)

    for attempt in range(MAX_ATTEMPTS):
        print(f"Opening Firefly in your browser (attempt {attempt + 1}/{MAX_ATTEMPTS})...")
        fc.launch_browser(force=True)

        # wait for the tab to load; if IRSA served it an error page, it never connects
        deadline = time.time() + PAGE_CONNECT_TIMEOUT
        while time.time() < deadline:
            if page_connected(fc):
                print("Firefly viewer connected.")
                return fc
            time.sleep(1)

        print(f"Browser tab didn't connect within {PAGE_CONNECT_TIMEOUT} seconds, opening a new one...")

    raise RuntimeError("Could not get the Firefly viewer to open in the browser.")


def show_in_firefly(results, image_url, target):
    """Show the results table and one image in IRSA's Firefly web viewer."""
    fc = open_firefly_viewer()
    with_retries("Reset viewer", fc.reinit_viewer)

    # show the list of matching SPHEREx images as a table
    results_file = os.path.join(tempfile.gettempdir(), 'spherex_results.xml')
    results.write(results_file, format='votable', overwrite=True)
    with_retries("Show results table", fc.show_table, file_input=results_file, title='SPHEREx images')

    # Each file is a multi-extension FITS (MEF):
    #   0 PRIMARY (header only), 1 IMAGE (MJy/sr), 2 FLAGS, 3 VARIANCE, 4 ZODI, 5 EPSF (table), 6 WCS-WAVE (table)
    # By default Firefly shows every image extension; multiImageIdx picks just one.
    # It counts only the extensions that hold image data (not the empty PRIMARY), so IMAGE is 0, FLAGS 1, VARIANCE 2, ZODI 3.
    with_retries("Show SPHEREx image", fc.show_fits_image, file_input=image_url, plot_id='spherex_image',
                 multiImageIdx=0, title='SPHEREx IMAGE (MJy/sr)')
    with_retries("Center on target", fc.set_pan, 'spherex_image', x=target.ra.deg, y=target.dec.deg, coord='J2000')


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

## to define coordinates of interest
ra_deg = 304.693508808
dec_deg = 42.4436872991

coord = SkyCoord(ra_deg, dec_deg, unit='deg')
search_radius = 1 * u.arcsec

for attempt in range(MAX_ATTEMPTS):
    try:
        results = Irsa.query_sia(pos=(coord, search_radius), collection=['spherex_dr1', 'spherex_qr3'])

        break

    except (DALServiceError, http.client.HTTPException, urllib.error.URLError) as e:
        short_error = str(e).splitlines()[0]
        print(f"Attempt {attempt + 1}/{MAX_ATTEMPTS} failed: {short_error} Retrying in {WAIT_SECONDS} seconds...")
        time.sleep(WAIT_SECONDS)

else:
    raise RuntimeError("All attempts to query IRSA failed.")

print(len(results))

spherex_image_url = results['access_url'][0]
print(f"Downloading {spherex_image_url}")
hdulist = download_spherex_image(spherex_image_url)

fig = plot_spherex_image(hdulist, coord, title=results['obs_id'][0])
plt.show()

if SHOW_IN_FIREFLY:
    show_in_firefly(results, spherex_image_url, coord)
