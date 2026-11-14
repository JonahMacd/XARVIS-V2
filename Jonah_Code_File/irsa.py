"""Talking to IRSA: timeouts, retries, SPHEREx image search, catalogs (Gator) and cutouts.

IRSA is unreliable at times: it answers "502 Proxy Error", or accepts a connection and never replies.
Every request here has a timeout and is retried, so neither can stop the program.
"""

import io
import threading
import time

import requests
from requests.adapters import HTTPAdapter

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astroquery.ipac.irsa import Irsa
from pyvo.dal.exceptions import DALAccessError     # base of all pyvo errors: 502s, timeouts, bad replies

from . import config as cfg


class _TimeoutAdapter(HTTPAdapter):
    """Gives every request a timeout. astroquery/pyvo send theirs with none, so a stalled IRSA reply
    would make Python wait forever; with a timeout it fails and gets retried instead."""
    def send(self, request, **kwargs):
        if kwargs.get('timeout') is None:
            kwargs['timeout'] = cfg.REQUEST_TIMEOUT
        return super().send(request, **kwargs)


Irsa._session.mount('https://', _TimeoutAdapter())     # used by Irsa.query_sia
Irsa._session.mount('http://', _TimeoutAdapter())

_thread_local = threading.local()
_download_slots = threading.BoundedSemaphore(cfg.MAX_CONCURRENT_DOWNLOADS)


def _session():
    """One reusable connection pool per thread (reusing connections is much faster than reconnecting)."""
    if not hasattr(_thread_local, 'session'):
        session = requests.Session()
        session.mount('https://', _TimeoutAdapter(pool_maxsize=cfg.MAX_CONCURRENT_DOWNLOADS))
        _thread_local.session = session
    return _thread_local.session


class BadResponse(Exception):
    """IRSA answered, but not with what we asked for (e.g. an error page)."""


RETRY_ERRORS = (DALAccessError, requests.RequestException, OSError, BadResponse)


def with_retries(label, func, *args, quiet=False, **kwargs):
    """Call func(*args, **kwargs), retrying on IRSA errors up to MAX_ATTEMPTS times."""
    for attempt in range(cfg.MAX_ATTEMPTS):
        try:
            return func(*args, **kwargs)
        except RETRY_ERRORS as e:
            if not quiet:
                message = str(e).splitlines()[0] if str(e) else type(e).__name__
                print(f'{label}: attempt {attempt + 1}/{cfg.MAX_ATTEMPTS} failed: {message[:120]}')
            time.sleep(cfg.WAIT_SECONDS)
    raise RuntimeError(f'{label}: all {cfg.MAX_ATTEMPTS} attempts to reach IRSA failed.')


def query_images(coord):
    """SPHEREx images covering `coord` (astropy Table), sorted by time."""
    results = with_retries('SPHEREx image search', Irsa.query_sia,
                           pos=(coord, 1 * u.arcsec), collection=cfg.COLLECTIONS)
    results.sort('t_min')
    return results


CATALOG_COLUMNS = {'fp_psc': 'designation,ra,dec,j_m,j_msigcom,h_m,h_msigcom,k_m,k_msigcom',
                   'allwise_p3as_psd': 'designation,ra,dec,w1mpro,w1sigmpro,w2mpro,w2sigmpro'}


def query_catalog(coord, catalog, radius):
    """Catalog sources within `radius` (astropy Table), sorted by distance, with a 'dist_arcsec' column.

    Uses IRSA's Gator service rather than astroquery's Irsa.query_region (IRSA's TAP service):
    same catalogs, but TAP has been failing most requests while Gator answers reliably.
    """
    params = {'catalog': catalog, 'spatial': 'cone', 'objstr': f'{coord.ra.deg} {coord.dec.deg}',
              'radius': radius.to_value(u.arcsec), 'radunits': 'arcsec', 'outfmt': 1,     # 1 = IPAC table
              'selcols': CATALOG_COLUMNS[catalog]}

    def once():
        response = _session().get(cfg.GATOR_URL, params=params)
        if response.status_code != 200 or not response.text.lstrip().startswith('\\'):  # IPAC tables start with '\'
            raise BadResponse(f'HTTP {response.status_code}')
        return Table.read(response.text, format='ipac')

    table = with_retries(f'Catalog {catalog}', once)
    table['dist_arcsec'] = SkyCoord(table['ra'], table['dec'], unit='deg').separation(coord).arcsec
    table.sort('dist_arcsec')
    return table


def fetch_cutout(image_url, coord, size):
    """Small cutout (HDUList) of one SPHEREx image around `coord`, cut server-side by IRSA.
    Kept in memory only: nothing is written to disk."""
    params = {'center': f'{coord.ra.deg},{coord.dec.deg}', 'size': f'{size.to_value(u.arcsec)}arcsec'}

    def once():
        with _download_slots:                       # at most MAX_CONCURRENT_DOWNLOADS at a time, program-wide
            response = _session().get(image_url, params=params)
        if response.status_code != 200:
            raise BadResponse(f'HTTP {response.status_code}')
        return response.content

    content = with_retries('Cutout', once, quiet=True)
    with fits.open(io.BytesIO(content), memmap=False) as full:
        hdulist = fits.HDUList([fits.PrimaryHDU(header=full[0].header)])
        for name in ['IMAGE', 'FLAGS', 'VARIANCE', 'ZODI', 'WCS-WAVE']:      # drop the large EPSF table
            hdu = full[name]
            if isinstance(hdu, fits.CompImageHDU):          # FLAGS is compressed; keep it as a plain image
                hdu = fits.ImageHDU(data=hdu.data, header=hdu.header, name=name)
            hdulist.append(hdu.copy())
    return hdulist
