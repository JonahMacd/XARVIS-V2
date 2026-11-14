"""Save / load the measurements of a run as a small table (one row per frame, ~100-200 KB per target).

Only measurements are saved - no pixels (those are in the disk cache while they're still useful, so re-running
the analysis is fast). One file per target, replaced on every run, so results never pile up.
Open it with load_results(), or with any program that reads CSV-like text (it's an ECSV file).
"""

import re
from datetime import datetime
from pathlib import Path

import numpy as np
from astropy.table import Table

from . import config as cfg

# per-frame measurements worth keeping (only the ones a frame actually has are saved)
FRAME_COLUMNS = ['obs_id', 'date_obs', 'mjd', 'detector', 'wavelength_um', 'flux_mjy', 'flux_err_mjy', 'snr',
                 'n_bad_in_aperture', 'object_ra', 'object_dec', 'object_err_arcsec', 'expected_ra', 'expected_dec',
                 'alignment_px', 'rotation_deg']


def results_dir():
    return Path(cfg.RESULTS_DIR) if cfg.RESULTS_DIR else Path(__file__).resolve().parent.parent / 'results'


def results_path(name):
    return results_dir() / (re.sub(r'[^A-Za-z0-9_+-]+', '_', name).strip('_') + '.ecsv')


def save_results(results, settings):
    """Write one row per frame: the frame's measurements plus its movement / brightness results.
    `settings` = what was run (target, window ...), stored in the file's header. Returns the path."""
    frames, n = results['frames'], len(results['frames'])
    table = Table()
    table['T'] = np.arange(1, n + 1)
    for column in FRAME_COLUMNS:
        if column in frames[0]:
            table[column] = [f[column] for f in frames]
    for prefix in ('movement', 'brightness'):
        for name, values in (results.get(prefix) or {}).items():
            if isinstance(values, np.ndarray) and values.shape == (n,):    # per-frame arrays only
                table[name if prefix == 'movement' and name != 'flagged' else f'{prefix}_{name}'] = values

    table.meta.update(target=results['tracked']['name'], saved=datetime.now().isoformat(timespec='seconds'),
                      **{k: str(v) for k, v in settings.items()})
    path = results_path(results['tracked']['name'])
    path.parent.mkdir(parents=True, exist_ok=True)
    table.write(path, format='ascii.ecsv', overwrite=True)
    print(f'Results saved: {path} ({path.stat().st_size / 1e3:.0f} KB, {n} frames)')
    return path


def load_results(name_or_path):
    """Load a saved results table, by target name (e.g. '433 Eros') or file path."""
    path = Path(name_or_path)
    return Table.read(path if path.suffix == '.ecsv' else results_path(str(name_or_path)), format='ascii.ecsv')
