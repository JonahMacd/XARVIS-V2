"""Fit and plot an object's apparent motion across SPHEREx frames."""

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from matplotlib.patches import Ellipse

from . import config as cfg


def _observations(frames: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, SkyCoord]:
    """Return usable times and tangent-plane offsets, plus their original-frame indices and origin."""
    if not frames:
        raise ValueError("frames must contain at least one measured position")

    times = np.asarray([float(frame["mjd"]) for frame in frames], dtype=float)
    ra = np.asarray([float(frame["object_ra"]) for frame in frames], dtype=float)
    dec = np.asarray([float(frame["object_dec"]) for frame in frames], dtype=float)
    reliable = np.asarray([frame.get("object_found", True) and frame.get("n_bad_in_aperture", 0) == 0
                           for frame in frames], dtype=bool)
    valid = reliable & np.isfinite(times) & np.isfinite(ra) & np.isfinite(dec)
    indices = np.flatnonzero(valid)
    if not len(indices):
        raise ValueError("frames contain no reliable finite positions")

    origin_index = indices[np.argmin(times[indices])]
    origin = SkyCoord(ra[origin_index], dec[origin_index], unit="deg")
    coords = SkyCoord(ra[indices], dec[indices], unit="deg")
    east, north = origin.spherical_offsets_to(coords)
    offsets = np.full((len(frames), 2), np.nan, dtype=float)
    offsets[indices, 0] = east.to_value("arcsec")
    offsets[indices, 1] = north.to_value("arcsec")
    return times, offsets, indices, origin


def get_pos_and_speed(frames: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Get per-frame sky offsets and angular velocities.

    Positions are east/north offsets in arcseconds from the first reliable observation.
    Speeds are east/north angular rates in arcseconds per day, estimated by finite
    differences between distinct timestamps. Simultaneous frames share one rate.
    Unusable frames have NaN rows; physical distances and linear speeds are not
    available from sky positions alone.
    """
    times, positions, indices, _ = _observations(frames)
    speeds = np.full_like(positions, np.nan)

    valid_times = times[indices]
    unique_times, epoch_index = np.unique(valid_times, return_inverse=True)
    if len(unique_times) < 2:
        return positions, speeds

    epoch_positions = np.zeros((len(unique_times), 2), dtype=float)
    epoch_counts = np.bincount(epoch_index)
    np.add.at(epoch_positions, epoch_index, positions[indices])
    epoch_positions /= epoch_counts[:, None]
    epoch_speeds = np.gradient(epoch_positions, unique_times, axis=0,
                               edge_order=2 if len(unique_times) >= 3 else 1)
    speeds[indices] = epoch_speeds[epoch_index]
    return positions, speeds


def _position_uncertainties(frames: Sequence[Mapping[str, Any]], indices: np.ndarray) -> np.ndarray:
    """Combine centroid and available alignment errors, filling missing errors conservatively."""
    errors = np.asarray([float(frames[i].get("object_err_arcsec", np.nan)) for i in indices])
    alignment = np.asarray([float(frames[i].get("alignment_px", np.nan)) for i in indices])
    alignment = np.where(np.isfinite(alignment) & (alignment >= 0),
                         alignment * cfg.PIXEL_ARCSEC, 0.0)
    sigma = np.hypot(errors, alignment)
    known = np.isfinite(sigma) & (sigma > 0)
    if known.any():
        sigma[~known] = np.median(sigma[known])
    else:
        sigma[:] = 1.0
    return sigma


def orbital_prediction(
    frames: Sequence[Mapping[str, Any]] | None = None,
    prediction_days: float = 0.0,
    show_plot: bool = True,
) -> dict[str, Any]:
    """Fit and plot the measured angular trajectory, optionally extrapolating in time.

    This is a weighted polynomial fit to apparent east/north motion, not a physical
    Keplerian orbit. SPHEREx frames provide direction and time but not distance or
    observer ephemerides, so they cannot determine a six-dimensional orbital state
    or support Gauss-Radau/JPL ephemeris propagation by themselves. A quadratic
    angular model is used when at least three distinct epochs are available; with
    two epochs the fit is linear. Extrapolation is disabled by default.

    Returns the observations, fitted/predicted positions, rates, covariance, and
    Matplotlib figure. All angular quantities are in arcseconds (rates per day).
    """
    if frames is None:
        raise ValueError("frames are required to fit an apparent trajectory")
    prediction_days = float(prediction_days)
    if not np.isfinite(prediction_days) or prediction_days < 0:
        raise ValueError("prediction_days must be a finite non-negative number")

    times, all_positions, indices, origin = _observations(frames)
    fit_times = times[indices]
    positions = all_positions[indices]
    unique_times = np.unique(fit_times)
    if len(unique_times) < 2:
        raise ValueError("at least two distinct observation times are required to fit motion")

    degree = min(2, len(unique_times) - 1)
    n_coefficients = degree + 1
    sigma = _position_uncertainties(frames, indices)
    center_time = float(np.average(fit_times, weights=1 / sigma**2))
    time_scale = float(np.max(np.abs(fit_times - center_time)))
    if time_scale == 0:
        raise ValueError("observation times must span a non-zero interval")

    scaled_time = (fit_times - center_time) / time_scale
    design = np.column_stack((np.ones(len(indices)), scaled_time, 0.5 * scaled_time**2))[:, :n_coefficients]
    weighted_design = design / sigma[:, None]
    weighted_positions = positions / sigma[:, None]
    coefficients, _, rank, _ = np.linalg.lstsq(weighted_design, weighted_positions, rcond=None)
    if rank < n_coefficients:
        raise ValueError("observations do not constrain the requested motion model")

    residuals = positions - design @ coefficients
    covariance_base = np.linalg.inv(weighted_design.T @ weighted_design)
    degrees_of_freedom = max(len(indices) - n_coefficients, 1)
    reduced_chi_squared = np.sum((residuals / sigma[:, None]) ** 2, axis=0) / degrees_of_freedom
    covariance_axes = [covariance_base * max(1.0, float(scale)) for scale in reduced_chi_squared]
    covariance = np.zeros((2 * n_coefficients, 2 * n_coefficients), dtype=float)
    covariance[:n_coefficients, :n_coefficients] = covariance_axes[0]
    covariance[n_coefficients:, n_coefficients:] = covariance_axes[1]

    velocity = coefficients[1] / time_scale
    acceleration = (coefficients[2] / time_scale**2 if degree == 2
                    else np.full(2, np.nan, dtype=float))

    start_time, last_time = float(np.min(fit_times)), float(np.max(fit_times))
    end_time = last_time + prediction_days
    sample_count = max(100, 4 * len(indices))
    prediction_times = np.linspace(start_time, end_time, sample_count)
    prediction_scaled_time = (prediction_times - center_time) / time_scale
    prediction_design = np.column_stack((np.ones(sample_count), prediction_scaled_time,
                                         0.5 * prediction_scaled_time**2))[:, :n_coefficients]
    predicted_positions = prediction_design @ coefficients

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(positions[:, 0], positions[:, 1], s=24, color="black", label="observations", zorder=3)
    observed_path = prediction_times <= last_time
    ax.plot(predicted_positions[observed_path, 0], predicted_positions[observed_path, 1],
            color="tab:blue", label="weighted fit")
    if prediction_days > 0:
        ax.plot(predicted_positions[~observed_path, 0], predicted_positions[~observed_path, 1],
                color="tab:orange", linestyle="--", label="extrapolation")

    endpoint_design = prediction_design[-1]
    endpoint_east_var = max(float(endpoint_design @ covariance_axes[0] @ endpoint_design), 0.0)
    endpoint_north_var = max(float(endpoint_design @ covariance_axes[1] @ endpoint_design), 0.0)
    if endpoint_east_var > 0 and endpoint_north_var > 0:
        ax.add_patch(Ellipse(predicted_positions[-1], 2 * np.sqrt(endpoint_east_var),
                             2 * np.sqrt(endpoint_north_var), facecolor="tab:orange",
                             edgecolor="tab:orange", alpha=0.2, label="1-sigma endpoint uncertainty"))

    ax.set_xlabel("East offset (arcsec; east to the left)")
    ax.set_ylabel("North offset (arcsec)")
    ax.set_title("Apparent sky-track fit (not a physical orbit)")
    ax.invert_xaxis()
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    if show_plot:
        plt.show()

    return {
        "reference_coord": origin,
        "observation_times_mjd": fit_times,
        "observed_positions_arcsec": positions,
        "predicted_times_mjd": prediction_times,
        "predicted_positions_arcsec": predicted_positions,
        "residuals_arcsec": residuals,
        "velocity_arcsec_per_day": velocity,
        "acceleration_arcsec_per_day2": acceleration,
        "covariance": covariance,
        "fit_degree": degree,
        "prediction_days": prediction_days,
        "figure": fig,
    }
