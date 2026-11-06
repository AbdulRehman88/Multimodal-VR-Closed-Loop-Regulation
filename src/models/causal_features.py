"""Causal motion-only features for latent-state estimation."""

import numpy as np


FEATURE_NAMES = (
    "speed_mean",
    "speed_std",
    "speed_max",
    "speed_q90",
    "speed_rms",
    "acceleration_mean",
    "acceleration_std",
    "acceleration_q90",
    "velocity_x_std",
    "velocity_y_std",
    "velocity_z_std",
    "position_x_std",
    "position_y_std",
    "position_z_std",
)


def extract_causal_features(
    position,
    velocity,
    sampling_rate_hz=90,
    window_duration_s=1.0,
    update_rate_hz=10,
):
    """Extract trailing-window motion features at causal update times.

    Each window includes observations no later than its timestamp.
    Windows are required to be complete; no future padding is used.
    """

    position = np.asarray(position, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)

    fs = int(sampling_rate_hz)
    fc = int(update_rate_hz)

    if fs <= 0 or fc <= 0 or fs % fc != 0:
        raise ValueError("Invalid sampling or update rate.")

    if not np.isfinite(window_duration_s) or window_duration_s <= 0:
        raise ValueError("Invalid window duration.")

    if (
        position.ndim != 2
        or position.shape[1] != 3
        or velocity.shape != position.shape
        or not np.isfinite(position).all()
        or not np.isfinite(velocity).all()
    ):
        raise ValueError("Invalid motion observations.")

    window = int(round(fs * window_duration_s))
    interval = fs // fc

    if window < 2 or len(position) < window:
        raise ValueError("Insufficient observations for a full window.")

    indices = np.arange(window - 1, len(position), interval)
    features = []

    for end in indices:
        start = end - window + 1

        p = position[start:end + 1]
        v = velocity[start:end + 1]

        speed = np.linalg.norm(v, axis=1)
        acceleration = np.linalg.norm(
            np.diff(v, axis=0) * fs,
            axis=1,
        )

        vector = [
            speed.mean(),
            speed.std(),
            speed.max(),
            np.quantile(speed, 0.90),
            np.sqrt(np.mean(speed ** 2)),
            acceleration.mean(),
            acceleration.std(),
            np.quantile(acceleration, 0.90),
            *v.std(axis=0),
            *p.std(axis=0),
        ]

        features.append(vector)

    result = np.asarray(features, dtype=np.float64)

    assert result.shape == (len(indices), len(FEATURE_NAMES))
    assert np.isfinite(result).all()

    return {
        "features": result,
        "sample_indices": indices,
        "feature_names": FEATURE_NAMES,
        "window_samples": window,
        "update_interval_samples": interval,
        "observation_source": "motion_only",
    }
