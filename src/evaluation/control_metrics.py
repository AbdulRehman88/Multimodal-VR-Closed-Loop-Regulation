"""Standardized metrics for closed-loop controller evaluation."""

import numpy as np


def evaluate_controller(run, hold_duration_s=1.0,
                        abrupt_threshold=0.05):
    """Evaluate a controller trajectory with fixed metric definitions."""

    state = np.asarray(run["state"], dtype=np.float64)
    stimulus = np.asarray(run["stimulus"], dtype=np.float64)

    fs = int(run["sampling_rate_hz"])
    fc = int(run["control_rate_hz"])

    if fs <= 0 or fc <= 0 or fs % fc != 0:
        raise ValueError("Invalid sampling/control frequencies.")

    if state.ndim != 1 or stimulus.shape != state.shape:
        raise ValueError("State and stimulus dimensions must match.")

    if len(state) < 2 or not np.isfinite(state).all():
        raise ValueError("Invalid state trajectory.")

    if not np.isfinite(stimulus).all():
        raise ValueError("Invalid stimulus trajectory.")

    if np.any((state < 0) | (state > 10)):
        raise ValueError("State outside [0, 10].")

    if np.any((stimulus < 0) | (stimulus > 1)):
        raise ValueError("Stimulus outside [0, 1].")

    if not np.isfinite(hold_duration_s) or hold_duration_s <= 0:
        raise ValueError("Invalid recovery hold duration.")

    if not np.isfinite(abrupt_threshold) or abrupt_threshold <= 0:
        raise ValueError("Invalid abrupt transition threshold.")

    lower, upper = 4.0, 6.0
    inside = (state >= lower) & (state <= upper)

    hold_samples = int(np.ceil(hold_duration_s * fs))

    # Earliest entry followed by continuous residence.
    recovery_index = None

    if hold_samples <= len(state):
        runs = np.convolve(
            inside.astype(np.int32),
            np.ones(hold_samples, dtype=np.int32),
            mode="valid"
        )
        valid = np.flatnonzero(runs == hold_samples)
        if valid.size:
            recovery_index = int(valid[0])

    recovery_time = (
        float(recovery_index / fs)
        if recovery_index is not None
        else float("nan")
    )

    state_stability = float(100.0 * inside.mean())

    # Overshoot: crossing the opposite state boundary.
    initial = float(state[0])

    if initial < lower:
        overshoot = float(max(0.0, np.max(state - upper)))
    elif initial > upper:
        overshoot = float(max(0.0, np.max(lower - state)))
    else:
        overshoot = float("nan")

    # Analyze changes at actual controller-update instants.
    interval = fs // fc
    updates = stimulus[::interval]

    # Include the first adjustment from the initial stimulus.
    initial_stimulus = 0.5
    changes = np.abs(
        np.diff(np.concatenate(([initial_stimulus], updates)))
    )

    duration_s = len(state) / fs

    total_variation = float(changes.sum())
    variation_per_s = total_variation / duration_s
    mean_update_change = (
        float(changes.mean()) if changes.size else 0.0
    )

    abrupt_transitions = int(
        np.count_nonzero(changes > abrupt_threshold + 1e-12)
    )

    return {
        "recovery_time_s": recovery_time,
        "recovered": recovery_index is not None,
        "state_stability_pct": state_stability,
        "opposite_boundary_overshoot": overshoot,
        "stimulus_total_variation": total_variation,
        "stimulus_variation_per_s": variation_per_s,
        "mean_stimulus_update_change": mean_update_change,
        "abrupt_transitions": abrupt_transitions,
        "abrupt_threshold": float(abrupt_threshold),
    }
