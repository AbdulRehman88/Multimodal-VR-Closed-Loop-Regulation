"""Reproducible baseline controllers for simulated VR adaptation."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ControllerParameters:
    lower_boundary: float = 4.0
    upper_boundary: float = 6.0
    neutral_stimulus: float = 0.5
    increment: float = 0.05
    decrement: float = 0.10
    smoothing: float = 0.80
    proportional_gain: float = 0.20
    minimum_stimulus: float = 0.0
    maximum_stimulus: float = 1.0

    def __post_init__(self):
        values = vars(self)
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError("Controller parameters must be finite.")
        if not self.lower_boundary < self.upper_boundary:
            raise ValueError("Invalid state boundaries.")
        if not 0 <= self.minimum_stimulus < self.maximum_stimulus <= 1:
            raise ValueError("Invalid stimulus limits.")
        if not self.minimum_stimulus <= self.neutral_stimulus <= self.maximum_stimulus:
            raise ValueError("Neutral stimulus outside limits.")
        if self.increment < 0 or self.decrement < 0:
            raise ValueError("Controller steps must be nonnegative.")
        if self.proportional_gain < 0:
            raise ValueError("Proportional gain must be nonnegative.")
        if not 0 <= self.smoothing < 1:
            raise ValueError("Smoothing must be in [0, 1).")


def neutral_controller(state, current_stimulus, params):
    """Non-adaptive reference controller."""
    return params.neutral_stimulus


def threshold_controller(state, current_stimulus, params):
    """Reactive threshold controller without smoothing."""
    if not math.isfinite(state) or not math.isfinite(current_stimulus):
        raise ValueError("Controller inputs must be finite.")
    if not params.minimum_stimulus <= current_stimulus <= params.maximum_stimulus:
        raise ValueError("Current stimulus outside limits.")

    if state < params.lower_boundary:
        target = current_stimulus + params.increment
    elif state > params.upper_boundary:
        target = current_stimulus - params.decrement
    else:
        target = current_stimulus

    return float(min(params.maximum_stimulus,
                     max(params.minimum_stimulus, target)))


def smoothed_threshold_controller(state, current_stimulus, params):
    """Threshold controller with explicit output smoothing."""
    target = threshold_controller(state, current_stimulus, params)

    output = (
        params.smoothing * current_stimulus
        + (1.0 - params.smoothing) * target
    )

    return float(min(params.maximum_stimulus,
                     max(params.minimum_stimulus, output)))


def proportional_controller(
    state,
    current_stimulus,
    params,
    gain=None,
    target_state=5.0,
):
    """Proportional feedback controller with bounded stimulus."""

    if gain is None:
        gain = params.proportional_gain

    if not all(math.isfinite(v) for v in
               (state, current_stimulus, gain, target_state)):
        raise ValueError("Controller inputs must be finite.")

    if gain < 0:
        raise ValueError("Proportional gain must be nonnegative.")

    if not params.minimum_stimulus <= current_stimulus <= params.maximum_stimulus:
        raise ValueError("Current stimulus outside limits.")

    target = (
        current_stimulus
        - gain * (state - target_state)
    )

    return float(min(
        params.maximum_stimulus,
        max(params.minimum_stimulus, target)
    ))
