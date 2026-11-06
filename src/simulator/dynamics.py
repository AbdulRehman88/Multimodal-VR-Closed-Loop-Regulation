"""Causal latent-state dynamics for the Multimodal-VR simulator."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class StateParameters:
    baseline: float
    relaxation_rate: float
    stimulus_gain: float
    reference_stimulus: float = 0.5
    minimum: float = 0.0
    maximum: float = 10.0

    def __post_init__(self):
        values = vars(self)
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError("State parameters must be finite.")
        if not self.minimum < self.maximum:
            raise ValueError("Invalid state bounds.")
        if not self.minimum <= self.baseline <= self.maximum:
            raise ValueError("Baseline outside state bounds.")
        if self.relaxation_rate < 0:
            raise ValueError("Relaxation rate must be nonnegative.")
        if self.stimulus_gain < 0:
            raise ValueError("Stimulus gain must be nonnegative.")
        if not 0 <= self.reference_stimulus <= 1:
            raise ValueError("Reference stimulus must be in [0, 1].")


def state_step(
    state: float,
    stimulus: float,
    dt: float,
    params: StateParameters,
    disturbance: float = 0.0,
) -> float:
    """One forward-time Euler step; disturbance is an external input."""

    if not all(math.isfinite(v) for v in
               (state, stimulus, dt, disturbance)):
        raise ValueError("Inputs must be finite.")

    if dt <= 0:
        raise ValueError("Time step must be positive.")

    if not params.minimum <= state <= params.maximum:
        raise ValueError("State outside configured bounds.")

    if not 0 <= stimulus <= 1:
        raise ValueError("Stimulus must be in [0, 1].")

    derivative = (
        -params.relaxation_rate * (state - params.baseline)
        + params.stimulus_gain
        * (stimulus - params.reference_stimulus)
        + disturbance
    )

    next_state = state + dt * derivative

    return float(min(params.maximum,
                     max(params.minimum, next_state)))
