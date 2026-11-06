"""Causal, state-dependent 3D motion generation."""

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class MotionParameters:
    frequency_hz: float = 0.3
    damping: float = 1.5
    baseline_noise: float = 0.02
    state_sensitivity: float = 0.08
    reference_state: float = 5.0
    maximum_speed: float = 2.0

    def __post_init__(self):
        if not all(math.isfinite(v) for v in vars(self).values()):
            raise ValueError("Motion parameters must be finite.")
        if self.frequency_hz <= 0 or self.damping < 0:
            raise ValueError("Invalid frequency or damping.")
        if self.baseline_noise < 0 or self.state_sensitivity < 0:
            raise ValueError("Noise parameters must be nonnegative.")
        if self.maximum_speed <= 0:
            raise ValueError("Maximum speed must be positive.")


def motion_step(position, velocity, state, dt, params, rng):
    """Advance motion one step using only current state and past motion."""

    position = np.asarray(position, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)

    if position.shape != (3,) or velocity.shape != (3,):
        raise ValueError("Position and velocity must be 3D.")

    if not np.isfinite(position).all() or not np.isfinite(velocity).all():
        raise ValueError("Motion inputs must be finite.")

    if not math.isfinite(state) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid state or time step.")

    omega = 2.0 * np.pi * params.frequency_hz

    deviation = state - params.reference_state

    noise_scale = (
        params.baseline_noise
        + params.state_sensitivity * max(deviation, 0.0)
        + 0.35 * params.state_sensitivity * max(-deviation, 0.0)
    )

    acceleration = (
        -(omega ** 2) * position
        - params.damping * velocity
        + noise_scale * rng.normal(size=3)
    )

    next_velocity = velocity + dt * acceleration

    speed = float(np.linalg.norm(next_velocity))
    if speed > params.maximum_speed:
        next_velocity *= params.maximum_speed / speed

    next_position = position + dt * next_velocity

    return next_position, next_velocity
