"""Causal coupled latent-state and 3D motion simulation."""

from dataclasses import dataclass
import numpy as np

from .dynamics import StateParameters, state_step
from .motion import MotionParameters, motion_step


@dataclass(frozen=True)
class CoupledParameters:
    sampling_rate_hz: int = 90
    duration_s: float = 12.0
    state_disturbance_sd: float = 0.05

    def __post_init__(self):
        if self.sampling_rate_hz <= 0 or self.duration_s <= 0:
            raise ValueError("Invalid sampling rate or duration.")
        if self.state_disturbance_sd < 0:
            raise ValueError("Invalid disturbance standard deviation.")


def simulate_episode(
    seed: int,
    state_params: StateParameters,
    motion_params: MotionParameters,
    coupled_params: CoupledParameters,
    initial_state: float = 5.0,
    stimulus=None,
):
    """Generate an episode using only current and previous states."""

    n = round(
        coupled_params.sampling_rate_hz * coupled_params.duration_s
    )
    dt = 1.0 / coupled_params.sampling_rate_hz

    if stimulus is None:
        stimulus = np.full(n, 0.5, dtype=np.float64)
    else:
        stimulus = np.asarray(stimulus, dtype=np.float64)

    if stimulus.shape != (n,):
        raise ValueError("Stimulus must contain exactly n samples.")

    if not np.isfinite(stimulus).all():
        raise ValueError("Stimulus contains non-finite values.")

    if np.any((stimulus < 0) | (stimulus > 1)):
        raise ValueError("Stimulus outside [0, 1].")

    seed_sequence = np.random.SeedSequence(seed)
    state_seed, motion_seed = seed_sequence.spawn(2)

    state_rng = np.random.default_rng(state_seed)
    motion_rng = np.random.default_rng(motion_seed)

    states = np.empty(n, dtype=np.float64)
    positions = np.empty((n, 3), dtype=np.float64)
    velocities = np.empty((n, 3), dtype=np.float64)

    state = float(initial_state)
    position = np.zeros(3, dtype=np.float64)
    velocity = np.zeros(3, dtype=np.float64)

    for t in range(n):
        states[t] = state
        positions[t] = position
        velocities[t] = velocity

        position, velocity = motion_step(
            position, velocity, state, dt,
            motion_params, motion_rng
        )

        disturbance = float(
            state_rng.normal(0, coupled_params.state_disturbance_sd)
        )

        state = state_step(
            state, float(stimulus[t]), dt,
            state_params, disturbance
        )

    return {
        "state": states,
        "position": positions,
        "velocity": velocities,
        "stimulus": stimulus.copy(),
        "sampling_rate_hz": coupled_params.sampling_rate_hz,
    }
