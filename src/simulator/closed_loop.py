"""Causal closed-loop simulation with matched disturbances."""

import numpy as np

from .dynamics import StateParameters, state_step
from .motion import MotionParameters, motion_step
from .coupled import CoupledParameters
from src.evaluation.resource_matching import (
    ResourceBudget, bounded_control_step,
)
from src.controllers.baselines import (
    ControllerParameters,
    neutral_controller,
    threshold_controller,
    smoothed_threshold_controller,
    proportional_controller,
)

CONTROLLERS = {
    "neutral": neutral_controller,
    "threshold": threshold_controller,
    "smoothed_threshold": smoothed_threshold_controller,
    "proportional": proportional_controller,
}


def simulate_closed_loop(
    seed,
    controller_name,
    state_params,
    motion_params,
    coupled_params,
    controller_params,
    initial_state=5.0,
    control_rate_hz=10,
    resource_budget=None,
    maximum_update_change=None,
):
    """Simulate one episode with causal oracle-state feedback."""

    if controller_name not in CONTROLLERS:
        raise ValueError("Unknown controller.")

    fs = coupled_params.sampling_rate_hz
    if not isinstance(control_rate_hz, int) or control_rate_hz <= 0:
        raise ValueError("Invalid control rate.")
    if fs % control_rate_hz != 0:
        raise ValueError("Control rate must divide sampling rate.")

    interval = fs // control_rate_hz
    n = round(fs * coupled_params.duration_s)
    dt = 1.0 / fs

    seed_sequence = np.random.SeedSequence(seed)
    state_seed, motion_seed = seed_sequence.spawn(2)

    state_rng = np.random.default_rng(state_seed)
    motion_rng = np.random.default_rng(motion_seed)

    # Pre-generated disturbances ensure matched noise across controllers.
    state_noise = state_rng.normal(
        0.0, coupled_params.state_disturbance_sd, size=n
    )
    motion_noise = motion_rng.normal(size=(n, 3))

    states = np.empty(n, dtype=np.float64)
    positions = np.empty((n, 3), dtype=np.float64)
    velocities = np.empty((n, 3), dtype=np.float64)
    stimuli = np.empty(n, dtype=np.float64)

    state = float(initial_state)
    position = np.zeros(3, dtype=np.float64)
    velocity = np.zeros(3, dtype=np.float64)
    stimulus = controller_params.neutral_stimulus

    controller = CONTROLLERS[controller_name]

    if maximum_update_change is not None:
        if resource_budget is not None:
            raise ValueError(
                "Specify either resource_budget or maximum_update_change."
            )
        if (
            not np.isfinite(maximum_update_change)
            or not 0 < maximum_update_change <= 1
        ):
            raise ValueError("Invalid maximum_update_change.")

    if resource_budget is not None:
        if not isinstance(resource_budget, ResourceBudget):
            raise TypeError("resource_budget must be ResourceBudget.")
        remaining_budget = resource_budget.maximum_total_variation
    else:
        remaining_budget = None

    for t in range(n):
        # Controller sees the current state, never a future state.
        if t % interval == 0:
            proposed = controller(
                state, stimulus, controller_params
            )

            if resource_budget is None:
                if maximum_update_change is None:
                    stimulus = proposed
                else:
                    delta = np.clip(
                        proposed - stimulus,
                        -maximum_update_change,
                        maximum_update_change,
                    )
                    stimulus = float(np.clip(
                        stimulus + delta, 0.0, 1.0
                    ))
            else:
                stimulus, remaining_budget = bounded_control_step(
                    proposed,
                    stimulus,
                    remaining_budget,
                    resource_budget,
                )

        states[t] = state
        positions[t] = position
        velocities[t] = velocity
        stimuli[t] = stimulus

        # Motion update with common random numbers.
        omega = 2.0 * np.pi * motion_params.frequency_hz
        deviation = state - motion_params.reference_state
        noise_scale = (
            motion_params.baseline_noise
            + motion_params.state_sensitivity * max(deviation, 0.0)
            + 0.35 * motion_params.state_sensitivity
              * max(-deviation, 0.0)
        )

        acceleration = (
            -(omega ** 2) * position
            - motion_params.damping * velocity
            + noise_scale * motion_noise[t]
        )

        velocity = velocity + dt * acceleration
        speed = np.linalg.norm(velocity)

        if speed > motion_params.maximum_speed:
            velocity *= motion_params.maximum_speed / speed

        position = position + dt * velocity

        state = state_step(
            state, stimulus, dt, state_params,
            float(state_noise[t])
        )

    return {
        "state": states,
        "position": positions,
        "velocity": velocities,
        "stimulus": stimuli,
        "controller": controller_name,
        "observation_source": "ground_truth_oracle",
        "sampling_rate_hz": fs,
        "control_rate_hz": control_rate_hz,
    }
