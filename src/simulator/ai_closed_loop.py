"""Isolated motion-observed closed-loop simulation.

The AI receives only completed causal motion features and time.
True state is used for system evolution and retrospective metrics,
never as an AI model input.
"""

import numpy as np

from src.models.online_motion_features import OnlineMotionFeatures
from src.simulator.dynamics import state_step


def simulate_motion_feedback(
    seed,
    state_params,
    motion_params,
    coupled_params,
    controller_params,
    initial_state=5.0,
    control_rate_hz=10,
    model=None,
    policy="neutral",
    maximum_update_change=0.05,
    maximum_total_variation=None,
):
    if policy not in ("neutral", "ai_neutral_return"):
        raise ValueError("Unsupported feedback policy.")

    if policy == "ai_neutral_return" and model is None:
        raise ValueError("AI policy requires a fitted model.")

    fs = coupled_params.sampling_rate_hz
    fc = control_rate_hz

    if (
        not isinstance(fc, int)
        or fc <= 0
        or fs % fc != 0
    ):
        raise ValueError("Invalid control rate.")

    if (
        not np.isfinite(maximum_update_change)
        or not 0 < maximum_update_change <= 1
    ):
        raise ValueError("Invalid maximum update change.")

    if maximum_total_variation is not None:
        if (
            not np.isfinite(maximum_total_variation)
            or maximum_total_variation < 0
        ):
            raise ValueError("Invalid total-variation budget.")

        remaining_variation = float(maximum_total_variation)
    else:
        remaining_variation = None

    interval = fs // fc
    n = round(fs * coupled_params.duration_s)
    dt = 1.0 / fs

    seed_sequence = np.random.SeedSequence(seed)
    state_seed, motion_seed = seed_sequence.spawn(2)

    state_rng = np.random.default_rng(state_seed)
    motion_rng = np.random.default_rng(motion_seed)

    state_noise = state_rng.normal(
        0.0,
        coupled_params.state_disturbance_sd,
        size=n,
    )
    motion_noise = motion_rng.normal(size=(n, 3))

    states = np.empty(n)
    positions = np.empty((n, 3))
    velocities = np.empty((n, 3))
    stimuli = np.empty(n)

    state = float(initial_state)
    position = np.zeros(3, dtype=float)
    velocity = np.zeros(3, dtype=float)
    stimulus = float(controller_params.neutral_stimulus)

    observer = OnlineMotionFeatures(
        sampling_rate_hz=fs,
        update_rate_hz=fc,
        window_duration_s=1.0,
    )

    pending_observation = None
    decisions = []

    for t in range(n):
        # Decisions at t use observations completed at t-1
        # or earlier. Never use the current hidden state.
        if t % interval == 0:
            previous = stimulus
            predicted_class = None
            confidence = None
            observation_index = None

            if policy == "neutral" or pending_observation is None:
                proposed = controller_params.neutral_stimulus
            else:
                observation_index = pending_observation[
                    "sample_index"
                ]
                assert observation_index < t

                feature_vector = np.concatenate([
                    pending_observation["features"],
                    [pending_observation["time_s"]],
                ])

                assert feature_vector.shape == (15,)
                assert np.isfinite(feature_vector).all()

                probability = np.asarray(
                    model.predict_proba(
                        feature_vector.reshape(1, -1)
                    ),
                    dtype=float,
                )

                if (
                    probability.shape != (1, 3)
                    or not np.isfinite(probability).all()
                    or not np.allclose(probability.sum(), 1)
                ):
                    raise ValueError("Invalid model probabilities.")

                # Frozen model's class order: Low, Regulated,
                # Elevated. Validate rather than assume.
                classes = getattr(model, "classes_", None)
                if classes is None and hasattr(model, "steps"):
                    classes = model.steps[-1][1].classes_

                if classes is None or not np.array_equal(
                    classes, [0, 1, 2]
                ):
                    raise ValueError("Unexpected model class order.")

                predicted_class = int(np.argmax(probability[0]))
                confidence = float(probability[0].max())

                if predicted_class == 0:
                    proposed = stimulus + 0.05
                elif predicted_class == 2:
                    proposed = stimulus - 0.05
                else:
                    proposed = controller_params.neutral_stimulus

            delta = np.clip(
                proposed - stimulus,
                -maximum_update_change,
                maximum_update_change,
            )

            # Budget-safe neutral-return reserve.
            #
            # Invariant:
            # cumulative_variation
            # + abs(stimulus - neutral) <= budget
            #
            # Outward changes consume variation AND increase
            # the reserved amount, so only half the available
            # headroom may be spent on the outward change.
            #
            # Inward changes consume variation while reducing
            # the required reserve by the same amount.
            #
            # Reversal across neutral is split into successive
            # control decisions to preserve the invariant.
            if remaining_variation is not None:
                neutral = float(
                    controller_params.neutral_stimulus
                )
                distance = abs(stimulus - neutral)

                target = float(np.clip(
                    stimulus + delta,
                    controller_params.minimum_stimulus,
                    controller_params.maximum_stimulus,
                ))
                requested = target - stimulus

                moving_toward_neutral = (
                    distance > 1e-12
                    and requested * (neutral - stimulus) > 0
                )

                if moving_toward_neutral:
                    allowed = min(
                        abs(requested),
                        distance,
                        remaining_variation,
                    )
                else:
                    headroom = max(
                        0.0,
                        remaining_variation - distance,
                    )
                    allowed = min(
                        abs(requested),
                        0.5 * headroom,
                    )

                delta = float(
                    np.sign(requested) * allowed
                )
                stimulus = float(
                    np.clip(
                        stimulus + delta,
                        controller_params.minimum_stimulus,
                        controller_params.maximum_stimulus,
                    )
                )

                actual_change = abs(stimulus - previous)
                remaining_variation = max(
                    0.0,
                    remaining_variation - actual_change,
                )

                if (
                    abs(stimulus - neutral)
                    > remaining_variation + 1e-9
                ):
                    raise AssertionError(
                        "Neutral-return reserve violated."
                    )
            else:
                stimulus = float(np.clip(
                    stimulus + delta,
                    controller_params.minimum_stimulus,
                    controller_params.maximum_stimulus,
                ))

            decisions.append({
                "control_sample": t,
                "observation_sample": observation_index,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "stimulus_before": previous,
                "stimulus_after": stimulus,
            })

        states[t] = state
        positions[t] = position
        velocities[t] = velocity
        stimuli[t] = stimulus

        # Record the motion observable at sample t.
        # The newly completed feature is eligible only
        # at the next control update.
        observation = observer.observe(position, velocity)

        if observation is not None:
            pending_observation = observation

        # Match the existing simulator's motion update.
        omega = 2.0 * np.pi * motion_params.frequency_hz
        deviation = state - motion_params.reference_state

        noise_scale = (
            motion_params.baseline_noise
            + motion_params.state_sensitivity
              * max(deviation, 0.0)
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
            float(state_noise[t]),
        )

    return {
        "state": states,
        "position": positions,
        "velocity": velocities,
        "stimulus": stimuli,
        "controller": policy,
        "observation_source": (
            "motion_only" if policy != "neutral"
            else "neutral"
        ),
        "sampling_rate_hz": fs,
        "control_rate_hz": fc,
        "decisions": decisions,
    }
