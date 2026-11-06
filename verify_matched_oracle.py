from pathlib import Path
import csv
import sys
import numpy as np
from tqdm import tqdm

ROOT = Path(
    r"D:\Dr_Abdul_Rehman\Multimodal-VR-Publication-Release"
)
sys.path.insert(0, str(ROOT))

from src.simulator.dynamics import StateParameters, state_step
from src.simulator.motion import MotionParameters
from src.evaluation.control_metrics import evaluate_controller

def read_rows(path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def simulate_reserved_oracle(profile, seed, initial_state):
    state_params = StateParameters(
        baseline=float(profile["baseline"]),
        relaxation_rate=float(profile["relaxation_rate"]),
        stimulus_gain=float(profile["stimulus_gain"])
    )
    motion_params = MotionParameters(
        frequency_hz=float(profile["motion_frequency_hz"]),
        state_sensitivity=float(profile["motion_state_sensitivity"])
    )

    fs, fc, n = 90, 10, 1080
    dt = 1.0 / fs
    interval = fs // fc

    seed_sequence = np.random.SeedSequence(seed)
    state_seed, motion_seed = seed_sequence.spawn(2)

    state_noise = np.random.default_rng(state_seed).normal(
        0.0, 0.05, size=n
    )
    motion_noise = np.random.default_rng(motion_seed).normal(
        size=(n, 3)
    )

    states = np.empty(n)
    positions = np.empty((n, 3))
    velocities = np.empty((n, 3))
    stimuli = np.empty(n)

    state = float(initial_state)
    position = np.zeros(3)
    velocity = np.zeros(3)
    stimulus = 0.5
    remaining = 0.6

    for t in range(n):
        if t % interval == 0:
            previous = stimulus

            # Match AI latency: first feedback at 1.0 s.
            if t < 90:
                proposed = 0.5
            elif state < 4.0:
                proposed = stimulus + 0.05
            elif state > 6.0:
                proposed = stimulus - 0.05
            else:
                proposed = 0.5

            delta = float(np.clip(
                proposed - stimulus, -0.05, 0.05
            ))

            distance = abs(stimulus - 0.5)
            target = float(np.clip(stimulus + delta, 0.0, 1.0))
            requested = target - stimulus

            toward_neutral = (
                distance > 1e-12
                and requested * (0.5 - stimulus) > 0
            )

            if toward_neutral:
                allowed = min(
                    abs(requested), distance, remaining
                )
            else:
                headroom = max(
                    0.0, remaining - distance
                )
                allowed = min(
                    abs(requested), 0.5 * headroom
                )

            delta = float(np.sign(requested) * allowed)
            stimulus = float(np.clip(
                stimulus + delta, 0.0, 1.0
            ))

            remaining = max(
                0.0, remaining - abs(stimulus - previous)
            )

            assert abs(stimulus - 0.5) <= remaining + 1e-9

        states[t] = state
        positions[t] = position
        velocities[t] = velocity
        stimuli[t] = stimulus

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
            float(state_noise[t])
        )

    return {
        "state": states,
        "position": positions,
        "velocity": velocities,
        "stimulus": stimuli,
        "sampling_rate_hz": fs,
        "control_rate_hz": fc,
    }

def main():
    profiles = read_rows(
        ROOT / "configs/independent_synthetic_test_profiles.csv"
    )
    archived = read_rows(
        ROOT / "results/final_independent_evaluation/"
        "final_results/all_1200_policy_episodes.csv"
    )

    reference = {
        (int(r["profile"]), r["condition"]): r
        for r in archived
        if r["policy"] == "matched_reserved_oracle"
    }

    assert len(profiles) == 200
    assert len(reference) == 400

    mismatches = []
    max_error = 0.0
    checked = 0

    for profile in tqdm(
        profiles, desc="Verifying matched oracle",
        unit="profile"
    ):
        profile_id = int(profile["profile"])

        for condition, initial in (
            ("Low", 2.0), ("Elevated", 8.0)
        ):
            expected = reference[(profile_id, condition)]
            seed = int(expected["episode_seed"])

            run = simulate_reserved_oracle(
                profile, seed, initial
            )

            metrics = evaluate_controller(
                run, hold_duration_s=1.0
            )

            actual = np.array([
                metrics["recovery_time_s"]
                if metrics["recovered"] else 12.0,
                metrics["state_stability_pct"],
                metrics["stimulus_total_variation"],
                abs(run["stimulus"][-1] - 0.5),
                float(np.mean(
                    abs(run["stimulus"] - 0.5)
                ))
            ])

            stored = np.array([
                float(expected["recovery_s"]),
                float(expected["stability_pct"]),
                float(expected["variation"]),
                float(expected["final_neutral_distance"]),
                float(expected["mean_neutral_distance"])
            ])

            error = float(np.max(np.abs(actual - stored)))
            max_error = max(max_error, error)

            if not np.allclose(
                actual, stored, rtol=0, atol=1e-9
            ):
                mismatches.append(
                    (profile_id, condition, actual, stored)
                )

            checked += 1

    print("\nSTEP 200 — MATCHED ORACLE REPRODUCTION")
    print("Profiles checked:", len(profiles))
    print("Episodes checked:", checked)
    print("Metrics per episode:", 5)
    print("Maximum absolute error:", f"{max_error:.3e}")
    print("Mismatched episodes:", len(mismatches))

    for item in mismatches[:5]:
        print("MISMATCH:", item)

    print(
        "FINAL STATUS:",
        "PASS" if not mismatches else "FAIL"
    )
    print("Frozen files modified: NO")
    print("Model retrained: NO")
    print("Archived outcomes modified: NO")

    return 0 if not mismatches else 1

if __name__ == "__main__":
    sys.exit(main())
