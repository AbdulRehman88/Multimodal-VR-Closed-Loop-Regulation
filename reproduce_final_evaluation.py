from pathlib import Path
import sys
import csv
import json
import hashlib
import numpy as np
import joblib
from tqdm import tqdm

ROOT = Path(
    r"D:\Dr_Abdul_Rehman\Multimodal-VR-Publication-Release"
)
sys.path.insert(0, str(ROOT))

from src.simulator.dynamics import StateParameters
from src.simulator.motion import MotionParameters
from src.simulator.coupled import CoupledParameters
from src.simulator.ai_closed_loop import simulate_motion_feedback
from src.controllers.baselines import ControllerParameters
from src.evaluation.control_metrics import evaluate_controller
from verify_matched_oracle import simulate_reserved_oracle

def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

protocol = json.loads(
    (ROOT / "results/final_independent_evaluation/locked_protocol.json")
    .read_text(encoding="utf-8")
)

for relative, expected in protocol["integrity"]["source_sha256"].items():
    path = ROOT / relative
    assert path.is_file() and sha256(path) == expected, (
        f"Frozen integrity failure: {relative}"
    )

model = joblib.load(
    ROOT / "results/frozen_one_second_ai_baseline/"
    "motion_time_unweighted.joblib"
)

profiles = read_csv(
    ROOT / "configs/independent_synthetic_test_profiles.csv"
)
ledger = read_csv(
    ROOT / "results/final_independent_evaluation/"
    "final_results/episode_seed_ledger.csv"
)
archived = read_csv(
    ROOT / "results/final_independent_evaluation/"
    "final_results/all_1200_policy_episodes.csv"
)

seeds = {
    (int(r["profile"]), r["condition"]): int(r["episode_seed"])
    for r in ledger
}

reference = {
    (int(r["profile"]), r["condition"], r["policy"]): r
    for r in archived
}

assert len(profiles) == 200
assert len(seeds) == 400
assert len(reference) == 1200

coupled = CoupledParameters(
    sampling_rate_hz=90,
    duration_s=12.0,
    state_disturbance_sd=0.05
)
controller = ControllerParameters()

policies = (
    "neutral",
    "reserved_ai",
    "matched_reserved_oracle"
)

metric_map = {
    "recovery_s": "recovery_time_s",
    "stability_pct": "state_stability_pct",
    "variation": "stimulus_total_variation",
    "overshoot": "opposite_boundary_overshoot"
}

mismatches = []
maximum_error = 0.0
checked = 0

print("STEP 201 — FULL INDEPENDENT TEST REPRODUCTION")
print("Frozen SHA-256 integrity: PASS")
print("Profiles:", len(profiles))
print("Expected episodes:", len(reference))
print("Policies:", ", ".join(policies))

for profile in tqdm(
    profiles, desc="Reproducing final evaluation", unit="profile"
):
    pid = int(profile["profile"])

    state_params = StateParameters(
        baseline=float(profile["baseline"]),
        relaxation_rate=float(profile["relaxation_rate"]),
        stimulus_gain=float(profile["stimulus_gain"])
    )
    motion_params = MotionParameters(
        frequency_hz=float(profile["motion_frequency_hz"]),
        state_sensitivity=float(profile["motion_state_sensitivity"])
    )

    for condition, initial in (
        ("Low", 2.0),
        ("Elevated", 8.0)
    ):
        seed = seeds[(pid, condition)]

        for policy in policies:
            expected = reference[(pid, condition, policy)]

            assert int(expected["episode_seed"]) == seed

            if policy == "matched_reserved_oracle":
                run = simulate_reserved_oracle(
                    profile, seed, initial
                )
            else:
                run = simulate_motion_feedback(
                    seed=seed,
                    state_params=state_params,
                    motion_params=motion_params,
                    coupled_params=coupled,
                    controller_params=controller,
                    initial_state=initial,
                    control_rate_hz=10,
                    model=model if policy == "reserved_ai" else None,
                    policy=(
                        "ai_neutral_return"
                        if policy == "reserved_ai"
                        else "neutral"
                    ),
                    maximum_update_change=0.05,
                    maximum_total_variation=0.6
                )

            result = evaluate_controller(
                run, hold_duration_s=1.0
            )

            actual = np.array([
                (
                    result["recovery_time_s"]
                    if result["recovered"] else 12.0
                ),
                result["state_stability_pct"],
                result["stimulus_total_variation"],
                result["opposite_boundary_overshoot"],
                abs(float(run["stimulus"][-1]) - 0.5),
                float(np.mean(
                    np.abs(np.asarray(run["stimulus"]) - 0.5)
                ))
            ], dtype=float)

            stored = np.array([
                float(expected["recovery_s"]),
                float(expected["stability_pct"]),
                float(expected["variation"]),
                float(expected["overshoot"]),
                float(expected["final_neutral_distance"]),
                float(expected["mean_neutral_distance"])
            ], dtype=float)

            recovered_expected = bool(int(expected["recovered"]))
            status_ok = (
                bool(result["recovered"]) == recovered_expected
            )

            difference = float(
                np.max(np.abs(actual - stored))
            )
            maximum_error = max(maximum_error, difference)

            if (
                not status_ok
                or not np.allclose(
                    actual, stored, rtol=0, atol=1e-9
                )
            ):
                mismatches.append({
                    "profile": pid,
                    "condition": condition,
                    "policy": policy,
                    "recovered_match": status_ok,
                    "maximum_error": difference,
                    "actual": actual.tolist(),
                    "archived": stored.tolist()
                })

            checked += 1

print("\nFINAL REPRODUCTION AUDIT")
print("Profiles:", len(profiles))
print("Episodes verified:", checked)
print("Numeric metrics per episode:", 6)
print("Recovery status checked:", True)
print("Maximum absolute error:", f"{maximum_error:.3e}")
print("Mismatched episodes:", len(mismatches))

for mismatch in mismatches[:5]:
    print("MISMATCH:", mismatch)

passed = checked == 1200 and len(mismatches) == 0

print("\nSTEP 201:", "PASS" if passed else "FAIL")
print("Frozen source modified: NO")
print("Model retrained: NO")
print("Archived results overwritten: NO")

sys.exit(0 if passed else 1)
