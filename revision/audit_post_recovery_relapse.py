from pathlib import Path
import hashlib
import importlib.util
import json
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(r"D:\Dr_Abdul_Rehman\Multimodal-VR-Comparative-Revision")
sys.path.insert(0, str(ROOT))

from src.simulator.ai_closed_loop import simulate_motion_feedback
from src.simulator.dynamics import StateParameters
from src.simulator.motion import MotionParameters
from src.simulator.coupled import CoupledParameters
from src.controllers.baselines import ControllerParameters
from src.evaluation.control_metrics import evaluate_controller

CONFIG = ROOT / "revision/configs"
ARCHIVE = ROOT / "revision/results/confirmatory_200"
OUTPUT = ROOT / "revision/results/post_recovery_relapse_audit.json"

def require(ok, message):
    if not ok:
        raise AssertionError(message)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def array_sha(array):
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes()
    ).hexdigest()

print("STEP 239 - DIRECT POST-RECOVERY RELAPSE AUDIT")

require(not OUTPUT.exists(), "Step 239 report already exists")

audit_file = CONFIG / "confirmatory_results_integrity_audit.json"
require(
    sha(audit_file) ==
    "76a31abe9b8f6df5c60656e2f789afad99160801809013751d6eda7aff822b7d",
    "Frozen audit mismatch"
)

audit = json.loads(audit_file.read_text(encoding="utf-8"))

require(
    sha(ROOT / "revision/run_confirmatory_comparison.py") ==
    "a9a1d6e5b94ac30789f55e87b2306c61fda2a14a11215a0d0e14fd4e4e3923e1",
    "Frozen runner mismatch"
)

lock = json.loads(
    (CONFIG / "comparative_protocol_lock.json").read_text(
        encoding="utf-8"
    )
)
for relative, expected in lock["artifact_sha256"].items():
    require(
        sha(ROOT / relative) == expected,
        f"Frozen artifact mismatch: {relative}"
    )

import joblib
model = joblib.load(
    ROOT / "results/frozen_one_second_ai_baseline"
    / "motion_time_unweighted.joblib"
)

module_path = ROOT / "revision/simulator_B5_budget_only.py"
spec = importlib.util.spec_from_file_location(
    "step239_budget_simulator", module_path
)
budget_simulator = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = budget_simulator
spec.loader.exec_module(budget_simulator)

profiles = pd.read_csv(
    CONFIG / "confirmatory_profiles_200.csv"
).set_index("profile")

ledger = pd.read_csv(
    CONFIG / "confirmatory_episode_seeds.csv"
)

seeds = {
    (int(r.profile), str(r.condition)): int(r.episode_seed)
    for r in ledger.itertuples(index=False)
}
require(len(profiles) == 200, "Profile count mismatch")
require(len(seeds) == 400, "Seed count mismatch")

coupled = CoupledParameters(
    sampling_rate_hz=90,
    duration_s=12.0,
    state_disturbance_sd=0.05
)
controller = ControllerParameters()

records = []
verified = 0

for entry in tqdm(
    audit["checkpoint_inventory"],
    desc="Replaying B5/B6",
    unit="profile"
):
    path = ARCHIVE / entry["file"]
    require(
        sha(path) == entry["sha256"],
        f"Archived checkpoint changed: {path.name}"
    )

    archived = {
        (r["condition"], r["policy"]): r
        for r in json.loads(path.read_text(encoding="utf-8"))
    }

    pid = int(entry["profile"])
    p = profiles.loc[pid]

    sp = StateParameters(
        baseline=float(p["baseline"]),
        relaxation_rate=float(p["relaxation_rate"]),
        stimulus_gain=float(p["stimulus_gain"])
    )
    mp = MotionParameters(
        frequency_hz=float(p["motion_frequency_hz"]),
        state_sensitivity=float(p["motion_state_sensitivity"])
    )

    for condition, initial in (
        ("Low", 2.0),
        ("Elevated", 8.0)
    ):
        seed = seeds[(pid, condition)]

        common = dict(
            seed=seed,
            state_params=sp,
            motion_params=mp,
            coupled_params=coupled,
            controller_params=controller,
            initial_state=initial,
            control_rate_hz=10,
            model=model,
            policy="ai_neutral_return",
            maximum_update_change=0.05,
            maximum_total_variation=0.6
        )

        for policy in ("B5", "B6"):
            if policy == "B5":
                run = budget_simulator.simulate_motion_feedback(
                    **common, reserve_enabled=False
                )
            else:
                run = simulate_motion_feedback(**common)

            old = archived[(condition, policy)]

            require(
                array_sha(run["state"]) == old["state_sha256"],
                f"State replay mismatch: {pid}/{condition}/{policy}"
            )

            motion_hash = hashlib.sha256(
                np.ascontiguousarray(run["position"]).tobytes()
                + np.ascontiguousarray(run["velocity"]).tobytes()
            ).hexdigest()

            require(
                motion_hash == old["motion_sha256"],
                f"Motion replay mismatch: {pid}/{condition}/{policy}"
            )

            require(
                np.array_equal(
                    np.asarray(run["stimulus"])[::9],
                    np.asarray(old["stimulus_at_updates"])
                ),
                f"Action replay mismatch: {pid}/{condition}/{policy}"
            )

            state = np.asarray(run["state"], dtype=float)
            inside = (state >= 4.0) & (state <= 6.0)

            # The frozen definition: first 90-sample
            # uninterrupted residence inside [4,6].
            windows = np.convolve(
                inside.astype(np.int32),
                np.ones(90, dtype=np.int32),
                mode="valid"
            )
            qualifying = np.flatnonzero(windows == 90)

            recovered = qualifying.size > 0
            recovery_index = (
                int(qualifying[0]) if recovered else None
            )

            require(
                bool(old["recovered"]) == bool(recovered),
                "Recovery status mismatch"
            )

            expected_time = (
                recovery_index / 90.0 if recovered else 12.0
            )
            require(
                abs(expected_time - float(old["recovery_s"]))
                < 1e-10,
                "Recovery-time replay mismatch"
            )

            stability = 100.0 * inside.mean()
            require(
                abs(stability - float(old["stability_pct"]))
                < 1e-10,
                "Stability replay mismatch"
            )

            # Relapse means leaving [4,6] AFTER the complete
            # initial one-second qualifying recovery window.
            if recovered:
                after = inside[recovery_index + 90:]
                relapsed = bool(np.any(~after))
                post_recovery_outside_samples = int(
                    np.count_nonzero(~after)
                )
            else:
                relapsed = False
                post_recovery_outside_samples = None

            # Exploratory late-episode outcome; 3 seconds = 270 samples.
            final_3s_stability_pct = float(
                100.0 * inside[-270:].mean()
            )
            final_state_inside = bool(inside[-1])

            records.append({
                "profile": pid,
                "condition": condition,
                "policy": policy,
                "recovered": recovered,
                "relapsed_after_qualifying_window": relapsed,
                "post_recovery_outside_samples":
                    post_recovery_outside_samples,
                "final_3s_stability_pct":
                    final_3s_stability_pct,
                "final_state_inside": final_state_inside,
                "state_stability_pct": stability,
                "opposite_boundary_overshoot":
                    float(old["overshoot"]),
            })
            verified += 1

require(verified == 800, "Expected 800 exact trajectory replays")

df = pd.DataFrame(records)
summary = {}

for policy in ("B5", "B6"):
    subset = df[df["policy"] == policy]

    summary[policy] = {
        "episodes": int(len(subset)),
        "recovered_count": int(subset["recovered"].sum()),
        "relapsed_after_qualifying_window_count": int(
            subset["relapsed_after_qualifying_window"].sum()
        ),
        "final_state_outside_count": int(
            (~subset["final_state_inside"]).sum()
        ),
        "final_3s_stability_mean_pct": float(
            subset["final_3s_stability_pct"].mean()
        ),
        "opposite_boundary_overshoot_count": int(
            (subset["opposite_boundary_overshoot"] > 1e-10).sum()
        )
    }

# Profile-cluster CI for an explicitly exploratory metric.
pair = df.pivot(
    index=["profile", "condition"],
    columns="policy",
    values="final_3s_stability_pct"
)
require(len(pair) == 400, "Incomplete paired late stability")

effects = (
    pair["B6"] - pair["B5"]
).unstack("condition").mean(axis=1).to_numpy(float)

require(effects.shape == (200,), "Incorrect cluster count")

rng = np.random.default_rng(2026)
draws = rng.integers(0, 200, size=(10000, 200))
interval = np.quantile(
    effects[draws].mean(axis=1),
    [0.025, 0.975]
)

report = {
    "step": 239,
    "status": "PASS",
    "scientific_role": "posthoc_secondary_diagnostic",
    "replayed_trajectories": verified,
    "all_saved_state_motion_action_hashes_matched": True,
    "relapse_definition": (
        "Any state outside [4,6] after completing "
        "the first 90-sample qualifying recovery window"
    ),
    "late_stability_definition": (
        "Percent of final 270 samples inside [4,6]"
    ),
    "policy_summary": summary,
    "exploratory_B6_minus_B5_final_3s_stability": {
        "mean_difference_pp": float(effects.mean()),
        "profile_cluster_bootstrap_CI_95": [
            float(interval[0]), float(interval[1])
        ],
        "bootstrap_replicates": 10000,
        "independent_profiles": 200
    },
    "frozen_results_modified": False
}

OUTPUT.write_text(
    json.dumps(report, indent=2) + "\n",
    encoding="utf-8"
)

print("\nEXACT TRAJECTORY REPRODUCTION")
print("Verified trajectories:", verified)
print("State/motion/action hashes: EXACT MATCH")

print("\nPOST-RECOVERY AND LATE-STABILITY RESULTS")
for policy, result in summary.items():
    print(policy, json.dumps(result, indent=2))

print("\nEXPLORATORY B6 - B5 FINAL-3-SECOND STABILITY")
print("Mean improvement (pp):", round(float(effects.mean()), 6))
print(
    "95% profile-cluster CI:",
    [round(float(v), 6) for v in interval]
)

print("\nOriginal checkpoints modified: NO")
print("Newly declared confirmatory endpoint: NO")
print("Report SHA256:", sha(OUTPUT))
print("STEP 239: PASS")
