from pathlib import Path
import sys
import json
import hashlib

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(r"D:\Dr_Abdul_Rehman\Multimodal-VR-Comparative-Revision")
sys.path.insert(0, str(ROOT))

from revision.baseline_policies import (
    FixedSchedulePolicy,
    MotionThresholdPolicy,
)
from src.simulator.dynamics import StateParameters
from src.simulator.motion import MotionParameters
from src.simulator.coupled import CoupledParameters
from src.simulator.ai_closed_loop import simulate_motion_feedback
from src.controllers.baselines import ControllerParameters
from src.evaluation.control_metrics import evaluate_controller

DATA = Path(
    r"D:\Dr_Abdul_Rehman\Multimodal-VR"
    r"\results\motion_state_dataset\development_windows.csv"
)

EXPECTED_SHA = (
    "f4e5f898b66bba7cd1fe0f24d83c856"
    "a9a679d17e692e2913319bd02ba358797"
)

def require(value, message):
    if not value:
        raise AssertionError(message)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

print("STEP 221 - SIMPLE ADAPTIVE BASELINES")

require(digest(DATA) == EXPECTED_SHA, "Training-data integrity failure")

data = pd.read_csv(DATA)
train = data.loc[data["split"] == "train"].copy()

require(len(train) == 13320, "Incorrect training-window count")
require(train["profile"].nunique() == 60, "Incorrect training cohort")

states = train["target_state"].to_numpy(float)
train["state_class"] = np.where(
    states < 4, 0, np.where(states > 6, 2, 1)
)

medians = train.groupby("state_class")["speed_std"].median()

require(set(medians.index) == {0, 1, 2}, "Missing state class")
require(medians[1] < medians[0] < medians[2],
        "Training-class ordering does not support proposed rule")

lower = float((medians[1] + medians[0]) / 2)
upper = float((medians[0] + medians[2]) / 2)

policies = {
    "B1_fixed_schedule": FixedSchedulePolicy(),
    "B4_motion_threshold": MotionThresholdPolicy(lower, upper),
}

# Confirm which information each baseline actually uses.
probe = np.zeros((3, 15))
probe[:, 14] = [89 / 90, 179 / 90, 269 / 90]
probe[:, 1] = [0.0005, 0.002, 0.006]

b1 = policies["B1_fixed_schedule"].predict_proba(probe)
b4 = policies["B4_motion_threshold"].predict_proba(probe)

require(np.array_equal(np.argmax(b1, axis=1), [0, 1, 2]),
        "B1 schedule mismatch")
require(np.array_equal(np.argmax(b4, axis=1), [1, 0, 2]),
        "B4 class mapping mismatch")

# B1 must ignore motion features entirely.
altered = probe.copy()
altered[:, :14] = 1000
require(np.array_equal(
    b1,
    policies["B1_fixed_schedule"].predict_proba(altered)
), "B1 improperly uses motion")

# B4 must ignore time entirely.
altered = probe.copy()
altered[:, 14] = 1000
require(np.array_equal(
    b4,
    policies["B4_motion_threshold"].predict_proba(altered)
), "B4 improperly uses elapsed time")

profiles = (
    pd.read_csv(ROOT / "configs/frozen_development_profiles.csv")
)

validation = profiles.loc[profiles["split"] == "validation"]
require(len(validation) == 20, "Invalid validation profiles")

coupled = CoupledParameters(
    sampling_rate_hz=90,
    duration_s=12.0,
    state_disturbance_sd=0.05,
)
controller = ControllerParameters()

tested = 0

for _, profile in tqdm(
    validation.head(2).iterrows(),
    total=2,
    desc="B1/B4 smoke tests",
    unit="profile",
):
    pid = int(profile["profile"])

    sp = StateParameters(
        baseline=float(profile["baseline"]),
        relaxation_rate=float(profile["relaxation_rate"]),
        stimulus_gain=float(profile["stimulus_gain"]),
    )
    mp = MotionParameters(
        frequency_hz=float(profile["motion_frequency_hz"]),
        state_sensitivity=float(profile["motion_state_sensitivity"]),
    )

    for condition, initial, offset in (
        ("Low", 2.0, 0),
        ("Elevated", 8.0, 1),
    ):
        seed = 8200000 + 2 * pid + offset

        for name, policy in policies.items():
            run = simulate_motion_feedback(
                seed=seed,
                state_params=sp,
                motion_params=mp,
                coupled_params=coupled,
                controller_params=controller,
                initial_state=initial,
                control_rate_hz=10,
                model=policy,
                policy="ai_neutral_return",
                maximum_update_change=0.05,
                maximum_total_variation=0.6,
            )

            result = evaluate_controller(
                run, hold_duration_s=1.0
            )

            changes = np.abs(np.diff(
                np.r_[0.5, run["stimulus"][::9]]
            ))

            require(changes.max() <= 0.05 + 1e-9,
                    "Per-update limit violated")
            require(changes.sum() <= 0.6 + 1e-9,
                    "Variation budget violated")
            require(
                changes.sum() +
                abs(run["stimulus"][-1] - 0.5)
                <= 0.6 + 1e-9,
                "Neutral-return reserve violated"
            )
            require(np.all(run["stimulus"][:90] == 0.5),
                    "Pre-feedback intervention")

            for decision in run["decisions"]:
                observed = decision["observation_sample"]
                if observed is not None:
                    require(
                        observed < decision["control_sample"],
                        "Temporal causality violated"
                    )

            recovery = (
                result["recovery_time_s"]
                if result["recovered"] else 12.0
            )

            print(
                f"{name} profile={pid} condition={condition} "
                f"recovery={recovery:.6f} "
                f"stability={result['state_stability_pct']:.6f} "
                f"variation={result['stimulus_total_variation']:.6f}"
            )
            tested += 1

require(tested == 8, "Unexpected number of smoke tests")

manifest = {
    "step": 221,
    "training_data_sha256": EXPECTED_SHA,
    "B1": {
        "policy": "fixed_schedule",
        "cycle_seconds": 4,
        "one_second_phases": [
            "increase",
            "neutral",
            "decrease",
            "neutral"
        ],
        "motion_used": False,
        "hidden_state_used": False,
        "initial_condition_used": False,
    },
    "B4": {
        "policy": "single_feature_threshold",
        "feature": "speed_std",
        "regulated_median": float(medians[1]),
        "low_median": float(medians[0]),
        "elevated_median": float(medians[2]),
        "regulated_low_threshold": lower,
        "low_elevated_threshold": upper,
        "elapsed_time_used": False,
        "threshold_source": "training_class_median_midpoints",
    },
    "shared_constraints": {
        "sampling_hz": 90,
        "control_hz": 10,
        "first_control_s": 1.0,
        "max_update_change": 0.05,
        "variation_budget": 0.6,
        "neutral_return_reserve": True,
    }
}

output = ROOT / "revision/configs/B1_B4_frozen_protocol.json"
require(not output.exists(), "Refusing to overwrite frozen protocol")

output.write_text(
    json.dumps(manifest, indent=2) + "\n",
    encoding="utf-8",
)

print("\nB4 thresholds")
print("Regulated/Low:", f"{lower:.10f}")
print("Low/Elevated:", f"{upper:.10f}")
print("Development episodes tested:", tested)
print("Protocol:", output)
print("Original source modified: NO")
print("Independent test accessed: NO")
print("STEP 221: PASS")
