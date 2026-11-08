from pathlib import Path
import sys
import csv
import json
import hashlib

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(r"D:\Dr_Abdul_Rehman\Multimodal-VR-Comparative-Revision")
sys.path.insert(0, str(ROOT))

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
MODEL_PATH = (
    ROOT / "results/frozen_one_second_ai_baseline"
    / "motion_time_unweighted.joblib"
)
MANIFEST_PATH = (
    ROOT / "results/frozen_one_second_ai_baseline"
    / "model_manifest.json"
)
OUTPUT = ROOT / "revision/configs/B8_sham_motion_protocol.json"

EXPECTED_DATA_SHA = (
    "f4e5f898b66bba7cd1fe0f24d83c856"
    "a9a679d17e692e2913319bd02ba358797"
)
EXPECTED_MODEL_SHA = (
    "1884007ac6581b6653c3227e464256"
    "b5ccc56c1e30acf847f543371312f36467"
)

def require(condition, message):
    if not condition:
        raise AssertionError(message)

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

print("STEP 223 - TIME-PRESERVED SHAM-MOTION CONTROL")

require(sha256(DATA) == EXPECTED_DATA_SHA, "Training data hash mismatch")
require(sha256(MODEL_PATH) == EXPECTED_MODEL_SHA, "Frozen model hash mismatch")
require(not OUTPUT.exists(), "Refusing to overwrite sham protocol")

manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8-sig"))
features = manifest["feature_names"]
motion_features = features[:-1]

require(len(motion_features) == 14, "Invalid motion feature count")
require(features[-1] == "time_s", "Invalid elapsed-time feature")

data = pd.read_csv(DATA)
training = data.loc[data["split"] == "train"].copy()

require(len(training) == 13320, "Unexpected training sample count")
require(training["profile"].nunique() == 60, "Unexpected training profiles")

# Build complete, time-indexed donor trajectories using
# training data only. Target state and labels are excluded.
donors = []

for (profile, condition), group in training.groupby(
    ["profile", "condition"], sort=True
):
    group = group.sort_values("sample_index")

    indices = group["sample_index"].to_numpy(dtype=int)
    expected = np.arange(89, 1080, 9, dtype=int)

    require(
        np.array_equal(indices, expected),
        f"Incomplete donor trajectory: {profile}/{condition}"
    )

    values = group[motion_features].to_numpy(dtype=float)

    require(
        values.shape == (111, 14),
        "Unexpected donor feature dimensions"
    )
    require(np.isfinite(values).all(), "Nonfinite donor features")

    donors.append({
        "profile": int(profile),
        "condition": str(condition),
        "values": values.copy(),
    })

require(len(donors) == 120, "Expected 120 complete donor trajectories")

model = joblib.load(MODEL_PATH)

class ShamMotionAdapter:
    """Replace motion with a fixed independent donor trace.

    Retain the recipient's own elapsed time.
    No current-episode state or outcome is accessed.
    """

    def __init__(self, fitted_model, donor):
        self.model = fitted_model
        self.donor = donor
        self.classes_ = np.asarray(fitted_model.classes_)

        require(
            np.array_equal(self.classes_, [0, 1, 2]),
            "Unexpected classifier classes"
        )

    def transform(self, X):
        X = np.asarray(X, dtype=float)

        require(
            X.ndim == 2 and X.shape[1] == 15,
            "Expected original 15-dimensional input"
        )

        result = X.copy()
        sample_indices = np.rint(result[:, 14] * 90).astype(int)

        for row, index in enumerate(sample_indices):
            require(
                89 <= index <= 1079 and (index - 89) % 9 == 0,
                "Invalid causal observation timestamp"
            )

            donor_index = (index - 89) // 9
            result[row, :14] = self.donor["values"][donor_index]

        require(
            np.array_equal(result[:, 14], X[:, 14]),
            "Elapsed time was modified"
        )

        return result

    def predict_proba(self, X):
        return self.model.predict_proba(self.transform(X))


def choose_donor(seed):
    # Selection is determined before trajectory generation.
    # It uses only the fixed episode seed, not the true state,
    # predicted class, condition, or evaluation outcome.
    key = f"sham-motion-v1|{int(seed)}".encode("ascii")
    digest = hashlib.sha256(key).digest()
    index = int.from_bytes(digest[:8], "big") % len(donors)
    return donors[index]


# Feature-isolation verification.
probe = np.zeros((3, 15), dtype=float)
probe[:, 14] = np.array([89, 98, 107]) / 90

donor = choose_donor(8500001)
adapter = ShamMotionAdapter(model, donor)

sham = adapter.transform(probe)

require(
    np.array_equal(sham[:, 14], probe[:, 14]),
    "Time preservation failed"
)
require(
    np.array_equal(sham[0, :14], donor["values"][0]),
    "Sham feature replacement failed"
)

altered = probe.copy()
altered[:, :14] = 999.0

require(
    np.array_equal(
        adapter.predict_proba(probe),
        adapter.predict_proba(altered)
    ),
    "Sham model still depends on current-episode motion"
)

require(
    not np.array_equal(sham[:, :14], probe[:, :14]),
    "Motion replacement ineffective"
)

# Development-validation smoke tests only.
with (
    ROOT / "configs/frozen_development_profiles.csv"
).open(encoding="utf-8-sig", newline="") as f:
    profiles = list(csv.DictReader(f))

validation = [p for p in profiles if p["split"] == "validation"]
require(len(validation) == 20, "Incorrect development-validation cohort")

coupled = CoupledParameters(
    sampling_rate_hz=90,
    duration_s=12.0,
    state_disturbance_sd=0.05,
)
controller = ControllerParameters()

checked = 0
different_actions = 0

for p in tqdm(
    validation[:2],
    desc="Sham-motion smoke tests",
    unit="profile"
):
    pid = int(p["profile"])

    sp = StateParameters(
        baseline=float(p["baseline"]),
        relaxation_rate=float(p["relaxation_rate"]),
        stimulus_gain=float(p["stimulus_gain"]),
    )

    mp = MotionParameters(
        frequency_hz=float(p["motion_frequency_hz"]),
        state_sensitivity=float(p["motion_state_sensitivity"]),
    )

    for condition, initial, offset in (
        ("Low", 2.0, 0),
        ("Elevated", 8.0, 1),
    ):
        seed = 8500000 + 2 * pid + offset
        selected = choose_donor(seed)

        kwargs = dict(
            seed=seed,
            state_params=sp,
            motion_params=mp,
            coupled_params=coupled,
            controller_params=controller,
            initial_state=initial,
            control_rate_hz=10,
            policy="ai_neutral_return",
            maximum_update_change=0.05,
            maximum_total_variation=0.6,
        )

        genuine = simulate_motion_feedback(
            **kwargs, model=model
        )

        sham_run = simulate_motion_feedback(
            **kwargs,
            model=ShamMotionAdapter(model, selected)
        )

        for name, run in (
            ("B6_genuine", genuine),
            ("B8_sham", sham_run),
        ):
            changes = np.abs(np.diff(
                np.r_[0.5, run["stimulus"][::9]]
            ))
            spent = float(changes.sum())

            require(
                np.isfinite(run["state"]).all(),
                "Nonfinite simulated state"
            )
            require(
                changes.max() <= 0.05 + 1e-9,
                "Per-update limit violated"
            )
            require(
                spent <= 0.6 + 1e-9,
                "Variation limit violated"
            )
            require(
                spent + abs(run["stimulus"][-1] - 0.5)
                <= 0.6 + 1e-9,
                "Return reserve violated"
            )
            require(
                np.all(run["stimulus"][:90] == 0.5),
                "Pre-observation intervention"
            )

            for decision in run["decisions"]:
                observation = decision["observation_sample"]
                if observation is not None:
                    require(
                        observation < decision["control_sample"],
                        "Temporal causality violation"
                    )

            metrics = evaluate_controller(
                run, hold_duration_s=1.0
            )

            recovery = (
                metrics["recovery_time_s"]
                if metrics["recovered"] else 12.0
            )

            print(
                f"{name} profile={pid} condition={condition} "
                f"recovery={recovery:.6f} "
                f"stability={metrics['state_stability_pct']:.6f} "
                f"variation={spent:.6f}"
            )

            checked += 1

        if not np.array_equal(
            genuine["stimulus"], sham_run["stimulus"]
        ):
            different_actions += 1

require(checked == 8, "Incorrect smoke-test count")

protocol = {
    "step": 223,
    "baseline": "B8_time_preserved_sham_motion",
    "purpose": "current_trajectory_motion_information_falsification",
    "source": str(DATA),
    "training_data_sha256": EXPECTED_DATA_SHA,
    "frozen_model_sha256": EXPECTED_MODEL_SHA,
    "donor_split": "train_only",
    "donor_profiles": 60,
    "donor_trajectories": 120,
    "donor_conditions": ["Low", "Elevated"],
    "donor_selection": "SHA256(sham-motion-v1|episode_seed), modulo 120",
    "donor_selection_independent_of_current_state": True,
    "recipient_motion_used": False,
    "recipient_elapsed_time_preserved": True,
    "donor_temporal_sequence_preserved": True,
    "donor_true_state_used_for_matching": False,
    "observation_window_samples": 90,
    "control_frequency_hz": 10,
    "first_feedback_time_s": 1.0,
    "variation_budget": 0.6,
    "maximum_update_change": 0.05,
    "neutral_return_reserve": True,
    "comparison": "B6_genuine_motion_time",
    "diagnostic_only": True,
}

OUTPUT.write_text(
    json.dumps(protocol, indent=2) + "\n",
    encoding="utf-8"
)

print("\nDonor trajectories:", len(donors))
print("Development simulations:", checked)
print("Cases with different intervention trajectories:", different_actions)
print("Donor selection independent of evaluation state: PASS")
print("Current-episode motion excluded from sham predictions: PASS")
print("Current elapsed time preserved: PASS")
print("Causal timing and resource constraints: PASS")
print("Frozen model retrained: NO")
print("Original source modified: NO")
print("Independent test accessed: NO")
print("Protocol:", OUTPUT)
print("STEP 223: PASS")
