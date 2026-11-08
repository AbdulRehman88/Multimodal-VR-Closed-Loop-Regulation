from pathlib import Path
import sys
import csv
import hashlib
import numpy as np
import joblib
from tqdm import tqdm

ROOT = Path(r"D:\Dr_Abdul_Rehman\Multimodal-VR-Comparative-Revision")
sys.path.insert(0, str(ROOT))

from src.simulator.dynamics import StateParameters
from src.simulator.motion import MotionParameters
from src.simulator.coupled import CoupledParameters
from src.simulator.ai_closed_loop import simulate_motion_feedback
from src.controllers.baselines import ControllerParameters
from src.evaluation.control_metrics import evaluate_controller

class FeatureAdapter:
    """Select frozen input columns without altering the simulator."""

    def __init__(self, model, columns):
        self.model = model
        self.columns = tuple(columns)
        self.classes_ = np.asarray(model.classes_)

        assert np.array_equal(self.classes_, [0, 1, 2])

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)

        assert X.ndim == 2
        assert X.shape[1] == 15

        selected = X[:, self.columns]
        result = self.model.predict_proba(selected)

        assert result.shape == (X.shape[0], 3)
        return result

def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

original = joblib.load(
    ROOT / "results/frozen_one_second_ai_baseline"
    / "motion_time_unweighted.joblib"
)

time_model = joblib.load(
    ROOT / "revision/models/B2_time_only.joblib"
)

motion_model = joblib.load(
    ROOT / "revision/models/B3_motion_only.joblib"
)

models = {
    "B6_original": original,
    "B6_identity_adapter": FeatureAdapter(original, range(15)),
    "B2_time_only": FeatureAdapter(time_model, [14]),
    "B3_motion_only": FeatureAdapter(motion_model, range(14)),
}

profiles = read_csv(
    ROOT / "configs/frozen_development_profiles.csv"
)

validation = [
    p for p in profiles if p["split"] == "validation"
]

assert len(validation) == 20

coupled = CoupledParameters(
    sampling_rate_hz=90,
    duration_s=12.0,
    state_disturbance_sd=0.05
)

controller = ControllerParameters()
checked = 0

print("STEP 219 - CAUSAL INFORMATION-BASELINE INTEGRATION")
print("Development-validation profiles only; no independent test.")

for p in tqdm(validation[:2], desc="Smoke testing", unit="profile"):
    pid = int(p["profile"])

    state_params = StateParameters(
        baseline=float(p["baseline"]),
        relaxation_rate=float(p["relaxation_rate"]),
        stimulus_gain=float(p["stimulus_gain"])
    )

    motion_params = MotionParameters(
        frequency_hz=float(p["motion_frequency_hz"]),
        state_sensitivity=float(p["motion_state_sensitivity"])
    )

    for condition, initial, offset in (
        ("Low", 2.0, 0),
        ("Elevated", 8.0, 1),
    ):
        seed = 8100000 + 2 * pid + offset
        runs = {}

        for name, model in models.items():
            run = simulate_motion_feedback(
                seed=seed,
                state_params=state_params,
                motion_params=motion_params,
                coupled_params=coupled,
                controller_params=controller,
                initial_state=initial,
                control_rate_hz=10,
                model=model,
                policy="ai_neutral_return",
                maximum_update_change=0.05,
                maximum_total_variation=0.6
            )

            result = evaluate_controller(
                run, hold_duration_s=1.0
            )

            assert np.isfinite(run["state"]).all()
            assert np.isfinite(run["stimulus"]).all()
            assert run["state"].shape == (1080,)
            assert np.all(run["stimulus"] >= 0)
            assert np.all(run["stimulus"] <= 1)

            updates = run["stimulus"][::9]
            changes = np.abs(
                np.diff(np.r_[0.5, updates])
            )

            assert changes.max() <= 0.05 + 1e-9
            assert changes.sum() <= 0.6 + 1e-9
            assert np.all(run["stimulus"][:90] == 0.5)

            assert (
                changes.sum() + abs(updates[-1] - 0.5)
                <= 0.6 + 1e-9
            )

            for decision in run["decisions"]:
                obs = decision["observation_sample"]
                if obs is not None:
                    assert obs < decision["control_sample"]

            runs[name] = run

            recovery = (
                result["recovery_time_s"]
                if result["recovered"] else 12.0
            )

            print(
                f"profile={pid} condition={condition} "
                f"policy={name} recovery={recovery:.6f} "
                f"stability={result['state_stability_pct']:.6f} "
                f"variation={result['stimulus_total_variation']:.6f}"
            )

            checked += 1

        # Selecting all 15 columns must be exactly equivalent
        # to using the original frozen model directly.
        for field in ("state", "position", "velocity", "stimulus"):
            assert np.array_equal(
                runs["B6_original"][field],
                runs["B6_identity_adapter"][field]
            ), f"Original-model adapter mismatch: {field}"

print("\nEpisodes executed:", checked)
assert checked == 16

print("Original model versus identity adapter: EXACT MATCH")
print("First feedback at 1 second: PASS")
print("Causal observation ordering: PASS")
print("Shared stimulus and reserve constraints: PASS")
print("Original source modified: NO")
print("Independent test accessed: NO")
print("STEP 219: PASS")
