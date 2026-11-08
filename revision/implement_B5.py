from pathlib import Path
import sys
import csv
import hashlib
import importlib.util
import numpy as np
import joblib
from tqdm import tqdm

ROOT = Path(r"D:\Dr_Abdul_Rehman\Multimodal-VR-Comparative-Revision")
sys.path.insert(0, str(ROOT))

from src.simulator.ai_closed_loop import simulate_motion_feedback
from src.simulator.dynamics import StateParameters
from src.simulator.motion import MotionParameters
from src.simulator.coupled import CoupledParameters
from src.controllers.baselines import ControllerParameters
from src.evaluation.control_metrics import evaluate_controller

SOURCE = ROOT / "src/simulator/ai_closed_loop.py"
TARGET = ROOT / "revision/simulator_B5_budget_only.py"

def require(condition, message):
    if not condition:
        raise AssertionError(message)

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

print("STEP 222 - B5 RESERVE-MECHANISM ABLATION")

require(SOURCE.is_file(), "Original simulator missing")
require(not TARGET.exists(), "Refusing to overwrite B5 implementation")

original_hash = sha256(SOURCE)
source = SOURCE.read_text(encoding="utf-8-sig")

signature = (
    "    maximum_total_variation=None,\n"
    "):"
)

replacement_signature = (
    "    maximum_total_variation=None,\n"
    "    reserve_enabled=True,\n"
    "):"
)

require(source.count(signature) == 1, "Simulator signature mismatch")
source = source.replace(signature, replacement_signature, 1)

old_outward = (
    "                else:\n"
    "                    headroom = max(\n"
    "                        0.0,\n"
    "                        remaining_variation - distance,\n"
    "                    )\n"
    "                    allowed = min(\n"
    "                        abs(requested),\n"
    "                        0.5 * headroom,\n"
    "                    )"
)

new_outward = (
    "                else:\n"
    "                    if reserve_enabled:\n"
    "                        headroom = max(\n"
    "                            0.0,\n"
    "                            remaining_variation - distance,\n"
    "                        )\n"
    "                        allowed = min(\n"
    "                            abs(requested),\n"
    "                            0.5 * headroom,\n"
    "                        )\n"
    "                    else:\n"
    "                        allowed = min(\n"
    "                            abs(requested),\n"
    "                            remaining_variation,\n"
    "                        )"
)

require(source.count(old_outward) == 1, "Reserve update block mismatch")
source = source.replace(old_outward, new_outward, 1)

old_check = (
    "                if (\n"
    "                    abs(stimulus - neutral)\n"
    "                    > remaining_variation + 1e-9\n"
    "                ):"
)

new_check = (
    "                if reserve_enabled and (\n"
    "                    abs(stimulus - neutral)\n"
    "                    > remaining_variation + 1e-9\n"
    "                ):"
)

require(source.count(old_check) == 1, "Reserve assertion block mismatch")
source = source.replace(old_check, new_check, 1)

TARGET.write_text(source, encoding="utf-8")

require(sha256(SOURCE) == original_hash, "Original simulator changed")

spec = importlib.util.spec_from_file_location(
    "simulator_B5_budget_only", TARGET
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

model = joblib.load(
    ROOT / "results/frozen_one_second_ai_baseline"
    / "motion_time_unweighted.joblib"
)

with (
    ROOT / "configs/frozen_development_profiles.csv"
).open(encoding="utf-8-sig", newline="") as f:
    profiles = list(csv.DictReader(f))

validation = [p for p in profiles if p["split"] == "validation"]
require(len(validation) == 20, "Unexpected validation cohort")

coupled = CoupledParameters(
    sampling_rate_hz=90,
    duration_s=12.0,
    state_disturbance_sd=0.05,
)
controller = ControllerParameters()

checked = 0
changed = 0

for p in tqdm(validation[:2], desc="B5 smoke test", unit="profile"):
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
        seed = 8300000 + 2 * pid + offset

        kwargs = dict(
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
            maximum_total_variation=0.6,
        )

        original = simulate_motion_feedback(**kwargs)

        reproduced = module.simulate_motion_feedback(
            **kwargs, reserve_enabled=True
        )

        budget_only = module.simulate_motion_feedback(
            **kwargs, reserve_enabled=False
        )

        # Modified simulator must reproduce the frozen original
        # exactly when the neutral-return reserve is enabled.
        for field in ("state", "position", "velocity", "stimulus"):
            require(
                np.array_equal(original[field], reproduced[field]),
                f"Original-controller mismatch: {field}"
            )

        for name, run in (
            ("B6_reserved", reproduced),
            ("B5_budget_only", budget_only),
        ):
            updates = run["stimulus"][::9]
            changes = np.abs(np.diff(np.r_[0.5, updates]))
            spent = float(changes.sum())
            distance = abs(float(updates[-1]) - 0.5)

            require(np.isfinite(run["state"]).all(),
                    "Nonfinite state trajectory")
            require(changes.max() <= 0.05 + 1e-9,
                    "Maximum update violated")
            require(spent <= 0.6 + 1e-9,
                    "Total variation exceeded")
            require(np.all(run["stimulus"][:90] == 0.5),
                    "Pre-observation intervention detected")

            for d in run["decisions"]:
                obs = d["observation_sample"]
                if obs is not None:
                    require(
                        obs < d["control_sample"],
                        "Future observation accessed"
                    )

            if name == "B6_reserved":
                require(
                    spent + distance <= 0.6 + 1e-9,
                    "Original reserve invariant violated"
                )

            result = evaluate_controller(
                run, hold_duration_s=1.0
            )

            recovery = (
                result["recovery_time_s"]
                if result["recovered"] else 12.0
            )

            print(
                f"{name} profile={pid} condition={condition} "
                f"recovery={recovery:.6f} "
                f"stability={result['state_stability_pct']:.6f} "
                f"variation={spent:.6f} "
                f"return_feasible={spent + distance <= 0.6 + 1e-9}"
            )

            checked += 1

        if not np.array_equal(
            reproduced["stimulus"], budget_only["stimulus"]
        ):
            changed += 1

require(checked == 8, "Unexpected smoke-test count")
require(sha256(SOURCE) == original_hash, "Frozen simulator modified")

print("\nOriginal simulator SHA256:", original_hash)
print("B5 implementation SHA256:", sha256(TARGET))
print("Original vs reserve-enabled copy: EXACT MATCH")
print("Controller episodes checked:", checked)
print("Profile-condition cases with different actions:", changed)
print("Shared update/budget/causality constraints: PASS")
print("Frozen original modified: NO")
print("Independent test accessed: NO")
print("STEP 222: PASS")
