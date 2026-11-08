from pathlib import Path
import sys, ast, hashlib, json, csv, importlib.util
import numpy as np
import pandas as pd
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
from revision.baseline_policies import (
    FixedSchedulePolicy, MotionThresholdPolicy
)

def require(ok, message):
    if not ok:
        raise AssertionError(message)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def extract_definitions(path, names, namespace):
    # Compile only requested definitions; never execute
    # module-level smoke tests or file-generation code.
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    found = [
        node for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name in names
    ]
    require(
        {n.name for n in found} == set(names),
        f"Missing definitions in {path}"
    )
    module = ast.fix_missing_locations(
        ast.Module(body=found, type_ignores=[])
    )
    exec(compile(module, str(path), "exec"), namespace)
    return namespace

print("STEP 231 - NINE-POLICY DEVELOPMENT PARITY")

config = ROOT / "revision/configs"
lock = config / "comparative_protocol_lock.json"
require(
    sha(lock) == "3d97a915e3a0fe62becda2ec6d423317680c3ece5daac7b2a9479bfc3a8c1b2c",
    "Frozen protocol mismatch"
)

manifest = json.loads(lock.read_text(encoding="utf-8"))
for relative, expected in manifest["artifact_sha256"].items():
    require(
        sha(ROOT / relative) == expected,
        f"Frozen artifact changed: {relative}"
    )

expected_cohort_hashes = {
    "confirmatory_profiles_200.csv":
        "e2ab295d7adfafb0926947007d3cfa61541afb9d0c55c1f71df193f33aec5b23",
    "confirmatory_episode_seeds.csv":
        "05fe01c4d0cc28d272603c774a25ad811c77c1e14481d456295345c79a5f6143",
    "confirmatory_population_addendum.json":
        "f3540238f0caf1bc8a41311688938ecbdac01e714e102fa803cd64df508d42ef",
}
for name, expected in expected_cohort_hashes.items():
    require(sha(config / name) == expected,
            f"Confirmatory cohort integrity failure: {name}")

frozen_model = joblib.load(
    ROOT / "results/frozen_one_second_ai_baseline"
    / "motion_time_unweighted.joblib"
)
b2_model = joblib.load(ROOT / "revision/models/B2_time_only.joblib")
b3_model = joblib.load(ROOT / "revision/models/B3_motion_only.joblib")

# Load the EXACT earlier FeatureAdapter definition,
# without executing its module-level tests.
feature_ns = {"np": np}
extract_definitions(
    ROOT / "revision/test_information_controllers.py",
    ["FeatureAdapter"], feature_ns
)
FeatureAdapter = feature_ns["FeatureAdapter"]

# Use the EXACT frozen sham adapter and donor-selection
# definitions without executing the original smoke tests.
training_file = Path(
    r"D:\Dr_Abdul_Rehman\Multimodal-VR"
    r"\results\motion_state_dataset\development_windows.csv"
)
require(
    sha(training_file)
    == "f4e5f898b66bba7cd1fe0f24d83c856a9a679d17e692e2913319bd02ba358797",
    "Training donor dataset mismatch"
)

data = pd.read_csv(training_file)
training = data.loc[data["split"] == "train"]
features = [
    "speed_mean", "speed_std", "speed_max", "speed_q90",
    "speed_rms", "acceleration_mean", "acceleration_std",
    "acceleration_q90", "velocity_x_std", "velocity_y_std",
    "velocity_z_std", "position_x_std", "position_y_std",
    "position_z_std"
]
donors = []
expected_indices = np.arange(89, 1080, 9, dtype=int)

for (pid, condition), group in training.groupby(
    ["profile", "condition"], sort=True
):
    group = group.sort_values("sample_index")
    require(
        np.array_equal(
            group["sample_index"].to_numpy(dtype=int),
            expected_indices
        ),
        "Donor indexing mismatch"
    )
    values = group[features].to_numpy(dtype=float)
    require(values.shape == (111, 14), "Donor shape mismatch")
    donors.append({
        "profile": int(pid),
        "condition": str(condition),
        "values": values.copy()
    })

require(len(donors) == 120, "Incorrect donor count")

sham_ns = {
    "np": np,
    "hashlib": hashlib,
    "require": require,
    "donors": donors,
}
extract_definitions(
    ROOT / "revision/implement_sham_motion.py",
    ["ShamMotionAdapter", "choose_donor"],
    sham_ns
)
ShamMotionAdapter = sham_ns["ShamMotionAdapter"]
choose_donor = sham_ns["choose_donor"]

b5_simulator = load_module(
    "step231_b5",
    ROOT / "revision/simulator_B5_budget_only.py"
)
oracle_module = load_module(
    "step231_oracle",
    ROOT / "verify_matched_oracle.py"
)

b4 = MotionThresholdPolicy(
    lower_threshold=0.0015767791,
    upper_threshold=0.0034363065
)

models = {
    "B1": FixedSchedulePolicy(),
    "B2": FeatureAdapter(b2_model, [14]),
    "B3": FeatureAdapter(b3_model, list(range(14))),
    "B4": b4,
    "B6": frozen_model,
}

with (
    ROOT / "configs/frozen_development_profiles.csv"
).open(encoding="utf-8-sig", newline="") as f:
    profiles = list(csv.DictReader(f))

validation = [p for p in profiles if p["split"] == "validation"]
require(len(validation) == 20, "Validation split mismatch")

coupled = CoupledParameters(
    sampling_rate_hz=90,
    duration_s=12.0,
    state_disturbance_sd=0.05
)
controller = ControllerParameters()

policies = [f"B{i}" for i in range(9)]
counts = {p: 0 for p in policies}
records = []

for profile in tqdm(
    validation[:2], desc="Nine-policy parity", unit="profile"
):
    pid = int(profile["profile"])
    sp = StateParameters(
        baseline=float(profile["baseline"]),
        relaxation_rate=float(profile["relaxation_rate"]),
        stimulus_gain=float(profile["stimulus_gain"])
    )
    mp = MotionParameters(
        frequency_hz=float(profile["motion_frequency_hz"]),
        state_sensitivity=float(profile["motion_state_sensitivity"])
    )

    for condition, initial, offset in (
        ("Low", 2.0, 0),
        ("Elevated", 8.0, 1)
    ):
        seed = 8700000 + 2 * pid + offset

        common = dict(
            seed=seed,
            state_params=sp,
            motion_params=mp,
            coupled_params=coupled,
            controller_params=controller,
            initial_state=initial,
            control_rate_hz=10,
            maximum_update_change=0.05,
            maximum_total_variation=0.6
        )

        runs = {}
        runs["B0"] = simulate_motion_feedback(
            **common, model=None, policy="neutral"
        )

        for policy, model in models.items():
            runs[policy] = simulate_motion_feedback(
                **common,
                model=model,
                policy="ai_neutral_return"
            )

        runs["B5"] = b5_simulator.simulate_motion_feedback(
            **common,
            model=frozen_model,
            policy="ai_neutral_return",
            reserve_enabled=False
        )

        runs["B7"] = oracle_module.simulate_reserved_oracle(
            profile, seed, initial
        )

        runs["B8"] = simulate_motion_feedback(
            **common,
            model=ShamMotionAdapter(
                frozen_model, choose_donor(seed)
            ),
            policy="ai_neutral_return"
        )

        # Preserve original B6 exactly when the
        # isolated simulator's reserve is enabled.
        b6_copy = b5_simulator.simulate_motion_feedback(
            **common,
            model=frozen_model,
            policy="ai_neutral_return",
            reserve_enabled=True
        )
        for key in ("state", "position", "velocity", "stimulus"):
            require(
                np.array_equal(runs["B6"][key], b6_copy[key]),
                f"B6 exact-parity failure: {key}"
            )

        require(set(runs) == set(policies),
                "Missing policy result")

        # Every policy must share identical pre-feedback
        # simulated state and motion (first 90 samples).
        for policy, run in runs.items():
            for field in ("state", "position", "velocity"):
                require(
                    np.array_equal(
                        run[field][:90],
                        runs["B6"][field][:90]
                    ),
                    f"Pre-feedback mismatch: {policy}/{field}"
                )

            stimulus = np.asarray(run["stimulus"])
            require(
                stimulus.shape == (1080,),
                f"Incorrect trajectory: {policy}"
            )
            require(
                np.all(stimulus[:90] == 0.5),
                f"Early intervention: {policy}"
            )
            require(
                np.all(np.isfinite(run["state"])),
                f"Nonfinite state: {policy}"
            )

            updates = stimulus[::9]
            increments = np.abs(np.diff(np.r_[0.5, updates]))
            variation = float(increments.sum())
            terminal_distance = abs(float(stimulus[-1]) - 0.5)

            require(
                float(increments.max()) <= 0.05 + 1e-9,
                f"Update bound exceeded: {policy}"
            )
            require(
                variation <= 0.6 + 1e-9,
                f"Total variation exceeded: {policy}"
            )
            if policy != "B5":
                require(
                    variation + terminal_distance <= 0.6 + 1e-9,
                    f"Return reserve violated: {policy}"
                )

            for decision in run.get("decisions", []):
                observation = decision.get("observation_sample")
                if observation is not None:
                    require(
                        observation < decision["control_sample"],
                        f"Causal timing violated: {policy}"
                    )

            metrics = evaluate_controller(
                run, hold_duration_s=1.0
            )
            recovery = (
                float(metrics["recovery_time_s"])
                if metrics["recovered"] else 12.0
            )

            require(
                np.isfinite(recovery)
                and 0 <= recovery <= 12,
                f"Recovery metric invalid: {policy}"
            )

            records.append({
                "profile": pid,
                "condition": condition,
                "policy": policy,
                "recovery_s": recovery,
                "stability_pct": float(
                    metrics["state_stability_pct"]
                ),
                "variation": variation,
                "terminal_neutral_distance": terminal_distance
            })
            counts[policy] += 1

print("\nDEVELOPMENT EPISODES PER POLICY")
for policy in policies:
    print(policy, counts[policy])
    require(counts[policy] == 4,
            f"Incorrect episode count: {policy}")

require(len(records) == 36, "Expected 36 development episodes")

print("\nCHECKS")
print("Original B6 vs reserved simulator copy: EXACT")
print("All nine policies share pre-feedback trajectories: PASS")
print("Matched seed and profile parameters: PASS")
print("Per-update bound: PASS")
print("Cumulative resource budget: PASS")
print("Reserve invariant (except B5): PASS")
print("Independent cohort SHA256 integrity: PASS")
print("Independent cohort simulations executed: NO")
print("Frozen code/model files modified: NO")
print("STEP 231: PASS")
