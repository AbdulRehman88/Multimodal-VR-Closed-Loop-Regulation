from pathlib import Path
import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import os
import sys

import joblib
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
from revision.baseline_policies import FixedSchedulePolicy, MotionThresholdPolicy

CONFIG = ROOT / "revision/configs"
RESULTS = ROOT / "revision/results/confirmatory_200"
RUNNER_LOCK = CONFIG / "confirmatory_runner_lock.json"
POLICIES = tuple(f"B{i}" for i in range(9))

EXPECTED = {
    "comparative_protocol_lock.json":
        "3d97a915e3a0fe62becda2ec6d423317680c3ece5daac7b2a9479bfc3a8c1b2c",
    "confirmatory_profiles_200.csv":
        "e2ab295d7adfafb0926947007d3cfa61541afb9d0c55c1f71df193f33aec5b23",
    "confirmatory_episode_seeds.csv":
        "05fe01c4d0cc28d272603c774a25ad811c77c1e14481d456295345c79a5f6143",
    "confirmatory_population_addendum.json":
        "f3540238f0caf1bc8a41311688938ecbdac01e714e102fa803cd64df508d42ef",
}

def require(ok, message):
    if not ok:
        raise AssertionError(message)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value

def definitions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    selected = [
        n for n in tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef))
        and n.name in names
    ]
    require({n.name for n in selected} == set(names),
            f"Definitions missing: {path}")
    module = ast.fix_missing_locations(
        ast.Module(body=selected, type_ignores=[])
    )
    exec(compile(module, str(path), "exec"), namespace)
    return namespace

def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def numeric_values(obj):
    if isinstance(obj, dict):
        for value in obj.values():
            yield from numeric_values(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from numeric_values(value)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield float(obj)

def frozen_thresholds():
    protocol = json.loads(
        (CONFIG / "B1_B4_frozen_protocol.json").read_text(
            encoding="utf-8-sig"
        )
    )
    values = list(numeric_values(protocol))
    thresholds = []
    for approximate in (0.0015767791, 0.0034363065):
        matches = [
            v for v in values
            if abs(v - approximate) < 5e-8
        ]
        require(
            len(matches) >= 1
            and max(matches) - min(matches) < 1e-12,
            "Cannot uniquely recover full-precision B4 threshold"
        )
        thresholds.append(matches[0])
    return thresholds

def integrity():
    for name, expected in EXPECTED.items():
        require(sha(CONFIG / name) == expected,
                f"Frozen input mismatch: {name}")
    protocol = json.loads(
        (CONFIG / "comparative_protocol_lock.json").read_text(
            encoding="utf-8"
        )
    )
    for relative, expected in protocol["artifact_sha256"].items():
        require(
            sha(ROOT / relative) == expected,
            f"Implementation changed: {relative}"
        )

def initialize():
    integrity()
    model = joblib.load(
        ROOT / "results/frozen_one_second_ai_baseline"
        / "motion_time_unweighted.joblib"
    )
    b2 = joblib.load(ROOT / "revision/models/B2_time_only.joblib")
    b3 = joblib.load(ROOT / "revision/models/B3_motion_only.joblib")

    feature_ns = {"np": np}
    definitions(
        ROOT / "revision/test_information_controllers.py",
        ["FeatureAdapter"], feature_ns
    )
    FeatureAdapter = feature_ns["FeatureAdapter"]

    donor_file = Path(
        r"D:\Dr_Abdul_Rehman\Multimodal-VR"
        r"\results\motion_state_dataset\development_windows.csv"
    )
    require(
        sha(donor_file)
        == "f4e5f898b66bba7cd1fe0f24d83c856a9a679d17e692e2913319bd02ba358797",
        "Training donor dataset changed"
    )

    data = pd.read_csv(donor_file)
    train = data.loc[data["split"] == "train"]

    columns = [
        "speed_mean", "speed_std", "speed_max", "speed_q90",
        "speed_rms", "acceleration_mean", "acceleration_std",
        "acceleration_q90", "velocity_x_std", "velocity_y_std",
        "velocity_z_std", "position_x_std", "position_y_std",
        "position_z_std"
    ]

    donors = []
    expected_indices = np.arange(89, 1080, 9)
    for (pid, condition), group in train.groupby(
        ["profile", "condition"], sort=True
    ):
        group = group.sort_values("sample_index")
        require(
            np.array_equal(
                group["sample_index"].to_numpy(int),
                expected_indices
            ),
            "Sham donor time alignment mismatch"
        )
        donors.append({
            "profile": int(pid),
            "condition": str(condition),
            "values": group[columns].to_numpy(float).copy()
        })
    require(len(donors) == 120, "Sham donor count mismatch")

    sham_ns = {
        "np": np, "hashlib": hashlib,
        "donors": donors, "require": require
    }
    definitions(
        ROOT / "revision/implement_sham_motion.py",
        ["ShamMotionAdapter", "choose_donor"],
        sham_ns
    )

    lower, upper = frozen_thresholds()
    models = {
        "B1": FixedSchedulePolicy(),
        "B2": FeatureAdapter(b2, [14]),
        "B3": FeatureAdapter(b3, list(range(14))),
        "B4": MotionThresholdPolicy(lower, upper),
        "B6": model,
    }

    b5 = import_file(
        "confirmatory_b5",
        ROOT / "revision/simulator_B5_budget_only.py"
    )
    oracle = import_file(
        "confirmatory_oracle",
        ROOT / "verify_matched_oracle.py"
    )

    return models, b5, oracle, sham_ns, (lower, upper)

def simulate_case(profile, condition, seed, context):
    models, b5, oracle, sham_ns, _ = context

    initial = {"Low": 2.0, "Elevated": 8.0}[condition]
    state_params = StateParameters(
        baseline=float(profile["baseline"]),
        relaxation_rate=float(profile["relaxation_rate"]),
        stimulus_gain=float(profile["stimulus_gain"])
    )
    motion_params = MotionParameters(
        frequency_hz=float(profile["motion_frequency_hz"]),
        state_sensitivity=float(profile["motion_state_sensitivity"])
    )

    common = dict(
        seed=int(seed),
        state_params=state_params,
        motion_params=motion_params,
        coupled_params=CoupledParameters(
            sampling_rate_hz=90,
            duration_s=12.0,
            state_disturbance_sd=0.05
        ),
        controller_params=ControllerParameters(),
        initial_state=initial,
        control_rate_hz=10,
        maximum_update_change=0.05,
        maximum_total_variation=0.6
    )

    runs = {
        "B0": simulate_motion_feedback(
            **common, model=None, policy="neutral"
        )
    }

    for policy, model in models.items():
        runs[policy] = simulate_motion_feedback(
            **common, model=model, policy="ai_neutral_return"
        )

    runs["B5"] = b5.simulate_motion_feedback(
        **common,
        model=models["B6"],
        policy="ai_neutral_return",
        reserve_enabled=False
    )
    runs["B7"] = oracle.simulate_reserved_oracle(
        profile, int(seed), initial
    )
    runs["B8"] = simulate_motion_feedback(
        **common,
        model=sham_ns["ShamMotionAdapter"](
            models["B6"],
            sham_ns["choose_donor"](seed)
        ),
        policy="ai_neutral_return"
    )

    require(set(runs) == set(POLICIES),
            "Incomplete policy evaluation")

    # Independent code-path parity: the original B6 must
    # agree exactly with the reserve-enabled B5 fork.
    parity = b5.simulate_motion_feedback(
        **common,
        model=models["B6"],
        policy="ai_neutral_return",
        reserve_enabled=True
    )
    for key in ("state", "position", "velocity", "stimulus"):
        require(
            np.array_equal(runs["B6"][key], parity[key]),
            f"B6 reproduction failure: {key}"
        )

    records = []
    reference = runs["B6"]

    for policy in POLICIES:
        run = runs[policy]
        stimulus = np.asarray(run["stimulus"], dtype=float)
        require(stimulus.shape == (1080,),
                "Incorrect stimulus length")
        require(np.isfinite(stimulus).all(),
                "Nonfinite stimulus")
        require(np.all(stimulus[:90] == 0.5),
                f"Early intervention: {policy}")

        for field in ("state", "position", "velocity"):
            require(
                np.array_equal(
                    run[field][:90], reference[field][:90]
                ),
                f"Pre-feedback mismatch: {policy}/{field}"
            )

        updates = stimulus[::9]
        changes = np.abs(np.diff(np.r_[0.5, updates]))
        variation = float(changes.sum())
        terminal = abs(float(stimulus[-1]) - 0.5)

        require(
            changes.max() <= 0.05 + 1e-9,
            f"Update limit: {policy}"
        )
        require(
            variation <= 0.6 + 1e-9,
            f"Variation limit: {policy}"
        )
        if policy != "B5":
            require(
                variation + terminal <= 0.6 + 1e-9,
                f"Reserve invariant: {policy}"
            )

        decisions = run.get("decisions", [])
        for decision in decisions:
            observation = decision.get("observation_sample")
            if observation is not None:
                require(
                    observation < decision["control_sample"],
                    f"Noncausal observation: {policy}"
                )

        metric = evaluate_controller(
            run, hold_duration_s=1.0
        )
        recovery = (
            float(metric["recovery_time_s"])
            if metric["recovered"] else 12.0
        )

        records.append({
            "profile": int(profile["profile"]),
            "condition": condition,
            "episode_seed": int(seed),
            "policy": policy,
            "recovered": bool(metric["recovered"]),
            "recovery_s": recovery,
            "stability_pct": float(
                metric["state_stability_pct"]
            ),
            "variation": variation,
            "terminal_neutral_distance": terminal,
            "neutral_return_feasible": bool(
                variation + terminal <= 0.6 + 1e-9
            ),
            "overshoot": float(
                metric["opposite_boundary_overshoot"]
            ),
            "stimulus_at_updates": updates.tolist(),
            "decisions": json_safe(decisions),
            "state_sha256": hashlib.sha256(
                np.ascontiguousarray(run["state"]).tobytes()
            ).hexdigest(),
            "motion_sha256": hashlib.sha256(
                np.ascontiguousarray(run["position"]).tobytes()
                + np.ascontiguousarray(run["velocity"]).tobytes()
            ).hexdigest()
        })

    return records

def validation_preflight(context):
    with (
        ROOT / "configs/frozen_development_profiles.csv"
    ).open(encoding="utf-8-sig", newline="") as f:
        profiles = [
            p for p in csv.DictReader(f)
            if p["split"] == "validation"
        ]
    require(len(profiles) == 20, "Validation count mismatch")
    checked = 0

    for profile in tqdm(
        profiles[:2], desc="Runner preflight", unit="profile"
    ):
        pid = int(profile["profile"])
        for condition, offset in (("Low", 0), ("Elevated", 1)):
            records = simulate_case(
                profile, condition,
                8800000 + 2 * pid + offset,
                context
            )
            require(len(records) == 9, "Policy count mismatch")
            checked += len(records)

    require(checked == 36, "Preflight episode count mismatch")
    return checked

def atomic_json(path, data):
    require(not path.exists(), f"Refusing overwrite: {path}")
    temporary = path.with_suffix(".tmp")
    require(not temporary.exists(),
            f"Unresolved temporary output: {temporary}")
    with temporary.open("x", encoding="utf-8") as f:
        json.dump(json_safe(data), f, separators=(",", ":"))
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)

def execute(context):
    require(RUNNER_LOCK.exists(), "Runner not locked")
    saved = json.loads(RUNNER_LOCK.read_text(encoding="utf-8"))
    require(
        saved["runner_sha256"] == sha(Path(__file__)),
        "Runner was modified after preflight"
    )
    require(
        saved["cohort_sha256"]
        == EXPECTED["confirmatory_profiles_200.csv"],
        "Cohort lock mismatch"
    )

    profiles = pd.read_csv(
        CONFIG / "confirmatory_profiles_200.csv"
    )
    ledger = pd.read_csv(
        CONFIG / "confirmatory_episode_seeds.csv"
    )
    require(len(profiles) == 200, "Cohort count mismatch")
    require(len(ledger) == 400, "Seed count mismatch")

    seeds = {
        (int(r.profile), str(r.condition)): int(r.episode_seed)
        for r in ledger.itertuples(index=False)
    }
    require(len(seeds) == 400, "Duplicate seed key")

    RESULTS.mkdir(parents=True, exist_ok=True)
    completed = 0

    for item in tqdm(
        profiles.to_dict("records"),
        desc="Confirmatory evaluation",
        unit="profile"
    ):
        pid = int(item["profile"])
        output = RESULTS / f"profile_{pid}.json"

        if output.exists():
            saved_records = json.loads(
                output.read_text(encoding="utf-8")
            )
            require(
                len(saved_records) == 18,
                f"Incomplete checkpoint: {pid}"
            )
            actual_keys = {
                (r["condition"], r["policy"])
                for r in saved_records
            }
            expected_keys = {
                (condition, policy)
                for condition in ("Low", "Elevated")
                for policy in POLICIES
            }
            require(
                actual_keys == expected_keys,
                f"Checkpoint policies mismatch: {pid}"
            )
            completed += 1
            continue

        records = []
        for condition in ("Low", "Elevated"):
            seed = seeds[(pid, condition)]
            records.extend(
                simulate_case(item, condition, seed, context)
            )

        require(len(records) == 18,
                f"Incomplete profile: {pid}")
        atomic_json(output, records)
        completed += 1

    require(completed == 200, "Incomplete confirmatory run")
    print("Profiles completed:", completed)
    print("Expected policy episodes:", completed * 18)
    print("Checkpoint directory:", RESULTS)
    print("Performance comparisons calculated: NO")
    print("STEP 233: EXECUTION COMPLETE")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("prepare", "run"), required=True
    )
    args = parser.parse_args()

    context = initialize()
    print("Frozen inputs and 16 implementations: PASS")
    print("Full-precision B4 thresholds:", context[4])

    if args.mode == "prepare":
        require(not RUNNER_LOCK.exists(),
                "Runner is already locked")
        require(not RESULTS.exists(),
                "Confirmatory output directory already exists")

        checked = validation_preflight(context)
        lock = {
            "step": 232,
            "runner_sha256": sha(Path(__file__)),
            "cohort_sha256":
                EXPECTED["confirmatory_profiles_200.csv"],
            "seed_ledger_sha256":
                EXPECTED["confirmatory_episode_seeds.csv"],
            "development_episodes_checked": checked,
            "confirmatory_episodes_executed": 0,
            "policies": list(POLICIES),
            "status": "locked_before_confirmatory_execution"
        }
        atomic_json(RUNNER_LOCK, lock)

        print("Development episodes verified:", checked)
        print("Runner SHA256:", lock["runner_sha256"])
        print("Runner lock SHA256:", sha(RUNNER_LOCK))
        print("Confirmatory controller simulations: NONE")
        print("STEP 232: PASS")
    else:
        execute(context)

if __name__ == "__main__":
    main()
