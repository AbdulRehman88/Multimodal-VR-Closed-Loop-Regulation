from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import math
import re
import sys

import numpy as np
from tqdm import tqdm

ROOT = Path(r"D:\Dr_Abdul_Rehman\Multimodal-VR-Comparative-Revision")
CONFIG = ROOT / "revision/configs"
RESULTS = ROOT / "revision/results/confirmatory_200"
REPORT = CONFIG / "confirmatory_results_integrity_audit.json"

POLICIES = tuple(f"B{i}" for i in range(9))
CONDITIONS = ("Low", "Elevated")
TOL = 1e-8

def require(ok, message):
    if not ok:
        raise AssertionError(message)

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

print("STEP 234 - FORENSIC CONFIRMATORY RESULTS AUDIT")

require(not REPORT.exists(), "Audit report already exists")
require(RESULTS.is_dir(), "Confirmatory results directory missing")

expected_hashes = {
    "comparative_protocol_lock.json":
        "3d97a915e3a0fe62becda2ec6d423317680c3ece5daac7b2a9479bfc3a8c1b2c",
    "confirmatory_profiles_200.csv":
        "e2ab295d7adfafb0926947007d3cfa61541afb9d0c55c1f71df193f33aec5b23",
    "confirmatory_episode_seeds.csv":
        "05fe01c4d0cc28d272603c774a25ad811c77c1e14481d456295345c79a5f6143",
    "confirmatory_population_addendum.json":
        "f3540238f0caf1bc8a41311688938ecbdac01e714e102fa803cd64df508d42ef",
    "confirmatory_runner_lock.json":
        "8ba80114fb9a3349f9a0816a7681147d6edc713fd77a5f02e7ac6837c20c79c2",
}

for name, expected in expected_hashes.items():
    require(
        sha(CONFIG / name) == expected,
        f"Frozen configuration changed: {name}"
    )

runner = ROOT / "revision/run_confirmatory_comparison.py"
require(
    sha(runner)
    == "a9a1d6e5b94ac30789f55e87b2306c61fda2a14a11215a0d0e14fd4e4e3923e1",
    "Frozen runner changed"
)

lock = json.loads(
    (CONFIG / "comparative_protocol_lock.json").read_text(
        encoding="utf-8"
    )
)

for relative, expected in lock["artifact_sha256"].items():
    require(
        sha(ROOT / relative) == expected,
        f"Frozen implementation changed: {relative}"
    )

profiles = read_csv(CONFIG / "confirmatory_profiles_200.csv")
seed_rows = read_csv(CONFIG / "confirmatory_episode_seeds.csv")

require(len(profiles) == 200, "Incorrect profile count")
require(len(seed_rows) == 400, "Incorrect seed ledger count")

profile_ids = [int(p["profile"]) for p in profiles]
require(len(set(profile_ids)) == 200, "Duplicate profile IDs")

ledger = {
    (int(r["profile"]), r["condition"]): int(r["episode_seed"])
    for r in seed_rows
}
require(len(ledger) == 400, "Duplicate seed-ledger key")

expected_cases = {
    (pid, condition)
    for pid in profile_ids
    for condition in CONDITIONS
}
require(set(ledger) == expected_cases, "Seed ledger incomplete")

files = sorted(RESULTS.glob("profile_*.json"))
require(len(files) == 200, "Expected exactly 200 checkpoints")

counts = Counter()
seen = set()
inventory = []
reserve_failures_B5 = 0

for path in tqdm(files, desc="Integrity audit", unit="profile"):
    match = re.fullmatch(r"profile_(\d+)\.json", path.name)
    require(match is not None, f"Invalid checkpoint name: {path.name}")
    pid = int(match.group(1))
    require(pid in profile_ids, f"Unexpected profile: {pid}")

    records = json.loads(path.read_text(encoding="utf-8"))
    require(
        isinstance(records, list) and len(records) == 18,
        f"Incomplete checkpoint: {pid}"
    )

    local_keys = set()

    for r in records:
        policy = r["policy"]
        condition = r["condition"]
        key = (pid, condition, policy)

        require(
            int(r["profile"]) == pid,
            f"Profile mismatch: {key}"
        )
        require(
            condition in CONDITIONS and policy in POLICIES,
            f"Invalid condition/policy: {key}"
        )
        require(
            int(r["episode_seed"]) == ledger[(pid, condition)],
            f"Seed mismatch: {key}"
        )
        require(key not in seen, f"Duplicate episode: {key}")

        seen.add(key)
        local_keys.add((condition, policy))
        counts[policy] += 1

        stimulus = np.asarray(
            r["stimulus_at_updates"], dtype=float
        )

        require(
            stimulus.shape == (120,),
            f"Stimulus update count mismatch: {key}"
        )
        require(
            np.isfinite(stimulus).all(),
            f"Nonfinite stimulus: {key}"
        )
        require(
            (stimulus >= -TOL).all()
            and (stimulus <= 1 + TOL).all(),
            f"Stimulus out of range: {key}"
        )

        # Controller decisions at t=0,9,...,81 must be neutral.
        require(
            np.allclose(stimulus[:10], 0.5, rtol=0, atol=TOL),
            f"Intervention before first observation: {key}"
        )

        changes = np.abs(
            np.diff(np.r_[0.5, stimulus])
        )
        variation = float(changes.sum())
        terminal_distance = abs(float(stimulus[-1]) - 0.5)
        feasible = variation + terminal_distance <= 0.6 + TOL

        require(
            changes.max() <= 0.05 + TOL,
            f"Update-size violation: {key}"
        )
        require(
            variation <= 0.6 + TOL,
            f"Variation-budget violation: {key}"
        )
        require(
            abs(variation - float(r["variation"])) <= TOL,
            f"Stored variation mismatch: {key}"
        )
        require(
            abs(terminal_distance -
                float(r["terminal_neutral_distance"])) <= TOL,
            f"Stored terminal distance mismatch: {key}"
        )
        require(
            bool(r["neutral_return_feasible"]) == feasible,
            f"Stored return-feasibility mismatch: {key}"
        )

        if policy != "B5":
            require(feasible, f"Reserve invariant violated: {key}")
        elif not feasible:
            reserve_failures_B5 += 1

        for field, minimum, maximum in (
            ("recovery_s", 0.0, 12.0),
            ("stability_pct", 0.0, 100.0),
        ):
            value = float(r[field])
            require(
                math.isfinite(value)
                and minimum - TOL <= value <= maximum + TOL,
                f"Invalid outcome {field}: {key}"
            )

        require(
            isinstance(r["recovered"], bool),
            f"Invalid recovery indicator: {key}"
        )
        require(
            r["recovered"] or
            abs(float(r["recovery_s"]) - 12.0) <= TOL,
            f"Nonrecovery handling mismatch: {key}"
        )
        require(
            math.isfinite(float(r["overshoot"])),
            f"Invalid overshoot: {key}"
        )

        for field in ("state_sha256", "motion_sha256"):
            require(
                isinstance(r[field], str)
                and re.fullmatch("[0-9a-f]{64}", r[field]) is not None,
                f"Malformed trajectory hash: {key}/{field}"
            )

        for decision in r["decisions"]:
            observation = decision.get("observation_sample")
            if observation is not None:
                require(
                    observation < decision["control_sample"],
                    f"Temporal causality violation: {key}"
                )

    expected_local = {
        (condition, policy)
        for condition in CONDITIONS
        for policy in POLICIES
    }
    require(local_keys == expected_local,
            f"Policy coverage mismatch: {pid}")

    inventory.append({
        "profile": pid,
        "file": path.name,
        "sha256": sha(path),
        "episodes": len(records)
    })

expected_episodes = {
    (pid, condition, policy)
    for pid in profile_ids
    for condition in CONDITIONS
    for policy in POLICIES
}

require(seen == expected_episodes, "Missing or unexpected episodes")
require(len(seen) == 3600, "Incorrect total episode count")

for policy in POLICIES:
    require(counts[policy] == 400,
            f"Wrong episode count for {policy}")

report = {
    "step": 234,
    "status": "PASS",
    "profiles": 200,
    "conditions": list(CONDITIONS),
    "policies": list(POLICIES),
    "episodes": len(seen),
    "episodes_per_policy": dict(sorted(counts.items())),
    "validation": {
        "frozen_hashes": "PASS",
        "profile_coverage": "PASS",
        "seed_matching": "PASS",
        "complete_policy_coverage": "PASS",
        "stimulus_action_limits": "PASS",
        "total_variation_limits": "PASS",
        "reserve_invariant_except_B5": "PASS",
        "saved_metrics_finite_and_bounded": "PASS",
        "decision_timestamp_order": "PASS"
    },
    "B5_episodes_without_reserved_return_capacity":
        reserve_failures_B5,
    "checkpoint_inventory": inventory,
    "runner_sha256": sha(runner),
    "protocol_sha256": sha(
        CONFIG / "comparative_protocol_lock.json"
    ),
    "comparative_statistics_computed": False
}

REPORT.write_text(
    json.dumps(report, indent=2) + "\n",
    encoding="utf-8"
)

print("\nRESULTS INTEGRITY")
print("Verified checkpoints:", len(files))
print("Verified unique episodes:", len(seen))
print("Episodes per policy:")
for policy in POLICIES:
    print(f"  {policy}: {counts[policy]}")

print("\nB5 episodes without neutral-return capacity:",
      reserve_failures_B5)
print("All other reserve invariants: PASS")
print("Frozen implementation integrity: PASS")
print("Checkpoint inventory: COMPLETE")
print("Comparative statistics computed: NO")
print("Audit report:", REPORT)
print("Audit SHA256:", sha(REPORT))
print("STEP 234: PASS")
