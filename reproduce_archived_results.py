from pathlib import Path
import csv
import hashlib
import json
import sys

import numpy as np

ROOT = Path(
    r"D:\Dr_Abdul_Rehman\Multimodal-VR-Publication-Release"
)
BASE = ROOT / "results/final_independent_evaluation/final_results"
MANIFEST = ROOT / "STAGING_MANIFEST.json"

def read_csv(name):
    with (BASE / name).open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        return list(csv.DictReader(f))

def check(condition, message):
    if not condition:
        raise AssertionError(message)

def mean(values):
    return float(np.mean(values))

print("STEP 198 - ARCHIVED RESULTS REPRODUCTION")

manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
for relative, metadata in manifest["files"].items():
    path = ROOT / relative
    check(path.is_file(), f"Missing: {relative}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    check(actual == metadata["sha256"], f"Modified: {relative}")
print("Frozen staging integrity: PASS")

episodes = read_csv("all_1200_policy_episodes.csv")
ledger = read_csv("episode_seed_ledger.csv")
audits = read_csv("all_400_ai_intervention_audits.csv")
contrasts = read_csv("profile_cluster_bootstrap_contrasts.csv")
adverse = read_csv("all_unfavorable_ai_episodes.csv")

check(len(episodes) == 1200, "Expected 1200 episodes")
check(len(ledger) == 400, "Expected 400 seeds")
check(len(audits) == 400, "Expected 400 AI audits")
check(len(contrasts) == 18, "Expected 18 contrasts")
check(len(adverse) == 65, "Expected 65 adverse episodes")

policies = (
    "neutral",
    "reserved_ai",
    "matched_reserved_oracle",
)

records = {}
for row in episodes:
    key = (int(row["profile"]), row["condition"])
    check(row["policy"] in policies, "Unknown policy")
    check(
        (key, row["policy"]) not in records,
        "Duplicate policy episode"
    )
    records[key, row["policy"]] = row

keys = sorted(set(key for key, policy in records))
profiles = sorted(set(profile for profile, condition in keys))

check(len(keys) == 400, "Expected 400 profile-condition keys")
check(len(profiles) == 200, "Expected 200 profiles")
check(
    all(
        sum(profile == p for profile, condition in keys) == 2
        for p in profiles
    ),
    "Each profile must have two conditions"
)
check(
    all(
        (key, policy) in records
        for key in keys for policy in policies
    ),
    "Missing paired controller episode"
)

seed_map = {}
for row in ledger:
    profile = int(row["profile"])
    condition = row["condition"]
    payload = f"final-ai-v1|20261103|{profile}|{condition}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    expected = int.from_bytes(digest[:8], "big") % (2**32)
    actual = int(row["episode_seed"])
    check(expected == actual, "Seed derivation mismatch")
    seed_map[profile, condition] = actual

check(len(seed_map) == 400, "Seed ledger is incomplete")

for key in keys:
    for policy in policies:
        check(
            int(records[key, policy]["episode_seed"]) == seed_map[key],
            "Episode seed disagreement"
        )
print("Paired episodes and deterministic seeds: PASS")

expected_means = {
    "neutral": (2.2619166666666666, 81.15069444444445, 0.0),
    "reserved_ai": (2.0363333333333333, 83.03055555555555, 0.5885),
    "matched_reserved_oracle": (
        1.8945, 84.2125, 0.5535
    ),
}

for policy in policies:
    rows = [records[key, policy] for key in keys]
    actual = (
        mean([float(r["recovery_s"]) for r in rows]),
        mean([float(r["stability_pct"]) for r in rows]),
        mean([float(r["variation"]) for r in rows]),
    )
    check(
        np.allclose(
            actual, expected_means[policy],
            rtol=0, atol=1e-6
        ),
        f"Published means disagree for {policy}"
    )
    print(
        f"{policy}: recovery={actual[0]:.6f}, "
        f"stability={actual[1]:.6f}, "
        f"variation={actual[2]:.6f}"
    )

comparison_map = {
    "AI_minus_neutral": "neutral",
    "AI_minus_matched_reserved_oracle": "matched_reserved_oracle",
}

for row in contrasts:
    condition = row["condition"]
    comparator = comparison_map[row["comparison"]]
    metric = row["metric"]

    selected = [
        k for k in keys
        if condition == "Both" or k[1] == condition
    ]

    differences = {}
    for key in selected:
        a = float(records[key, "reserved_ai"][metric])
        b = float(records[key, comparator][metric])
        differences.setdefault(key[0], []).append(a - b)

    profile_diffs = np.array([
        mean(differences[p]) for p in sorted(differences)
    ])

    check(
        abs(mean(profile_diffs) - float(row["mean_difference"]))
        < 1e-6,
        f"Contrast mismatch: {condition}, {metric}"
    )

    positive = int(np.count_nonzero(profile_diffs > 1e-10))
    negative = int(np.count_nonzero(profile_diffs < -1e-10))
    check(
        positive == int(row["profiles_positive"]) and
        negative == int(row["profiles_negative"]),
        f"Profile sign-count mismatch: {condition}, {metric}"
    )

print("All 18 paired contrast means and sign counts: PASS")

# Independently reproduce the primary recovery bootstrap.
profile_differences = np.array([
    mean([
        float(records[(p, c), "reserved_ai"]["recovery_s"]) -
        float(records[(p, c), "neutral"]["recovery_s"])
        for c in ("Low", "Elevated")
    ])
    for p in profiles
])

rng = np.random.default_rng(2026)
indices = rng.integers(
    0, len(profiles), size=(10000, len(profiles))
)
replicates = profile_differences[indices].mean(axis=1)
ci = np.quantile(replicates, [0.025, 0.975])

reference = next(
    r for r in contrasts
    if r["condition"] == "Both"
    and r["comparison"] == "AI_minus_neutral"
    and r["metric"] == "recovery_s"
)

check(
    np.allclose(
        ci,
        [float(reference["CI_low"]), float(reference["CI_high"])],
        atol=1e-6, rtol=0
    ),
    "Primary recovery bootstrap CI mismatch"
)

print("Primary 10,000-replicate bootstrap CI: PASS")
print("Recovery CI:", [round(x, 8) for x in ci])

check(
    sum(int(a["opposite_actions"]) > 0 for a in audits) == 167,
    "Opposite-direction episode count mismatch"
)
check(
    sum(int(a["opposite_actions"]) for a in audits) == 745,
    "Opposite-direction action count mismatch"
)
check(
    sum(int(a["ai_reserve_boundary_reached"]) for a in audits) == 373,
    "AI reserve-boundary count mismatch"
)

# Independently calculate the formal adverse-outcome definition.
adverse_keys = set()
for key in keys:
    ai = records[key, "reserved_ai"]
    neutral = records[key, "neutral"]
    worse_recovery = (
        float(ai["recovery_s"]) >
        float(neutral["recovery_s"]) + 1e-10
    )
    worse_stability = (
        float(ai["stability_pct"]) <
        float(neutral["stability_pct"]) - 1e-10
    )
    if worse_recovery or worse_stability:
        adverse_keys.add(key)

archived_adverse_keys = {
    (int(r["profile"]), r["condition"]) for r in adverse
}

check(
    adverse_keys == archived_adverse_keys,
    "Adverse-episode membership mismatch"
)

print("Adverse, directional, and reserve audits: PASS")
print("Adverse episodes:", len(adverse_keys))

print("\nSTEP 198: PASS")
print("Frozen data modified: NO")
print("Model retrained: NO")
print("Simulator rerun: NO")
print("Independent full simulation reproduction: PENDING")
