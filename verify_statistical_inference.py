from pathlib import Path
import csv
import sys
import numpy as np
from tqdm import tqdm

ROOT = Path(r"D:\Dr_Abdul_Rehman\Multimodal-VR-Publication-Release")
BASE = ROOT / "results/final_independent_evaluation/final_results"

def load(name):
    with (BASE / name).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def require(condition, message):
    if not condition:
        raise AssertionError(message)

episodes = load("all_1200_policy_episodes.csv")
audits = load("all_400_ai_intervention_audits.csv")
effects = load("profile_cluster_bootstrap_contrasts.csv")
safety = load("final_safety_rate_confidence_intervals.csv")

E = {(int(r["profile"]), r["condition"], r["policy"]): r
     for r in episodes}
A = {(int(r["profile"]), r["condition"]): r for r in audits}
profiles = sorted({int(r["profile"]) for r in episodes})

require(len(profiles) == 200, "Expected 200 independent profiles")
require(len(E) == 1200 and len(A) == 400, "Incomplete evidence")
require(len(effects) == 18 and len(safety) == 18,
        "Unexpected statistical table length")

rng = np.random.default_rng(2026)
indices = rng.integers(0, 200, size=(10000, 200))

def bootstrap_ci(values):
    means = values[indices].mean(axis=1)
    return np.quantile(means, [0.025, 0.975])

def conditions(label):
    return ("Low", "Elevated") if label == "Both" else (label,)

comparators = {
    "AI_minus_neutral": "neutral",
    "AI_minus_matched_reserved_oracle": "matched_reserved_oracle",
}

print("STEP 207 - COMPLETE STATISTICAL INFERENCE AUDIT")
print("Bootstrap: 10000 profile-cluster replicates, seed 2026")

max_effect_ci_error = {}
for r in tqdm(effects, desc="Effect intervals", unit="contrast"):
    comparator = comparators[r["comparison"]]
    metric = r["metric"]
    conds = conditions(r["condition"])

    values = np.array([
        np.mean([
            float(E[p, c, "reserved_ai"][metric]) -
            float(E[p, c, comparator][metric])
            for c in conds
        ])
        for p in profiles
    ])

    require(abs(values.mean() - float(r["mean_difference"])) < 1e-9,
            f"Effect mean mismatch: {r}")

    positive = int(np.sum(values > 1e-10))
    negative = int(np.sum(values < -1e-10))
    require(positive == int(r["profiles_positive"]),
            f"Positive profile count mismatch: {r}")
    require(negative == int(r["profiles_negative"]),
            f"Negative profile count mismatch: {r}")

    calculated = bootstrap_ci(values)
    stored = np.array([float(r["CI_low"]), float(r["CI_high"])])
    error = float(np.max(np.abs(calculated - stored)))

    max_effect_ci_error[metric] = max(
        max_effect_ci_error.get(metric, 0.0), error
    )

    # Independent RNG streams can produce slightly different
    # percentile limits despite the same bootstrap method.
    tolerance = 0.03 if metric == "stability_pct" else 0.002
    require(error <= tolerance,
            f"Effect bootstrap disagreement exceeds tolerance: {r}, error={error}")

print("All 18 paired effects and bootstrap intervals: PASS")

safety_metrics = {
    "adverse_recovery":
        lambda ai, ne, a:
        float(ai["recovery_s"]) > float(ne["recovery_s"]) + 1e-10,
    "adverse_stability":
        lambda ai, ne, a:
        float(ai["stability_pct"]) < float(ne["stability_pct"]) - 1e-10,
    "adverse_either":
        lambda ai, ne, a:
        (float(ai["recovery_s"]) > float(ne["recovery_s"]) + 1e-10)
        or (float(ai["stability_pct"]) < float(ne["stability_pct"]) - 1e-10),
    "opposite_episode":
        lambda ai, ne, a: int(a["opposite_actions"]) > 0,
    "unnecessary_episode":
        lambda ai, ne, a: int(a["unnecessary_actions"]) > 0,
    "ai_reserve_boundary_reached":
        lambda ai, ne, a: int(a["ai_reserve_boundary_reached"]) > 0,
}

maximum_safety_error = 0.0

for r in tqdm(safety, desc="Safety intervals", unit="interval"):
    conds = conditions(r["condition"])
    test = safety_metrics[r["metric"]]

    matrix = np.array([
        [
            bool(test(E[p, c, "reserved_ai"],
                      E[p, c, "neutral"], A[p, c]))
            for c in conds
        ]
        for p in profiles
    ], dtype=float)

    count = int(matrix.sum())
    rate = 100 * matrix.mean()
    ci = 100 * bootstrap_ci(matrix.mean(axis=1))

    require(count == int(r["count"]), f"Safety count mismatch: {r}")
    require(int(r["episodes"]) == 200 * len(conds),
            f"Episode denominator mismatch: {r}")
    require(abs(rate - float(r["rate_pct"])) < 1e-9,
            f"Safety rate mismatch: {r}")

    stored = np.array([
        float(r["CI_low_pct"]), float(r["CI_high_pct"])
    ])
    error = float(np.max(np.abs(ci - stored)))
    maximum_safety_error = max(maximum_safety_error, error)
    require(error < 1e-8,
            f"Safety bootstrap interval mismatch: {r}, error={error}")

print("All 18 safety-rate bootstrap intervals: PASS")

print("\nSTATISTICAL REPRODUCIBILITY SUMMARY")
print("Paired effect intervals:", len(effects), "PASS")
print("Safety-rate intervals:", len(safety), "PASS")
print("Maximum safety CI difference:", f"{maximum_safety_error:.3e}")
print("Maximum paired CI differences by metric:")
for metric, error in max_effect_ci_error.items():
    print(f"  {metric}: {error:.8f}")

print("\nSTEP 207: PASS")
print("Published results modified: NO")
print("Model retrained: NO")
print("Simulation rerun: NO")
