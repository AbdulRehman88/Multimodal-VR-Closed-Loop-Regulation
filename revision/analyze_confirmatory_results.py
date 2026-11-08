from pathlib import Path
import hashlib
import json
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(r"D:\Dr_Abdul_Rehman\Multimodal-VR-Comparative-Revision")
CONFIG = ROOT / "revision/configs"
SOURCE = ROOT / "revision/results/confirmatory_200"
OUTPUT = ROOT / "revision/results/statistical_analysis"

AUDIT = CONFIG / "confirmatory_results_integrity_audit.json"
EXPECTED_AUDIT_SHA = (
    "76a31abe9b8f6df5c60656e2f789afad99160801809013751d6eda7aff822b7d"
)

POLICIES = [f"B{i}" for i in range(9)]
CONDITIONS = ["Low", "Elevated"]

PRIMARY = ["B1", "B2", "B3", "B4"]
SECONDARY = ["B0", "B5", "B7", "B8"]

N_BOOT = 10000
N_PERM = 100000
BOOT_SEED = 2026
PERM_SEED = 2027

def require(condition, message):
    if not condition:
        raise AssertionError(message)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

print("STEP 235 - PREDEFINED COMPARATIVE STATISTICAL ANALYSIS")

require(sha(AUDIT) == EXPECTED_AUDIT_SHA,
        "Step 234 audit integrity mismatch")
require(not OUTPUT.exists(),
        "Statistical output directory already exists")

audit = json.loads(AUDIT.read_text(encoding="utf-8"))
require(audit["status"] == "PASS", "Audit did not pass")
require(audit["episodes"] == 3600, "Unexpected episode count")
require(len(audit["checkpoint_inventory"]) == 200,
        "Incomplete checkpoint inventory")

# Verify every result file is unchanged since the forensic audit.
rows = []
for entry in tqdm(
    audit["checkpoint_inventory"],
    desc="Verifying frozen results",
    unit="profile"
):
    path = SOURCE / entry["file"]
    require(path.is_file(), f"Missing checkpoint: {path.name}")
    require(sha(path) == entry["sha256"],
            f"Checkpoint modified: {path.name}")

    records = json.loads(path.read_text(encoding="utf-8"))
    require(len(records) == 18, "Incomplete checkpoint")

    for r in records:
        rows.append({
            "profile": int(r["profile"]),
            "condition": r["condition"],
            "policy": r["policy"],
            "recovery_s": float(r["recovery_s"]),
            "stability_pct": float(r["stability_pct"]),
            "variation": float(r["variation"]),
            "recovered": int(r["recovered"]),
            "neutral_return_feasible": int(
                r["neutral_return_feasible"]
            ),
            "overshoot": float(r["overshoot"])
        })

df = pd.DataFrame(rows)
require(len(df) == 3600, "Incorrect number of records")
require(
    not df.duplicated(
        ["profile", "condition", "policy"]
    ).any(),
    "Duplicate profile-condition-policy"
)
require(
    df["profile"].nunique() == 200,
    "Incorrect independent-unit count"
)
require(set(df["policy"]) == set(POLICIES),
        "Policy coverage mismatch")
require(set(df["condition"]) == set(CONDITIONS),
        "Condition coverage mismatch")

counts = df.groupby("policy").size()
require(all(counts == 400), "Unequal policy coverage")

# Matrix orientation: 200 profiles x 2 conditions x 9 policies.
profile_ids = sorted(df["profile"].unique())
index = pd.MultiIndex.from_product(
    [profile_ids, CONDITIONS],
    names=["profile", "condition"]
)

def matrix(metric):
    pivot = df.pivot(
        index=["profile", "condition"],
        columns="policy",
        values=metric
    ).reindex(index=index, columns=POLICIES)
    require(not pivot.isna().any().any(),
            f"Incomplete metric: {metric}")
    return pivot.to_numpy(dtype=float).reshape(200, 2, 9)

recovery = matrix("recovery_s")
stability = matrix("stability_pct")
variation = matrix("variation")
recovered = matrix("recovered")
feasible = matrix("neutral_return_feasible")
overshoot = matrix("overshoot")

# Identical profile bootstrap draws are used across all
# primary and secondary contrasts and endpoints.
rng_boot = np.random.default_rng(BOOT_SEED)
draws = rng_boot.integers(0, 200, size=(N_BOOT, 200))

def bootstrap_ci(profile_effects):
    values = np.asarray(profile_effects, dtype=float)
    require(values.shape == (200,), "Incorrect cluster dimension")
    samples = values[draws].mean(axis=1)
    return np.quantile(samples, [0.025, 0.975])

def paired_effects(policy, endpoint):
    j = POLICIES.index(policy)
    ai = POLICIES.index("B6")

    if endpoint == "recovery_s":
        # Positive: B6 recovers faster.
        differences = recovery[:, :, j] - recovery[:, :, ai]
    elif endpoint == "stability_pct":
        # Positive: B6 has greater stability.
        differences = stability[:, :, ai] - stability[:, :, j]
    else:
        raise ValueError(endpoint)

    return differences.mean(axis=1), differences

# Two-sided paired sign-flip randomization test.
# The profile is the unit of sign-flipping.
# Monte Carlo plus-one correction prevents zero p-values.
rng_perm = np.random.default_rng(PERM_SEED)

def permutation_pvalue(profile_effects):
    values = np.asarray(profile_effects, dtype=float)
    observed = abs(float(values.mean()))
    extreme = 0
    chunk_size = 5000

    for start in range(0, N_PERM, chunk_size):
        size = min(chunk_size, N_PERM - start)
        signs = rng_perm.integers(
            0, 2, size=(size, 200), dtype=np.int8
        )
        signs = 2 * signs.astype(np.float64) - 1.0

        null_effects = np.abs(
            (signs @ values) / 200.0
        )
        extreme += int(
            np.count_nonzero(null_effects >= observed - 1e-12)
        )

    return (extreme + 1) / (N_PERM + 1)

print("\nPRIMARY CONFIRMATORY CONTRASTS")

primary_rows = []

for comparator in tqdm(
    PRIMARY,
    desc="Primary comparisons",
    unit="comparator"
):
    for endpoint in ("recovery_s", "stability_pct"):
        profile_effects, differences = paired_effects(
            comparator, endpoint
        )

        ci = bootstrap_ci(profile_effects)
        p = permutation_pvalue(profile_effects)

        primary_rows.append({
            "comparator": comparator,
            "endpoint": endpoint,
            "direction": "positive_favors_B6",
            "profiles": 200,
            "episodes_per_policy": 400,
            "mean_improvement": float(profile_effects.mean()),
            "CI_low": float(ci[0]),
            "CI_high": float(ci[1]),
            "p_permutation": float(p),
            "profiles_favor_B6": int(
                np.count_nonzero(profile_effects > 1e-10)
            ),
            "profiles_favor_comparator": int(
                np.count_nonzero(profile_effects < -1e-10)
            ),
            "profiles_tied": int(
                np.count_nonzero(abs(profile_effects) <= 1e-10)
            ),
            "low_condition_improvement": float(
                differences[:, 0].mean()
            ),
            "elevated_condition_improvement": float(
                differences[:, 1].mean()
            )
        })

# Holm step-down adjustment over the exactly eight
# prespecified primary tests.
order = sorted(
    range(len(primary_rows)),
    key=lambda i: primary_rows[i]["p_permutation"]
)

adjusted_previous = 0.0
m = len(primary_rows)
require(m == 8, "Incorrect number of primary tests")

for rank, index_i in enumerate(order):
    raw = primary_rows[index_i]["p_permutation"]
    adjusted = min(1.0, (m - rank) * raw)
    adjusted_previous = max(
        adjusted_previous, adjusted
    )
    primary_rows[index_i]["p_holm"] = float(
        adjusted_previous
    )
    primary_rows[index_i]["holm_significant_0_05"] = bool(
        adjusted_previous < 0.05
    )

primary_df = pd.DataFrame(primary_rows)

print(
    primary_df[
        [
            "comparator", "endpoint",
            "mean_improvement", "CI_low", "CI_high",
            "p_permutation", "p_holm"
        ]
    ].round(6).to_string(index=False)
)

# Secondary comparisons: descriptive point estimates
# and cluster bootstrap CIs, not confirmatory p-values.
print("\nSECONDARY COMPARISONS")

secondary_rows = []

for comparator in SECONDARY:
    for endpoint in ("recovery_s", "stability_pct"):
        profile_effects, differences = paired_effects(
            comparator, endpoint
        )
        ci = bootstrap_ci(profile_effects)

        secondary_rows.append({
            "comparator": comparator,
            "endpoint": endpoint,
            "mean_improvement": float(profile_effects.mean()),
            "CI_low": float(ci[0]),
            "CI_high": float(ci[1]),
            "low_condition_improvement": float(
                differences[:, 0].mean()
            ),
            "elevated_condition_improvement": float(
                differences[:, 1].mean()
            ),
            "inference_role": "secondary_descriptive"
        })

secondary_df = pd.DataFrame(secondary_rows)

print(
    secondary_df[
        [
            "comparator", "endpoint",
            "mean_improvement", "CI_low", "CI_high"
        ]
    ].round(6).to_string(index=False)
)

print("\nPOLICY SUMMARY")

summary = (
    df.groupby("policy", as_index=False)
    .agg(
        episodes=("profile", "size"),
        recovery_s=("recovery_s", "mean"),
        stability_pct=("stability_pct", "mean"),
        variation=("variation", "mean"),
        recovery_rate_pct=("recovered", "mean"),
        neutral_return_feasible_pct=(
            "neutral_return_feasible", "mean"
        ),
        overshoot=("overshoot", "mean")
    )
)

summary["recovery_rate_pct"] *= 100
summary["neutral_return_feasible_pct"] *= 100

print(summary.round(6).to_string(index=False))

# Condition-specific policy summaries are important because
# pooled averages may conceal asymmetric behavior.
by_condition = (
    df.groupby(["condition", "policy"], as_index=False)
    .agg(
        episodes=("profile", "size"),
        recovery_s=("recovery_s", "mean"),
        stability_pct=("stability_pct", "mean"),
        variation=("variation", "mean"),
        recovery_rate_pct=("recovered", "mean")
    )
)
by_condition["recovery_rate_pct"] *= 100

# Paired adverse episodes are evaluated against the same
# profile-condition, preserving the matched design.
paired_adverse = []

for comparator in ["B0", "B1", "B2", "B3", "B4", "B5", "B7", "B8"]:
    j = POLICIES.index(comparator)
    ai = POLICIES.index("B6")

    worse_recovery = (
        recovery[:, :, ai] > recovery[:, :, j] + 1e-10
    )
    worse_stability = (
        stability[:, :, ai] < stability[:, :, j] - 1e-10
    )

    paired_adverse.append({
        "comparator": comparator,
        "paired_episodes": 400,
        "B6_worse_recovery_count": int(
            worse_recovery.sum()
        ),
        "B6_worse_stability_count": int(
            worse_stability.sum()
        ),
        "B6_worse_either_count": int(
            np.logical_or(
                worse_recovery, worse_stability
            ).sum()
        ),
        "B6_better_both_count": int(
            np.logical_and(
                recovery[:, :, ai]
                < recovery[:, :, j] - 1e-10,
                stability[:, :, ai]
                > stability[:, :, j] + 1e-10
            ).sum()
        )
    })

adverse_df = pd.DataFrame(paired_adverse)

# Mechanism outcome: inability to guarantee an additional
# return to neutral from the terminal action.
b5 = df[df["policy"] == "B5"]
require(
    int((b5["neutral_return_feasible"] == 0).sum()) == 357,
    "B5 reserve result disagrees with forensic audit"
)

OUTPUT.mkdir(parents=True, exist_ok=False)

artifacts = {
    "primary_holm_comparisons.csv": primary_df,
    "secondary_bootstrap_comparisons.csv": secondary_df,
    "policy_summary.csv": summary,
    "policy_summary_by_condition.csv": by_condition,
    "paired_adverse_outcomes.csv": adverse_df
}

for name, frame in artifacts.items():
    frame.to_csv(
        OUTPUT / name,
        index=False,
        float_format="%.17g"
    )

report = {
    "step": 235,
    "status": "PASS",
    "original_audit_sha256": sha(AUDIT),
    "population": "reconstructed_independent_uniform_synthetic",
    "independent_profiles": 200,
    "paired_conditions": CONDITIONS,
    "total_policy_episodes": 3600,
    "primary_comparators": PRIMARY,
    "secondary_comparators": SECONDARY,
    "primary_endpoints": [
        "recovery_s", "stability_pct"
    ],
    "effect_direction": "positive_favors_B6",
    "paired_unit": "profile_mean_across_two_conditions",
    "bootstrap_replicates": N_BOOT,
    "bootstrap_seed": BOOT_SEED,
    "bootstrap_method": "profile_cluster_percentile",
    "permutation_replicates": N_PERM,
    "permutation_seed": PERM_SEED,
    "permutation_method": "two_sided_profile_sign_flip",
    "permutation_pvalue_correction": "plus_one",
    "multiple_testing": "Holm_eight_primary_tests",
    "secondary_inference": "descriptive_unadjusted_CI",
    "nonrecovery_handling": "12_second_horizon",
    "B5_without_reserved_neutral_return_capacity": 357,
    "checkpoint_hashes_verified": 200,
    "outputs": {}
}

for name in artifacts:
    report["outputs"][name] = sha(OUTPUT / name)

report_path = OUTPUT / "statistical_analysis_manifest.json"

report_path.write_text(
    json.dumps(report, indent=2) + "\n",
    encoding="utf-8"
)

print("\nFINAL ANALYSIS AUDIT")
print("Verified checkpoint hashes: 200/200")
print("Independent clusters: 200")
print("Primary Holm-adjusted tests: 8")
print("Bootstrap replicates:", N_BOOT)
print("Sign-flip permutations per test:", N_PERM)
print("B5 without reserved return capacity: 357/400")
print("Frozen controller results modified: NO")
print("Analysis directory:", OUTPUT)
print("Analysis manifest SHA256:", sha(report_path))
print("STEP 235: PASS")
