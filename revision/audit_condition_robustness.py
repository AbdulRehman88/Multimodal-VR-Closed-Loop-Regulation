from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(r"D:\Dr_Abdul_Rehman\Multimodal-VR-Comparative-Revision")
BASE = ROOT / "revision/results"
SOURCE = BASE / "confirmatory_200"
ANALYSIS = BASE / "statistical_analysis"
OUT = BASE / "condition_robustness"

EXPECTED_MANIFEST = (
    "6a9685ea20bd0261a213b300a9b5df739bac93753a028a58af161e2003076ce4"
)
EXPECTED_AUDIT = (
    "76a31abe9b8f6df5c60656e2f789afad99160801809013751d6eda7aff822b7d"
)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def require(condition, message):
    if not condition:
        raise AssertionError(message)

print("STEP 236 - CONDITION-SPECIFIC ROBUSTNESS AUDIT")

require(not OUT.exists(), "Step 236 results already exist")

manifest_path = ANALYSIS / "statistical_analysis_manifest.json"
audit_path = (
    ROOT / "revision/configs"
    / "confirmatory_results_integrity_audit.json"
)

require(
    sha(manifest_path) == EXPECTED_MANIFEST,
    "Statistical analysis manifest changed"
)
require(
    sha(audit_path) == EXPECTED_AUDIT,
    "Confirmatory audit changed"
)

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
audit = json.loads(audit_path.read_text(encoding="utf-8"))

for name, expected in manifest["outputs"].items():
    require(
        sha(ANALYSIS / name) == expected,
        f"Step 235 result changed: {name}"
    )

records = []
for entry in tqdm(
    audit["checkpoint_inventory"],
    desc="Robustness verification",
    unit="profile"
):
    path = SOURCE / entry["file"]
    require(
        sha(path) == entry["sha256"],
        f"Checkpoint changed: {path.name}"
    )
    for r in json.loads(path.read_text(encoding="utf-8")):
        records.append({
            "profile": int(r["profile"]),
            "condition": r["condition"],
            "policy": r["policy"],
            "recovery_s": float(r["recovery_s"]),
            "stability_pct": float(r["stability_pct"]),
            "variation": float(r["variation"]),
            "recovered": bool(r["recovered"]),
            "return_feasible": bool(
                r["neutral_return_feasible"]
            ),
            "overshoot": float(r["overshoot"])
        })

df = pd.DataFrame(records)
require(len(df) == 3600, "Episode count mismatch")

policies = [f"B{i}" for i in range(9)]
conditions = ["Low", "Elevated"]
profiles = sorted(df["profile"].unique())

require(len(profiles) == 200, "Profile count mismatch")

rng = np.random.default_rng(2026)
draws = rng.integers(0, 200, size=(10000, 200))

results = []
adverse = []

for condition in conditions:
    subset = df[df["condition"] == condition]
    table = subset.pivot(
        index="profile", columns="policy",
        values=["recovery_s", "stability_pct", "variation"]
    ).reindex(index=profiles)

    require(not table.isna().any().any(),
            f"Incomplete {condition} data")

    for comparator in policies:
        if comparator == "B6":
            continue

        for endpoint in ("recovery_s", "stability_pct"):
            ai = table[(endpoint, "B6")].to_numpy(float)
            other = table[(endpoint, comparator)].to_numpy(float)

            effects = (
                other - ai if endpoint == "recovery_s"
                else ai - other
            )

            ci = np.quantile(
                effects[draws].mean(axis=1),
                [0.025, 0.975]
            )

            results.append({
                "condition": condition,
                "comparator": comparator,
                "endpoint": endpoint,
                "mean_improvement": effects.mean(),
                "CI_low": ci[0],
                "CI_high": ci[1],
                "profiles_favor_B6": int(
                    (effects > 1e-10).sum()
                ),
                "profiles_favor_comparator": int(
                    (effects < -1e-10).sum()
                ),
                "profiles_tied": int(
                    (abs(effects) <= 1e-10).sum()
                ),
                "inference_role": "secondary_condition_specific"
            })

        ai_rec = table[("recovery_s", "B6")].to_numpy(float)
        co_rec = table[("recovery_s", comparator)].to_numpy(float)
        ai_st = table[("stability_pct", "B6")].to_numpy(float)
        co_st = table[("stability_pct", comparator)].to_numpy(float)

        worse_rec = ai_rec > co_rec + 1e-10
        worse_st = ai_st < co_st - 1e-10

        adverse.append({
            "condition": condition,
            "comparator": comparator,
            "episodes": 200,
            "B6_worse_recovery": int(worse_rec.sum()),
            "B6_worse_stability": int(worse_st.sum()),
            "B6_worse_either": int(
                np.logical_or(worse_rec, worse_st).sum()
            )
        })

condition_df = pd.DataFrame(results)
adverse_df = pd.DataFrame(adverse)

summary = df.groupby(
    ["condition", "policy"], as_index=False
).agg(
    episodes=("profile", "size"),
    recovery_s=("recovery_s", "mean"),
    stability_pct=("stability_pct", "mean"),
    variation=("variation", "mean"),
    recovery_rate=("recovered", "mean"),
    feasible_rate=("return_feasible", "mean"),
    overshoot=("overshoot", "mean")
)
summary["recovery_rate"] *= 100
summary["feasible_rate"] *= 100

# Exact paired B5/B6 resource-expenditure audit.
paired = df[df["policy"].isin(["B5", "B6"])].pivot(
    index=["profile", "condition"],
    columns="policy",
    values=["variation", "recovery_s", "stability_pct"]
)

var_delta = (
    paired[("variation", "B6")]
    - paired[("variation", "B5")]
)

b5 = df[df["policy"] == "B5"]

mechanism = {
    "B5_nonreturn_feasible_count": int(
        (~b5["return_feasible"]).sum()
    ),
    "B5_nonrecovered_count": int(
        (~b5["recovered"]).sum()
    ),
    "B5_positive_overshoot_count": int(
        (b5["overshoot"] > 1e-10).sum()
    ),
    "B6_minus_B5_variation_mean": float(var_delta.mean()),
    "B6_minus_B5_variation_min": float(var_delta.min()),
    "B6_minus_B5_variation_max": float(var_delta.max()),
    "B5_B6_equal_variation_episode_count": int(
        (var_delta.abs() <= 1e-10).sum()
    )
}

require(
    mechanism["B5_nonreturn_feasible_count"] == 357,
    "Reserve-mechanism audit mismatch"
)

OUT.mkdir(parents=True, exist_ok=False)

condition_df.to_csv(
    OUT / "condition_specific_contrasts.csv", index=False
)
adverse_df.to_csv(
    OUT / "condition_specific_adverse.csv", index=False
)
summary.to_csv(
    OUT / "condition_specific_policy_summary.csv", index=False
)

report_path = OUT / "mechanism_audit.json"
report_path.write_text(
    json.dumps(mechanism, indent=2) + "\n",
    encoding="utf-8"
)

print("\nCONDITION-SPECIFIC POLICY SUMMARY")
print(summary.round(5).to_string(index=False))

print("\nPRIMARY COMPARATORS BY CONDITION")
print(
    condition_df[
        condition_df["comparator"].isin(
            ["B1", "B2", "B3", "B4"]
        )
    ][[
        "condition", "comparator", "endpoint",
        "mean_improvement", "CI_low", "CI_high",
        "profiles_favor_B6", "profiles_favor_comparator"
    ]].round(5).to_string(index=False)
)

print("\nADVERSE EPISODES AGAINST PRIMARY COMPARATORS")
print(
    adverse_df[
        adverse_df["comparator"].isin(
            ["B1", "B2", "B3", "B4"]
        )
    ].to_string(index=False)
)

print("\nRESERVE-MECHANISM AUDIT")
print(json.dumps(mechanism, indent=2))

print("\nOriginal checkpoints unchanged: PASS")
print("Step 235 analysis unchanged: PASS")
print("Independent profiles: 200")
print("Condition-specific results: SECONDARY")
print("STEP 236: PASS")
