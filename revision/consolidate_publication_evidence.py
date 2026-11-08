from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(r"D:\Dr_Abdul_Rehman\Multimodal-VR-Comparative-Revision")
BASE = ROOT / "revision/results"
STATS = BASE / "statistical_analysis"
OUT = BASE / "publication_evidence"

def require(ok, message):
    if not ok:
        raise AssertionError(message)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

print("STEP 240 - PUBLICATION EVIDENCE CONSOLIDATION")

sources = {
    "statistics": (
        STATS / "statistical_analysis_manifest.json",
        "6a9685ea20bd0261a213b300a9b5df739bac93753a028a58af161e2003076ce4"
    ),
    "endpoint_dependence": (
        BASE / "endpoint_dependence_audit.json",
        "fe3b814014aa94d4aa62a170b4783970a5c53aebcee52233bd3210b52a609127"
    ),
    "post_recovery": (
        BASE / "post_recovery_relapse_audit.json",
        "2f6d6e37349551bdc0c48fbbf173aaae66ab6a950b74bcabc8e7ab268d03b264"
    ),
}

require(not OUT.exists(), "Publication evidence already exists")

for name, (path, expected) in sources.items():
    require(path.is_file(), f"Missing {name}")
    require(sha(path) == expected, f"Modified {name}")

stats = json.loads(sources["statistics"][0].read_text())
dependence = json.loads(sources["endpoint_dependence"][0].read_text())
relapse = json.loads(sources["post_recovery"][0].read_text())

for filename, expected in stats["outputs"].items():
    require(
        sha(STATS / filename) == expected,
        f"Statistical result modified: {filename}"
    )

primary = pd.read_csv(STATS / "primary_holm_comparisons.csv")
secondary = pd.read_csv(STATS / "secondary_bootstrap_comparisons.csv")
policy = pd.read_csv(STATS / "policy_summary.csv")

require(len(primary) == 8, "Primary comparison count mismatch")
require(len(secondary) == 8, "Secondary comparison count mismatch")
require(len(policy) == 9, "Policy summary incomplete")

# Recovery is retained as the single nonredundant
# primary effectiveness presentation.
primary_recovery = primary[
    primary["endpoint"] == "recovery_s"
].copy()

require(
    set(primary_recovery["comparator"]) ==
    {"B1", "B2", "B3", "B4"},
    "Unexpected primary comparators"
)
require(
    (primary_recovery["mean_improvement"] > 0).all(),
    "Unexpected recovery effect direction"
)

primary_recovery["analysis_role"] = (
    "prespecified_primary_endpoint_presentation"
)
primary_recovery["multiplicity_note"] = (
    "Holm adjustment inherited from original eight-test family"
)

condition_rows = []

for item in dependence["condition_interactions"]:
    if item["endpoint"] != "recovery_s":
        continue
    condition_rows.append({
        **item,
        "analysis_role": "secondary_condition_interaction"
    })

require(len(condition_rows) == 4, "Interaction count mismatch")
conditions = pd.DataFrame(condition_rows)

reserve = relapse["policy_summary"]

require(reserve["B5"]["episodes"] == 400, "B5 count mismatch")
require(reserve["B6"]["episodes"] == 400, "B6 count mismatch")
require(
    reserve["B5"]["relapsed_after_qualifying_window_count"] == 77,
    "B5 relapse mismatch"
)
require(
    reserve["B6"]["relapsed_after_qualifying_window_count"] == 0,
    "B6 relapse mismatch"
)
require(
    relapse["replayed_trajectories"] == 800,
    "Replay verification incomplete"
)
require(
    dependence["B5_B6_resource_audit"]["equal_total_variation"]
    == 400,
    "B5/B6 resource parity mismatch"
)

OUT.mkdir(parents=True, exist_ok=False)

primary_recovery.to_csv(
    OUT / "table_primary_recovery.csv",
    index=False, float_format="%.17g"
)

conditions.to_csv(
    OUT / "table_condition_interactions.csv",
    index=False, float_format="%.17g"
)

policy.to_csv(
    OUT / "table_policy_summary.csv",
    index=False, float_format="%.17g"
)

secondary.to_csv(
    OUT / "table_secondary_comparisons.csv",
    index=False, float_format="%.17g"
)

mechanism = {
    "analysis_role": "reserve_ablation_and_posthoc_diagnostics",
    "B5": reserve["B5"],
    "B6": reserve["B6"],
    "resource_parity": dependence["B5_B6_resource_audit"],
    "exploratory_final_3s_stability":
        relapse["exploratory_B6_minus_B5_final_3s_stability"],
    "neutral_return_capacity_failures_B5": 357,
    "interpretation_limits": [
        "Return capacity is not actual terminal neutrality",
        "Late-window stability and relapse are post hoc diagnostics",
        "All findings concern simulated latent states",
        "Equal variation does not imply equal intervention trajectories"
    ]
}

(OUT / "reserve_mechanism.json").write_text(
    json.dumps(mechanism, indent=2) + "\n",
    encoding="utf-8"
)

manifest = {
    "step": 240,
    "status": "PASS",
    "source_sha256": {
        name: expected
        for name, (_, expected) in sources.items()
    },
    "primary_effectiveness_presentation": "recovery_s",
    "stability_interpretation": (
        "Distinct definition but algebraically linked to recovery "
        "for B0-B4 and B6-B8 in this cohort"
    ),
    "primary_inference": (
        "Original Holm-adjusted eight-test results preserved; "
        "no new p-value correction applied"
    ),
    "condition_interactions": "secondary",
    "relapse_and_late_stability": "posthoc",
    "population": "reconstructed_distribution_matched_synthetic",
    "source_simulations_modified": False,
    "files": {}
}

for path in sorted(OUT.iterdir()):
    if path.is_file():
        manifest["files"][path.name] = sha(path)

manifest_file = OUT / "publication_evidence_manifest.json"
manifest_file.write_text(
    json.dumps(manifest, indent=2) + "\n",
    encoding="utf-8"
)

print("\nPRIMARY RECOVERY COMPARISONS")
print(
    primary_recovery[
        ["comparator", "mean_improvement",
         "CI_low", "CI_high", "p_holm"]
    ].round(6).to_string(index=False)
)

print("\nCONDITION INTERACTIONS")
print(
    conditions[
        ["comparator", "low_mean_improvement",
         "elevated_mean_improvement",
         "elevated_minus_low"]
    ].round(6).to_string(index=False)
)

print("\nRESERVE MECHANISM")
print("B5 relapses:", reserve["B5"]["relapsed_after_qualifying_window_count"])
print("B6 relapses:", reserve["B6"]["relapsed_after_qualifying_window_count"])
print("B5/B6 equal-variation cases: 400/400")
print("All 800 B5/B6 trajectory replays matched: PASS")

print("\nPublication evidence:", OUT)
print("Manifest SHA256:", sha(manifest_file))
print("Original statistical results modified: NO")
print("Original simulation results modified: NO")
print("STEP 240: PASS")
