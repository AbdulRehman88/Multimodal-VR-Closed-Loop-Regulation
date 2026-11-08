from pathlib import Path
import hashlib
import json
import csv

ROOT = Path(r"D:\Dr_Abdul_Rehman\Multimodal-VR-Comparative-Revision")
OUT = ROOT / "revision/configs/comparative_protocol_lock.json"

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def require(condition, message):
    if not condition:
        raise AssertionError(message)

print("STEP 224 - FREEZE COMPARATIVE PROTOCOL")

require(not OUT.exists(), "Protocol already frozen")

files = [
    "src/simulator/ai_closed_loop.py",
    "src/simulator/dynamics.py",
    "src/simulator/motion.py",
    "src/models/online_motion_features.py",
    "src/models/causal_features.py",
    "src/evaluation/control_metrics.py",
    "results/frozen_one_second_ai_baseline/motion_time_unweighted.joblib",
    "revision/models/B2_time_only.joblib",
    "revision/models/B3_motion_only.joblib",
    "revision/models/B2_time_only_manifest.json",
    "revision/models/B3_motion_only_manifest.json",
    "revision/baseline_policies.py",
    "revision/simulator_B5_budget_only.py",
    "revision/implement_sham_motion.py",
    "revision/configs/B1_B4_frozen_protocol.json",
    "revision/configs/B8_sham_motion_protocol.json",
]

hashes = {}

for relative in files:
    path = ROOT / relative
    require(path.is_file(), f"Missing required artifact: {relative}")
    hashes[relative] = sha(path)

require(
    hashes["src/simulator/ai_closed_loop.py"]
    == "6faebd1c0d656a0c37ae851a403eaebb27067da776b9c51a9dc02add2c2fe7aa",
    "Original simulator integrity failure"
)

require(
    hashes["results/frozen_one_second_ai_baseline/motion_time_unweighted.joblib"]
    == "1884007ac6581b6653c3227e464256b5ccc56c1e30acf847f543371312f36467",
    "Original estimator integrity failure"
)

require(
    hashes["revision/models/B2_time_only.joblib"]
    == "21630fa2246541d2a7879b503b2240bdc7c3d806c3918181ac3c95a7f910e760",
    "B2 model integrity failure"
)

require(
    hashes["revision/models/B3_motion_only.joblib"]
    == "867466ef2fce36cf90420f947598ecec1ca5e56a0541886c7c6655bbcfe7e6a4",
    "B3 model integrity failure"
)

development = Path(
    r"D:\Dr_Abdul_Rehman\Multimodal-VR"
    r"\results\motion_state_dataset\development_windows.csv"
)
require(
    sha(development)
    == "f4e5f898b66bba7cd1fe0f24d83c856a9a679d17e692e2913319bd02ba358797",
    "Development dataset integrity failure"
)

manifest = {
    "protocol": "Comparative Revision - Independent Confirmatory Simulation",
    "version": "2.0-preconfirmatory",
    "original_release": "10.5281/zenodo.23999113",
    "status": "frozen_before_new_confirmatory_evaluation",
    "policies": {
        "B0": "neutral",
        "B1": "fixed_schedule",
        "B2": "time_only_logistic_regression",
        "B3": "motion_only_logistic_regression",
        "B4": "single_feature_motion_threshold",
        "B5": "motion_time_AI_without_neutral_return_reserve",
        "B6": "original_frozen_reserved_AI",
        "B7": "matched_reserved_privileged_state_oracle",
        "B8": "time_preserved_sham_motion_diagnostic"
    },
    "simulation": {
        "sampling_hz": 90,
        "duration_s": 12,
        "control_hz": 10,
        "first_action_s": 1.0,
        "conditions": {"Low": 2.0, "Elevated": 8.0},
        "maximum_update_change": 0.05,
        "maximum_total_variation": 0.6,
        "neutral": 0.5,
        "reserve_enabled": "All adaptive policies except B5"
    },
    "new_cohort": {
        "profiles": 200,
        "episodes_per_policy": 400,
        "generation": "Use original population generator with new independent random seed",
        "selection_after_outcome_inspection": False
    },
    "primary_endpoints": [
        "recovery_time_s",
        "state_stability_pct"
    ],
    "primary_confirmatory_contrasts": [
        "B6_minus_B1",
        "B6_minus_B2",
        "B6_minus_B3",
        "B6_minus_B4"
    ],
    "secondary_contrasts": [
        "B6_minus_B0",
        "B6_minus_B7",
        "B6_minus_B5",
        "B6_minus_B8"
    ],
    "statistics": {
        "independent_unit": "synthetic_profile",
        "paired_conditions": ["Low", "Elevated"],
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 2026,
        "confidence_intervals": "95% profile-cluster percentile",
        "primary_multiplicity": "Holm correction across 8 endpoint-by-comparator tests",
        "unfavorable_outcomes_retained": True,
        "nonrecovery_time_s": 12.0
    },
    "training_dataset_sha256": sha(development),
    "artifact_sha256": hashes,
    "integrity": {
        "original_release_unchanged": True,
        "original_AI_model_unchanged": True,
        "no_confirmatory_cohort_inspected": True,
        "no_post_confirmatory_tuning": True
    }
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(
    json.dumps(manifest, indent=2) + "\n",
    encoding="utf-8"
)

print("Frozen artifact hashes:", len(hashes))
print("Protocol SHA256:", sha(OUT))
print("New confirmatory cohort accessed: NO")

print("\nPROFILE-GENERATOR DISCOVERY")
candidates = []

for base in (ROOT / "src", ROOT / "experiments"):
    if base.exists():
        for path in base.rglob("*.py"):
            name = path.name.lower()
            if any(term in name for term in (
                "profile", "population", "generate", "independent"
            )):
                candidates.append(path.relative_to(ROOT).as_posix())

for item in sorted(candidates):
    print(item)

print("Generator candidate count:", len(candidates))
print("Protocol:", OUT)
print("STEP 224: PASS")
