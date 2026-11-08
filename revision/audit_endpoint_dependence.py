from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(r"D:\Dr_Abdul_Rehman\Multimodal-VR-Comparative-Revision")
CONFIG = ROOT / "revision/configs"
SOURCE = ROOT / "revision/results/confirmatory_200"
OUTPUT = ROOT / "revision/results/endpoint_dependence_audit.json"

def require(ok, message):
    if not ok:
        raise AssertionError(message)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

print("STEP 237 - ENDPOINT DEPENDENCE AND CONDITION INTERACTIONS")

require(not OUTPUT.exists(), "Step 237 report already exists")

audit_path = CONFIG / "confirmatory_results_integrity_audit.json"
require(
    sha(audit_path) ==
    "76a31abe9b8f6df5c60656e2f789afad99160801809013751d6eda7aff822b7d",
    "Frozen checkpoint audit mismatch"
)

audit = json.loads(audit_path.read_text(encoding="utf-8"))
rows = []
trajectories = {}

for item in tqdm(
    audit["checkpoint_inventory"],
    desc="Endpoint integrity",
    unit="profile"
):
    path = SOURCE / item["file"]
    require(sha(path) == item["sha256"],
            f"Checkpoint changed: {path.name}")

    for r in json.loads(path.read_text(encoding="utf-8")):
        key = (int(r["profile"]), r["condition"], r["policy"])

        rows.append({
            "profile": key[0],
            "condition": key[1],
            "policy": key[2],
            "recovery_s": float(r["recovery_s"]),
            "stability_pct": float(r["stability_pct"]),
            "recovered": bool(r["recovered"]),
            "variation": float(r["variation"])
        })
        trajectories[key] = np.asarray(
            r["stimulus_at_updates"], dtype=float
        )

df = pd.DataFrame(rows)
require(len(df) == 3600, "Unexpected episode count")

# Check whether stability is algebraically determined by
# recovery time for the completed-recovery episodes.
df["stability_implied_by_recovery"] = (
    100.0 * (1.0 - df["recovery_s"] / 12.0)
)
df["dependence_residual"] = (
    df["stability_pct"]
    - df["stability_implied_by_recovery"]
)

print("\nENDPOINT DEPENDENCE BY POLICY")
dependence = []

for policy, group in df.groupby("policy", sort=True):
    recovered = group[group["recovered"]]
    error = np.abs(recovered["dependence_residual"])

    record = {
        "policy": policy,
        "recovered_episodes": int(len(recovered)),
        "max_abs_residual_pp": float(error.max()),
        "exact_within_1e_8": int((error <= 1e-8).sum())
    }
    dependence.append(record)
    print(
        f"{policy}: recovered={record['recovered_episodes']} "
        f"max_error_pp={record['max_abs_residual_pp']:.10f} "
        f"matching={record['exact_within_1e_8']}"
    )

print("\nCONDITION-DEPENDENT PRIMARY EFFECTS")
policies = ["B1", "B2", "B3", "B4"]
rng = np.random.default_rng(2026)
draws = rng.integers(0, 200, size=(10000, 200))
interactions = []

for comparator in policies:
    for endpoint in ("recovery_s", "stability_pct"):
        relevant = df[
            df["policy"].isin([comparator, "B6"])
        ].pivot(
            index=["profile", "condition"],
            columns="policy",
            values=endpoint
        )
        require(not relevant.isna().any().any(),
                "Missing paired observations")

        improvement = (
            relevant[comparator] - relevant["B6"]
            if endpoint == "recovery_s"
            else relevant["B6"] - relevant[comparator]
        )

        by_profile = improvement.unstack("condition").sort_index()
        require(len(by_profile) == 200,
                "Incorrect number of paired profiles")

        low = by_profile["Low"].to_numpy()
        elevated = by_profile["Elevated"].to_numpy()
        interaction = elevated - low

        ci = np.quantile(
            interaction[draws].mean(axis=1),
            [0.025, 0.975]
        )

        interactions.append({
            "comparator": comparator,
            "endpoint": endpoint,
            "low_mean_improvement": float(low.mean()),
            "elevated_mean_improvement": float(elevated.mean()),
            "elevated_minus_low": float(interaction.mean()),
            "interaction_CI_low": float(ci[0]),
            "interaction_CI_high": float(ci[1]),
            "role": "secondary_descriptive"
        })

        print(
            f"{comparator} {endpoint}: "
            f"Low={low.mean():+.6f} "
            f"Elevated={elevated.mean():+.6f} "
            f"Interaction={interaction.mean():+.6f} "
            f"95%CI=[{ci[0]:+.6f},{ci[1]:+.6f}]"
        )

print("\nB5/B6 INTERVENTION ALLOCATION AUDIT")
variation_equal = 0
action_identical = 0
action_different = 0
budget_difference_max = 0.0

for pid in sorted(df["profile"].unique()):
    for condition in ("Low", "Elevated"):
        a = trajectories[(pid, condition, "B5")]
        b = trajectories[(pid, condition, "B6")]

        require(a.shape == b.shape == (120,),
                "Unexpected intervention trajectory size")

        va = float(np.abs(np.diff(np.r_[0.5, a])).sum())
        vb = float(np.abs(np.diff(np.r_[0.5, b])).sum())
        difference = abs(va - vb)
        budget_difference_max = max(
            budget_difference_max, difference
        )

        if difference <= 1e-10:
            variation_equal += 1

        if np.array_equal(a, b):
            action_identical += 1
        else:
            action_different += 1

require(variation_equal == 400,
        "B5/B6 total variation mismatch")

print("Equal cumulative variation:", variation_equal, "/400")
print("Identical intervention trajectories:", action_identical)
print("Different intervention trajectories:", action_different)
print("Max variation difference:", budget_difference_max)

report = {
    "step": 237,
    "status": "PASS",
    "original_audit_sha256": sha(audit_path),
    "episodes": len(df),
    "endpoint_dependence": dependence,
    "condition_interactions": interactions,
    "B5_B6_resource_audit": {
        "equal_total_variation": variation_equal,
        "identical_action_trajectories": action_identical,
        "different_action_trajectories": action_different,
        "max_total_variation_difference":
            budget_difference_max
    },
    "statistical_role": "secondary_diagnostic",
    "frozen_results_modified": False
}

OUTPUT.write_text(
    json.dumps(report, indent=2) + "\n",
    encoding="utf-8"
)

print("\nReport SHA256:", sha(OUTPUT))
print("Original checkpoints unchanged: PASS")
print("STEP 237: PASS")
