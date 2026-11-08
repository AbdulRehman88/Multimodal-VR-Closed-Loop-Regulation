from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(r"D:\Dr_Abdul_Rehman\Multimodal-VR-Comparative-Revision")
CONFIG = ROOT / "revision/configs"
LOCK = CONFIG / "comparative_protocol_lock.json"

PROFILES = CONFIG / "confirmatory_profiles_200.csv"
LEDGER = CONFIG / "confirmatory_episode_seeds.csv"
ADDENDUM = CONFIG / "confirmatory_population_addendum.json"

def require(condition, message):
    if not condition:
        raise AssertionError(message)

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

print("STEP 228 - CONFIRMATORY COHORT GENERATION")

require(LOCK.is_file(), "Frozen comparative protocol missing")

require(
    sha256(LOCK)
    == "3d97a915e3a0fe62becda2ec6d423317680c3ece5daac7b2a9479bfc3a8c1b2c",
    "Frozen comparative protocol SHA256 mismatch"
)

for path in (PROFILES, LEDGER, ADDENDUM):
    require(not path.exists(), f"Output already exists: {path}")

development = pd.read_csv(
    ROOT / "configs/frozen_development_profiles.csv"
)
original_test = pd.read_csv(
    ROOT / "configs/independent_synthetic_test_profiles.csv"
)

require(len(development) == 100, "Development count mismatch")
require(len(original_test) == 200, "Original test count mismatch")

ranges = {
    "baseline": (4.5, 5.5),
    "relaxation_rate": (0.3, 0.8),
    "stimulus_gain": (0.5, 1.5),
    "motion_frequency_hz": (0.1, 0.5),
    "motion_state_sensitivity": (0.02, 0.15),
}

# Verify that the archived parameter samples are
# consistent with the declared reconstructed bounds.
for source in (development, original_test):
    for key, (low, high) in ranges.items():
        values = source[key].to_numpy(dtype=float)
        require(
            np.isfinite(values).all()
            and np.all(values >= low)
            and np.all(values < high),
            f"Archived parameter outside proposed bounds: {key}"
        )

SEED = 20261107
N = 200
rng = np.random.default_rng(SEED)

profiles = pd.DataFrame({
    "profile": np.arange(200000, 200000 + N)
})

for key, (low, high) in ranges.items():
    profiles[key] = rng.uniform(low, high, N)

existing_ids = (
    set(development["profile"].astype(int))
    | set(original_test["profile"].astype(int))
)

new_ids = set(profiles["profile"].astype(int))

require(len(new_ids) == N, "Duplicate profile IDs")
require(
    new_ids.isdisjoint(existing_ids),
    "Original/new profile IDs overlap"
)

for key, (low, high) in ranges.items():
    values = profiles[key].to_numpy(dtype=float)
    require(np.isfinite(values).all(), f"Nonfinite {key}")
    require(
        np.all(values >= low) and np.all(values < high),
        f"Sampling bounds violated: {key}"
    )

# Paired episode seeds: same profile-condition seed
# must be reused for ALL compared policies.
rows = []

for pid in profiles["profile"]:
    for offset, condition in enumerate(("Low", "Elevated")):
        rows.append({
            "profile": int(pid),
            "condition": condition,
            "episode_seed": (
                900000000
                + 2 * (int(pid) - 200000)
                + offset
            )
        })

ledger = pd.DataFrame(rows)

require(len(ledger) == 400, "Incorrect ledger size")
require(
    ledger["episode_seed"].nunique() == 400,
    "Duplicate episode seed"
)
require(
    not ledger.duplicated(["profile", "condition"]).any(),
    "Duplicate profile-condition case"
)

CONFIG.mkdir(parents=True, exist_ok=True)

profiles.to_csv(
    PROFILES, index=False, float_format="%.17g"
)
ledger.to_csv(LEDGER, index=False)

require(len(pd.read_csv(PROFILES)) == N,
        "Profile file persistence failure")
require(len(pd.read_csv(LEDGER)) == 400,
        "Seed file persistence failure")

addendum = {
    "step": 228,
    "role": "Pre-evaluation population-provenance clarification",
    "parent_protocol_sha256": sha256(LOCK),
    "legacy_simulator_used": False,
    "original_profile_generator_recovered": False,
    "population_definition": (
        "Reconstructed independent uniform synthetic population"
    ),
    "distribution_bounds": {
        key: [low, high]
        for key, (low, high) in ranges.items()
    },
    "basis": (
        "Archived development and original independent-test "
        "profile parameter ranges and observed low correlations"
    ),
    "scientific_limitation": (
        "The archived samples support but cannot prove the "
        "original sampler used independent uniform draws. "
        "The new population is therefore documented explicitly "
        "rather than described as an exact historical reproduction."
    ),
    "parent_protocol_clarification": (
        "The phrase 'Use original population generator' in "
        "the frozen Step 224 protocol is superseded solely "
        "for cohort-generation provenance. All policy, "
        "resource, endpoint and statistical rules remain fixed."
    ),
    "generator": "numpy.random.default_rng",
    "generator_seed": SEED,
    "sampling_order": list(ranges),
    "profiles": N,
    "profile_ids": [200000, 200199],
    "initial_states": {
        "Low": 2.0,
        "Elevated": 8.0
    },
    "paired_seeds": (
        "900000000 + 2*(profile-200000) + condition_offset"
    ),
    "profile_csv_sha256": sha256(PROFILES),
    "episode_ledger_sha256": sha256(LEDGER),
    "all_policy_outcomes_unexamined": True,
    "post_evaluation_sampling_tuning_allowed": False
}

ADDENDUM.write_text(
    json.dumps(addendum, indent=2) + "\n",
    encoding="utf-8"
)

require(
    sha256(LOCK) == addendum["parent_protocol_sha256"],
    "Original frozen protocol changed"
)

print("\nGENERATED COHORT")
print("New profiles:", len(profiles))
print("Profile-condition cases:", len(ledger))
print("Original profile-ID overlap:", len(new_ids & existing_ids))
print("Sampling seed:", SEED)

print("\nPARAMETER SUMMARY")
print(
    profiles[list(ranges)].describe()
    .loc[["min", "mean", "std", "max"]]
    .round(6).to_string()
)

print("\nFROZEN ARTIFACTS")
print("Profiles SHA256:", sha256(PROFILES))
print("Seed ledger SHA256:", sha256(LEDGER))
print("Provenance addendum SHA256:", sha256(ADDENDUM))

print("\nORIGINAL PROTOCOL UNCHANGED: PASS")
print("LEGACY GENERATOR USED: NO")
print("POLICY SIMULATIONS EXECUTED: NO")
print("CONTROLLER/MODEL RETRAINING: NO")
print("STEP 228: PASS")
