from pathlib import Path
import hashlib
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from tqdm import tqdm

ROOT = Path(r"D:\Dr_Abdul_Rehman\Multimodal-VR-Comparative-Revision")
DATA = Path(r"D:\Dr_Abdul_Rehman\Multimodal-VR\results\motion_state_dataset\development_windows.csv")
OUT = ROOT / "revision" / "models"
MANIFEST = ROOT / "results" / "frozen_one_second_ai_baseline" / "model_manifest.json"

EXPECTED_SHA = "f4e5f898b66bba7cd1fe0f24d83c856a9a679d17e692e2913319bd02ba358797"

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def require(condition, message):
    if not condition:
        raise AssertionError(message)

print("STEP 218 - FROZEN TRAINING-DATA ABLATIONS")

require(DATA.is_file(), "Original training dataset missing")
require(sha256(DATA) == EXPECTED_SHA, "Training dataset SHA-256 mismatch")

manifest = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))
features = manifest["feature_names"]
require(len(features) == 15, "Frozen feature manifest mismatch")
require(features[-1] == "time_s", "Expected elapsed time as final feature")

df = pd.read_csv(DATA)
require(len(df) == 17760, "Unexpected total window count")
require(set(features).issubset(df.columns), "Missing original features")

train = df.loc[df["split"] == "train"].copy()
val = df.loc[df["split"] == "validation"].copy()

require(len(train) == 13320, "Unexpected training window count")
require(len(val) == 4440, "Unexpected validation window count")
require(train["profile"].nunique() == 60, "Unexpected training profiles")
require(val["profile"].nunique() == 20, "Unexpected validation profiles")
require(
    set(train["profile"]).isdisjoint(set(val["profile"])),
    "Training-validation profile overlap"
)

def labels(frame):
    x = frame["target_state"].to_numpy(dtype=float)
    require(np.isfinite(x).all(), "Invalid target state")
    return np.where(x < 4, 0, np.where(x > 6, 2, 1))

y_train = labels(train)
y_val = labels(val)

require(set(np.unique(y_train)) == {0, 1, 2}, "Missing training classes")

experiments = [
    ("B2_time_only", ["time_s"]),
    ("B3_motion_only", features[:-1]),
]

OUT.mkdir(parents=True, exist_ok=True)

for name, selected in tqdm(experiments, desc="Training ablations", unit="model"):
    model_path = OUT / f"{name}.joblib"
    metadata_path = OUT / f"{name}_manifest.json"

    require(
        not model_path.exists() and not metadata_path.exists(),
        f"STOP: Existing artifacts for {name}"
    )

    X_train = train[selected].to_numpy(dtype=float)
    X_val = val[selected].to_numpy(dtype=float)

    require(np.isfinite(X_train).all(), "Nonfinite training features")
    require(np.isfinite(X_val).all(), "Nonfinite validation features")

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight=None,
            random_state=2026,
            max_iter=1000,
        ),
    )

    model.fit(X_train, y_train)

    require(
        np.array_equal(model.classes_, [0, 1, 2]),
        f"Invalid class order: {name}"
    )

    predictions = model.predict(X_val)
    ba = balanced_accuracy_score(y_val, predictions)
    matrix = confusion_matrix(y_val, predictions, labels=[0, 1, 2])

    joblib.dump(model, model_path)

    metadata = {
        "baseline": name,
        "role": "supplementary_information_ablation",
        "training_data": str(DATA),
        "training_data_sha256": EXPECTED_SHA,
        "training_profiles": 60,
        "training_windows": 13320,
        "validation_profiles": 20,
        "validation_windows": 4440,
        "validation_used_for_fitting": False,
        "independent_test_used": False,
        "features": selected,
        "feature_count": len(selected),
        "labels": {
            "0": "state < 4",
            "1": "4 <= state <= 6",
            "2": "state > 6",
        },
        "classifier": "StandardScaler + LogisticRegression",
        "C": 1.0,
        "class_weight": None,
        "random_state": 2026,
        "max_iter": 1000,
        "validation_balanced_accuracy": float(ba),
        "validation_confusion_matrix": matrix.tolist(),
        "model_sha256": sha256(model_path),
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\n{name}")
    print("Features:", len(selected))
    print("Validation balanced accuracy:", f"{ba:.6f}")
    print("Validation confusion matrix:")
    print(matrix)
    print("Model SHA256:", metadata["model_sha256"])

print("\nSTEP 218: PASS")
print("Original training dataset modified: NO")
print("Frozen 15-feature model retrained: NO")
print("Independent test accessed: NO")
