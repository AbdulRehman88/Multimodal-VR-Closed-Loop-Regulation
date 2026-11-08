"""Time-preserved training-donor motion control.

The sham adapter and deterministic donor-selection algorithm are
identical to the validated B8 implementation.
Training donors are loaded explicitly from repository-local data.
"""

import hashlib
import numpy as np
import pandas as pd

donors = []

def require(condition, message):
    if not condition:
        raise AssertionError(message)

class ShamMotionAdapter:
    """Replace motion with a fixed independent donor trace.

    Retain the recipient's own elapsed time.
    No current-episode state or outcome is accessed.
    """

    def __init__(self, fitted_model, donor):
        self.model = fitted_model
        self.donor = donor
        self.classes_ = np.asarray(fitted_model.classes_)

        require(
            np.array_equal(self.classes_, [0, 1, 2]),
            "Unexpected classifier classes"
        )

    def transform(self, X):
        X = np.asarray(X, dtype=float)

        require(
            X.ndim == 2 and X.shape[1] == 15,
            "Expected original 15-dimensional input"
        )

        result = X.copy()
        sample_indices = np.rint(result[:, 14] * 90).astype(int)

        for row, index in enumerate(sample_indices):
            require(
                89 <= index <= 1079 and (index - 89) % 9 == 0,
                "Invalid causal observation timestamp"
            )

            donor_index = (index - 89) // 9
            result[row, :14] = self.donor["values"][donor_index]

        require(
            np.array_equal(result[:, 14], X[:, 14]),
            "Elapsed time was modified"
        )

        return result

    def predict_proba(self, X):
        return self.model.predict_proba(self.transform(X))


def choose_donor(seed):
    # Selection is determined before trajectory generation.
    # It uses only the fixed episode seed, not the true state,
    # predicted class, condition, or evaluation outcome.
    key = f"sham-motion-v1|{int(seed)}".encode("ascii")
    digest = hashlib.sha256(key).digest()
    index = int.from_bytes(digest[:8], "big") % len(donors)
    return donors[index]

def load_training_donors(path):
    """Load the 120 time-aligned trajectories from training data."""
    global donors

    path = __import__("pathlib").Path(path)
    expected_hash = (
        "f4e5f898b66bba7cd1fe0f24d83c856"
        "a9a679d17e692e2913319bd02ba358797"
    )
    require(
        hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash,
        "Development dataset hash mismatch"
    )

    frame = pd.read_csv(path)
    training = frame.loc[frame["split"] == "train"]

    require(len(training) == 13320, "Wrong training row count")
    require(
        training["profile"].nunique() == 60,
        "Wrong training profile count"
    )

    feature_columns = [
        "speed_mean", "speed_std", "speed_max",
        "speed_q90", "speed_rms", "acceleration_mean",
        "acceleration_std", "acceleration_q90",
        "velocity_x_std", "velocity_y_std",
        "velocity_z_std", "position_x_std",
        "position_y_std", "position_z_std"
    ]

    expected_indices = np.arange(89, 1080, 9)
    loaded = []

    for (pid, condition), group in training.groupby(
        ["profile", "condition"], sort=True
    ):
        group = group.sort_values("sample_index")

        require(
            np.array_equal(
                group["sample_index"].to_numpy(int),
                expected_indices
            ),
            "Sham donor timestamps misaligned"
        )

        values = group[feature_columns].to_numpy(float).copy()
        require(values.shape == (111, 14), "Wrong donor dimensions")
        require(np.isfinite(values).all(), "Nonfinite donor features")

        loaded.append({
            "profile": int(pid),
            "condition": str(condition),
            "values": values
        })

    require(len(loaded) == 120, "Wrong donor trajectory count")
    donors = loaded
    return donors
