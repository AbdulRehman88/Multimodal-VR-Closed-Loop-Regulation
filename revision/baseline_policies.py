"""Frozen, interpretable supplementary controller policies.

These adapters use the unchanged 15-input simulator interface.
Only their permitted observations differ.
"""

import numpy as np


class FixedSchedulePolicy:
    """B1: predetermined 4-second cycle, independent of motion."""

    classes_ = np.array([0, 1, 2])

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        assert X.ndim == 2 and X.shape[1] == 15

        result = np.zeros((len(X), 3), dtype=float)

        for i, row in enumerate(X):
            time_s = float(row[14])

            # The first completed observation ends at 89/90 s.
            # Each phase lasts one second.
            phase = int(np.floor(
                (time_s - 89.0 / 90.0) + 1e-9
            )) % 4

            # Request increase, neutral, decrease, neutral.
            selected_class = (0, 1, 2, 1)[phase]
            result[i, selected_class] = 1.0

        return result


class MotionThresholdPolicy:
    """B4: two frozen speed-variability thresholds."""

    classes_ = np.array([0, 1, 2])

    def __init__(self, lower_threshold, upper_threshold):
        self.lower = float(lower_threshold)
        self.upper = float(upper_threshold)

        if not 0 < self.lower < self.upper:
            raise ValueError("Invalid motion thresholds.")

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        assert X.ndim == 2 and X.shape[1] == 15

        speed_std = X[:, 1]

        classes = np.where(
            speed_std < self.lower,
            1,
            np.where(speed_std < self.upper, 0, 2),
        )

        result = np.zeros((len(X), 3), dtype=float)
        result[np.arange(len(X)), classes] = 1.0
        return result
