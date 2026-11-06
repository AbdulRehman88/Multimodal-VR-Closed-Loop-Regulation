"""Strictly causal rolling temporal features from motion windows."""

from collections import deque
import numpy as np

from src.models.causal_features import FEATURE_NAMES


class OnlineTemporalFeatures:
    """Combine observations at t-90, t-45, and t samples.

    A temporal vector first becomes available at sample 179.
    All inputs must originate from the motion-only feature stream.
    """

    def __init__(self, sampling_rate_hz=90, update_rate_hz=10):
        self.fs = int(sampling_rate_hz)
        self.fc = int(update_rate_hz)

        if self.fs != 90 or self.fc != 10:
            raise ValueError(
                "Current representation requires 90 Hz and 10 Hz."
            )

        self.interval = self.fs // self.fc
        self.selected = (
            FEATURE_NAMES.index("acceleration_mean"),
            FEATURE_NAMES.index("speed_mean"),
        )

        self.history = deque(maxlen=11)
        self.last_index = None

    def observe(self, output):
        if not isinstance(output, dict):
            raise ValueError("Expected an online motion-feature record.")

        if output.get("observation_source") != "motion_only":
            raise ValueError("Only motion observations are permitted.")

        if tuple(output.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("Unexpected feature schema.")

        index = int(output["sample_index"])
        vector = np.asarray(output["features"], dtype=np.float64)

        if vector.shape != (len(FEATURE_NAMES),):
            raise ValueError("Unexpected feature dimension.")

        if not np.isfinite(vector).all():
            raise ValueError("Nonfinite motion features.")

        expected = 89 if self.last_index is None else (
            self.last_index + self.interval
        )

        if index != expected:
            raise ValueError(
                f"Nonconsecutive update: expected {expected}, got {index}"
            )

        if not np.isclose(
            float(output["time_s"]),
            index / self.fs,
            rtol=0,
            atol=1e-9,
        ):
            raise ValueError("Invalid observation timestamp.")

        self.last_index = index
        self.history.append((index, vector.copy()))

        if len(self.history) < 11:
            return None

        first_index, first = self.history[0]
        middle_index, middle = self.history[5]
        last_index, last = self.history[10]

        assert middle_index - first_index == 45
        assert last_index - first_index == 90

        first = first[list(self.selected)]
        middle = middle[list(self.selected)]
        last = last[list(self.selected)]

        raw = np.concatenate([first, middle, last])

        relative = np.concatenate([
            (middle - first) / (np.abs(first) + 1e-4),
            (last - first) / (np.abs(first) + 1e-4),
        ])

        combined = np.concatenate([raw, relative])

        assert combined.shape == (10,)
        assert np.isfinite(combined).all()

        return {
            "sample_index": index,
            "time_s": index / self.fs,
            "features": combined,
            "observation_source": "motion_only",
            "earliest_control_sample": index + 1,
        }
