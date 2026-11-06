"""Streaming motion-only features for causal feedback."""

from collections import deque
import numpy as np

from src.models.causal_features import (
    FEATURE_NAMES,
    extract_causal_features,
)


class OnlineMotionFeatures:
    """Observe motion at 90 Hz; emit features at 10 Hz."""

    def __init__(
        self,
        sampling_rate_hz=90,
        update_rate_hz=10,
        window_duration_s=1.0,
    ):
        fs = int(sampling_rate_hz)
        fc = int(update_rate_hz)

        if fs <= 0 or fc <= 0 or fs % fc:
            raise ValueError("Invalid sampling/control rate.")

        if not np.isfinite(window_duration_s) or window_duration_s <= 0:
            raise ValueError("Invalid window duration.")

        self.fs = fs
        self.fc = fc
        self.interval = fs // fc
        self.window = int(round(fs * window_duration_s))

        if self.window < 2:
            raise ValueError("Window requires >=2 samples.")

        self.positions = deque(maxlen=self.window)
        self.velocities = deque(maxlen=self.window)
        self.sample_index = -1

    def observe(self, position, velocity):
        """Consume exactly one current observation.

        Returns None before a complete window is available,
        or between update boundaries.
        """
        p = np.asarray(position, dtype=np.float64)
        v = np.asarray(velocity, dtype=np.float64)

        if (
            p.shape != (3,)
            or v.shape != (3,)
            or not np.isfinite(p).all()
            or not np.isfinite(v).all()
        ):
            raise ValueError("Invalid observed motion.")

        self.sample_index += 1
        self.positions.append(p.copy())
        self.velocities.append(v.copy())

        index = self.sample_index

        if index < self.window - 1:
            return None

        if (index - (self.window - 1)) % self.interval:
            return None

        result = extract_causal_features(
            np.asarray(self.positions),
            np.asarray(self.velocities),
            sampling_rate_hz=self.fs,
            window_duration_s=self.window / self.fs,
            update_rate_hz=self.fc,
        )

        assert result["features"].shape == (
            1, len(FEATURE_NAMES)
        )

        return {
            "sample_index": index,
            "time_s": index / self.fs,
            "features": result["features"][0].copy(),
            "feature_names": FEATURE_NAMES,
            "observation_source": "motion_only",
        }
