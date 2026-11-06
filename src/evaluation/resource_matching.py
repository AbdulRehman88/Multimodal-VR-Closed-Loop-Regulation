"""Utilities for fair, budget-aware controller comparisons."""

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class ResourceBudget:
    maximum_total_variation: float = 0.6
    maximum_update_change: float = 0.05

    def __post_init__(self):
        if not all(math.isfinite(v) for v in vars(self).values()):
            raise ValueError("Budget values must be finite.")
        if self.maximum_total_variation < 0:
            raise ValueError("Total variation budget must be nonnegative.")
        if not 0 < self.maximum_update_change <= 1:
            raise ValueError("Invalid per-update change limit.")


def stimulus_effort(stimulus, fs, fc):
    """Measure effort at controller-update instants."""
    x = np.asarray(stimulus, dtype=np.float64)

    if x.ndim != 1 or x.size < 2 or not np.isfinite(x).all():
        raise ValueError("Invalid stimulus trajectory.")

    if fs <= 0 or fc <= 0 or fs % fc:
        raise ValueError("Invalid sampling/control rates.")

    if np.any((x < 0) | (x > 1)):
        raise ValueError("Stimulus outside [0, 1].")

    updates = x[::fs // fc]
    initial_stimulus = 0.5
    changes = np.abs(
        np.diff(np.concatenate(([initial_stimulus], updates)))
    )

    return {
        "total_variation": float(changes.sum()),
        "maximum_update_change": (
            float(changes.max()) if changes.size else 0.0
        ),
        "mean_update_change": (
            float(changes.mean()) if changes.size else 0.0
        ),
    }


def check_budget(stimulus, fs, fc, budget):
    """Assess whether a completed trajectory meets its budget."""
    effort = stimulus_effort(stimulus, fs, fc)

    return {
        **effort,
        "total_variation_pass": (
            effort["total_variation"]
            <= budget.maximum_total_variation + 1e-10
        ),
        "update_change_pass": (
            effort["maximum_update_change"]
            <= budget.maximum_update_change + 1e-10
        ),
    }


def bounded_control_step(
    proposed_stimulus,
    current_stimulus,
    remaining_variation,
    budget,
):
    """Apply a causal, per-update and remaining-budget constraint."""

    values = (
        proposed_stimulus,
        current_stimulus,
        remaining_variation,
    )

    if not all(math.isfinite(v) for v in values):
        raise ValueError("Control inputs must be finite.")

    if not 0 <= current_stimulus <= 1:
        raise ValueError("Current stimulus outside [0, 1].")

    if remaining_variation < 0:
        raise ValueError("Remaining budget cannot be negative.")

    proposed = float(np.clip(proposed_stimulus, 0.0, 1.0))

    allowed = min(
        budget.maximum_update_change,
        remaining_variation,
    )

    delta = float(np.clip(
        proposed - current_stimulus,
        -allowed,
        allowed,
    ))

    output = float(np.clip(current_stimulus + delta, 0.0, 1.0))
    remaining = max(0.0, remaining_variation - abs(output - current_stimulus))

    return output, remaining
