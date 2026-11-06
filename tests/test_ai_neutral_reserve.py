import numpy as np

from src.simulator.ai_closed_loop import (
    simulate_motion_feedback,
)
from src.simulator.dynamics import StateParameters
from src.simulator.motion import MotionParameters
from src.simulator.coupled import CoupledParameters
from src.controllers.baselines import ControllerParameters


BUDGET = 0.6


def settings():
    return dict(
        seed=20261105,
        state_params=StateParameters(
            baseline=5.0,
            relaxation_rate=0.55,
            stimulus_gain=1.0,
        ),
        motion_params=MotionParameters(),
        coupled_params=CoupledParameters(),
        controller_params=ControllerParameters(),
        initial_state=2.0,
        control_rate_hz=10,
        maximum_update_change=0.05,
    )


class AlwaysLow:
    classes_ = np.array([0, 1, 2])

    def predict_proba(self, X):
        assert X.shape == (1, 15)
        return np.array([[1.0, 0.0, 0.0]])


class LowThenRegulated:
    classes_ = np.array([0, 1, 2])

    def __init__(self, low_predictions=10):
        self.calls = 0
        self.low_predictions = low_predictions

    def predict_proba(self, X):
        assert X.shape == (1, 15)
        self.calls += 1
        if self.calls <= self.low_predictions:
            return np.array([[1.0, 0.0, 0.0]])
        return np.array([[0.0, 1.0, 0.0]])


class AlternatingDirections:
    classes_ = np.array([0, 1, 2])

    def __init__(self):
        self.calls = 0

    def predict_proba(self, X):
        self.calls += 1
        direction = (self.calls // 4) % 3
        probabilities = np.zeros((1, 3))
        probabilities[0, direction] = 1.0
        return probabilities


def assert_reserve(run):
    decisions = run["decisions"]
    cumulative = 0.0
    for decision in decisions:
        before = decision["stimulus_before"]
        after = decision["stimulus_after"]
        change = abs(after - before)
        cumulative += change

        assert change <= 0.05 + 1e-10
        assert cumulative <= BUDGET + 1e-9
        assert (
            cumulative + abs(after - 0.5)
            <= BUDGET + 1e-9
        )


def test_always_low_preserves_return_reserve():
    run = simulate_motion_feedback(
        policy="ai_neutral_return",
        model=AlwaysLow(),
        maximum_total_variation=BUDGET,
        **settings()
    )
    assert_reserve(run)

    # A constant outward command may spend at most
    # half the budget, reserving the other half.
    assert np.isclose(run["stimulus"][-1], 0.8)
    used = sum(
        abs(d["stimulus_after"] - d["stimulus_before"])
        for d in run["decisions"]
    )
    assert np.isclose(used, 0.3)


def test_neutral_return_remains_possible():
    model = LowThenRegulated(low_predictions=10)
    run = simulate_motion_feedback(
        policy="ai_neutral_return",
        model=model,
        maximum_total_variation=BUDGET,
        **settings()
    )
    assert_reserve(run)

    assert np.isclose(run["stimulus"][-1], 0.5)

    used = sum(
        abs(d["stimulus_after"] - d["stimulus_before"])
        for d in run["decisions"]
    )
    assert used <= BUDGET + 1e-9


def test_direction_reversals_preserve_budget():
    run = simulate_motion_feedback(
        policy="ai_neutral_return",
        model=AlternatingDirections(),
        maximum_total_variation=BUDGET,
        **settings()
    )
    assert_reserve(run)


def test_unrestricted_trajectory_unchanged():
    a = simulate_motion_feedback(
        policy="ai_neutral_return",
        model=AlwaysLow(),
        **settings()
    )
    b = simulate_motion_feedback(
        policy="ai_neutral_return",
        model=AlwaysLow(),
        maximum_total_variation=None,
        **settings()
    )

    for key in ("state", "position", "velocity", "stimulus"):
        np.testing.assert_array_equal(a[key], b[key])
