import numpy as np
import pytest

from src.simulator.ai_closed_loop import simulate_motion_feedback
from src.simulator.closed_loop import simulate_closed_loop
from src.simulator.dynamics import StateParameters
from src.simulator.motion import MotionParameters
from src.simulator.coupled import CoupledParameters
from src.controllers.baselines import ControllerParameters


class AlwaysLow:
    classes_ = np.array([0, 1, 2])

    def predict_proba(self, X):
        assert X.shape == (1, 15)
        return np.array([[1.0, 0.0, 0.0]])


def inputs():
    return dict(
        seed=2026,
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


def total_variation(run):
    updates = run["stimulus"][::9]
    return np.abs(
        np.diff(np.r_[0.5, updates])
    ).sum()


@pytest.mark.parametrize("budget", [0.0, 0.1, 0.3, 0.6])
def test_cumulative_budget_is_enforced(budget):
    run = simulate_motion_feedback(
        policy="ai_neutral_return",
        model=AlwaysLow(),
        maximum_total_variation=budget,
        **inputs(),
    )

    assert total_variation(run) <= budget + 1e-10

    updates = run["stimulus"][::9]
    assert np.all(
        np.abs(np.diff(np.r_[0.5, updates]))
        <= 0.05 + 1e-12
    )

    assert np.allclose(run["stimulus"][:90], 0.5)


def test_budget_exhaustion_prevents_later_changes():
    run = simulate_motion_feedback(
        policy="ai_neutral_return",
        model=AlwaysLow(),
        maximum_total_variation=0.3,
        **inputs(),
    )

    # A total budget of 0.3 reserves half for
    # return to neutral under persistent Low output.
    assert np.isclose(total_variation(run), 0.15)
    assert np.isclose(run["stimulus"][-1], 0.65)


def test_unrestricted_behavior_unchanged():
    a = simulate_motion_feedback(
        policy="ai_neutral_return",
        model=AlwaysLow(),
        **inputs(),
    )
    b = simulate_motion_feedback(
        policy="ai_neutral_return",
        model=AlwaysLow(),
        maximum_total_variation=None,
        **inputs(),
    )

    for field in ("state", "position", "velocity", "stimulus"):
        np.testing.assert_array_equal(a[field], b[field])


def test_neutral_matches_reference_with_budget():
    common = inputs()

    new = simulate_motion_feedback(
        policy="neutral",
        maximum_total_variation=0.6,
        **common,
    )

    old = simulate_closed_loop(
        controller_name="neutral",
        **common,
    )

    for field in ("state", "position", "velocity", "stimulus"):
        np.testing.assert_array_equal(
            new[field], old[field]
        )


@pytest.mark.parametrize("invalid", [-0.1, np.nan, np.inf])
def test_invalid_budget_rejected(invalid):
    with pytest.raises(ValueError):
        simulate_motion_feedback(
            policy="neutral",
            maximum_total_variation=invalid,
            **inputs(),
        )
