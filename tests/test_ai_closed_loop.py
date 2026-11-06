import numpy as np
import pytest

from src.simulator.ai_closed_loop import simulate_motion_feedback
from src.simulator.closed_loop import simulate_closed_loop
from src.simulator.dynamics import StateParameters
from src.simulator.motion import MotionParameters
from src.simulator.coupled import CoupledParameters
from src.controllers.baselines import ControllerParameters


def settings():
    return dict(
        seed=20261105,
        state_params=StateParameters(
            baseline=4.9,
            relaxation_rate=0.55,
            stimulus_gain=1.0,
        ),
        motion_params=MotionParameters(
            frequency_hz=0.3,
            state_sensitivity=0.08,
        ),
        coupled_params=CoupledParameters(),
        controller_params=ControllerParameters(),
        control_rate_hz=10,
    )


@pytest.mark.parametrize("initial", [2.0, 5.0, 8.0])
def test_neutral_exact_oracle_simulator_match(initial):
    common = settings()
    common["initial_state"] = initial

    reference = simulate_closed_loop(
        controller_name="neutral",
        maximum_update_change=0.05,
        **common
    )

    new = simulate_motion_feedback(
        policy="neutral",
        maximum_update_change=0.05,
        **common
    )

    for name in ("state", "position", "velocity", "stimulus"):
        np.testing.assert_array_equal(
            new[name], reference[name]
        )


class MockModel:
    classes_ = np.array([0, 1, 2])

    def __init__(self):
        self.seen = []

    def predict_proba(self, X):
        assert X.shape == (1, 15)
        self.seen.append(X.copy())
        return np.array([[0.1, 0.8, 0.1]])


def test_ai_first_action_timing_and_neutral_prefix():
    common = settings()
    common["initial_state"] = 8.0

    neutral = simulate_motion_feedback(
        policy="neutral", **common
    )
    model = MockModel()
    ai = simulate_motion_feedback(
        policy="ai_neutral_return",
        model=model,
        **common
    )

    assert len(model.seen) == 110

    decisions = ai["decisions"]
    assert len(decisions) == 120
    assert all(
        d["predicted_class"] is None
        for d in decisions[:10]
    )

    assert decisions[10]["control_sample"] == 90
    assert decisions[10]["observation_sample"] == 89
    assert decisions[11]["observation_sample"] == 98

    np.testing.assert_array_equal(
        ai["state"][:90], neutral["state"][:90]
    )
    np.testing.assert_array_equal(
        ai["stimulus"][:90], neutral["stimulus"][:90]
    )


def test_unsupported_policy_rejected():
    with pytest.raises(ValueError):
        simulate_motion_feedback(
            policy="oracle",
            **settings()
        )


def test_ai_requires_model():
    with pytest.raises(ValueError):
        simulate_motion_feedback(
            policy="ai_neutral_return",
            **settings()
        )
