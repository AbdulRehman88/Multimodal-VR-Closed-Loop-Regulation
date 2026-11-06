import numpy as np
import pytest

from src.models.causal_features import (
    extract_causal_features,
)
from src.models.online_motion_features import (
    OnlineMotionFeatures,
)


def make_motion(seed=2026, n=1080):
    rng = np.random.default_rng(seed)
    velocity = rng.normal(size=(n, 3)) * 0.02
    position = np.cumsum(velocity / 90, axis=0)
    return position, velocity


def test_streaming_matches_offline():
    position, velocity = make_motion()

    offline = extract_causal_features(
        position, velocity
    )

    online = OnlineMotionFeatures()
    emitted = []

    for p, v in zip(position, velocity):
        output = online.observe(p, v)
        if output is not None:
            emitted.append(output)

    assert len(emitted) == len(
        offline["sample_indices"]
    )

    actual_indices = np.array([
        item["sample_index"] for item in emitted
    ])

    np.testing.assert_array_equal(
        actual_indices,
        offline["sample_indices"]
    )

    actual_features = np.stack([
        item["features"] for item in emitted
    ])

    np.testing.assert_allclose(
        actual_features,
        offline["features"],
        rtol=1e-12,
        atol=1e-12,
    )


def test_first_decision_timing():
    position, velocity = make_motion(n=181)
    online = OnlineMotionFeatures()
    emitted_indices = []

    for p, v in zip(position, velocity):
        output = online.observe(p, v)
        if output is not None:
            emitted_indices.append(
                output["sample_index"]
            )

    assert emitted_indices == [
        89, 98, 107, 116, 125,
        134, 143, 152, 161, 170, 179
    ]

    # Each feature is available strictly before
    # the next control update at indices 90,99,...
    assert all(
        (index + 1) % 9 == 0
        for index in emitted_indices
    )


def test_future_observations_cannot_change_past():
    position, velocity = make_motion(n=360)

    first = OnlineMotionFeatures()
    past = []

    for p, v in zip(position[:180], velocity[:180]):
        result = first.observe(p, v)
        if result is not None:
            past.append(result["features"].copy())

    changed_position = position.copy()
    changed_velocity = velocity.copy()

    changed_position[180:] += 1000.0
    changed_velocity[180:] *= -1000.0

    second = OnlineMotionFeatures()
    reproduced = []

    for p, v in zip(
        changed_position[:180],
        changed_velocity[:180],
    ):
        result = second.observe(p, v)
        if result is not None:
            reproduced.append(
                result["features"].copy()
            )

    np.testing.assert_array_equal(
        np.stack(past),
        np.stack(reproduced)
    )


def test_invalid_motion_rejected():
    online = OnlineMotionFeatures()

    with pytest.raises(ValueError):
        online.observe(
            [np.nan, 0, 0], [0, 0, 0]
        )

    with pytest.raises(ValueError):
        online.observe(
            [0, 0], [0, 0, 0]
        )


def test_input_arrays_are_copied():
    online = OnlineMotionFeatures()

    position = np.array([1.0, 2.0, 3.0])
    velocity = np.array([0.1, 0.2, 0.3])

    online.observe(position, velocity)
    position[:] = 999.0
    velocity[:] = 999.0

    np.testing.assert_array_equal(
        online.positions[0],
        [1.0, 2.0, 3.0],
    )
    np.testing.assert_array_equal(
        online.velocities[0],
        [0.1, 0.2, 0.3],
    )
