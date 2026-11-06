import numpy as np
import pytest

from src.models.causal_features import extract_causal_features
from src.models.online_motion_features import OnlineMotionFeatures
from src.models.online_temporal_features import OnlineTemporalFeatures


def motion(seed=2026, n=1080):
    rng = np.random.default_rng(seed)
    velocity = rng.normal(0, 0.02, size=(n, 3))
    position = np.cumsum(velocity / 90, axis=0)
    return position, velocity


def offline_expected(position, velocity):
    result = extract_causal_features(position, velocity)
    features = result["features"]
    indices = result["sample_indices"]

    names = result["feature_names"]
    selected = [
        names.index("acceleration_mean"),
        names.index("speed_mean"),
    ]

    expected = {}

    for j in range(10, len(indices)):
        first = features[j - 10, selected]
        middle = features[j - 5, selected]
        last = features[j, selected]

        raw = np.concatenate([first, middle, last])
        relative = np.concatenate([
            (middle - first) / (np.abs(first) + 1e-4),
            (last - first) / (np.abs(first) + 1e-4),
        ])

        expected[int(indices[j])] = np.concatenate([
            raw, relative
        ])

    return expected


def test_full_stream_matches_offline():
    p, v = motion()

    online = OnlineMotionFeatures()
    temporal = OnlineTemporalFeatures()
    observed = {}

    for position, velocity in zip(p, v):
        item = online.observe(position, velocity)

        if item is not None:
            output = temporal.observe(item)

            if output is not None:
                observed[output["sample_index"]] = output["features"]

    expected = offline_expected(p, v)

    assert list(observed) == list(expected)
    assert len(observed) == 101

    for index in expected:
        np.testing.assert_allclose(
            observed[index],
            expected[index],
            rtol=1e-12,
            atol=1e-12,
        )


def test_first_prediction_can_affect_sample_180():
    p, v = motion(n=181)
    online = OnlineMotionFeatures()
    temporal = OnlineTemporalFeatures()
    outputs = []

    for position, velocity in zip(p, v):
        item = online.observe(position, velocity)
        if item is not None:
            result = temporal.observe(item)
            if result is not None:
                outputs.append(result)

    assert len(outputs) == 1
    assert outputs[0]["sample_index"] == 179
    assert outputs[0]["earliest_control_sample"] == 180
    assert np.isclose(outputs[0]["time_s"], 179 / 90)


def test_rejects_forged_observation_source():
    p, v = motion(n=90)
    item = None
    online = OnlineMotionFeatures()

    for position, velocity in zip(p, v):
        item = online.observe(position, velocity)

    assert item is not None
    item["observation_source"] = "ground_truth_oracle"

    with pytest.raises(ValueError):
        OnlineTemporalFeatures().observe(item)


def test_rejects_missing_or_duplicate_updates():
    p, v = motion(n=110)
    online = OnlineMotionFeatures()
    temporal = OnlineTemporalFeatures()
    items = []

    for position, velocity in zip(p, v):
        item = online.observe(position, velocity)
        if item is not None:
            items.append(item)

    temporal.observe(items[0])

    with pytest.raises(ValueError):
        temporal.observe(items[0])

    with pytest.raises(ValueError):
        temporal.observe(items[2])


def test_future_motion_does_not_modify_past_representation():
    p, v = motion(n=270)

    def compute(position, velocity):
        online = OnlineMotionFeatures()
        temporal = OnlineTemporalFeatures()
        results = {}

        for pos, vel in zip(position, velocity):
            item = online.observe(pos, vel)
            if item is not None:
                result = temporal.observe(item)
                if result is not None:
                    results[result["sample_index"]] = result["features"]
        return results

    original = compute(p, v)

    changed_p = p.copy()
    changed_v = v.copy()
    changed_p[180:] += 1000
    changed_v[180:] *= -1000

    modified = compute(changed_p, changed_v)

    np.testing.assert_array_equal(
        original[179],
        modified[179],
    )
