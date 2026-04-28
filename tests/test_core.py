import numpy as np

from various_speed.core import SpeedTransformConfig
from various_speed.core import clean_near_zero_actions
from various_speed.core import compute_replay_metrics
from various_speed.core import segment_actions
from various_speed.core import transform_episode


def _episode(length: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actions = np.zeros((length, 7), dtype=np.float32)
    actions[:, 0] = 1.0
    actions[:, 3] = 0.1
    actions[: length // 2, 6] = -1.0
    actions[length // 2 :, 6] = 1.0
    states = np.zeros((length, 8), dtype=np.float32)
    frames = np.arange(length, dtype=np.int64)
    return actions, states, frames


def test_clean_near_zero_actions_does_not_touch_gripper():
    actions = np.zeros((2, 7), dtype=np.float32)
    actions[:, 6] = [-1.0, 1.0]

    cleaned, mask = clean_near_zero_actions(actions, transl_eps=1e-3, rot_eps=1e-3)

    np.testing.assert_array_equal(cleaned[:, 6], actions[:, 6])
    np.testing.assert_array_equal(mask, np.ones((2, 2), dtype=bool))


def test_gripper_only_actions_do_not_create_motion_segments():
    actions = np.zeros((4, 7), dtype=np.float32)
    actions[:, 6] = [-1.0, -1.0, 1.0, 1.0]

    segments = segment_actions(actions, SpeedTransformConfig())

    assert segments == [(0, 4, 0)]


def test_slow_transform_pads_every_repeated_observation():
    actions, states, frames = _episode(length=6)

    transformed, metrics = transform_episode(actions, states, frames, 0.5, SpeedTransformConfig())

    assert len(transformed["action"]) == 12
    assert int(transformed["is_padded"].sum()) == 6
    assert float(metrics["padded_ratio"]) == 0.5
    assert metrics["gripper_switch_delta"] == 0


def test_fast_transform_preserves_integrated_motion_and_gripper_switches():
    actions, states, frames = _episode(length=10)

    transformed, metrics = transform_episode(actions, states, frames, 2.0, SpeedTransformConfig())

    assert len(transformed["action"]) <= 6
    assert metrics["gripper_switch_delta"] == 0
    assert metrics["integrated_translation_l2_error"] < 1e-5
    assert metrics["integrated_rotation_l2_error"] < 1e-5


def test_replay_metrics_report_path_ratios_and_target_speed():
    actions, _states, _frames = _episode(length=6)

    metrics = compute_replay_metrics(actions, actions, target_speed=1.0)

    assert metrics["target_speed"] == 1.0
    assert metrics["translation_path_ratio"] == 1.0
    assert metrics["rotation_path_ratio"] == 1.0
