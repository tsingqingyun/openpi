from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

MOTION_CLASS_NAMES = {
    0: "still",
    1: "translate",
    2: "rotate",
    3: "translate_rotate",
}


@dataclass(frozen=True)
class SpeedTransformConfig:
    """Configuration for action-space speed augmentation.

    The action convention is LIBERO's 7D delta action:
    [x, y, z, roll, pitch, yaw, gripper].  The gripper channel is never used for
    segmentation, and is copied discretely during resampling.
    """

    transl_eps: float = 1e-4
    rot_eps: float = 1e-4
    clean_transl_eps: float = 1e-4
    clean_rot_eps: float = 1e-4
    direction_cos_threshold: float = -0.25
    min_segment_len: int = 1
    keep_still_segments: bool = True
    fps: int = 20


def _as_float32_2d(values: Iterable[np.ndarray], expected_dim: int) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != expected_dim:
        raise ValueError(f"Expected shape (T, {expected_dim}), got {arr.shape}")
    return arr


def clean_near_zero_actions(
    actions: np.ndarray,
    transl_eps: float = 1e-4,
    rot_eps: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Zero tiny translation/rotation noise without touching gripper actions.

    Returns the cleaned actions and a boolean mask of shape (T, 2), where the
    columns indicate whether translation / rotation was zeroed.
    """

    cleaned = np.asarray(actions, dtype=np.float32).copy()
    if cleaned.ndim != 2 or cleaned.shape[1] < 7:
        raise ValueError(f"Expected 7D actions, got {cleaned.shape}")

    transl_norm = np.linalg.norm(cleaned[:, :3], axis=1)
    rot_norm = np.linalg.norm(cleaned[:, 3:6], axis=1)
    transl_zeroed = transl_norm < transl_eps
    rot_zeroed = rot_norm < rot_eps
    cleaned[transl_zeroed, :3] = 0.0
    cleaned[rot_zeroed, 3:6] = 0.0
    return cleaned, np.stack([transl_zeroed, rot_zeroed], axis=1)


def _motion_class(action: np.ndarray, transl_eps: float, rot_eps: float) -> int:
    has_translation = float(np.linalg.norm(action[:3])) >= transl_eps
    has_rotation = float(np.linalg.norm(action[3:6])) >= rot_eps
    if has_translation and has_rotation:
        return 3
    if has_translation:
        return 1
    if has_rotation:
        return 2
    return 0


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 1.0
    return float(np.dot(a, b) / denom)


def segment_actions(
    actions: np.ndarray,
    config: SpeedTransformConfig,
) -> list[tuple[int, int, int]]:
    """Segment an episode by translation/rotation motion type.

    Gripper open/close is intentionally excluded from every boundary decision.
    A segment is returned as (start, end, motion_class), with end exclusive.
    """

    if len(actions) == 0:
        return []

    classes = np.asarray(
        [_motion_class(a, config.transl_eps, config.rot_eps) for a in actions],
        dtype=np.int32,
    )

    segments: list[tuple[int, int, int]] = []
    start = 0
    for i in range(1, len(actions)):
        boundary = classes[i] != classes[i - 1]
        if not boundary and classes[i] in (1, 3):
            boundary = (
                _cosine(actions[i - 1, :3], actions[i, :3])
                < config.direction_cos_threshold
            )
        if not boundary and classes[i] in (2, 3):
            boundary = (
                _cosine(actions[i - 1, 3:6], actions[i, 3:6])
                < config.direction_cos_threshold
            )
        if boundary:
            segments.append((start, i, int(classes[i - 1])))
            start = i
    segments.append((start, len(actions), int(classes[-1])))

    if config.min_segment_len <= 1:
        return segments

    merged: list[tuple[int, int, int]] = []
    for seg in segments:
        if not merged or (seg[1] - seg[0]) >= config.min_segment_len:
            merged.append(seg)
            continue
        prev_start, _prev_end, prev_cls = merged[-1]
        merged[-1] = (prev_start, seg[1], prev_cls)
    return merged


def _interp_cumulative(cumulative: np.ndarray, x: float) -> np.ndarray:
    x = float(np.clip(x, 0.0, cumulative.shape[0] - 1))
    left = int(np.floor(x))
    right = min(left + 1, cumulative.shape[0] - 1)
    alpha = np.float32(x - left)
    return (1.0 - alpha) * cumulative[left] + alpha * cumulative[right]


def _segment_boundaries(src_actions: np.ndarray, speed: float) -> np.ndarray:
    """Build resampling boundaries, preserving gripper switches as hard anchors."""

    n_src = len(src_actions)
    n_out = max(1, int(round(n_src / speed)))
    boundaries = [float(x) for x in np.linspace(0.0, float(n_src), n_out + 1)]

    # Gripper is intentionally excluded from motion segmentation, but fast
    # resampling can otherwise hide a close/open transition inside a large bin.
    # Adding the source switch time as an interval boundary preserves the event
    # as a discrete controller step without making it a segmentation criterion.
    switch_indices = np.flatnonzero(np.abs(np.diff(src_actions[:, 6])) > 0.5) + 1
    boundaries.extend(float(i) for i in switch_indices)

    deduped = sorted(set(boundaries))
    return np.asarray(deduped, dtype=np.float32)


def _resample_segment(
    actions: np.ndarray,
    states: np.ndarray,
    source_frame_indices: np.ndarray,
    start: int,
    end: int,
    motion_class: int,
    speed: float,
) -> dict[str, np.ndarray]:
    src_actions = actions[start:end]
    src_states = states[start:end]
    src_frames = source_frame_indices[start:end]
    n_src = end - start
    if n_src <= 0:
        raise ValueError("Cannot resample an empty segment")

    boundaries = _segment_boundaries(src_actions, speed)
    n_out = len(boundaries) - 1
    cumulative = np.concatenate(
        [np.zeros((1, 6), dtype=np.float32), np.cumsum(src_actions[:, :6], axis=0)],
        axis=0,
    )

    out_actions = np.zeros((n_out, 7), dtype=np.float32)
    out_states = np.zeros((n_out, states.shape[1]), dtype=np.float32)
    out_source_frames = np.zeros(n_out, dtype=np.int64)
    out_source_steps = np.zeros(n_out, dtype=np.int64)
    observation_mask = np.ones(n_out, dtype=np.int8)

    used_sources: set[int] = set()
    for j in range(n_out):
        left_t = float(boundaries[j])
        right_t = float(boundaries[j + 1])
        out_actions[j, :6] = _interp_cumulative(cumulative, right_t) - _interp_cumulative(
            cumulative, left_t
        )

        source_local = min(int(np.floor(left_t)), n_src - 1)
        grip_local = min(max(int(np.ceil(right_t) - 1), 0), n_src - 1)
        out_actions[j, 6] = src_actions[grip_local, 6]
        out_states[j] = src_states[source_local]
        out_source_frames[j] = int(src_frames[source_local])
        out_source_steps[j] = start + source_local

        if speed < 1.0 and source_local in used_sources:
            observation_mask[j] = 0
        used_sources.add(source_local)

    return {
        "action": out_actions,
        "state": out_states,
        "source_frame_index": out_source_frames,
        "source_step_index": out_source_steps,
        "observation_mask": observation_mask,
        "segment_id": np.full(n_out, -1, dtype=np.int32),
        "motion_class": np.full(n_out, motion_class, dtype=np.int8),
    }


def transform_episode(
    actions: np.ndarray,
    states: np.ndarray,
    source_frame_indices: np.ndarray,
    speed: float,
    config: SpeedTransformConfig,
) -> tuple[dict[str, np.ndarray], dict[str, float | int]]:
    """Clean, segment, and resample one episode for a target speed."""

    if speed <= 0:
        raise ValueError(f"Speed must be positive, got {speed}")
    actions = _as_float32_2d(actions, 7)
    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 2:
        raise ValueError(f"Expected 2D states, got {states.shape}")
    source_frame_indices = np.asarray(source_frame_indices, dtype=np.int64)
    if len(actions) != len(states) or len(actions) != len(source_frame_indices):
        raise ValueError("actions, states, and source_frame_indices must have equal length")

    cleaned_actions, clean_mask = clean_near_zero_actions(
        actions, config.clean_transl_eps, config.clean_rot_eps
    )
    segments = segment_actions(cleaned_actions, config)
    if not config.keep_still_segments:
        segments = [s for s in segments if s[2] != 0]

    pieces = []
    for segment_id, (start, end, motion_class) in enumerate(segments):
        piece = _resample_segment(
            cleaned_actions,
            states,
            source_frame_indices,
            start,
            end,
            motion_class,
            speed,
        )
        piece["segment_id"][:] = segment_id
        pieces.append(piece)

    if not pieces:
        raise ValueError("Episode produced no segments")

    merged = {
        key: np.concatenate([piece[key] for piece in pieces], axis=0)
        for key in pieces[0].keys()
    }
    source_steps = merged["source_step_index"]
    merged["cleaned_translation"] = clean_mask[source_steps, 0].astype(np.int8)
    merged["cleaned_rotation"] = clean_mask[source_steps, 1].astype(np.int8)
    merged["speed"] = np.full(len(merged["action"]), float(speed), dtype=np.float32)
    merged["action_mask"] = np.ones(len(merged["action"]), dtype=np.int8)
    merged["is_padded"] = (1 - merged["observation_mask"]).astype(np.int8)

    metrics = compute_replay_metrics(actions, merged["action"], speed)
    metrics.update(
        {
            "segment_count": len(segments),
            "segments": len(segments),
            "source_frames": int(len(actions)),
            "output_frames": int(len(merged["action"])),
            "padded_frames": int(np.sum(merged["is_padded"])),
            "padded_ratio": float(np.mean(merged["is_padded"])),
            "cleaned_translation_frames": int(np.sum(clean_mask[:, 0])),
            "cleaned_rotation_frames": int(np.sum(clean_mask[:, 1])),
        }
    )
    return merged, metrics


def compute_replay_metrics(
    source_actions: np.ndarray,
    replay_actions: np.ndarray,
    target_speed: float | None = None,
) -> dict[str, float | int]:
    """Offline controller metrics from action sequences.

    This does not claim task success; it checks whether speed augmentation kept
    the same integrated 6D command while changing the number of controller
    steps as expected.
    """

    source_actions = _as_float32_2d(source_actions, 7)
    replay_actions = _as_float32_2d(replay_actions, 7)
    source_motion = source_actions[:, :6].sum(axis=0)
    replay_motion = replay_actions[:, :6].sum(axis=0)
    source_steps = len(source_actions)
    replay_steps = len(replay_actions)
    actual_speed = source_steps / max(replay_steps, 1)
    translation_path_source = float(np.linalg.norm(source_actions[:, :3], axis=1).sum())
    translation_path_replay = float(np.linalg.norm(replay_actions[:, :3], axis=1).sum())
    rotation_path_source = float(np.linalg.norm(source_actions[:, 3:6], axis=1).sum())
    rotation_path_replay = float(np.linalg.norm(replay_actions[:, 3:6], axis=1).sum())
    gripper_switches_source = int(np.sum(np.abs(np.diff(source_actions[:, 6])) > 0.5))
    gripper_switches_replay = int(np.sum(np.abs(np.diff(replay_actions[:, 6])) > 0.5))

    out: dict[str, float | int] = {
        "target_speed": float(target_speed) if target_speed is not None else 1.0,
        "source_steps": int(source_steps),
        "replay_steps": int(replay_steps),
        "actual_speed": float(actual_speed),
        "speed_error": float(abs(actual_speed - target_speed)) if target_speed is not None else 0.0,
        "translation_path_source": translation_path_source,
        "translation_path_replay": translation_path_replay,
        "translation_path_ratio": translation_path_replay / max(translation_path_source, 1e-12),
        "rotation_path_source": rotation_path_source,
        "rotation_path_replay": rotation_path_replay,
        "rotation_path_ratio": rotation_path_replay / max(rotation_path_source, 1e-12),
        "integrated_translation_l2_error": float(
            np.linalg.norm(source_motion[:3] - replay_motion[:3])
        ),
        "integrated_rotation_l2_error": float(np.linalg.norm(source_motion[3:] - replay_motion[3:])),
        "gripper_switches_source": gripper_switches_source,
        "gripper_switches_replay": gripper_switches_replay,
        "gripper_switch_delta": int(gripper_switches_replay - gripper_switches_source),
    }
    return out
