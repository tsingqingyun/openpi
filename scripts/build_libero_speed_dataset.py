#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from various_speed.core import SpeedTransformConfig, transform_episode


def _episode_paths(dataset_root: Path, limit: int | None) -> list[Path]:
    paths = sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet"))
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No parquet episodes found under {dataset_root / 'data'}")
    return paths


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=4)
        f.write("\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _speed_label(speed: float) -> str:
    text = f"{speed:g}".replace(".", "p")
    return f"{text}x"


def _speed_name(speeds: list[float]) -> str:
    labels = []
    for speed in speeds:
        text = f"{speed:.2f}".rstrip("0").rstrip(".")
        if "." not in text:
            text += ".0"
        labels.append(text.replace(".", "p"))
    return "_".join(labels)


def _safe_name(value: str) -> str:
    out = []
    for char in value.strip().lower():
        if char.isalnum():
            out.append(char)
        elif char in {"-", "_"}:
            out.append(char)
        else:
            out.append("_")
    return "_".join(part for part in "".join(out).split("_") if part)


def _timestamp(value: str | None) -> str:
    if value:
        return _safe_name(value)
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_output_root(dst: Path, args: argparse.Namespace, speeds: list[float], episode_count: int) -> Path:
    if not args.auto_name:
        return dst.resolve()

    suite = _safe_name(args.task_suite_name)
    tag = _safe_name(args.run_tag)
    episodes = f"ep{episode_count}" if args.max_episodes is not None else "full"
    name = f"{suite}_speed_{_speed_name(speeds)}_{episodes}_{tag}_{_timestamp(args.timestamp)}"
    return (dst / name).resolve()


def _stack_column(df: pd.DataFrame, name: str) -> np.ndarray:
    return np.stack(df[name].to_numpy()).astype(np.float32)


def _video_keys(info: dict) -> list[str]:
    return [
        name
        for name, feature in info.get("features", {}).items()
        if feature.get("dtype") == "video"
    ]


def _read_video(path: Path) -> list[np.ndarray]:
    import imageio.v3 as iio

    if not path.exists():
        raise FileNotFoundError(f"Missing video: {path}")
    return [np.asarray(frame) for frame in iio.imiter(path)]


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    import imageio.v3 as iio

    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, np.stack(frames), fps=fps)


def _copy_or_resample_videos(
    src_root: Path,
    dst_root: Path,
    src_episode_index: int,
    dst_episode_index: int,
    source_frame_index: np.ndarray,
    observation_mask: np.ndarray,
    video_keys: list[str],
    chunks_size: int,
    fps: int,
) -> None:
    if not video_keys:
        return

    src_chunk = src_episode_index // chunks_size
    dst_chunk = dst_episode_index // chunks_size
    for key in video_keys:
        src = (
            src_root
            / "videos"
            / f"chunk-{src_chunk:03d}"
            / key
            / f"episode_{src_episode_index:06d}.mp4"
        )
        dst = (
            dst_root
            / "videos"
            / f"chunk-{dst_chunk:03d}"
            / key
            / f"episode_{dst_episode_index:06d}.mp4"
        )
        frames = _read_video(src)
        zero = np.zeros_like(frames[0])
        out = []
        for frame_idx, valid in zip(source_frame_index, observation_mask):
            if int(valid) == 0:
                out.append(zero)
            else:
                out.append(frames[min(int(frame_idx), len(frames) - 1)])
        _write_video(dst, out, fps=fps)


def _numeric_stats(values: np.ndarray) -> dict:
    arr = np.asarray(values)
    if arr.ndim == 1:
        flat = arr.reshape(-1, 1)
    else:
        flat = arr.reshape(arr.shape[0], -1)
    return {
        "min": flat.min(axis=0).tolist(),
        "max": flat.max(axis=0).tolist(),
        "mean": flat.mean(axis=0).tolist(),
        "std": flat.std(axis=0).tolist(),
        "count": [int(arr.shape[0])],
    }


def _episode_stats(df: pd.DataFrame, episode_index: int) -> dict:
    stats = {}
    for col in df.columns:
        first = df[col].iloc[0]
        if isinstance(first, np.ndarray):
            stats[col] = _numeric_stats(np.stack(df[col].to_numpy()))
        elif np.issubdtype(df[col].dtype, np.number):
            stats[col] = _numeric_stats(df[col].to_numpy())
    return {"episode_index": int(episode_index), "stats": stats}


def _update_info(
    src_info: dict,
    total_episodes: int,
    total_frames: int,
    total_videos: int,
    chunks_size: int,
    fps: int,
) -> dict:
    info = json.loads(json.dumps(src_info))
    info["total_episodes"] = int(total_episodes)
    info["total_frames"] = int(total_frames)
    info["total_videos"] = int(total_videos)
    info["total_chunks"] = int(math.ceil(total_episodes / chunks_size))
    info["chunks_size"] = int(chunks_size)
    info["fps"] = int(fps)
    info["splits"] = {"train": f"0:{total_episodes}"}
    info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

    features = info.setdefault("features", {})
    scalar_i64 = {"dtype": "int64", "shape": [1], "names": None}
    scalar_f32 = {"dtype": "float32", "shape": [1], "names": None}
    scalar_i8 = {"dtype": "int8", "shape": [1], "names": None}
    features.update(
        {
            "speed": scalar_f32,
            "speed_index": scalar_i64,
            "speed_label": {"dtype": "string", "shape": [1], "names": None},
            "valid_mask": scalar_i8,
            "observation_mask": scalar_i8,
            "action_mask": scalar_i8,
            "is_padded": scalar_i8,
            "segment_id": scalar_i64,
            "motion_class": scalar_i64,
            "source_episode_index": scalar_i64,
            "source_frame_index": scalar_i64,
            "source_step_index": scalar_i64,
            "source_index": scalar_i64,
            "cleaned_translation": scalar_i8,
            "cleaned_rotation": scalar_i8,
        }
    )
    for feature in features.values():
        if feature.get("dtype") == "video":
            feature.setdefault("info", {})["video.fps"] = int(fps)
    return info


def build_dataset(args: argparse.Namespace) -> None:
    src_root = Path(args.src).resolve()
    speeds = [float(x) for x in args.speeds]
    paths = _episode_paths(src_root, args.max_episodes)
    dst_root = _resolve_output_root(Path(args.dst), args, speeds, len(paths))
    if dst_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{dst_root} exists; pass --overwrite to replace it")
        shutil.rmtree(dst_root)

    src_info = _load_json(src_root / "meta" / "info.json")
    source_episode_tasks = {
        int(row["episode_index"]): row.get("tasks", [])
        for row in _read_jsonl(src_root / "meta" / "episodes.jsonl")
    }
    chunks_size = int(src_info.get("chunks_size", 1000))
    fps = int(args.fps or src_info.get("fps", 20))
    video_keys = _video_keys(src_info) if args.write_videos else []
    config = SpeedTransformConfig(
        transl_eps=args.segment_transl_eps,
        rot_eps=args.segment_rot_eps,
        clean_transl_eps=args.clean_transl_eps,
        clean_rot_eps=args.clean_rot_eps,
        direction_cos_threshold=args.direction_cos_threshold,
        min_segment_len=args.min_segment_len,
        keep_still_segments=not args.drop_still_segments,
        fps=fps,
    )

    dst_root.mkdir(parents=True, exist_ok=True)
    all_episode_rows: list[dict] = []
    all_stats: list[dict] = []
    all_metrics: list[dict] = []
    global_index = 0
    dst_episode_index = 0

    for path in tqdm(paths, desc="episodes"):
        src_df = pd.read_parquet(path)
        src_episode_index = int(src_df["episode_index"].iloc[0])
        task_index = int(src_df["task_index"].iloc[0])
        actions = _stack_column(src_df, "action")
        states = _stack_column(src_df, "observation.state")
        source_frame_indices = src_df["frame_index"].to_numpy(dtype=np.int64)

        for speed_index, speed in enumerate(speeds):
            transformed, metrics = transform_episode(actions, states, source_frame_indices, speed, config)
            n = len(transformed["action"])
            out_df = pd.DataFrame(
                {
                    "observation.state": list(transformed["state"].astype(np.float32)),
                    "action": list(transformed["action"].astype(np.float32)),
                    "timestamp": (np.arange(n, dtype=np.float32) / float(fps)).astype(np.float32),
                    "frame_index": np.arange(n, dtype=np.int64),
                    "episode_index": np.full(n, dst_episode_index, dtype=np.int64),
                    "index": np.arange(global_index, global_index + n, dtype=np.int64),
                    "task_index": np.full(n, task_index, dtype=np.int64),
                    "speed": transformed["speed"].astype(np.float32),
                    "speed_index": np.full(n, speed_index, dtype=np.int64),
                    "speed_label": np.full(n, _speed_label(speed), dtype=object),
                    "valid_mask": transformed["observation_mask"].astype(np.int8),
                    "observation_mask": transformed["observation_mask"].astype(np.int8),
                    "action_mask": transformed["action_mask"].astype(np.int8),
                    "is_padded": transformed["is_padded"].astype(np.int8),
                    "segment_id": transformed["segment_id"].astype(np.int64),
                    "motion_class": transformed["motion_class"].astype(np.int64),
                    "source_episode_index": np.full(n, src_episode_index, dtype=np.int64),
                    "source_frame_index": transformed["source_frame_index"].astype(np.int64),
                    "source_step_index": transformed["source_step_index"].astype(np.int64),
                    "source_index": src_df["index"].to_numpy(dtype=np.int64)[
                        transformed["source_step_index"].astype(np.int64)
                    ],
                    "cleaned_translation": transformed["cleaned_translation"].astype(np.int8),
                    "cleaned_rotation": transformed["cleaned_rotation"].astype(np.int8),
                }
            )

            chunk = dst_episode_index // chunks_size
            out_path = dst_root / "data" / f"chunk-{chunk:03d}" / f"episode_{dst_episode_index:06d}.parquet"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_df.to_parquet(out_path, index=False)

            _copy_or_resample_videos(
                src_root,
                dst_root,
                src_episode_index,
                dst_episode_index,
                transformed["source_frame_index"],
                transformed["observation_mask"],
                video_keys,
                chunks_size,
                fps,
            )

            task_payload = source_episode_tasks.get(src_episode_index, [])
            all_episode_rows.append(
                {
                    "episode_index": dst_episode_index,
                    "tasks": task_payload,
                    "length": int(n),
                    "source_episode_index": int(src_episode_index),
                    "speed": float(speed),
                    "speed_label": _speed_label(speed),
                }
            )
            all_stats.append(_episode_stats(out_df, dst_episode_index))
            metrics.update(
                {
                    "episode_index": int(dst_episode_index),
                    "source_episode_index": int(src_episode_index),
                    "task_index": int(task_index),
                    "speed": float(speed),
                    "speed_label": _speed_label(speed),
                }
            )
            all_metrics.append(metrics)
            global_index += n
            dst_episode_index += 1

    src_meta = src_root / "meta"
    dst_meta = dst_root / "meta"
    dst_meta.mkdir(parents=True, exist_ok=True)
    for filename in ["tasks.jsonl", "modality.json"]:
        src_file = src_meta / filename
        if src_file.exists():
            shutil.copy2(src_file, dst_meta / filename)

    info = _update_info(
        src_info,
        total_episodes=dst_episode_index,
        total_frames=global_index,
        total_videos=dst_episode_index * len(video_keys),
        chunks_size=chunks_size,
        fps=fps,
    )
    _write_json(dst_meta / "info.json", info)
    _write_jsonl(dst_meta / "episodes.jsonl", all_episode_rows)
    _write_jsonl(dst_meta / "episodes_stats.jsonl", all_stats)
    _write_jsonl(dst_meta / "speed_metrics.jsonl", all_metrics)

    print(f"Wrote {dst_episode_index} episodes / {global_index} frames to {dst_root}")
    if not args.write_videos:
        print("Video writing was disabled; enable --write-videos for LeRobot video loading.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="Source LeRobot LIBERO dataset root")
    parser.add_argument("--dst", required=True, help="Output dataset root, or parent root with --auto-name")
    parser.add_argument(
        "--auto-name",
        action="store_true",
        help="Treat --dst as a parent directory and create a unique run subdirectory.",
    )
    parser.add_argument("--run-tag", default="smoke", help="Run tag used by --auto-name")
    parser.add_argument("--task-suite-name", default="libero_spatial", help="Suite name used by --auto-name")
    parser.add_argument(
        "--timestamp",
        default=None,
        help="Optional timestamp/name suffix for --auto-name. Defaults to current YYYYMMDD_HHMMSS.",
    )
    parser.add_argument("--speeds", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--write-videos", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--segment-transl-eps", type=float, default=1e-4)
    parser.add_argument("--segment-rot-eps", type=float, default=1e-4)
    parser.add_argument("--clean-transl-eps", type=float, default=1e-4)
    parser.add_argument("--clean-rot-eps", type=float, default=1e-4)
    parser.add_argument("--direction-cos-threshold", type=float, default=-0.25)
    parser.add_argument("--min-segment-len", type=int, default=1)
    parser.add_argument("--drop-still-segments", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())
