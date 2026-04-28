#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=4)
        f.write("\n")


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _safe_name(path: Path) -> str:
    return path.name.replace("/", "_")


def _numeric_stats(values: np.ndarray) -> dict:
    arr = np.asarray(values)
    flat = arr.reshape(-1, 1) if arr.ndim == 1 else arr.reshape(arr.shape[0], -1)
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


def _video_keys(info: dict) -> list[str]:
    return [name for name, feature in info.get("features", {}).items() if feature.get("dtype") == "video"]


def _task_mapping(dataset: Path) -> dict[int, str]:
    return {int(row["task_index"]): row["task"] for row in _read_jsonl(dataset / "meta" / "tasks.jsonl")}


def merge_datasets(inputs: list[Path], output: Path, *, overwrite: bool) -> None:
    inputs = [path.resolve() for path in inputs]
    output = output.resolve()
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
        shutil.rmtree(output)

    if len(inputs) < 2:
        raise ValueError("At least two input datasets are required")

    infos = [_load_json(path / "meta" / "info.json") for path in inputs]
    chunks_size = int(infos[0].get("chunks_size", 1000))
    video_keys = _video_keys(infos[0])
    for dataset, info in zip(inputs, infos, strict=True):
        if int(info.get("chunks_size", chunks_size)) != chunks_size:
            raise ValueError(f"{dataset} uses a different chunks_size")
        if _video_keys(info) != video_keys:
            raise ValueError(f"{dataset} has different video keys")

    output.mkdir(parents=True)
    all_tasks: dict[str, int] = {}
    task_rows: list[dict] = []
    episode_rows: list[dict] = []
    stats_rows: list[dict] = []
    speed_metrics_rows: list[dict] = []

    global_frame_index = 0
    global_episode_index = 0
    total_videos = 0

    for dataset_index, dataset in enumerate(inputs):
        source_name = _safe_name(dataset)
        old_tasks = _task_mapping(dataset)
        old_to_new_task = {}
        for old_task_index, task in sorted(old_tasks.items()):
            if task not in all_tasks:
                all_tasks[task] = len(all_tasks)
                task_rows.append({"task_index": all_tasks[task], "task": task})
            old_to_new_task[old_task_index] = all_tasks[task]

        info = infos[dataset_index]
        input_chunks_size = int(info.get("chunks_size", chunks_size))
        for episode in tqdm(_read_jsonl(dataset / "meta" / "episodes.jsonl"), desc=source_name):
            old_episode_index = int(episode["episode_index"])
            old_chunk = old_episode_index // input_chunks_size
            new_chunk = global_episode_index // chunks_size
            src_data = dataset / "data" / f"chunk-{old_chunk:03d}" / f"episode_{old_episode_index:06d}.parquet"
            dst_data = output / "data" / f"chunk-{new_chunk:03d}" / f"episode_{global_episode_index:06d}.parquet"

            frame = pd.read_parquet(src_data)
            length = len(frame)
            frame["episode_index"] = np.full(length, global_episode_index, dtype=np.int64)
            frame["index"] = np.arange(global_frame_index, global_frame_index + length, dtype=np.int64)
            frame["task_index"] = frame["task_index"].map(old_to_new_task).astype(np.int64)
            dst_data.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(dst_data, index=False)

            for key in video_keys:
                src_video = dataset / "videos" / f"chunk-{old_chunk:03d}" / key / f"episode_{old_episode_index:06d}.mp4"
                dst_video = (
                    output / "videos" / f"chunk-{new_chunk:03d}" / key / f"episode_{global_episode_index:06d}.mp4"
                )
                dst_video.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_video, dst_video)
                total_videos += 1

            episode_row = dict(episode)
            episode_row.update(
                {
                    "episode_index": global_episode_index,
                    "length": int(length),
                    "source_dataset": source_name,
                    "source_dataset_index": int(dataset_index),
                    "source_dataset_episode_index": old_episode_index,
                }
            )
            episode_rows.append(episode_row)
            stats_rows.append(_episode_stats(frame, global_episode_index))

            global_frame_index += length
            global_episode_index += 1

        metrics_path = dataset / "meta" / "speed_metrics.jsonl"
        if metrics_path.exists():
            for row in _read_jsonl(metrics_path):
                metric = dict(row)
                metric["source_dataset"] = source_name
                metric["source_dataset_index"] = int(dataset_index)
                speed_metrics_rows.append(metric)

    info = json.loads(json.dumps(infos[0]))
    info["total_episodes"] = int(global_episode_index)
    info["total_frames"] = int(global_frame_index)
    info["total_tasks"] = len(task_rows)
    info["total_videos"] = int(total_videos)
    info["total_chunks"] = int(np.ceil(global_episode_index / chunks_size))
    info["chunks_size"] = int(chunks_size)
    info["splits"] = {"train": f"0:{global_episode_index}"}
    info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

    _write_json(output / "meta" / "info.json", info)
    _write_jsonl(output / "meta" / "tasks.jsonl", task_rows)
    _write_jsonl(output / "meta" / "episodes.jsonl", episode_rows)
    _write_jsonl(output / "meta" / "episodes_stats.jsonl", stats_rows)
    if speed_metrics_rows:
        _write_jsonl(output / "meta" / "speed_metrics.jsonl", speed_metrics_rows)

    modality_path = inputs[0] / "meta" / "modality.json"
    if modality_path.exists():
        shutil.copy2(modality_path, output / "meta" / "modality.json")

    print(f"Wrote {global_episode_index} episodes / {global_frame_index} frames to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="Input LeRobot dataset roots")
    parser.add_argument("--output", required=True, help="Merged output LeRobot dataset root")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge_datasets([Path(path) for path in args.inputs], Path(args.output), overwrite=args.overwrite)
