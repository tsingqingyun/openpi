#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from various_speed.core import compute_replay_metrics


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


def _episode_paths(dataset_root: Path, limit: int | None) -> list[Path]:
    paths = sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet"))
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No parquet episodes found under {dataset_root / 'data'}")
    return paths


def _stack_column(df: pd.DataFrame, name: str) -> np.ndarray:
    return np.stack(df[name].to_numpy()).astype(np.float32)


def _source_episode_path(src_root: Path, episode_index: int, chunks_size: int) -> Path:
    return (
        src_root
        / "data"
        / f"chunk-{episode_index // chunks_size:03d}"
        / f"episode_{episode_index:06d}.parquet"
    )


def _load_info(root: Path) -> dict:
    with (root / "meta" / "info.json").open() as f:
        return json.load(f)


def _mean(rows: list[dict], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r and r[key] is not None]
    return float(np.mean(vals)) if vals else float("nan")


def _summarize(rows: list[dict]) -> dict:
    by_speed: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_speed[str(row.get("speed_label", row.get("speed", "unknown")))].append(row)

    summary = {"overall": {}, "by_speed": {}}
    keys = [
        "target_speed",
        "source_steps",
        "replay_steps",
        "actual_speed",
        "speed_error",
        "integrated_translation_l2_error",
        "integrated_rotation_l2_error",
        "translation_path_replay",
        "rotation_path_replay",
        "translation_path_ratio",
        "rotation_path_ratio",
        "gripper_switch_delta",
        "padded_ratio",
    ]
    for key in keys:
        summary["overall"][key] = _mean(rows, key)
    if "success" in rows[0]:
        summary["overall"]["success_rate"] = _mean(rows, "success")
        summary["overall"]["sim_steps"] = _mean(rows, "sim_steps")
        summary["overall"]["hit_max_steps_rate"] = _mean(rows, "hit_max_steps")
    if "source_success" in rows[0]:
        summary["overall"]["source_success_rate"] = _mean(rows, "source_success")
        summary["overall"]["source_sim_steps"] = _mean(rows, "source_sim_steps")

    for speed_label, items in sorted(by_speed.items()):
        summary["by_speed"][speed_label] = {key: _mean(items, key) for key in keys}
        if "success" in items[0]:
            summary["by_speed"][speed_label]["success_rate"] = _mean(items, "success")
            summary["by_speed"][speed_label]["sim_steps"] = _mean(items, "sim_steps")
            summary["by_speed"][speed_label]["hit_max_steps_rate"] = _mean(items, "hit_max_steps")
        if "source_success" in items[0]:
            summary["by_speed"][speed_label]["source_success_rate"] = _mean(items, "source_success")
            summary["by_speed"][speed_label]["source_sim_steps"] = _mean(items, "source_sim_steps")
        summary["by_speed"][speed_label]["episodes"] = len(items)
    return summary


def offline_replay(args: argparse.Namespace) -> list[dict]:
    dataset_root = Path(args.dataset).resolve()
    src_root = Path(args.source_dataset).resolve() if args.source_dataset else None
    info = _load_info(dataset_root)
    chunks_size = int(info.get("chunks_size", 1000))
    src_chunks_size = int(_load_info(src_root).get("chunks_size", 1000)) if src_root else chunks_size

    rows: list[dict] = []
    for path in tqdm(_episode_paths(dataset_root, args.max_episodes), desc="offline replay"):
        df = pd.read_parquet(path)
        replay_actions = _stack_column(df, "action")
        speed = float(df["speed"].iloc[0]) if "speed" in df else None
        speed_label = str(df["speed_label"].iloc[0]) if "speed_label" in df else str(speed)

        if src_root is not None and "source_episode_index" in df:
            src_episode_index = int(df["source_episode_index"].iloc[0])
            src_path = _source_episode_path(src_root, src_episode_index, src_chunks_size)
            src_df = pd.read_parquet(src_path)
            source_actions = _stack_column(src_df, "action")
            metrics = compute_replay_metrics(source_actions, replay_actions, speed)
        else:
            metrics = compute_replay_metrics(replay_actions, replay_actions, speed)

        metrics.update(
            {
                "episode_index": int(df["episode_index"].iloc[0]),
                "source_episode_index": int(df["source_episode_index"].iloc[0])
                if "source_episode_index" in df
                else None,
                "task_index": int(df["task_index"].iloc[0]),
                "speed": speed,
                "speed_label": speed_label,
                "padded_frames": int(df["is_padded"].sum()) if "is_padded" in df else 0,
                "padded_ratio": float(df["is_padded"].mean()) if "is_padded" in df else 0.0,
            }
        )
        rows.append(metrics)
    return rows


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = math.sqrt(max(1.0 - float(quat[3] * quat[3]), 0.0))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / den


def _get_libero_env(task, resolution: int, seed: int):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_description = task.language
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task_description


def _rollout_actions(task, initial_state: np.ndarray, actions: np.ndarray, args: argparse.Namespace) -> dict:
    env = None
    done = False
    reward = 0.0
    step = -1
    exception = None
    try:
        env, _task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        env.reset()
        _obs = env.set_init_state(initial_state)
        for _ in range(args.num_steps_wait):
            _obs, reward, done, _info = env.step(LIBERO_DUMMY_ACTION)

        max_actions = actions[: args.max_controller_steps]
        for step, action in enumerate(max_actions):
            _obs, reward, done, _info = env.step(action.tolist())
            if done:
                break
    except Exception as exc:  # noqa: BLE001 - diagnostics should survive simulator errors.
        exception = repr(exc)
    finally:
        if env is not None:
            env.close()

    sim_steps = int(step + 1) if step >= 0 else 0
    return {
        "success": float(bool(done)),
        "sim_steps": sim_steps,
        "hit_max_steps": float(not done and exception is None and sim_steps >= min(len(actions), args.max_controller_steps)),
        "final_reward": float(reward),
        "exception": exception,
    }


def sim_replay(args: argparse.Namespace, rows: list[dict]) -> list[dict]:
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    if args.libero_package:
        sys.path.insert(0, str(Path(args.libero_package).resolve()))

    from libero.libero import benchmark

    dataset_root = Path(args.dataset).resolve()
    src_root = Path(args.source_dataset).resolve() if args.source_dataset else None
    if args.compare_source and src_root is None:
        raise ValueError("--compare-source requires --source-dataset")
    src_chunks_size = int(_load_info(src_root).get("chunks_size", 1000)) if src_root else 1000

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    paths = _episode_paths(dataset_root, args.max_sim_episodes or args.max_episodes)

    row_by_episode = {int(r["episode_index"]): r for r in rows}
    for path in tqdm(paths, desc="sim replay"):
        df = pd.read_parquet(path)
        episode_index = int(df["episode_index"].iloc[0])
        task_index = int(df["task_index"].iloc[0])
        task = task_suite.get_task(task_index)
        initial_states = task_suite.get_task_init_states(task_index)
        source_episode_index = int(df["source_episode_index"].iloc[0]) if "source_episode_index" in df else 0
        init_idx = source_episode_index % len(initial_states)
        initial_state = initial_states[init_idx]
        row = row_by_episode[episode_index]

        if args.compare_source and src_root is not None:
            src_path = _source_episode_path(src_root, source_episode_index, src_chunks_size)
            src_df = pd.read_parquet(src_path)
            source_actions = _stack_column(src_df, "action")
            source_result = _rollout_actions(task, initial_state, source_actions, args)
            for key, value in source_result.items():
                row[f"source_{key}"] = value

        actions = _stack_column(df, "action")
        result = _rollout_actions(task, initial_state, actions, args)
        row.update(result)
        row["init_state_index"] = int(init_idx)
    return rows


def write_outputs(args: argparse.Namespace, rows: list[dict]) -> None:
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "replay_metrics.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    summary = _summarize(rows)
    with (out_dir / "replay_summary.json").open("w") as f:
        json.dump(summary, f, indent=4)
        f.write("\n")
    print(json.dumps(summary, indent=2))
    print(f"Wrote replay metrics to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay/check variable-speed LIBERO datasets.")
    parser.add_argument("--dataset", required=True, help="Processed speed dataset root")
    parser.add_argument("--source-dataset", default=None, help="Original source dataset root")
    parser.add_argument("--out", required=True, help="Output directory for replay metrics")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--sim", action="store_true", help="Also replay actions in LIBERO sim")
    parser.add_argument("--compare-source", action="store_true", help="Replay source actions before each processed episode")
    parser.add_argument("--max-sim-episodes", type=int, default=None)
    parser.add_argument("--task-suite-name", default="libero_spatial")
    parser.add_argument(
        "--libero-package",
        default="/mnt/data/fangyu/code/github/LIBERO",
        help="Path containing the libero Python package",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--max-controller-steps", type=int, default=1000)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    replay_rows = offline_replay(parsed)
    if parsed.sim:
        replay_rows = sim_replay(parsed, replay_rows)
    write_outputs(parsed, replay_rows)
