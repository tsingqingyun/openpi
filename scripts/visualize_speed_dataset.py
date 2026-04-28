#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm


ACTION_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def _episode_paths(dataset_root: Path, limit: int | None) -> list[Path]:
    paths = sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet"))
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No parquet episodes found under {dataset_root / 'data'}")
    return paths


def _stack_column(df: pd.DataFrame, name: str) -> np.ndarray:
    return np.stack(df[name].to_numpy()).astype(np.float32)


def _plot_actions(df: pd.DataFrame, out_path: Path) -> None:
    actions = _stack_column(df, "action")
    fig, axes = plt.subplots(7, 1, figsize=(12, 10), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(actions[:, i], linewidth=1.2)
        ax.set_ylabel(ACTION_LABELS[i])
        ax.grid(True, alpha=0.2)
    title = f"episode={int(df['episode_index'].iloc[0])}"
    if "speed_label" in df:
        title += f"  speed={df['speed_label'].iloc[0]}"
    if "source_episode_index" in df:
        title += f"  source={int(df['source_episode_index'].iloc[0])}"
    axes[0].set_title(title)
    axes[-1].set_xlabel("controller step")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _find_video(dataset_root: Path, episode_index: int, key: str) -> Path | None:
    for chunk in sorted((dataset_root / "videos").glob("chunk-*")):
        candidate = chunk / key / f"episode_{episode_index:06d}.mp4"
        if candidate.exists():
            return candidate
    return None


def _contact_sheet(df: pd.DataFrame, dataset_root: Path, out_path: Path, video_key: str, samples: int) -> None:
    episode_index = int(df["episode_index"].iloc[0])
    video_path = _find_video(dataset_root, episode_index, video_key)
    if video_path is None:
        return

    frames = [np.asarray(frame) for frame in iio.imiter(video_path)]
    if not frames:
        return
    indices = np.linspace(0, len(frames) - 1, num=min(samples, len(frames)), dtype=int)
    thumbs = []
    for idx in indices:
        img = Image.fromarray(frames[idx]).resize((160, 160))
        draw = ImageDraw.Draw(img)
        mask = int(df["observation_mask"].iloc[idx]) if "observation_mask" in df else 1
        label = f"t={idx} m={mask}"
        draw.rectangle((0, 0, 78, 18), fill=(0, 0, 0))
        draw.text((4, 3), label, fill=(255, 255, 255))
        thumbs.append(img)

    cols = min(5, len(thumbs))
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * 160, rows * 160), color=(255, 255, 255))
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % cols) * 160, (i // cols) * 160))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def visualize(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset).resolve()
    out_dir = Path(args.out).resolve()
    by_source: dict[int, list[Path]] = defaultdict(list)
    for path in _episode_paths(dataset_root, None):
        df_head = pd.read_parquet(path, columns=["episode_index", "source_episode_index"])
        source = int(df_head["source_episode_index"].iloc[0]) if "source_episode_index" in df_head else int(df_head["episode_index"].iloc[0])
        by_source[source].append(path)

    selected = []
    for _source, paths in sorted(by_source.items()):
        selected.extend(paths)
        if len(selected) >= args.num_demos:
            break

    for path in tqdm(selected[: args.num_demos], desc="visualize"):
        df = pd.read_parquet(path)
        episode_index = int(df["episode_index"].iloc[0])
        speed_label = str(df["speed_label"].iloc[0]) if "speed_label" in df else "speed"
        stem = f"episode_{episode_index:06d}_{speed_label}"
        _plot_actions(df, out_dir / f"{stem}_actions.png")
        _contact_sheet(
            df,
            dataset_root,
            out_dir / f"{stem}_{args.video_key.replace('/', '_')}.jpg",
            args.video_key,
            args.frames,
        )
    print(f"Wrote visualizations to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize variable-speed LIBERO episodes.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-demos", type=int, default=20)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--video-key", default="observation.images.image")
    return parser.parse_args()


if __name__ == "__main__":
    visualize(parse_args())
