"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import json
import pathlib

import numpy as np
import polars as pl
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def _local_lerobot_episode_paths(repo_id: str) -> list[pathlib.Path]:
    root = pathlib.Path(repo_id)
    paths = sorted((root / "data").glob("chunk-*/episode_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet episodes found under {root / 'data'}")
    return paths


def _local_lerobot_total_frames(repo_id: str) -> int:
    info_path = pathlib.Path(repo_id) / "meta" / "info.json"
    with info_path.open() as f:
        return int(json.load(f)["total_frames"])


def _stack_list_column(frame: pl.DataFrame, column: str) -> np.ndarray:
    return np.asarray(frame[column].to_list(), dtype=np.float32)


def _action_chunks(actions: np.ndarray, num_starts: int, action_horizon: int) -> np.ndarray:
    starts = np.arange(num_starts)[:, None]
    offsets = np.arange(action_horizon)[None, :]
    indices = np.minimum(starts + offsets, len(actions) - 1)
    return actions[indices]


def _can_use_fast_local_lerobot_stats(
    config: _config.TrainConfig,
    data_config: _config.DataConfig,
    max_frames: int | None,
) -> bool:
    if max_frames is not None:
        return False
    if not isinstance(config.data, _config.LeRobotVariousSpeedLiberoDataConfig):
        return False
    if config.data.extra_delta_transform:
        return False
    return data_config.repo_id is not None and pathlib.Path(data_config.repo_id).is_dir()


def _compute_fast_local_lerobot_stats(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
) -> tuple[dict[str, normalize.RunningStats], int]:
    """Compute stats from local LeRobot parquet data without decoding videos."""
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")

    total_frames = _local_lerobot_total_frames(data_config.repo_id)
    usable_frames = (total_frames // batch_size) * batch_size
    remaining = usable_frames
    stats = {key: normalize.RunningStats() for key in ["state", "actions"]}

    for path in tqdm.tqdm(
        _local_lerobot_episode_paths(data_config.repo_id),
        desc="Computing stats from parquet",
    ):
        if remaining <= 0:
            break

        frame = pl.read_parquet(path, columns=["observation.state", "action"])
        num_starts = min(len(frame), remaining)
        if num_starts <= 0:
            continue

        states = _stack_list_column(frame, "observation.state")
        actions = _stack_list_column(frame, "action")
        stats["state"].update(states[:num_starts])
        stats["actions"].update(_action_chunks(actions, num_starts, action_horizon))
        remaining -= num_starts

    return stats, usable_frames // batch_size


def main(config_name: str, max_frames: int | None = None, *, fast_local_lerobot: bool = True):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    keys = ["state", "actions"]
    if fast_local_lerobot and _can_use_fast_local_lerobot_stats(config, data_config, max_frames):
        stats, num_batches = _compute_fast_local_lerobot_stats(
            data_config,
            config.model.action_horizon,
            config.batch_size,
        )
    elif data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
        stats = {key: normalize.RunningStats() for key in keys}
        for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
            for key in keys:
                stats[key].update(np.asarray(batch[key]))
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )
        stats = {key: normalize.RunningStats() for key in keys}
        for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
            for key in keys:
                stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    if data_config.asset_id is None:
        raise ValueError("Data config must have an asset_id")
    output_path = config.assets_dirs / data_config.asset_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
