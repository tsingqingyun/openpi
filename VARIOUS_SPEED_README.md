# Various-Speed LIBERO Pipeline

Tools for creating and checking variable-speed LIBERO trajectories for pi0 /
pi0.5 experiments.

## 1. Build A Dataset

Recommended output root:

```bash
ROOT=/mnt/data/fangyu/dataset/VariousSpeed
SRC=/mnt/data/fangyu/dataset/IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot
cd /mnt/data/fangyu/code/VariousSpeed/openpi
```

Smoke build with automatic, non-overwriting run names:

```bash
uv run python scripts/build_libero_speed_dataset.py \
  --src "$SRC" \
  --dst "$ROOT" \
  --auto-name \
  --task-suite-name libero_spatial \
  --run-tag smoke \
  --speeds 0.5 1.0 2.0 \
  --max-episodes 2 \
  --write-videos
```

This creates a child directory like:

```text
/mnt/data/fangyu/dataset/VariousSpeed/libero_spatial_speed_0p5_1p0_2p0_ep2_smoke_20260428_173012
```

Naming format:

```text
{task_suite}_speed_{speeds}_{epN/full}_{run_tag}_{timestamp}
```

Use a fixed suffix when you want a reproducible run name:

```bash
uv run python scripts/build_libero_speed_dataset.py \
  --src "$SRC" \
  --dst "$ROOT" \
  --auto-name \
  --task-suite-name libero_spatial \
  --run-tag replay_debug \
  --timestamp test001 \
  --speeds 0.5 1.0 2.0 \
  --max-episodes 2 \
  --write-videos
```

Full spatial build:

```bash
uv run python scripts/build_libero_speed_dataset.py \
  --src "$SRC" \
  --dst "$ROOT" \
  --auto-name \
  --task-suite-name libero_spatial \
  --run-tag full \
  --speeds 0.5 1.0 2.0 \
  --write-videos
```

Do not pass `--overwrite` unless you intentionally want to replace an existing
run directory.

The transform:

- segments by translation/rotation only; gripper does not create motion segment boundaries,
- cleans near-zero translation/rotation noise while leaving gripper untouched,
- merges actions for speed > 1 and drops middle observations,
- preserves gripper switches as replay anchors during fast merge,
- splits actions for speed < 1, writes zero image frames for synthetic observations,
  and sets `observation_mask=0` / `is_padded=1`,
- stores `speed`, `speed_label`, `segment_id`, source indices, and masks in parquet.

## 2. Pick A Dataset

Use the latest generated dataset:

```bash
DATASET=$(ls -td /mnt/data/fangyu/dataset/VariousSpeed/libero_spatial_speed_* | head -1)
echo "$DATASET"
```

The `--dataset` argument must point to a single generated dataset directory that
contains `data/` and `meta/`, not the parent `VariousSpeed` directory.

## 3. Offline Replay

Offline replay checks only action sequence math. It does not start LIBERO sim and
does not need GPU.

```bash
uv run python scripts/replay_speed_dataset.py \
  --dataset "$DATASET" \
  --source-dataset "$SRC" \
  --out "$DATASET/replay"
```

Outputs:

```text
$DATASET/replay/replay_metrics.jsonl
$DATASET/replay/replay_summary.json
```

Metrics include step count, target speed, actual speed, speed error, integrated
6D delta error, path length ratio, gripper switch delta, and padded ratio.

## 4. Online LIBERO Action Replay

Online replay sends stored actions to the LIBERO controller through
`OffScreenRenderEnv.step(action)`. It does not run pi0/pi05 inference.

```bash
CUDA_VISIBLE_DEVICES=4,5 \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
NUMBA_DISABLE_JIT=1 \
MPLCONFIGDIR=/tmp/matplotlib \
uv run python scripts/replay_speed_dataset.py \
  --dataset "$DATASET" \
  --source-dataset "$SRC" \
  --out "$DATASET/sim_replay" \
  --sim \
  --compare-source \
  --task-suite-name libero_spatial \
  --max-sim-episodes 6
```

With logs:

```bash
CUDA_VISIBLE_DEVICES=4,5 \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
NUMBA_DISABLE_JIT=1 \
MPLCONFIGDIR=/tmp/matplotlib \
uv run python scripts/replay_speed_dataset.py \
  --dataset "$DATASET" \
  --source-dataset "$SRC" \
  --out "$DATASET/sim_replay" \
  --sim \
  --compare-source \
  --task-suite-name libero_spatial \
  --max-sim-episodes 6 \
  2>&1 | tee "$DATASET/sim_replay.log"
```

`--compare-source` first replays the original source actions from the same initial
state and records `source_success`. This separates variable-speed failures from
environment/action-interface failures.

Outputs:

```text
$DATASET/sim_replay/replay_metrics.jsonl
$DATASET/sim_replay/replay_summary.json
```

## 5. Visualize

```bash
uv run python scripts/visualize_speed_dataset.py \
  --dataset "$DATASET" \
  --out "$DATASET/visualizations" \
  --num-demos 20
```

This writes action plots and contact sheets. In contact sheets, `m=0` means the
frame is a zero-padded synthetic observation from split/slow trajectories. These
black frames are dataset observation padding only; online replay sends actions to
the controller and does not use those frames for control.

## 6. OpenPI Training Interface

This repo adds `pi0_libero_various_speed` and `pi05_libero_various_speed` configs
under `openpi/src/openpi/training/config.py`. They use the VariousSpeed parquet
schema directly:

- `observation.images.image`
- `observation.images.wrist_image`
- `observation.state`
- `action`
- `speed` / `speed_label`

The training prompt is text-conditioned as:

```text
Perform the task at 0.5x speed. <original task prompt>
```

OpenPI needs Python 3.11+. Example setup with Tsinghua PyPI and a GitHub mirror:

```bash
conda create -n openpi-vs python=3.11 -y
conda activate openpi-vs

pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
git config --global url."https://gh-proxy.com/https://github.com/".insteadOf "https://github.com/"

cd /mnt/data/fangyu/code/VariousSpeed/openpi
UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
UV_CACHE_DIR=/tmp/uv-cache \
uv sync
```

Generated datasets intentionally do not copy stale source normalization stats.
Before real training, recompute norm stats for the selected config and dataset.
