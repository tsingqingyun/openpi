# Various-Speed LIBERO Pipeline

This repo contains the current workflow for creating variable-speed LIBERO
LeRobot datasets and finetuning OpenPI pi0 / pi0.5 policies.

The main pipeline now targets four LIBERO suites:

- `libero_spatial`
- `libero_object`
- `libero_goal`
- `libero_10`

Each suite is converted into 0.5x / 1.0x / 2.0x speed variants, then the four
generated datasets are merged into one LeRobot dataset for joint training.

## 1. Environment

```bash
cd /mnt/data/fangyu/code/VariousSpeed/openpi

export ROOT=/mnt/data/fangyu/dataset/VariousSpeed
export BASE=/mnt/data/fangyu/dataset/IPEC-COMMUNITY
export UV_CACHE_DIR=/tmp/uv-cache
export HF_HOME=/tmp/hf-cache
export HF_DATASETS_CACHE=/tmp/hf-cache/datasets
```

The four source datasets expected by the commands below are:

```text
$BASE/libero_spatial_no_noops_1.0.0_lerobot
$BASE/libero_object_no_noops_1.0.0_lerobot
$BASE/libero_goal_no_noops_1.0.0_lerobot
$BASE/libero_10_no_noops_1.0.0_lerobot
```

If you use Weights & Biases, log in once:

```bash
uv run wandb login
```

Or, inside tmux, pass the key without echoing it:

```bash
read -s WANDB_API_KEY
export WANDB_API_KEY
```

## 2. Build Four Speed Datasets

Run the same speed processing on all four suites:

```bash
for SUITE in libero_spatial libero_object libero_goal libero_10; do
  uv run python scripts/build_libero_speed_dataset.py \
    --src "$BASE/${SUITE}_no_noops_1.0.0_lerobot" \
    --dst "$ROOT" \
    --auto-name \
    --task-suite-name "$SUITE" \
    --run-tag four_suite_v1 \
    --timestamp 20260429 \
    --speeds 0.5 1.0 2.0 \
    --clean-transl-eps 1e-4 \
    --clean-rot-eps 1e-4 \
    --min-segment-len 1 \
    --write-videos
done
```

This creates:

```text
$ROOT/libero_spatial_speed_0p5_1p0_2p0_full_four_suite_v1_20260429
$ROOT/libero_object_speed_0p5_1p0_2p0_full_four_suite_v1_20260429
$ROOT/libero_goal_speed_0p5_1p0_2p0_full_four_suite_v1_20260429
$ROOT/libero_10_speed_0p5_1p0_2p0_full_four_suite_v1_20260429
```

The speed transform:

- segments by translation/rotation only; gripper does not create motion segment boundaries,
- cleans near-zero translation/rotation noise while leaving gripper untouched,
- merges actions for speed > 1 and drops middle observations,
- preserves gripper switches as replay anchors during fast merge,
- splits actions for speed < 1, writes zero image frames for synthetic observations,
  and sets `observation_mask=0` / `is_padded=1`,
- stores `speed`, `speed_label`, `segment_id`, source indices, and masks in parquet.

Do not pass `--overwrite` unless you intentionally want to replace an existing
generated dataset.

## 3. Merge Four Datasets

Merge the four speed-processed datasets into one LeRobot dataset:

```bash
uv run python scripts/merge_lerobot_datasets.py \
  --inputs \
    "$ROOT/libero_spatial_speed_0p5_1p0_2p0_full_four_suite_v1_20260429" \
    "$ROOT/libero_object_speed_0p5_1p0_2p0_full_four_suite_v1_20260429" \
    "$ROOT/libero_goal_speed_0p5_1p0_2p0_full_four_suite_v1_20260429" \
    "$ROOT/libero_10_speed_0p5_1p0_2p0_full_four_suite_v1_20260429" \
  --output "$ROOT/libero_all_speed_0p5_1p0_2p0_full_v1" \
  --overwrite
```

The merge script renumbers:

- `episode_index`
- global frame `index`
- `task_index`
- video filenames

It also rewrites:

- `meta/info.json`
- `meta/tasks.jsonl`
- `meta/episodes.jsonl`
- `meta/episodes_stats.jsonl`
- `meta/speed_metrics.jsonl` when available

## 4. Automatic Merge + Norm Stats + Training

When the four speed-processing jobs are still running, use this script to wait
for them and continue automatically:

```bash
scripts/run_four_suite_merge_and_train.sh
```

Default settings:

```text
ROOT=/mnt/data/fangyu/dataset/VariousSpeed
RUN_TAG=four_suite_v1
TIMESTAMP=20260429
OUTPUT=$ROOT/libero_all_speed_0p5_1p0_2p0_full_v1
CONFIG_NAME=pi0_libero_various_speed_all
EXP_NAME=pi0_various_speed_all_gpu45
PROJECT_NAME=various_speed
GPU_IDS=4,5
NUM_WORKERS=8
WAIT_SECONDS=300
POLL_SECONDS=60
```

Recommended tmux command:

```bash
cd /mnt/data/fangyu/code/VariousSpeed/openpi

read -s WANDB_API_KEY
export WANDB_API_KEY

WAIT_SECONDS=43200 \
POLL_SECONDS=120 \
GPU_IDS=4,5 \
EXP_NAME=pi0_various_speed_all_gpu45 \
PROJECT_NAME=various_speed \
scripts/run_four_suite_merge_and_train.sh 2>&1 | tee four_suite_merge_train.log
```

This performs:

```text
wait for the four generated datasets
-> merge them into $ROOT/libero_all_speed_0p5_1p0_2p0_full_v1
-> compute pi0 norm stats
-> start JAX pi0 finetuning on GPU 4,5
```

Watch logs from another tmux pane:

```bash
tail -f four_suite_merge_train.log
```

## 5. Norm Stats

Generated datasets intentionally do not copy stale source normalization stats.
Always recompute norm stats before training a new dataset/config.

For four-suite pi0:

```bash
UV_CACHE_DIR=/tmp/uv-cache \
HF_HOME=/tmp/hf-cache \
HF_DATASETS_CACHE=/tmp/hf-cache/datasets \
uv run python scripts/compute_norm_stats.py \
  --config-name pi0_libero_various_speed_all
```

Output:

```text
assets/pi0_libero_various_speed_all/libero_various_speed_all_pi0/norm_stats.json
```

For four-suite pi0.5:

```bash
UV_CACHE_DIR=/tmp/uv-cache \
HF_HOME=/tmp/hf-cache \
HF_DATASETS_CACHE=/tmp/hf-cache/datasets \
uv run python scripts/compute_norm_stats.py \
  --config-name pi05_libero_various_speed_all
```

Output:

```text
assets/pi05_libero_various_speed_all/libero_various_speed_all_pi05/norm_stats.json
```

The norm-stat script has a local LeRobot fast path for these VariousSpeed
datasets. It reads parquet `observation.state` and `action` directly and skips
video decoding. This is correct for norm stats because only `state` and
`actions` are normalized. Training still uses the videos.

## 6. Training

The current pi0/pi0.5 training path uses JAX:

```text
scripts/train.py
```

Four-suite pi0 smoke test:

```bash
CUDA_VISIBLE_DEVICES=4,5 \
UV_CACHE_DIR=/tmp/uv-cache \
HF_HOME=/tmp/hf-cache \
HF_DATASETS_CACHE=/tmp/hf-cache/datasets \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi0_libero_various_speed_all \
  --exp-name=pi0_various_speed_all_smoke_gpu45 \
  --project-name=various_speed \
  --num-train-steps=10 \
  --save-interval=10 \
  --num-workers=8 \
  --wandb-enabled \
  --overwrite
```

Four-suite pi0 full training:

```bash
CUDA_VISIBLE_DEVICES=4,5 \
UV_CACHE_DIR=/tmp/uv-cache \
HF_HOME=/tmp/hf-cache \
HF_DATASETS_CACHE=/tmp/hf-cache/datasets \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi0_libero_various_speed_all \
  --exp-name=pi0_various_speed_all_gpu45 \
  --project-name=various_speed \
  --num-workers=8 \
  --wandb-enabled \
  --overwrite
```

Resume an interrupted full run:

```bash
CUDA_VISIBLE_DEVICES=4,5 \
UV_CACHE_DIR=/tmp/uv-cache \
HF_HOME=/tmp/hf-cache \
HF_DATASETS_CACHE=/tmp/hf-cache/datasets \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi0_libero_various_speed_all \
  --exp-name=pi0_various_speed_all_gpu45 \
  --project-name=various_speed \
  --num-workers=8 \
  --wandb-enabled \
  --resume
```

Do not use `--overwrite` when resuming.

Checkpoints:

```text
checkpoints/pi0_libero_various_speed_all/pi0_various_speed_all_gpu45/
```

If training runs out of GPU memory, lower the batch size:

```bash
--batch-size=16
```

then, if needed:

```bash
--batch-size=8
```

## 7. Single-Suite Spatial Reference

The earlier single-suite configs are still available:

- `pi0_libero_various_speed`
- `pi05_libero_various_speed`

They point to:

```text
/mnt/data/fangyu/dataset/VariousSpeed/libero_spatial_speed_0p5_1p0_2p0_full_v1_20260428_204431
```

Their norm stats are stored separately:

```text
assets/pi0_libero_various_speed/libero_various_speed_pi0/norm_stats.json
assets/pi05_libero_various_speed/libero_various_speed_pi05/norm_stats.json
```

## 8. Replay And Visualization

Pick a generated dataset:

```bash
DATASET="$ROOT/libero_spatial_speed_0p5_1p0_2p0_full_four_suite_v1_20260429"
SRC="$BASE/libero_spatial_no_noops_1.0.0_lerobot"
```

Offline replay:

```bash
uv run python scripts/replay_speed_dataset.py \
  --dataset "$DATASET" \
  --source-dataset "$SRC" \
  --out "$DATASET/replay"
```

Online LIBERO action replay:

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

Visualize:

```bash
uv run python scripts/visualize_speed_dataset.py \
  --dataset "$DATASET" \
  --out "$DATASET/visualizations" \
  --num-demos 20
```

In contact sheets, `m=0` means the frame is a zero-padded synthetic observation
from split/slow trajectories. These black frames are dataset observation padding
only; online replay sends actions to the controller and does not use those frames
for control.

## 9. Serving

After training, serve a checkpoint like:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_libero_various_speed_all \
  --policy.dir=checkpoints/pi0_libero_various_speed_all/pi0_various_speed_all_gpu45/30000
```
