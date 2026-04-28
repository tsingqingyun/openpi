#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/data/fangyu/dataset/VariousSpeed}"
RUN_TAG="${RUN_TAG:-four_suite_v1}"
TIMESTAMP="${TIMESTAMP:-20260429}"
OUTPUT="${OUTPUT:-$ROOT/libero_all_speed_0p5_1p0_2p0_full_v1}"

CONFIG_NAME="${CONFIG_NAME:-pi0_libero_various_speed_all}"
EXP_NAME="${EXP_NAME:-pi0_various_speed_all_gpu45}"
PROJECT_NAME="${PROJECT_NAME:-various_speed}"

GPU_IDS="${GPU_IDS:-4,5}"
NUM_WORKERS="${NUM_WORKERS:-8}"
WANDB_ENABLED="${WANDB_ENABLED:-1}"
OVERWRITE_OUTPUT="${OVERWRITE_OUTPUT:-1}"
OVERWRITE_TRAIN="${OVERWRITE_TRAIN:-1}"
WAIT_SECONDS="${WAIT_SECONDS:-300}"
POLL_SECONDS="${POLL_SECONDS:-60}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export HF_HOME="${HF_HOME:-/tmp/hf-cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/hf-cache/datasets}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

inputs=(
  "$ROOT/libero_spatial_speed_0p5_1p0_2p0_full_${RUN_TAG}_${TIMESTAMP}"
  "$ROOT/libero_object_speed_0p5_1p0_2p0_full_${RUN_TAG}_${TIMESTAMP}"
  "$ROOT/libero_goal_speed_0p5_1p0_2p0_full_${RUN_TAG}_${TIMESTAMP}"
  "$ROOT/libero_10_speed_0p5_1p0_2p0_full_${RUN_TAG}_${TIMESTAMP}"
)

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

dataset_ready() {
  local dataset="$1"
  [[ -f "$dataset/meta/info.json" ]] || return 1
  [[ -f "$dataset/meta/episodes.jsonl" ]] || return 1
  [[ -f "$dataset/meta/tasks.jsonl" ]] || return 1
  [[ -d "$dataset/data" ]] || return 1
  [[ -d "$dataset/videos" ]] || return 1

  python - "$dataset" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
info = json.loads((root / "meta" / "info.json").read_text())
expected_episodes = int(info["total_episodes"])
expected_videos = int(info.get("total_videos", 0))
data_files = list((root / "data").glob("chunk-*/episode_*.parquet"))
video_files = list((root / "videos").glob("chunk-*/*/episode_*.mp4"))
if len(data_files) != expected_episodes:
    raise SystemExit(1)
if expected_videos and len(video_files) != expected_videos:
    raise SystemExit(1)
PY
}

wait_for_inputs() {
  local deadline=$((SECONDS + WAIT_SECONDS))
  while true; do
    local all_ready=1
    for dataset in "${inputs[@]}"; do
      if dataset_ready "$dataset"; then
        log "ready: $dataset"
      else
        log "waiting: $dataset"
        all_ready=0
      fi
    done

    if [[ "$all_ready" -eq 1 ]]; then
      return 0
    fi
    if (( SECONDS >= deadline )); then
      log "timed out waiting for input datasets"
      return 1
    fi
    sleep "$POLL_SECONDS"
  done
}

main() {
  cd /mnt/data/fangyu/code/VariousSpeed/openpi

  log "waiting for four speed-processed datasets"
  wait_for_inputs

  log "merging datasets into $OUTPUT"
  merge_args=()
  if [[ "$OVERWRITE_OUTPUT" == "1" ]]; then
    merge_args+=(--overwrite)
  fi
  uv run python scripts/merge_lerobot_datasets.py \
    --inputs "${inputs[@]}" \
    --output "$OUTPUT" \
    "${merge_args[@]}"

  log "computing norm stats for $CONFIG_NAME"
  uv run python scripts/compute_norm_stats.py --config-name "$CONFIG_NAME"

  log "starting finetune: config=$CONFIG_NAME exp=$EXP_NAME gpus=$GPU_IDS"
  train_args=()
  if [[ "$WANDB_ENABLED" == "1" ]]; then
    train_args+=(--wandb-enabled)
  else
    train_args+=(--no-wandb-enabled)
  fi
  if [[ "$OVERWRITE_TRAIN" == "1" ]]; then
    train_args+=(--overwrite)
  fi

  uv run scripts/train.py "$CONFIG_NAME" \
    --exp-name="$EXP_NAME" \
    --project-name="$PROJECT_NAME" \
    --num-workers="$NUM_WORKERS" \
    "${train_args[@]}"
}

main "$@"
