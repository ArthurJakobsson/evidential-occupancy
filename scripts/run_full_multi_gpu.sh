#!/usr/bin/env bash
# Run the full nuScenes data-generation pipeline across N GPUs in parallel.
#
# Each GPU processes a contiguous shard of scenes through all three steps
# (transmissions-reflections -> scene-flow -> temporal-accumulation). Shards are
# self-contained, so the GPUs never contend for the same output files.
#
# Prereqs:
#   1) pixi install              (build the env once)
#   2) cp paths.env.example paths.env && edit it for this machine
#   3) source paths.env          (sets NUSCENES_ROOT, NUSCENES_EXTRA_ROOT, NUM_GPUS)
#
# Usage:
#   source paths.env
#   bash scripts/run_full_multi_gpu.sh
#
# Resumable: if it crashes, just run it again — missing_only=true skips finished frames.
set -euo pipefail

# repo root = parent of this script's dir
cd "$(dirname "$0")/.."

NUM_GPUS="${NUM_GPUS:-5}"
CONFIG_NAME="${CONFIG_NAME:-full}"
LOG_DIR="${LOG_DIR:-logs}"
PIXI="${PIXI:-pixi}"   # set PIXI=~/.pixi/bin/pixi if pixi is not on PATH
mkdir -p "$LOG_DIR"

echo "=================================================="
echo " NUSCENES_ROOT       = ${NUSCENES_ROOT:-<unset!>}"
echo " NUSCENES_EXTRA_ROOT = ${NUSCENES_EXTRA_ROOT:-<unset!>}"
echo " NUM_GPUS            = ${NUM_GPUS}"
echo " config              = ${CONFIG_NAME}"
echo " logs                = ${LOG_DIR}/"
echo "=================================================="

# Step 0: build the shared annotation cache ONCE (single process) so the parallel
# scene-flow workers don't race to write it.
echo "[prime] building sample-annotation cache (one-time)..."
"$PIXI" run python scripts/generate_shard.py \
    --config-name "$CONFIG_NAME" --gpu-index 0 --num-gpus "$NUM_GPUS" --prime-cache \
    2>&1 | tee "$LOG_DIR/prime.log"

# Step 1: launch one worker per GPU.
echo "[launch] starting ${NUM_GPUS} GPU workers..."
pids=()
for g in $(seq 0 $((NUM_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES="$g" "$PIXI" run python scripts/generate_shard.py \
        --config-name "$CONFIG_NAME" --gpu-index "$g" --num-gpus "$NUM_GPUS" \
        > "$LOG_DIR/gpu_${g}.log" 2>&1 &
    pids+=("$!")
    echo "  gpu ${g} -> pid ${pids[-1]}  (log: $LOG_DIR/gpu_${g}.log)"
done

echo "[wait] watch progress with:  tail -f $LOG_DIR/gpu_*.log"
fail=0
for g in $(seq 0 $((NUM_GPUS - 1))); do
    if wait "${pids[$g]}"; then
        echo "  gpu ${g}: OK"
    else
        echo "  gpu ${g}: FAILED -> see $LOG_DIR/gpu_${g}.log"
        fail=1
    fi
done

if [ "$fail" -eq 0 ]; then
    echo "[done] all GPUs finished. Outputs in: ${NUSCENES_EXTRA_ROOT}/reflection_and_transmission_multi_frame/"
else
    echo "[done] some shards failed; fix and re-run this script (it resumes)."
fi
exit "$fail"
