#!/usr/bin/env bash
set -euo pipefail

# Put the local MVBench data under this directory before running the script.
# VLMEvalKit reads official datasets from $LMUData and expects MVBench.tsv there.
MV_BENCH_DATA_ROOT="${HOME}/LMUData"
MODEL="Qwen/Qwen2.5-VL-3B-Instruct"

if [[ ! -f "${MV_BENCH_DATA_ROOT}/MVBench.tsv" ]]; then
  echo "MVBench.tsv not found in ${MV_BENCH_DATA_ROOT}." >&2
  echo "Please prepare the local MVBench dataset before evaluation." >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=0 \
LMUData="${MV_BENCH_DATA_ROOT}" \
MAX_PIXELS=1003520 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=16 \
swift eval \
  --model "${MODEL}" \
  --infer_backend vllm \
  --eval_backend VLMEvalKit \
  --eval_dataset MVBench
