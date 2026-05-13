#!/usr/bin/env bash
set -euo pipefail

# EvalScope/VLMEvalKit supports both MVBench and MVBench_MP4.
# Override EVAL_DATASET=MVBench_MP4 if your local data is prepared in mp4 format.
MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
INFER_BACKEND="${INFER_BACKEND:-vllm}"
EVAL_DATASET="${EVAL_DATASET:-MVBench}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-eval_output/mvbench}"
EVAL_NUM_PROC="${EVAL_NUM_PROC:-16}"
EVAL_LIMIT="${EVAL_LIMIT:-}"
EVAL_GENERATION_CONFIG="${EVAL_GENERATION_CONFIG:-{\"max_tokens\": 1024, \"temperature\": 0}}"

cmd=(
  swift eval
  --model "${MODEL}"
  --infer_backend "${INFER_BACKEND}"
  --eval_backend VLMEvalKit
  --eval_dataset "${EVAL_DATASET}"
  --eval_output_dir "${EVAL_OUTPUT_DIR}"
  --eval_num_proc "${EVAL_NUM_PROC}"
  --eval_generation_config "${EVAL_GENERATION_CONFIG}"
)

if [[ -n "${EVAL_LIMIT}" ]]; then
  cmd+=(--eval_limit "${EVAL_LIMIT}")
fi

if [[ "${INFER_BACKEND}" == "vllm" ]]; then
  cmd+=(
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
    --vllm_max_model_len "${VLLM_MAX_MODEL_LEN:-8192}"
    --vllm_limit_mm_per_prompt "${VLLM_LIMIT_MM_PER_PROMPT:-{\"image\": 5, \"video\": 2}}"
  )
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
MAX_PIXELS="${MAX_PIXELS:-1003520}" \
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-50176}" \
FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-16}" \
"${cmd[@]}" "$@"
