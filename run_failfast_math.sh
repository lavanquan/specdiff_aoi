#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/home/qula0496/quan/.venv/bin/python"
FAILFAST_SCRIPT="$SCRIPT_DIR/failfast.py"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Error: Python interpreter not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$FAILFAST_SCRIPT" ]]; then
  echo "Error: failfast.py not found at: $FAILFAST_SCRIPT" >&2
  exit 1
fi

# Default GPU mapping; override by exporting CUDA_VISIBLE_DEVICES before running.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# Avoid W&B service/socket failures (e.g., BrokenPipeError) unless explicitly overridden.
export WANDB_MODE="${WANDB_MODE:-disabled}"

cmd=(
  "$PYTHON_BIN" "$FAILFAST_SCRIPT"
  --multi_gpu
  --num_drafters 3
  --target_gpu 0
  --drafter_gpus 1 2 3
  --dataset_name aime
  --num_questions 90
  --target_model_name Qwen/Qwen2.5-7B-Instruct
  --dllm_dir /home/qula0496/quan/Fast_dLLM_v2_1.5B
  --output_dir ./outputs
  --max_new_tokens 1024
  --spec_len 8
  --drafter_thresholds 0.9
  --run_dllm_sf
  --baseline_sweep
  --overwrite
  --timing_out ./outputs/system_timing.jsonl
#   --optimize_spec_len
)

# Optional extra args appended at the end, useful for quick overrides.
if (( $# > 0 )); then
  cmd+=("$@")
fi

echo "Working directory: $SCRIPT_DIR"
echo "WANDB_MODE: $WANDB_MODE"
printf "Command: "
printf "%q " "${cmd[@]}"
echo

cd "$SCRIPT_DIR"
exec "${cmd[@]}"
