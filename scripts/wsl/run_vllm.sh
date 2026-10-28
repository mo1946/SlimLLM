#!/usr/bin/env bash
# One local engine only. Stop a resident Transformers model before using this.
set -euo pipefail
project=/mnt/b/python/SlimLLM
venv=/opt/slimllm-venv
[[ "$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")" == "$project" ]] || exit 1
[[ -x "$venv/bin/vllm" ]] || { echo 'Install the separately approved Linux environment first.' >&2; exit 1; }
export HF_HOME="$project/.cache/huggingface" HF_HUB_CACHE="$project/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET=1
export TORCH_HOME="$project/.cache/torch/linux" XDG_CACHE_HOME="$project/.cache/linux"
export CUDA_CACHE_PATH="$project/.cache/cuda/linux" TRITON_CACHE_DIR="$project/.cache/triton/linux"
export TORCHINDUCTOR_CACHE_DIR="$project/.cache/torchinductor/linux" VLLM_CACHE_ROOT="$project/.cache/vllm"
export TMPDIR=/opt/slimllm-cache/tmp
export PYTHONDONTWRITEBYTECODE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_WORKER_MULTIPROC_METHOD=spawn OMP_NUM_THREADS=4
mkdir -p "$TMPDIR" "$TORCH_HOME" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" \
    "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT"
if [[ "${1:-}" == --help || "${1:-}" == --help=all ]]; then exec "$venv/bin/vllm" serve "$1"; fi
[[ $# == 0 ]] || { echo 'Usage: bash run_vllm.sh [--help]' >&2; exit 2; }
free_mib=$(/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
(( free_mib >= 6144 )) || { echo 'Need >=6144 MiB free; stop other model processes before loading.' >&2; exit 1; }
stamp=$(date -u +%Y%m%dT%H%M%SZ)
log="$project/.cache/tmp/vllm_server_${stamp}.log"
pid_file="$project/.cache/tmp/vllm_server_${stamp}.pid"
[[ ! -e "$log" && ! -e "$pid_file" ]] || exit 1
(set -o noclobber; printf '%s\n' "$$" > "$pid_file")
exec > >(tee -a "$log") 2>&1
printf 'SlimLLM vLLM PID=%s; log=%s\n' "$$" "$log"
exec "$venv/bin/vllm" serve "$project/.cache/huggingface/qwen3-1.7b" \
    --host 127.0.0.1 --port 8001 --served-model-name slimllm-base \
    --enable-lora --lora-modules "slimllm-lora=$project/outputs/qlora/final_adapter" \
    --max-loras 1 --max-lora-rank 8 --dtype float16 \
    --tensor-parallel-size 1 --pipeline-parallel-size 1 \
    --gpu-memory-utilization 0.70 --max-model-len 2560 \
    --max-num-seqs 2 --max-num-batched-tokens 2048 --enable-chunked-prefill \
    --structured-outputs-config '{"backend":"guidance","disable_fallback":true}' \
    --no-enable-prefix-caching --enforce-eager --cpu-offload-gb 0 --swap-space 0 --seed 42
