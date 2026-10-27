#!/usr/bin/env bash
# Default is read-only. Run install only after the separate Linux dependency approval.
set -euo pipefail
mode=${1:-check}
project=/mnt/b/python/SlimLLM
venv=/opt/slimllm-venv
if [[ "$mode" != check && "$mode" != install ]]; then
    echo 'Usage: bash setup_vllm.sh [check|install]' >&2
    exit 2
fi
[[ "$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")" == "$project" ]] || { echo 'Unexpected project path.' >&2; exit 1; }
[[ "$(uname -r)" == *microsoft* ]] || { echo 'This script is for WSL2 only.' >&2; exit 1; }
source /etc/os-release
[[ "$ID" == ubuntu && "$VERSION_ID" == 24.04 ]] || { echo 'Expected Ubuntu 24.04.' >&2; exit 1; }
[[ -f "$project/.cache/huggingface/qwen3-1.7b/config.json" ]] || exit 1
[[ -f "$project/outputs/qlora/final_adapter/adapter_model.safetensors" ]] || exit 1
export PYTHONDONTWRITEBYTECODE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export HF_HOME="$project/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET=1
export PIP_CACHE_DIR="$project/.cache/pip/linux"
export TORCH_HOME="$project/.cache/torch/linux"
export XDG_CACHE_HOME="$project/.cache/linux"
export CUDA_CACHE_PATH="$project/.cache/cuda/linux"
export TRITON_CACHE_DIR="$project/.cache/triton/linux"
export TORCHINDUCTOR_CACHE_DIR="$project/.cache/torchinductor/linux"
export VLLM_CACHE_ROOT="$project/.cache/vllm"
# Native Linux permissions are required by apt/gpgv. This directory is still
# physically inside B:\python\SlimLLM\.wsl\Ubuntu\ext4.vhdx.
export TMPDIR=/opt/slimllm-cache/tmp
if [[ "$mode" == check ]]; then
    python3 -B --version
    printf 'Independent environment: %s\n' "$venv"
    printf 'Physical Linux disk: B:\\python\\SlimLLM\\.wsl\\Ubuntu\\ext4.vhdx\n'
    df -h /mnt/b /mnt/c
    if [[ -x "$venv/bin/python" ]]; then
        "$venv/bin/python" -B -m pip list --format=json
    else
        echo 'Linux environment not yet created; no installation performed.'
    fi
    exit 0
fi
[[ "$EUID" == 0 ]] || { echo 'Use the explicitly approved WSL root command.' >&2; exit 1; }
free_b_kib=$(df -Pk /mnt/b | awk 'END {print $4}')
free_c_kib=$(df -Pk /mnt/c | awk 'END {print $4}')
(( free_b_kib >= 40 * 1024 * 1024 && free_c_kib >= 6 * 1024 * 1024 )) || { echo 'Insufficient disk budget; do not auto-clean.' >&2; exit 1; }
if [[ -e "$venv" && ! -f "$venv/pyvenv.cfg" ]]; then
    echo 'Existing environment path is not a venv; inspect it without overwriting.' >&2
    exit 1
fi
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$TORCH_HOME" "$XDG_CACHE_HOME" \
    "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
log="$project/.cache/tmp/vllm_install_${stamp}.log"
report="$project/reports/vllm_pip_install_${stamp}.json"
lock="$project/reports/vllm_linux_packages_${stamp}.txt"
[[ ! -e "$log" && ! -e "$report" && ! -e "$lock" ]] || exit 1
exec > >(tee -a "$log") 2>&1
export DEBIAN_FRONTEND=noninteractive
TMPDIR=/tmp apt-get -o APT::Update::Error-Mode=any -o Acquire::Retries=1 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 update
TMPDIR=/tmp apt-get -o Dpkg::Use-Pty=0 -o Acquire::Retries=1 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 install -y --no-install-recommends python3-venv libnuma1 libgomp1 gcc python3-dev
if [[ ! -e "$venv" ]]; then python3 -B -m venv "$venv"; fi
"$venv/bin/python" -B -m pip install \
    --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu128 \
    --only-binary=:all: --no-cache-dir --no-compile \
    -r "$project/requirements/requirements-vllm.txt" --report "$report"
"$venv/bin/python" -B -m pip check
"$venv/bin/python" -B -m pip freeze > "$lock"
echo 'Linux dependencies installed. Model loading and GPU checks are separate; no weights downloaded or copied.'
