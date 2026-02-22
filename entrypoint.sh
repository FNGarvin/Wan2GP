#!/usr/bin/env bash
# FNGarvin - Wan2GP Project
# License: TODO: INSERT LICENSE
# Year: 2026
#
# entrypoint.sh
# Container entrypoint. Detects GPU model and VRAM at runtime, selects the
# appropriate WanGP profile (1-5) and attention mode (sage2/sage/sdpa), then
# launches wgp.py. Persistent data directories are routed to /workspace/ for
# RunPod pod-volume persistence.
#
# Runtime overrides (env vars):
#   WGP_PROFILE   — override auto-detected profile (1-5)
#   WGP_ATTENTION — override auto-detected attention mode
#   WGP_ARGS      — additional wgp.py arguments (e.g. "--compile --teacache 2.0")
#   CUDA_VISIBLE_DEVICES — standard NVIDIA env; respected automatically

set -euo pipefail

export HOME=/home/user
export PYTHONUNBUFFERED=1

# ── Persistent cache dirs (RunPod pod volume) ────────────────────────────────
export HF_HOME=/workspace/.cache/huggingface
export TRITON_CACHE_DIR=/workspace/.cache/triton

# ── CPU thread tuning ────────────────────────────────────────────────────────
_nproc=$(nproc)
export OMP_NUM_THREADS=$_nproc
export MKL_NUM_THREADS=$_nproc
export OPENBLAS_NUM_THREADS=$_nproc
export NUMEXPR_NUM_THREADS=$_nproc

# ── TF32 acceleration (Ampere+) ──────────────────────────────────────────────
export TORCH_ALLOW_TF32_CUBLAS=1
export TORCH_ALLOW_TF32_CUDNN=1

# ── Audio dummy (suppress ALSA/PulseAudio noise in headless containers) ──────
export SDL_AUDIODRIVER=dummy
export PULSE_RUNTIME_PATH=/tmp/pulse-runtime

# ─────────────────────────────────────────────────────────────────────────────
# GPU detection helpers (ported from run-docker-cuda-deb.sh)
# ─────────────────────────────────────────────────────────────────────────────

_detect_gpu_name() {
    nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1
}

_detect_vram_gb() {
    local mb
    mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    echo $(( mb / 1024 ))
}

# Maps GPU name → WanGP profile number (1-5).
# Profile definitions (from WanGP UI):
#   1 HighRAM_HighVRAM  48GB+ RAM, 24GB+ VRAM (fastest short video; RTX 3090/4090)
#   2 HighRAM_LowVRAM   48GB+ RAM, 12GB+ VRAM (recommended; most versatile)
#   3 LowRAM_HighVRAM   32GB+ RAM, 24GB+ VRAM (RTX 3090/4090 with limited RAM)
#   4 LowRAM_LowVRAM    32GB+ RAM, 12GB+ VRAM (default)
#   5 VeryLowRAM_LowVRAM 16GB+ RAM, 10GB+ VRAM (fail-safe; slow)
_map_profile() {
    local name="$1" vram_gb="$2"
    case "$name" in
        *"RTX 50"*|*"5090"*|*"5080"*|*"5070"*|\
        *"A100"*|*"A800"*|*"H100"*|*"H800"*)
            [ "$vram_gb" -ge 24 ] && echo 1 || echo 2 ;;
        *"RTX 40"*|*"4090"*|*"RTX 30"*|*"3090"*)
            [ "$vram_gb" -ge 24 ] && echo 3 || echo 2 ;;
        *"4080"*|*"4070"*|*"3080"*|*"3070"*|\
        *"RTX 20"*|*"2080"*|*"2070"*)
            [ "$vram_gb" -ge 12 ] && echo 2 || echo 4 ;;
        *"4060"*|*"3060"*|*"2060"*|*"GTX 16"*|*"1660"*|*"1650"*)
            [ "$vram_gb" -ge 10 ] && echo 4 || echo 5 ;;
        *"GTX 10"*|*"1080"*|*"1070"*|*"1060"*|*"Tesla"*)
            echo 5 ;;
        *)
            echo 4 ;;  # safe default
    esac
}

# Maps GPU name → attention mode (sage2 / sage / sdpa)
_map_attention() {
    local name="$1"
    case "$name" in
        *"RTX 50"*|*"RTX 40"*|*"RTX 30"*|\
        *"A100"*|*"A800"*|*"H100"*|*"H800"*)
            echo sage2 ;;
        *"RTX 20"*)
            echo sage ;;
        *)
            echo sdpa ;;
    esac
}

# ─────────────────────────────────────────────────────────────────────────────
# Runtime GPU detection
# ─────────────────────────────────────────────────────────────────────────────

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[WARN] nvidia-smi not found — GPU passthrough may be missing." >&2
    PROFILE=${WGP_PROFILE:-4}
    ATTN=${WGP_ATTENTION:-sdpa}
    GPU_NAME="unknown"
    VRAM_GB=0
else
    GPU_NAME=$(_detect_gpu_name)
    VRAM_GB=$(_detect_vram_gb)
    PROFILE=${WGP_PROFILE:-$(_map_profile "$GPU_NAME" "$VRAM_GB")}
    ATTN=${WGP_ATTENTION:-$(_map_attention "$GPU_NAME")}
fi

echo "[INFO] GPU: ${GPU_NAME} | VRAM: ${VRAM_GB}GB | Profile: ${PROFILE} | Attention: ${ATTN}"

# ─────────────────────────────────────────────────────────────────────────────
# Launch
# ─────────────────────────────────────────────────────────────────────────────

exec su -p user -c "cd /workspace/wan2gp && \
    python3 wgp.py \
        --listen \
        --profile ${PROFILE} \
        --attention ${ATTN} \
        ${WGP_ARGS:-} \
        $*"

# EOF entrypoint.sh
