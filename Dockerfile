# syntax=docker/dockerfile:1.4
# Multi-stage Dockerfile for RunPod / cloud deployment.
#
# Stage 1 (deps):  apt + uv + PyTorch + pip requirements + SageAttention compile.
#                  Cache-stable across source-code-only changes.
# Stage 2 (runtime): copies project source into image (required for RunPod;
#                    no host bind mount available). Persistent data (models,
#                    outputs, caches) routes to /workspace/ which maps to the
#                    RunPod pod volume.
#
# Build args:
#   CUDA_ARCHITECTURES  Semicolon-separated SM targets for SageAttention.
#                       Default covers Ampere, Ada, Hopper, and Blackwell
#                       (via forward-compat PTX JIT on sm_120).
#
# Local build (Podman):
#   podman build --build-arg CUDA_ARCHITECTURES="8.0;8.6;8.9;9.0+PTX" \
#     -t wan2gp:local .

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — deps
# Everything expensive lives here and is layer-cached between builds.
# Invalidated only when apt packages, PyTorch version, or SageAttention tag change.
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 AS deps

# Broad SM target set:
#   8.0  — A100/A800 (Ampere data center)
#   8.6  — RTX 3060-3090 (Ampere consumer)
#   8.9  — RTX 4060-4090 (Ada Lovelace)
#   9.0+PTX — H100/H800 (Hopper) + forward-compat PTX for Blackwell JIT (sm_120)
ARG CUDA_ARCHITECTURES="8.0;8.6;8.9;9.0+PTX"
ENV DEBIAN_FRONTEND=noninteractive

# System packages — single layer, cleanup in same RUN to avoid bloating intermediate layers
# python3-dev provides Python.h, required for packages with C/Cython extensions (e.g. insightface)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip git wget curl cmake ninja-build \
    libgl1 libglib2.0-0 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Install uv (fast pip resolver/installer)
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin:${PATH}"

# PyTorch — installed first so requirements.txt doesn't pull a generic CPU build.
# CUDA 12.8 matches the base image; change both together if upgrading.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
    torch==2.10.0+cu128 \
    torchvision==0.25.0+cu128 \
    torchaudio==2.10.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# Project requirements
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
    --index-strategy unsafe-best-match \
    -r /tmp/requirements.txt

# ── SageAttention 2++ ────────────────────────────────────────────────────────
# Two-pass NVCC strategy (ported from comfy_con).
# NVCC is a cross-compiler — no physical GPU is required on the build host.
# TORCH_CUDA_ARCH_LIST bypasses SageAttention's runtime gpu-detection probe.
#
# Why two passes instead of one?
#   Compiling sm_80/86/89 and sm_90 in a single NVCC invocation can produce
#   PTX that is only valid for the highest SM, causing assembler failures for
#   lower SM targets at link time. Separating the passes eliminates this.
#
# Pass 1: Ampere (sm_80, sm_86) + Ada (sm_89)
# Pass 2: Hopper (sm_90) + forward-compat PTX — Blackwell (sm_120) JIT-recompiles
#         this PTX via the CUDA driver on first load.
#
# Pinned to v2.2.0 (eb615cf6) for reproducibility. Bump tag via single-line PR.
ENV FORCE_CUDA="1"
RUN --mount=type=cache,target=/tmp/sa_cache \
    git clone --branch v2.2.0 --depth 1 \
    https://github.com/thu-ml/SageAttention.git /tmp/SageAttention && \
    cd /tmp/SageAttention && \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9" MAX_JOBS=4 python3 setup.py build_ext && \
    TORCH_CUDA_ARCH_LIST="9.0+PTX"     MAX_JOBS=4 python3 setup.py build_ext && \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"             python3 setup.py install && \
    rm -rf /tmp/SageAttention

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — runtime
# Thin application layer. Invalidated on every source change; deps stage is not.
# ─────────────────────────────────────────────────────────────────────────────
FROM deps AS runtime

# Non-root user — uid 1000 is the RunPod convention
RUN useradd -u 1000 -ms /bin/bash user

# Project source baked into the image (no host bind mount for cloud/RunPod deployments)
WORKDIR /workspace/wan2gp
COPY --chown=user:user . .

# Persistent data directories live under /workspace/ which maps to the RunPod
# pod volume — survives container restarts, not image rebuilds.
#   /workspace/models            — downloaded model weights
#   /workspace/outputs           — generated video/image output
#   /workspace/.cache/huggingface — HF hub cache (large; keep persistent)
#   /workspace/.cache/triton     — compiled Triton kernels (expensive; keep persistent)
RUN mkdir -p \
    /workspace/models \
    /workspace/outputs \
    /workspace/.cache/huggingface \
    /workspace/.cache/triton && \
    chown -R user:user /workspace

COPY --chown=user:user entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 7860
ENTRYPOINT ["/entrypoint.sh"]
