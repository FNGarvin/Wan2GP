# syntax=docker/dockerfile:1.4
# Multi-stage Dockerfile for cloud/RunPod deployment.
#
# Stages:
#   sage-compile — apt + uv + PyTorch + SageAttention two-pass NVCC build.
#                  Cache-stable; only rebuilds when these deps change.
#                  Preserves dist/*.whl for use by the deps stage and
#                  for wheel extraction by the sage-wheels.yml workflow.
#   deps         — installs SageAttention wheel + project requirements
#                  on top of sage-compile. Rebuilds on requirements.txt changes.
#   runtime      — source copy + SSH + filebrowser + entrypoint.
#
# Build args (set at build time):
#   CUDA_ARCHITECTURES   Semicolon-separated SM targets. Default covers
#                        Ampere, Ada, Hopper, and Blackwell via PTX.
#   FILEBROWSER_VERSION  filebrowser release tag to install.
#
# Runtime env vars (set on pod, no image rebuild needed):
#   SSH_PUBLIC_KEY       Ed25519/RSA public key injected into /root/.ssh/authorized_keys
#   SSH_PORT             Port for sshd. sshd is skipped if unset.
#   FILEBROWSER_PORT     Port for filebrowser. Skipped if unset.
#   WGP_PROFILE          Override auto-detected WanGP profile (1-5)
#   WGP_ATTENTION        Override auto-detected attention mode (sage2/sage/sdpa)
#   WGP_ARGS             Extra arguments passed to wgp.py

ARG CUDA_ARCHITECTURES="8.0;8.6;8.9;9.0+PTX"
ARG FILEBROWSER_VERSION="2.32.0"

# ─────────────────────────────────────────────────────────────────────────────
# Stage: sage-compile
# Stable cache layer. Only rebuilds when the CUDA base image, apt packages,
# PyTorch version, or SageAttention tag/build script changes.
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 AS sage-compile

ARG CUDA_ARCHITECTURES
ENV DEBIAN_FRONTEND=noninteractive

# python3-dev provides Python.h for C/Cython extension builds (e.g. insightface).
# openssh-server is baked here so the apt step is one operation.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip \
    git wget curl cmake ninja-build \
    libgl1 libglib2.0-0 ffmpeg \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Configure sshd: root login via key only, no password auth.
# Host keys are generated at build time so sshd can start without extra setup.
RUN ssh-keygen -A && \
    sed -i \
    -e 's/#\?\(PermitRootLogin\).*/\1 yes/' \
    -e 's/#\?\(PubkeyAuthentication\).*/\1 yes/' \
    -e 's/#\?\(PasswordAuthentication\).*/\1 no/' \
    /etc/ssh/sshd_config && \
    mkdir -p /run/sshd

# uv — fast Python package installer and resolver
# Pinned version prevents non-deterministic 'ADD <URL>' cache busts.
RUN pip install --no-cache-dir uv==0.6.2
ENV PATH="/root/.local/bin:${PATH}"

# PyTorch first — prevents requirements.txt from pulling a generic CPU build.
# If upgrading CUDA, change both the index URL and the FROM image above.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
    torch==2.10.0+cu128 \
    torchvision==0.25.0+cu128 \
    torchaudio==2.10.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# ── SageAttention 2++ ────────────────────────────────────────────────────────
# Distributing a pre-built wheel avoids the 30-minute NVCC recompile tax.
# This wheel targets sm_80, sm_86, sm_89, sm_90, and includes +PTX for sm_120.
RUN pip install --no-cache-dir \
    "https://github.com/${REPO_OWNER}/releases/download/sage-${SAGE_VERSION}-cu128-cp310/sageattention-2.2.0-cp310-cp310-linux_x86_64.whl"

# ─────────────────────────────────────────────────────────────────────────────
# Stage: deps
# Adds project requirements on top of the sage-compile base.
# The sage-compile stage is a cache hit; only this stage rebuilds when
# requirements.txt changes.
# ─────────────────────────────────────────────────────────────────────────────
FROM sage-compile AS deps

# SageAttention is already installed in sage-compile's site-packages.
# /tmp/sa_dist/*.whl is preserved for sage-wheels.yml extraction.

# Project requirements.
# --index-strategy unsafe-best-match: uv stops at the first index that has a
# package name; this flag enables cross-index version resolution matching pip's
# default behaviour. Needed because the aiinfra nightly index appears first but
# does not have all versions (e.g. onnxruntime-gpu==1.22.0 is only on PyPI).
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
    --index-strategy unsafe-best-match \
    -r /tmp/requirements.txt

# ─────────────────────────────────────────────────────────────────────────────
# Stage: runtime
# Source copy + optional services + entrypoint.
# Rebuilds on every source-code push; deps stage is a cache hit.
# ─────────────────────────────────────────────────────────────────────────────
FROM deps AS runtime

ARG FILEBROWSER_VERSION

# Filebrowser — lightweight browser-based file manager (optional, env-gated)
RUN curl -fsSL \
    "https://github.com/filebrowser/filebrowser/releases/download/v${FILEBROWSER_VERSION}/linux-amd64-filebrowser.tar.gz" \
    -o /tmp/fb.tar.gz && \
    tar -xzf /tmp/fb.tar.gz -C /usr/local/bin/ filebrowser && \
    rm /tmp/fb.tar.gz

# Project source baked into image (no host bind mount for cloud deployments).
# /workspace/ maps to the pod network volume and persists across restarts.
WORKDIR /workspace/wan2gp
COPY . .

RUN mkdir -p \
    /workspace/models \
    /workspace/outputs \
    /workspace/.cache/huggingface \
    /workspace/.cache/triton

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 7860 22
ENTRYPOINT ["/entrypoint.sh"]
