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
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin:${PATH}"

# PyTorch first — prevents requirements.txt from pulling a generic CPU build.
# If upgrading CUDA, change both the index URL and the FROM image above.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
    torch==2.10.0+cu128 \
    torchvision==0.25.0+cu128 \
    torchaudio==2.10.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# ── SageAttention 2++ two-pass NVCC build ────────────────────────────────────
# NVCC is a cross-compiler — no physical GPU is required on the build host.
# TORCH_CUDA_ARCH_LIST bypasses SageAttention's runtime GPU-detection probe.
#
# Why two passes?
#   Single-pass compilation of sm_80–sm_90 can produce PTX valid only for the
#   highest SM, causing assembler failures for lower targets. Separate passes
#   eliminate this: sm_80/86/89 in pass 1, sm_90+PTX in pass 2.
#   The +PTX suffix embeds forward-compat PTX so Blackwell (sm_120) can
#   JIT-recompile it at first load via the CUDA driver.
#
# Why pip wheel instead of setup.py build_ext + bdist_wheel --skip-build?
#   setup.py install --skip-build and bdist_wheel --skip-build both silently
#   omit the compiled C extension .so files, causing 'cannot import sageattn
#   (unknown location)' at runtime. pip wheel builds a guaranteed binary wheel
#   with .so files included, and pip install from it is a complete installation.
#   Single-pass 8.0;8.6;8.9;9.0+PTX is safe here — SageAttention v2.2.0 uses
#   SM-specific extension modules that each target only their own SM
#   (_qattn_sm80, _qattn_sm89, _qattn_sm90), so no cross-arch PTX conflict.
#
# We intentionally select MAX_JOBS=1 for free CI runners (~2 vCPU / 7 GB RAM).
# On a machine with more resources you can increase this to speed up cold builds.
#
# Pinned to v2.2.0 (eb615cf6) for reproducibility. Bump tag via single-line PR.
# /tmp/sa_dist/*.whl is NOT deleted — the deps stage installs it, and the
# sage-wheels.yml workflow extracts it via a cache-hit build of this stage.
ENV FORCE_CUDA="1"
RUN pip install wheel packaging && \
    git clone --branch v2.2.0 --depth 1 \
    https://github.com/thu-ml/SageAttention.git /tmp/SageAttention && \
    cd /tmp/SageAttention && \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX" MAX_JOBS=1 \
    pip wheel --no-deps --no-build-isolation . -w /tmp/sa_dist/ && \
    pip install --no-deps /tmp/sa_dist/*.whl && \
    rm -rf /tmp/SageAttention

# ─────────────────────────────────────────────────────────────────────────────
# Stage: deps
# Adds project requirements on top of the sage-compile base.
# The sage-compile stage is a cache hit; only this stage rebuilds when
# requirements.txt changes.
# ─────────────────────────────────────────────────────────────────────────────
FROM sage-compile AS deps

# Install SageAttention from the wheel built in sage-compile (no recompile).
RUN pip install --no-deps /tmp/SageAttention/dist/*.whl && \
    rm -rf /tmp/SageAttention

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
