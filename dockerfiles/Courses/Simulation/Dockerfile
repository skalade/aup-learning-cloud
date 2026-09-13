# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Use a verified ROCm PyTorch image as base
ARG BASE_IMAGE=ghcr.io/amdresearch/auplc-base:latest
FROM ${BASE_IMAGE}

# Copy related rocm notebooks into the docker
USER root
RUN mkdir -p /ryzers/notebooks
COPY ./course_data /ryzers/notebooks

RUN chown -R jovyan:100 /ryzers

# ---------------------------------------------------------------------------
# ROSCon Humanoid Stairs Workshop (Genesis / Unitree G1) stack.
#
# The gslab wheel needs torch>=2.10 + genesis-world 1.4.0. auplc-base ships
# torch 2.9.1 (too old for gslab), so the workshop gets an isolated venv (no
# --system-site-packages) with its own newer torch plus a dedicated Jupyter
# kernel, mirroring the isolated-venv + kernelspec pattern used by the
# LocalInference course.
#
# The workshop's upstream INSTALL.md pins torch 2.12.0+rocm7.2, but the ROCm
# kernel driver (KFD) is host-global and the AUP fleet runs ROCm 7.13: a
# rocm7.2 torch userspace segfaults on first GPU dispatch against a 7.13 driver.
# So the venv torch MUST match the host ROCm. We install torch 2.10.0+rocm7.13.0
# from the same gfx1151 wheel index auplc-base uses (repo.amd.com), which both
# satisfies gslab's >=2.10 floor and matches the host driver.
# ---------------------------------------------------------------------------
ENV GSLAB_VENV=/opt/gslab-venv

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Install torch from the host-matched ROCm index FIRST (a pinned --index-url, not
# --extra-index-url, so a missing +rocm7.13.0 wheel fails loudly instead of
# silently pulling a PyPI/CUDA build). torch pulls its matching triton
# (triton==3.6.0+rocm7.13.0) from that same index.
#
# Then install gslab's real deps from PyPI and the wheel itself with --no-deps:
# gslab's metadata declares a "triton-rocm" requirement that only exists on the
# rocm7.2 pytorch index (wrong ROCm for this host and unavailable on PyPI). The
# `triton` module it actually needs is already satisfied by torch's triton above,
# so we install gslab's genuine deps explicitly and skip its phantom triton-rocm
# pin. Delete the 20 MB wheel in the SAME layer so it is not stranded.
RUN /usr/bin/python3 -m venv ${GSLAB_VENV} \
    && ${GSLAB_VENV}/bin/pip install --no-cache-dir --upgrade pip \
    && ${GSLAB_VENV}/bin/pip install --no-cache-dir \
         --index-url https://repo.amd.com/rocm/whl/gfx1151/ \
         "torch==2.10.0+rocm7.13.0" \
    && ${GSLAB_VENV}/bin/python -c "import torch; assert torch.version.hip, \
         'gslab venv torch is not a ROCm/HIP build'; \
         print('gslab torch', torch.__version__, 'hip', torch.version.hip)" \
    && ${GSLAB_VENV}/bin/pip install --no-cache-dir \
         "genesis-world==1.4.0" "rsl-rl-lib>=5.4.0" "tyro>=1.0.13" "wandb>=0.27.2" \
         jupyterlab ipykernel ipywidgets \
    && ${GSLAB_VENV}/bin/pip install --no-cache-dir --no-deps \
         /ryzers/notebooks/wheels/gslab-0.1.0-py3-none-any.whl \
    && rm -f /ryzers/notebooks/wheels/gslab-0.1.0-py3-none-any.whl

# Genesis pulls in system OpenGL at import time: pygel3d dlopens libGL.so.1, and
# pyglet's default backend dlopens the X11/libXrender stack. This is a headless
# JupyterHub server with no X display, so instead of dragging in all of X11 we run
# pyglet in headless mode (PYGLET_HEADLESS, below), which uses the EGL backend and
# skips the xlib import — needing only libGL + libEGL. Kept in its own layer AFTER
# the torch install so editing it does not invalidate the cached ROCm-torch layer.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libegl1 \
      libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Route pyglet (imported transitively by Genesis) to its headless EGL backend so
# `import genesis` and offscreen camera rendering work without an X server. Safe
# to set globally: nothing in the base image uses pyglet.
ENV PYGLET_HEADLESS=1

# Register the gslab-roscon kernelspec. The venv torch is rocm7.13, matching the
# host's /opt/rocm/lib (same as auplc-base), so the kernel inherits the container
# LD_LIBRARY_PATH rather than scrubbing it. Then a build-time smoke test (import +
# HIP-build only, so it passes on a GPU-less builder; the notebook env-check is
# the real runtime GPU gate).
RUN mkdir -p /usr/local/share/jupyter/kernels/gslab-roscon \
    && printf '%s\n' \
      '{' \
      '  "argv": ["/opt/gslab-venv/bin/python", "-m", "ipykernel_launcher", "-f", "{connection_file}"],' \
      '  "display_name": "gslab ROSCon (ROCm 7.13)",' \
      '  "language": "python"' \
      '}' \
      > /usr/local/share/jupyter/kernels/gslab-roscon/kernel.json \
    && ${GSLAB_VENV}/bin/python -c \
      "import torch, gslab, genesis; assert torch.version.hip; \
print('gslab workshop env OK:', torch.__version__, '| genesis', genesis.__version__)"

# Join the host render group by NUMBER so ROCm (/dev/kfd, /dev/dri/renderD*) is
# reachable for non-root jovyan whether the image runs under a K8s pod or a plain
# `docker run`. The render GID is host-assigned and owns those 0660 device nodes,
# so it cannot be derived inside the image; override HOST_RENDER_GID to match the
# target fleet (default 992 matches the render group observed on the AUP host fleet).
ARG HOST_RENDER_GID=992
RUN getent group "${HOST_RENDER_GID}" >/dev/null \
      || groupadd --gid "${HOST_RENDER_GID}" hostrender; \
    usermod -aG "$(getent group "${HOST_RENDER_GID}" | cut -d: -f1)" jovyan

USER jovyan
WORKDIR /ryzers/notebooks
