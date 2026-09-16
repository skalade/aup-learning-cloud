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
# torch 2.9.1 (too old for gslab), so the workshop gets an isolated venv with its
# own newer torch plus a dedicated Jupyter kernel, mirroring the isolated-venv +
# kernelspec pattern used by the LocalInference course. Built with uv (already in
# the base image) on Python 3.13, exactly as the upstream INSTALL.md prescribes.
#
# Two ROCm layers are in play and both must be 7.13 to match the host KFD driver
# (which is host-global and cannot be isolated per container):
#   * PyTorch finds the HIP runtime through the pip `rocm_sdk` packages that the
#     torch==2.11.0+rocm7.13.0 wheel pulls from repo.amd.com/gfx1151.
#   * Genesis compiles its physics kernels at runtime with `quadrants`, which
#     needs a classic /opt/rocm layout the pip SDK does not ship: an unversioned
#     libamdhip64.so alias (for bare-name dlopen) and an ld.lld of LLVM >=19 (its
#     AMDGPU code objects are v6; lld-18 rejects them). link_rocm_tree.sh builds
#     that <venv>/rocm tree from the pip SDK and links a good ld.lld (preferring
#     /opt/rocm/llvm/bin/ld.lld from the base image's system ROCm 7.13).
#
# (The previous gslab-0.1.0 bundle was validated for ROCm 7.2; on this 7.13 host
# quadrants segfaulted JIT-linking kernels. gslab-0.1.2 + the pip ROCm SDK +
# link_rocm_tree.sh is the upstream rocm7.13 fix.)
# ---------------------------------------------------------------------------
ENV GSLAB_VENV=/opt/gslab-venv

# Build the workshop venv per INSTALL.md: torch FIRST from the host-matched ROCm
# index (a pinned --index-url so a missing +rocm7.13.0 wheel fails loudly instead
# of silently pulling a PyPI/CUDA build; it also pulls the pip rocm_sdk packages),
# then gslab from the bundled wheel plus jupyter, resolving gslab's deps across
# PyPI and the AMD index (unsafe-best-match). Delete the 20 MB wheel in the SAME
# layer so it is not stranded.
RUN uv venv --python 3.13 ${GSLAB_VENV} \
    && VIRTUAL_ENV=${GSLAB_VENV} uv pip install --no-cache \
         --index-url https://repo.amd.com/rocm/whl/gfx1151/ \
         "torch==2.11.0+rocm7.13.0" \
    && ${GSLAB_VENV}/bin/python -c "import torch; assert torch.version.hip, \
         'gslab venv torch is not a ROCm/HIP build'; \
         print('gslab torch', torch.__version__, 'hip', torch.version.hip)" \
    && VIRTUAL_ENV=${GSLAB_VENV} uv pip install --no-cache \
         --index-url https://pypi.org/simple \
         --extra-index-url https://repo.amd.com/rocm/whl/gfx1151/ \
         --index-strategy unsafe-best-match \
         "gslab @ file:///ryzers/notebooks/wheels/gslab-0.1.2-py3-none-any.whl" \
         jupyterlab ipykernel ipywidgets plotly tqdm \
    && rm -f /ryzers/notebooks/wheels/gslab-0.1.2-py3-none-any.whl

# plotly 5's labextension targets JupyterLab 3 and the 4.x server rejects it, so
# FigureWidget never renders. plotly 6+ uses anywidget instead, and anywidget's
# frontend has to be visible to the system JupyterLab, not just the kernel venv.
# --no-deps bypasses PyGEL3D's plotly<6 pin; PyGEL3D only imports plotly in
# pygel3d/jupyter_display.py, which neither Genesis nor gslab references.
RUN VIRTUAL_ENV=${GSLAB_VENV} uv pip install --no-cache anywidget narwhals \
    && VIRTUAL_ENV=${GSLAB_VENV} uv pip install --no-cache --no-deps --upgrade "plotly==7.1.0" \
    && /usr/bin/python3 -m pip install --break-system-packages anywidget

# uv places the managed CPython 3.13 under /root/.local/share/uv/python and points
# ${GSLAB_VENV}/bin/python at it by symlink. /root is mode 700, so non-root jovyan
# cannot traverse it to exec the interpreter and the gslab-roscon kernel dies with
# "[Errno 13] Permission denied: '/opt/gslab-venv/bin/python'". Grant traversal-only
# (o+x, not o+r) on /root so jovyan can reach the known interpreter path without being
# able to list /root's contents. Everything below /root is already 0755.
RUN chmod o+x /root

# Assemble the <venv>/rocm tree Genesis' quadrants compiler needs (unversioned
# .so aliases + an ld.lld of LLVM >=19). The script self-verifies that ld.lld
# runs and libamdhip64.so loads by bare name, so this layer fails loudly if the
# base image lacks a new-enough linker.
RUN VENV=${GSLAB_VENV} bash /ryzers/notebooks/link_rocm_tree.sh

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

# Register the gslab-roscon kernelspec. quadrants (Genesis' compiler) runs
# ${ROCM_PATH}/llvm/bin/ld.lld and dlopens libamdhip64.so by bare name, so the
# kernel bakes in ROCM_PATH + LD_LIBRARY_PATH pointing at the <venv>/rocm tree
# built above (the two venv-rocm dirs only, per INSTALL.md — torch finds HIP via
# the rocm_sdk package, not this path). QD_OFFLINE_CACHE_MAX_SIZE_OF_FILES raises
# the compiled-kernel cache limit so kernels persist between runs. Then a
# build-time smoke test (import + HIP-build only, so it passes on a GPU-less
# builder; the notebook env-check is the real runtime GPU gate).
RUN mkdir -p /usr/local/share/jupyter/kernels/gslab-roscon \
    && printf '%s\n' \
      '{' \
      '  "argv": ["/opt/gslab-venv/bin/python", "-m", "ipykernel_launcher", "-f", "{connection_file}"],' \
      '  "display_name": "gslab ROSCon (ROCm 7.13)",' \
      '  "language": "python",' \
      '  "env": {' \
      '    "ROCM_PATH": "/opt/gslab-venv/rocm",' \
      '    "LD_LIBRARY_PATH": "/opt/gslab-venv/rocm/lib:/opt/gslab-venv/rocm/lib-gfx1151",' \
      '    "QD_OFFLINE_CACHE_MAX_SIZE_OF_FILES": "2147483647",' \
      '    "PYTHONDONTWRITEBYTECODE": "1"' \
      '  }' \
      '}' \
      > /usr/local/share/jupyter/kernels/gslab-roscon/kernel.json \
    && ROCM_PATH=${GSLAB_VENV}/rocm \
       LD_LIBRARY_PATH=${GSLAB_VENV}/rocm/lib:${GSLAB_VENV}/rocm/lib-gfx1151 \
       ${GSLAB_VENV}/bin/python -c \
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
