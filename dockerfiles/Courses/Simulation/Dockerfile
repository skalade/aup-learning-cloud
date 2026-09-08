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
RUN mkdir -p /opt/workspace/Simulation
COPY ./course_data /opt/workspace/Simulation

RUN chown -R jovyan:1000 /opt/workspace

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
WORKDIR /opt/workspace/Simulation
