#!/usr/bin/env bash
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set -euo pipefail

cp -r ../../../projects/LocalInference ./course_data
trap 'rm -rf course_data' EXIT

# outputs/ is local experiment scratch and the caches are host-side; neither
# belongs in the image, and outputs/ alone is tens of megabytes.
rm -rf course_data/outputs course_data/.ipynb_checkpoints
find course_data -name __pycache__ -type d -prune -exec rm -rf {} +

docker build ${BASE_IMAGE:+--build-arg BASE_IMAGE="$BASE_IMAGE"} \
  ${HOST_RENDER_GID:+--build-arg HOST_RENDER_GID="$HOST_RENDER_GID"} \
  -t ghcr.io/amdresearch/auplc-localinference:latest .
