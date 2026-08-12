#!/usr/bin/env bash
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set -euo pipefail

cp -r ../../../projects/Finetuning ./course_data
trap 'rm -rf course_data' EXIT

# Optionally BAKE the pre-staged workshop assets into the image so it is fully self-contained
# and can be distributed to every node without a shared mount. Set ASSETS_SRC to the prestaged
# tree (the output of prestage_shared.sh: hf_hub/ + checkpoints/reference/pretrained_model/).
# The assets are passed as a BuildKit named context so they never enter the git build context.
BUILD_EXTRA=()
if [ -n "${ASSETS_SRC:-}" ] && [ -d "${ASSETS_SRC}" ]; then
  echo "baking workshop assets into the image from: ${ASSETS_SRC}"
  BUILD_EXTRA+=(--build-context "assets=${ASSETS_SRC}" --build-arg "FINAL=with-assets")
else
  echo "ASSETS_SRC unset or missing -> building code-only image (assets NOT baked in)"
fi

DOCKER_BUILDKIT=1 docker build ${BASE_IMAGE:+--build-arg BASE_IMAGE="$BASE_IMAGE"} \
  ${BUILD_EXTRA[@]+"${BUILD_EXTRA[@]}"} \
  -t ghcr.io/amdresearch/auplc-finetuning:latest .
