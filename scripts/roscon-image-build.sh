#!/usr/bin/env bash
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
set -euo pipefail

# Build the four ROSCon course images via the auplc-installer:
#   local-inference, simulation, rl-learning, finetuning
#
# Wraps `./auplc-installer img build <target>`, which builds each image through
# dockerfiles/Makefile (GPU detection, image tagging, and save-image export are
# all handled by the installer). Builds continue past a failing image; a
# pass/fail summary is printed at the end and the script exits non-zero if any
# image failed.
#
# Extra arguments are forwarded to auplc-installer, e.g.:
#   scripts/roscon-image-build.sh --gpu=strix-halo
#
# Finetuning bakes in projects/Finetuning/mm2_workshop_assets.zip when present,
# otherwise builds a code-only image.

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
INSTALLER="$REPO_ROOT/auplc-installer"

TARGETS=(local-inference simulation rl-learning finetuning)

declare -a BUILT=() FAILED=()

for target in "${TARGETS[@]}"; do
  echo "=============================================================="
  echo ">> Building ${target}"
  echo "=============================================================="
  if "$INSTALLER" img build "$target" "$@"; then
    BUILT+=("$target")
  else
    echo "!! Build failed: ${target}" >&2
    FAILED+=("$target")
  fi
done

echo
echo "=============================================================="
echo "ROSCon image build summary"
echo "  built:  ${BUILT[*]:-<none>}"
echo "  failed: ${FAILED[*]:-<none>}"
echo "=============================================================="

if ((${#FAILED[@]})); then
  exit 1
fi
