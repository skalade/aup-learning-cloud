#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# WORKSHOP USER script -- runs INSIDE your notebook pod, needs NO sudo and NO kubectl.
#
# Copies the workshop's large inputs (base checkpoint + LIBERO dataset + fine-tuned checkpoint)
# from the shared storage your instructor mounted into this pod, straight into the caches the
# notebooks read. After it finishes, both notebooks find everything cached and download nothing.
#
# It accepts EITHER layout the organizer may have provided at the source location:
#   (a) the split-tar bundle  -> base/  libero/  tokenizer/  droid_dataset/  ft_checkpoint/
#   (b) an already-unpacked dir -> hf_hub/<models--...|datasets--...>/  checkpoints/reference/...
#
# Everything lands at the paths the notebooks expect:
#   $HF_HOME/hub                                (models--*, datasets--*)
#   $REFERENCE_POLICY (checkpoints/reference/pretrained_model)   the fine-tuned policy
#
# Usage (from a terminal in the pod, or `!bash scripts/fetch_assets.sh` in a cell):
#   scripts/fetch_assets.sh                 # auto-detects the source (see ASSETS_SRC below)
#   ASSETS_SRC=/path/to/assets scripts/fetch_assets.sh
# It is idempotent: items already present are skipped, so re-running is safe.
set -euo pipefail

HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
HUB="$HF_HOME/hub"
REFERENCE_POLICY="${REFERENCE_POLICY:-$HOME/checkpoints/reference/pretrained_model}"
REF_PARENT="$(dirname "$REFERENCE_POLICY")"

# Resolve the source. Explicit ASSETS_SRC wins; otherwise probe the usual mount points the
# organizer uses for the workshop's shared read-only asset volume.
ASSETS_SRC="${ASSETS_SRC:-${1:-}}"
if [ -z "$ASSETS_SRC" ]; then
  for cand in \
    /opt/auplc-assets/mm2_workshop_assets \
    /opt/auplc-assets/assets \
    /mnt/workshop-assets/mm2_workshop_assets \
    /mnt/workshop-assets \
    "$HOME/mm2_workshop_assets" \
    "$HOME/mm2_asset_bundle"; do
    if [ -d "$cand" ]; then ASSETS_SRC="$cand"; break; fi
  done
fi

if [ -z "$ASSETS_SRC" ] || [ ! -d "$ASSETS_SRC" ]; then
  cat >&2 <<EOF
No workshop assets found.
  Set ASSETS_SRC to where your instructor mounted them, e.g.:
    ASSETS_SRC=/opt/auplc-assets/mm2_workshop_assets scripts/fetch_assets.sh
  (If assets are not staged, the notebooks will download from Hugging Face at run time
   instead -- that needs network access and takes a while.)
EOF
  exit 2
fi

echo "fetching workshop assets"
echo "  source           = $ASSETS_SRC"
echo "  HF_HOME          = $HF_HOME"
echo "  REFERENCE_POLICY = $REFERENCE_POLICY"
mkdir -p "$HUB" "$REF_PARENT"

_have() {  # dir non-empty?
  [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null || true)" ]
}

# ---------------------------------------------------------------- (a) split-tar bundle -------
if ls "$ASSETS_SRC"/base/base.tar.part-* >/dev/null 2>&1; then
  echo "  detected split-tar bundle"
  # Extract each tar straight into the HF cache, but skip any item already present so re-running
  # is cheap (the cache dir names are fixed and known).
  _untar_hub() {  # <expected-cache-dirname> <parts...>
    local name="$1"; shift
    if _have "$HUB/$name"; then echo "  -> $name: already present, skipping"; return 0; fi
    echo "  -> $name (into HF cache)"
    cat "$@" | tar -C "$HUB" -xf -
  }
  _untar_hub models--allenai--MolmoAct2-DROID            "$ASSETS_SRC"/base/base.tar.part-*
  _untar_hub datasets--allenai--MolmoAct2-LIBERO-Dataset "$ASSETS_SRC"/libero/libero.tar.part-*
  # tokenizer is tiny -> always extract (cheap, and easy to miss)
  echo "  -> tokenizer (into HF cache)"
  tar -C "$HUB" -xf "$ASSETS_SRC"/tokenizer/tokenizer.tar
  if ls "$ASSETS_SRC"/droid_dataset/droid_dataset.tar.part-* >/dev/null 2>&1; then
    _untar_hub datasets--allenai--MolmoAct2-DROID-Dataset "$ASSETS_SRC"/droid_dataset/droid_dataset.tar.part-*
  fi
  if _have "$REFERENCE_POLICY"; then
    echo "  -> fine-tuned checkpoint: already present, skipping"
  else
    echo "  -> fine-tuned checkpoint (-> $REF_PARENT)"
    cat "$ASSETS_SRC"/ft_checkpoint/ft.tar.part-* | tar -C "$REF_PARENT" -xf -
  fi

# ------------------------------------------------------------- (b) unpacked assets dir -------
elif [ -d "$ASSETS_SRC/hf_hub" ]; then
  echo "  detected unpacked assets dir (hf_hub/); delegating to stage_assets.sh"
  ASSETS_DIR="$ASSETS_SRC" HF_HOME="$HF_HOME" REFERENCE_POLICY="$REFERENCE_POLICY" \
    "$(dirname "$0")/stage_assets.sh" "$ASSETS_SRC"

else
  echo "ERROR: $ASSETS_SRC has neither base/base.tar.part-* nor hf_hub/ -- unknown layout." >&2
  exit 2
fi

echo "=== verify ==="
du -sh "$HUB"/models--allenai--MolmoAct2-DROID            2>/dev/null || true
du -sh "$HUB"/datasets--allenai--MolmoAct2-LIBERO-Dataset 2>/dev/null || true
du -sh "$HUB"/datasets--allenai--MolmoAct2-DROID-Dataset  2>/dev/null || true
du -sh "$REFERENCE_POLICY"                                2>/dev/null || true
if [ -f "$REFERENCE_POLICY/config.json" ]; then
  echo "fine-tuned checkpoint: config.json present"
else
  echo "WARN: $REFERENCE_POLICY/config.json missing"
fi
echo "done -- open the notebooks and Run All; downloads are skipped (cache detected)."
