<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved. -->

# ROSCon 2026: Fine-tuning a Robot Policy (MolmoAct2 + LIBERO)

Fine-tune a real-robot vision-language-action model (**MolmoAct2**) on a new skill in the **LIBERO**
simulator with a short **LoRA** run, then drive the fine-tuned policy live in the simulator. It runs
in the browser on a single AMD **Strix Halo** machine as a JupyterHub course image.

The course is reproducible from public sources. You do **not** need
`mm2_workshop_assets.zip`: the notebooks download pinned Hugging Face revisions on first use and
reuse the persistent cache afterwards.

## Requirements

- A supported AMD ROCm GPU. The workshop target is Strix Halo (`gfx1151`).
- Docker and enough space for the image, public inputs, and generated checkpoints. Budget at least
  **150 GB free** to run all three notebooks.
- A reliable internet connection for the first run. `HF_TOKEN` is optional but recommended to
  avoid anonymous Hub rate limits.
- Persistent storage for `/home/jovyan`; otherwise every fresh container downloads the inputs again.

## Build

From the repository root:

```bash
make -C dockerfiles finetuning GPU_TARGET=gfx1151
```

This produces `ghcr.io/amdresearch/auplc-finetuning:latest` and
`ghcr.io/amdresearch/auplc-finetuning:latest-gfx1151`. The image contains the pinned software
stacks, notebooks, simulator, and helper scripts; large model and dataset files stay outside the
image in the user's persistent cache.

## Deploy

Deploy JupyterHub with the image:

```bash
sudo ./auplc-installer install --gpu=strix-halo
```

Set a token before starting the notebooks if you have one:

```bash
export HF_TOKEN=hf_...
```

Run the notebooks in order:

1. `0_overview.ipynb`
2. `1_finetune_molmoact2_libero.ipynb`
3. `2_interactive_sim_molmoact2_libero.ipynb`
4. `3_inference_fastwam_libero.ipynb`

Notebook 1 downloads the public MolmoAct2 DROID base and full LIBERO training dataset, then
evaluates the checkpoint it creates. Its DROID sanity check fetches only the selected episodes,
not the entire DROID dataset. Notebook 2 uses the newest local training output by default; set
`USE_PUBLIC_CHECKPOINT = True` in its setup cell to download AllenAI's fully trained public LIBERO
checkpoint for comparison. Notebook 3 downloads the public FastWAM checkpoint,
Wan T5/VAE/tokenizer components, and the LIBERO replay archive.

All downloads are idempotent. The code pins immutable Hub commits, so a later change to a
repository's `main` branch does not silently change the workshop.

## Useful overrides

- `STEPS=10000` — turn the short training smoke test into a longer run.
- `N_DROID_EPISODES=1` — reduce the open-loop DROID download and runtime.
- `USE_PUBLIC_CHECKPOINT = True` in notebook 2 — download the fully trained public comparison checkpoint.
- `POLICY_PATH=/path/to/pretrained_model` — use a compatible local LeRobot checkpoint.
- `FASTWAM_DOWNLOAD_DATASET=0` — skip FastWAM replay data when using only its interactive sim.
- `HF_HOME`, `CHECKPOINTS_DIR`, `FASTWAM_CACHE` — relocate persistent large files.

## Optional offline-event image

The old private bundle path remains available only for organizers who already have it:

```bash
make -C dockerfiles finetuning GPU_TARGET=gfx1151 \
  ASSETS_ZIP=/path/to/mm2_workshop_assets.zip
```

It is not required for home reproduction and is no longer selected automatically.
