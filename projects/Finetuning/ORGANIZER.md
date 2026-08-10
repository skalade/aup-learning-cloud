<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved. -->

# Fine-tuning course - organizer & developer guide

Everything needed to **stand up** the MolmoAct2 fine-tuning workshop and to **reproduce it
offline** on your own Strix Halo box before it ships to the AUP Learning Cloud cluster.

> **Workshop attendee?** You do not need this file. Your environment is already built and the
> assets are staged for you - follow **[README.md](README.md)** (no `sudo`, no building).

Two audiences, cleanly separated:

- **Organizer** - has `sudo`/root on the deploy host. Builds the course image, deploys
  JupyterHub, hosts the 65 GB assets, and wires them into user pods. Parts **A** and **B**.
- **Developer** - reproduces the whole thing offline on one Strix Halo to verify a change before
  merge. Part **C** (uses A + B locally).

The attendee experience these steps produce: pick the course on the spawn page, run one no-`sudo`
copy command, **Run All** on two notebooks - with **zero Hub downloads at the venue**.

---

## Part A - Build & deploy the course (organizer, needs `sudo`)

### A0. Prerequisites (once per box)

Follow the repo top-level `README.md` "Quick Start": OEM kernel for Ryzen AI APUs, Docker with
non-root access, `build-essential`, and the installer TUI deps. Budget **~150 GB free** (image +
65 GB bundle + the cache it expands into) on **Ubuntu 24.04**.

### A1. Clone the repo and check out the branch

```bash
git clone <your-fork>/aup-learning-cloud.git
cd aup-learning-cloud
git checkout finetuning-interactive-sim
```

### A2. Build the Finetuning course image

The notebooks, scripts and tests are baked into the image at build time (`projects/Finetuning/` ->
`/ryzers/notebooks`), so build from this branch. Layers on top of `ghcr.io/amdresearch/auplc-base`
(pulled automatically):

```bash
make -C dockerfiles finetuning GPU_TARGET=gfx1151
# -> ghcr.io/amdresearch/auplc-finetuning:latest
```

Rebuild the **single** Finetuning image in place with new layers when notebooks/scripts change -
do not fork a parallel image. The heavy dependency layers cache, so doc/script/notebook edits
rebuild in under a minute.

### A3. Deploy JupyterHub (k3s single node)

```bash
sudo ./auplc-installer install --gpu=strix-halo      # or: ./auplc-installer  (interactive TUI)
```

Wait for pods to be Ready (`kubectl get pods -A`). The dev deploy auto-logs-in as user `student`
and uses local images (no registry pull). The course is already registered in
`runtime/values.yaml` under the **ROSCON 2026** group as **"Fine-tuning on GPUs"**
(`Course-Finetuning` -> `auplc-finetuning`, strix-halo GPU, landing dir `/ryzers/notebooks`).

> Storage: single-node uses k3s `local-path`; each user gets a writable home PVC over
> `/home/jovyan`. Its 10Gi request is **not enforced**, so the ~64 GB of assets fit. Anything
> baked into the image under `/home/jovyan` is hidden by that PVC - assets must be delivered at
> runtime (Part B), not baked.

---

## Part B - Host & deliver the 65 GB assets (organizer)

The three large inputs (base checkpoint, LIBERO dataset, fine-tuned checkpoint) plus two small
helpers ship as a split-tar **bundle** (`mm2_workshop_assets/`, ~65 GB) that lives on OneDrive.
Get it onto persistent storage the cluster can reach, e.g.:

```
mm2_workshop_assets/
  base/           -> HF cache: models--allenai--MolmoAct2-DROID          (base checkpoint, ~21 GB)
  libero/         -> HF cache: datasets--allenai--MolmoAct2-LIBERO-Dataset (dataset, ~33 GB)
  tokenizer/      -> HF cache: models--allenai--MolmoAct2-FAST-Tokenizer   (required by the policy)
  droid_dataset/  -> HF cache: datasets--allenai--MolmoAct2-DROID-Dataset  (Step-3 open-loop, ~1.4 GB)
  ft_checkpoint/  -> our fine-tuned checkpoint  (pretrained_model, ~11.5 GB)
  unpack_bundle.sh  README.txt
```

Assets are **staged (copied), never re-hosted** in this repo - copy the folder from wherever the
workshop stores it. Pick one delivery path:

### Option 1 (recommended for a real multi-user workshop): mount + self-serve

Mount the asset folder **read-only into every user pod** at a fixed path (via the chart's
`singleuser.storage.extraVolumes` / `extraVolumeMounts`), e.g. `/opt/auplc-assets/mm2_workshop_assets`.
Attendees then run one no-`sudo` command in their pod:

```bash
scripts/fetch_assets.sh          # auto-detects /opt/auplc-assets/...; ASSETS_SRC=<path> to override
```

`fetch_assets.sh` copies from the mount into each user's own `~/.cache/huggingface` and
`~/checkpoints`. It accepts **either** the split-tar bundle **or** an already-unpacked dir
(`hf_hub/` + `checkpoints/`), and is idempotent. If you prefer to mount an unpacked dir, expand
the bundle once on the host first:

```bash
projects/Finetuning/scripts/unpack_bundle.sh /path/to/mm2_workshop_assets /path/to/assets
# mount /path/to/assets read-only; attendees run fetch_assets.sh (or set ASSETS_DIR and Run All)
```

### Option 2 (single-node / a few pods): push straight into the running pod

After the attendee spawns their server, stream the bundle from the host into their pod (needs
`kubectl`, so this is the organizer's job, not the attendee's):

```bash
projects/Finetuning/scripts/prestage_to_pod.sh /path/to/mm2_workshop_assets
# defaults: NAMESPACE=jupyterhub  NB_USER=student  POD=jupyter-student  CONTAINER=notebook
```

Files are written **as the pod user** (correct ownership) into `~/.cache/huggingface/hub` and
`~/checkpoints/reference/pretrained_model`. The attendee then just **Runs All** (nothing to copy).

### Reference checkpoint - staging, not baking (compliance)

We do **not** re-host model weights in the image or repo, and the home PVC hides anything baked
under `/home/jovyan`, so the fine-tuned checkpoint (LeRobot format: `config.json`,
`model.safetensors` ~11.5 GB, processor/normalizer stats, `train_config.json`) is provided at
**runtime** at `~/checkpoints/reference/pretrained_model` (`REFERENCE_POLICY`). It is part of the
bundle above; the delivery options place it. For scale, organizers may instead:

1. **Shared read-only mount** - one copy on a shared PV mounted into every pod (no per-user duplication).
2. **initContainer / pre-puller** - copy it onto each user PVC before the notebook starts.
3. **Hugging Face repo id** - publish the reference LoRA and set `POLICY_PATH`/`REFERENCE_POLICY` to the repo id.

---

## Part C - Developer local reproduction (offline, on one Strix Halo)

Verify the full pipeline end to end **before merge**, with no Hub downloads at run time. You
download the inputs **once** (to build the bundle); the staging path then replays them into the
caches so the attendee flow downloads nothing.

1. **Get the bundle** (`mm2_workshop_assets/`, ~65 GB) from OneDrive onto the box at a persistent
   path, e.g. `~/mm2_asset_bundle`.
2. **Build** the image (Part A2) and, if testing the full Hub path, **deploy** (Part A3).

Then validate one of two ways:

### C1. Fast path - run the notebooks headless in the course container (offline)

No Hub, no k8s - the tightest loop to check a notebook/script change:

```bash
BUNDLE=~/mm2_asset_bundle
docker run --rm -it --network=host --ipc=host --shm-size 16G \
  --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined \
  --group-add video --group-add render \
  -v "$BUNDLE":/opt/auplc-assets/mm2_workshop_assets:ro \
  ghcr.io/amdresearch/auplc-finetuning:latest bash -lc '
    scripts/fetch_assets.sh &&                                   # tar bundle -> HF cache + checkpoint
    export HF_HUB_OFFLINE=1 &&                                   # prove no network is used
    jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=-1 finetune_molmoact2_libero.ipynb &&
    jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=-1 interactive_sim_molmoact2_libero.ipynb'
```

### C2. Full path - deploy the Hub and mimic an attendee

Deploy (A3), spawn the **"Fine-tuning on GPUs"** pod, deliver assets (Part B, Option 1 or 2),
then open the two notebooks and **Run All** exactly as an attendee would.

---

## Notes / gotchas

- **The fine-tuned checkpoint reuses the base weights.** Loading it (eval or interactive) resolves
  `allenai/MolmoAct2-DROID` from the HF cache, so the base **must** be present. The bundle includes
  it and every delivery path places it - never ship only the fine-tuned checkpoint.
- **Fully offline after staging.** Set `HF_HUB_OFFLINE=1` to prove no Hub access is needed at run
  time.
- **LeRobot uses its own dataset root.** The Step-4 fine-tune loads the LIBERO dataset through
  LeRobot, which resolves datasets under `$HF_LEROBOT_HOME/{repo_id}`
  (`~/.cache/huggingface/lerobot/allenai/MolmoAct2-LIBERO-Dataset`) - **not** the standard HF hub
  cache. The staging scripts therefore also symlink that path to the hub-cached snapshot so offline
  training reuses the same blobs instead of triggering a second ~33 GB download. If you ever
  populate the hub cache by hand, recreate this link or offline training will fail with
  `LocalEntryNotFoundError`.
- **Assets are staged, never re-hosted** in this repo; they come from the OneDrive bundle.
- **Verified path (full pipeline, offline).** On a single Strix Halo iGPU (gfx1151), with the bundle
  staged and `HF_HUB_OFFLINE=1`, both notebooks run end to end with **no Hub access**:
  notebook 1 (preflight cache-hit -> open-loop DROID replay -> LoRA fine-tune, checkpoint saved ->
  closed-loop LIBERO eval `pc_success=100%`, `libero_object` task 3) in ~5 min, and notebook 2
  (interactive sim server loads the fine-tuned checkpoint and reaches the `idle` ready state).
