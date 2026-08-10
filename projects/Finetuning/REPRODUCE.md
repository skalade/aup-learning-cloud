<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved. -->

# Reproduce the MolmoAct2 fine-tuning workshop from scratch (offline)

End-to-end steps to bring up **both** notebooks on a fresh Strix Halo box using the pre-downloaded
asset bundle - no Hugging Face downloads at run time.

- `finetune_molmoact2_libero.ipynb` - LoRA fine-tune + open-loop + closed-loop LIBERO eval
- `interactive_sim_molmoact2_libero.ipynb` - drive the fine-tuned policy live in the LIBERO sim

The three large inputs (base checkpoint, LIBERO dataset, fine-tuned checkpoint) ship as a split-tar
**bundle** (`mm2_workshop_assets/`, ~65 GB) that you host on the box and load into the pod once.

---

## 0. Prerequisites (once per box)

Follow the repo top-level `README.md` "Quick Start" for the machine prep (OEM kernel for Ryzen AI
APUs, Docker with non-root access, `build-essential`, and the installer TUI deps). You need
**~150 GB free** (image + 65 GB bundle + the cache it expands into) and Ubuntu 24.04.

Get the bundle onto the box (from OneDrive) at a persistent path, e.g.:

```
/opt/auplc-assets/mm2_workshop_assets/
  base/  libero/  tokenizer/  droid_dataset/  ft_checkpoint/  unpack_bundle.sh  README.txt
```

## 1. Clone the repo and check out the branch

```bash
git clone <your-fork>/aup-learning-cloud.git
cd aup-learning-cloud
git checkout finetuning-interactive-sim
```

## 2. Build the Finetuning course image

The notebooks and the asset scripts are baked into the image at build time (from
`projects/Finetuning/` into `/ryzers/notebooks`), so build from this branch:

```bash
# Layers on top of ghcr.io/amdresearch/auplc-base (pulled automatically).
make -C dockerfiles finetuning GPU_TARGET=gfx1151
# -> ghcr.io/amdresearch/auplc-finetuning:latest
```

## 3. Deploy JupyterHub (k3s single node)

```bash
sudo ./auplc-installer install --gpu=strix-halo      # or: ./auplc-installer  (interactive TUI)
```

Wait for pods to be Ready (`kubectl get pods -A`). The dev deploy auto-logs-in as user `student`
and uses local images (no registry pull). Open the Hub in a browser.

> Storage note: single-node uses k3s `local-path`, whose PVC size request (10Gi) is **not
> enforced** - the 64 GB of assets fit in the default home volume. No capacity change needed.

## 4. Spawn the Finetuning pod (creates the user volume)

In the Hub UI pick **"Fine-tuning on GPUs"** (ROSCON 2026 group) and start the server. This creates
the persistent home volume the assets land in. Leave it running.

## 5. Load the assets into the pod (the key step)

From the repo on the host, stream the bundle straight into the running pod's HF cache and the
fine-tuned-checkpoint location:

```bash
projects/Finetuning/scripts/prestage_to_pod.sh /opt/auplc-assets/mm2_workshop_assets
# defaults: NAMESPACE=jupyterhub  NB_USER=student  POD=jupyter-student  CONTAINER=notebook
```

It writes (as the pod user, correct ownership) and prints a size check at the end:

```
~/.cache/huggingface/hub/models--allenai--MolmoAct2-DROID            (base checkpoint, ~21 GB)
~/.cache/huggingface/hub/datasets--allenai--MolmoAct2-LIBERO-Dataset (dataset, ~33 GB)
~/.cache/huggingface/hub/datasets--allenai--MolmoAct2-DROID-Dataset  (Step-3 open-loop, ~1.4 GB)
~/.cache/huggingface/hub/models--allenai--MolmoAct2-FAST-Tokenizer   (required by the policy)
~/checkpoints/reference/pretrained_model                             (fine-tuned checkpoint, ~11 GB)
```

**Alternative (mount instead of copy):** unpack the bundle to a host dir and mount it into the pod
read-only, then let the notebooks stage it on first run:

```bash
projects/Finetuning/scripts/unpack_bundle.sh /opt/auplc-assets/mm2_workshop_assets /opt/auplc-assets/assets
# mount /opt/auplc-assets/assets into the pod (singleuser.storage.extraVolumes) and set ASSETS_DIR
# to that mount; both notebooks copy it into the caches on first run (see README "Offline assets").
```

## 6. Notebook 1 - fine-tune + eval (`finetune_molmoact2_libero.ipynb`)

Open it in the pod and **Run All**. With the assets prestaged it runs fully offline:

1. **Step 1** ROCm/import smoke test.
2. **Step 2** preflight prints `CACHED` for the base + dataset (no download).
3. **Step 3** open-loop rollout on real DROID episodes (GT-vs-pred overlay) - uses the cached DROID-Dataset.
4. **Step 4** LoRA fine-tune, `STEPS=10` by default (just proves the loop; raise for real training).
5. **Step 5** closed-loop LIBERO eval. By default it evaluates **our fine-tuned reference checkpoint**
   (a strong policy) - verified `pc_success=100%` on `libero_object` task 3.

Useful env knobs (set in a cell / the pod env before launch): `STEPS`, `FT_MODE`
(`lora_vlm`|`action_expert_only`|`full`), `BATCH_SIZE`, `SUITE`, `TASK_ID`, `N_EPISODES`,
`PREFER_TRAINED=1` (eval the checkpoint this run just produced instead of the reference).

## 7. Notebook 2 - interactive sim (`interactive_sim_molmoact2_libero.ipynb`)

Open it and **Run All**. It loads the fine-tuned checkpoint, boots the synchronous LIBERO sim server
on an internal port, and embeds it inline via `jupyter-server-proxy` (no extra port to expose). Type
an instruction (e.g. *"pick up the object and place it in the basket"*) and watch the arm execute.

---

## Notes / gotchas

- **The fine-tuned checkpoint reuses the base weights.** Loading it (eval or interactive) resolves
  and reads `allenai/MolmoAct2-DROID` from the HF cache, so the base **must** be present. The bundle
  includes it and `prestage_to_pod.sh` places it - don't ship only the fine-tuned checkpoint.
- **Everything is offline after step 5.** No Hub access is needed at run time; you can set
  `HF_HUB_OFFLINE=1` to prove it.
- **Assets are staged, never re-hosted** in this repo. They come from the OneDrive bundle.
- **Verified path:** on a single Strix Halo iGPU, `lerobot-eval` on the bundle's fine-tuned
  checkpoint (offline) returned `pc_success=100.0%` (`libero_object`, task 3) in ~26 s.
