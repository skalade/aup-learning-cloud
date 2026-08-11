<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved. -->

# Fine-tuning course - organizer & developer guide

This guide sets up the MolmoAct2 fine-tuning workshop on one AMD **Strix Halo** machine, end to end:
build the course image, deploy the JupyterHub server, put the large files in place, and hand each
attendee a web address. When you are done, an attendee just opens that address, logs in, runs one
command, and clicks **Run All** - with no downloading at the venue.

> **Just attending?** You do not need this file - follow **[README.md](README.md)** instead.

Every step below says **where to run it**. Do them in order. You need a Strix Halo box running
**Ubuntu 24.04** where you have `sudo` (admin) rights, and about **120 GB free** disk.

---

## Step 1 - Prepare the machine (once per box)

**Run this on: the Strix Halo box (a terminal, with sudo).**

Install the prerequisites from the repository's top-level `README.md` "Quick Start": the OEM kernel
for Ryzen AI APUs, Docker set up for non-root use, `build-essential`, and the installer's terminal
UI dependencies. Then confirm Docker works without sudo:

```bash
docker run --rm hello-world
```

If that prints a success message, the machine is ready.

## Step 2 - Get the course code

**Run this on: the Strix Halo box.**

```bash
git clone <your-fork-url>/aup-learning-cloud.git
cd aup-learning-cloud
git checkout finetuning-interactive-sim
```

Everything in the rest of this guide is run from inside this `aup-learning-cloud` folder.

## Step 3 - Build the course image

**Run this on: the Strix Halo box, inside the `aup-learning-cloud` folder.**

This bakes the two notebooks and their helper scripts into a container image (the base model layers
are pulled automatically and cached, so later rebuilds take under a minute):

```bash
make -C dockerfiles finetuning GPU_TARGET=gfx1151
```

The result is a local image named `ghcr.io/amdresearch/auplc-finetuning:latest`. When you change a
notebook or script later, run this same command again to rebuild in place - do not create a second
image.

## Step 4 - Deploy the JupyterHub server

**Run this on: the Strix Halo box, inside the `aup-learning-cloud` folder.**

```bash
sudo ./auplc-installer install --gpu=strix-halo
```

This installs a small single-node Kubernetes (k3s) and the JupyterHub server on top of it, using the
image you just built. Wait until it finishes and reports the pods are ready. When it is done it
prints the **web address of the server** - it looks like `http://<this-machine's-address>:30890/`.
**Write that address down; it is what you give attendees in Step 7.**

The course "Fine-tuning on GPUs" is already registered, so it shows up on the login page
automatically. Each attendee who logs in gets their own private notebook server with its own GPU
slice and a writable home folder.

> **Already have another JupyterHub running on this same box?** Two servers cannot share the same
> name. On a fresh box this never happens. If it does (a shared dev box), the simplest fix is to
> deploy on a machine that does not already run one - that keeps this command exactly as written,
> with nothing to configure.

## Step 5 - Put the workshop files on the machine

**Run this on: any machine that can reach both your file store and the Strix Halo box.**

The large inputs ship as one folder called `mm2_workshop_assets` (about **47 GB**), split into ~4 GB
pieces so it copies easily. It is hosted on your team's shared drive (for this workshop, OneDrive).
Its layout is:

```
mm2_workshop_assets/
  base/           the base MolmoAct2-DROID checkpoint, stored in BF16   (~11 GB)
  libero/         the LIBERO training dataset                            (~33 GB)
  tokenizer/      a small tokenizer the model needs
  droid_dataset/  a small real-robot dataset used by notebook 1, Step 3  (~1.4 GB)
  ft_checkpoint/  the fine-tune "delta": the LoRA adapter + trained parts (~2.4 GB)
  reconstruct_reference.py  unpack_bundle.sh  README.txt
```

Copy that whole folder onto the Strix Halo box, for example:

```bash
scp -r /path/to/mm2_workshop_assets  <user>@<strix-halo-address>:/opt/auplc-assets/mm2_workshop_assets
```

> **Why is `ft_checkpoint/` so small, and what is the "delta"?** The fine-tuned policy is just the
> frozen base model plus a small set of weights that changed during fine-tuning. Instead of shipping
> the whole base model twice, we ship the base once (in BF16, the precision everyone runs in) and
> only the small delta. The full fine-tuned checkpoint is rebuilt automatically from base + delta in
> Step 6 - you do not do anything extra.

## Step 6 - Pre-stage the files once (before the workshop)

**Run this on: the Strix Halo box, once, before attendees arrive.**

Unpack the bundle one time into a shared, read-only folder and rebuild the full fine-tuned
checkpoint once. Every attendee's server then **links** to this single copy instantly - nothing is
downloaded or copied per person, and attendees never touch it.

```bash
sudo mkdir -p /opt/auplc-assets/assets
docker run --rm --user 0:0 -v /opt/auplc-assets:/opt/auplc-assets \
  --entrypoint bash ghcr.io/amdresearch/auplc-finetuning:latest \
  /ryzers/notebooks/scripts/prestage_shared.sh \
    /opt/auplc-assets/mm2_workshop_assets /opt/auplc-assets/assets
```

This writes `/opt/auplc-assets/assets` - the base model and datasets under `hf_hub/`, and the
rebuilt fine-tuned policy under `checkpoints/reference/pretrained_model/` - and makes it
world-readable. It takes a couple of minutes and prints `Shared assets ready` when done.

That is all you do. The standard install already mounts `/opt/auplc-assets` **read-only** into every
attendee's server, and the moment a server starts it links these files into that attendee's
workspace automatically - no command, no `sudo`, no downloading. Attendees just open the notebooks
and run.

> **Ran the installer before staging?** That is fine - the mount is part of the standard install, so
> any server started after Step 6 picks the files up automatically. If a server was already running,
> that attendee should restart it (**File → Hub Control Panel → Stop My Server**, then **Start**) so
> the one-time link step runs.

## Step 7 - Give attendees the address

Send each attendee:

- the web address from Step 4 (`http://<this-machine's-address>:30890/`), and
- their username and password.

They then follow **[README.md](README.md)**. Done.

---

## Developer check: reproduce the whole thing offline before you ship

**Run this on: a Strix Halo box, after Steps 1-6 above.**

To verify a change end to end without any network, run a notebook headless against the same shared
files the attendees use. This links them into a scratch cache exactly the way an attendee's server
does (zero-copy), then executes the notebook fully offline on the GPU:

```bash
docker run --rm --network=host --ipc=host --shm-size 16G \
  --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined \
  --group-add video --group-add render \
  -v /opt/auplc-assets:/opt/auplc-assets:ro \
  --tmpfs /home/jovyan:mode=0777 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  --entrypoint bash ghcr.io/amdresearch/auplc-finetuning:latest -lc '
    scripts/fetch_assets.sh                     # links the shared assets into this run (zero-copy)
    mkdir -p /home/jovyan/outputs
    /opt/train-venv/bin/python -m ipykernel install --user --name tv >/dev/null 2>&1
    jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=tv \
      --ExecutePreprocessor.timeout=-1 --output /home/jovyan/outputs/nb1.ipynb \
      finetune_molmoact2_libero.ipynb'
```

A healthy run shows the closed-loop LIBERO evaluation reporting success. Model weights and large
videos stay on the machine (never re-hosted).
