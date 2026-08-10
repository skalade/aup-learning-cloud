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

## Step 6 - Deliver the files to attendees

Pick **one** of the two options below.

### Option A - Multi-user workshop: mount the folder into everyone's server

**Run this on: the Strix Halo box (edit the chart, then re-run the installer or `helm upgrade`).**

Mount the `mm2_workshop_assets` folder **read-only** into every attendee's server at the fixed path
`/opt/auplc-assets/mm2_workshop_assets` (via the chart's `singleuser.storage.extraVolumes` and
`extraVolumeMounts`). Each attendee then runs `scripts/fetch_assets.sh` once from a terminal in their
server (this is exactly what README.md tells them). That script copies the files into their own
workspace and rebuilds the full fine-tuned checkpoint from base + delta.

### Option B - One or a few attendees: push the files straight into a running server

**Run this on: the Strix Halo box, inside the `aup-learning-cloud` folder.**

After an attendee has started their server (Steps 1-2 of README.md), stream the files directly into
it. This needs `kubectl` (admin), so it is your job, not the attendee's:

```bash
projects/Finetuning/scripts/prestage_to_pod.sh /opt/auplc-assets/mm2_workshop_assets
```

Defaults it uses: namespace `jupyterhub`, user `student`, pod `jupyter-student`, container
`notebook`. If your attendee's username or namespace differs, set them first, for example:

```bash
NAMESPACE=jupyterhub NB_USER=alice POD=jupyter-alice \
  projects/Finetuning/scripts/prestage_to_pod.sh /opt/auplc-assets/mm2_workshop_assets
```

This copies the files into the attendee's server and rebuilds the full fine-tuned checkpoint inside
it. With Option B the attendee can skip `fetch_assets.sh` - their files are already in place.

## Step 7 - Give attendees the address

Send each attendee:

- the web address from Step 4 (`http://<this-machine's-address>:30890/`), and
- their username and password.

They then follow **[README.md](README.md)**. Done.

---

## Developer check: reproduce the whole thing offline before you ship

**Run this on: a Strix Halo box, after Steps 1-5 above.**

To verify a change end to end without any network, run both notebooks headless against the staged
files. First unpack the bundle into a plain folder and stage it into a local cache:

```bash
# unpack the 47 GB bundle into ./assets (also rebuilds the full fine-tuned checkpoint)
/opt/auplc-assets/mm2_workshop_assets/unpack_bundle.sh /opt/auplc-assets/assets

# stage into the caches the notebooks read (base model, dataset, fine-tuned policy)
ASSETS_DIR=/opt/auplc-assets/assets \
  projects/Finetuning/scripts/stage_assets.sh
```

Then execute each notebook fully offline on the GPU (this is the same environment attendees get):

```bash
docker run --rm --network=host --ipc=host --shm-size 16G \
  --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined \
  --group-add video --group-add render \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v "$HOME/.cache":/home/jovyan/.cache \
  -v "$HOME/checkpoints":/home/jovyan/checkpoints \
  -v "$HOME/outputs":/home/jovyan/outputs \
  --entrypoint bash ghcr.io/amdresearch/auplc-finetuning:latest -c '
    cd /ryzers/notebooks
    /opt/train-venv/bin/python -m ipykernel install --user --name tv >/dev/null 2>&1
    jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=tv \
      --ExecutePreprocessor.timeout=-1 --output /home/jovyan/outputs/nb1.ipynb \
      finetune_molmoact2_libero.ipynb'
```

A healthy run shows the closed-loop LIBERO evaluation reporting success and writes the executed
notebook to `~/outputs`. Model weights and large videos stay on the machine (never re-hosted).
