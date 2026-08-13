<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved. -->

# Fine-tuning course - organizer & developer guide

> **Just attending?** You do not need this file - follow **[README.md](README.md)** instead.


---

## Step 1 - Prepare the machine (once per box)

**Run this on: the Strix Halo box (a terminal, with sudo).**

Install the prerequisites from the repository's top-level `README.md`

## Step 2 - Get the course code

**Run this on: the Strix Halo box.**

```bash
git clone https://github.com/skalade/aup-learning-cloud
```

Everything in the rest of this guide is run from inside this `aup-learning-cloud` folder.

## Step 3 - Put the workshop files on the machine

**Run this on: any machine that can reach both your file store and the Strix Halo box.**

The large inputs ship as one folder called `mm2_workshop_assets` (about **16 GB** with the default
LIBERO subset), split into ~4 GB pieces so it copies easily. It is hosted on a shared
drive. Its layout is:

```
mm2_workshop_assets/
  base/           the base MolmoAct2-DROID checkpoint, stored in BF16          (~11 GB)
  libero/         the LIBERO training dataset (default: ~1 GB task-diverse subset;
                  set USE_FULL_LIBERO=1 in the notebook to fetch the full ~33 GB set)
  tokenizer/      a small tokenizer the model needs
  droid_dataset/  a small real-robot dataset used by notebook 1, Step 3         (~1.4 GB)
  ft_checkpoint/  the fine-tune "delta": the LoRA adapter + trained parts       (~2.4 GB)
  reconstruct_reference.py  unpack_bundle.sh  README.txt
```

Copy that whole folder onto the Strix Halo box, for example:

```bash
scp -r /path/to/mm2_workshop_assets  <user>@<strix-halo-address>:/opt/auplc-assets/mm2_workshop_assets
```

## Step 4 - Build the self-contained course image

**Run this on: the Strix Halo box, inside the `aup-learning-cloud` folder.**

One command builds a FULLY self-contained image: the Docker build itself unpacks the raw workshop
bundle, includes the droid-trained checkpoint delivered by MolmoAct2, rebuilds the full fine-tuned checkpoint from our trained LoRA adapter, puts the sample droid data and the sample LIBERO data subset for testing and fine-tuning, and bakes everything - together with the two notebooks and their helper scripts - into a single container image. Point `ASSETS_SRC` at the raw bundle from Step 3:

```bash
make -C dockerfiles finetuning GPU_TARGET=gfx1151 ASSETS_SRC=/path/to/mm2_workshop_assets
```

## Step 5 - Deploy the JupyterHub server

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
slice and a writable home folder. The assets are already in the image, so attendees just open the
notebooks and click **Run All** - nothing is downloaded or linked at the venue.

> **Already have another JupyterHub running on this same box?** Two servers cannot share the same
> name. On a fresh box this never happens. If it does (a shared dev box), the simplest fix is to
> deploy on a machine that does not already run one - that keeps this command exactly as written,
> with nothing to configure.


