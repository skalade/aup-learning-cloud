<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved. -->

# ROSCon 2026: VLA Fine-tuning

LoRA fine-tune the **MolmoAct2** vision-language-action model on the **LIBERO** simulator, then
drive the fine-tuned policy live in the sim - all on a single AMD **Strix Halo** iGPU, from your
browser. Everything runs inside this course environment; there is nothing to install and **no
`sudo`** - you only need your AUP Learning Cloud login.

> Setting up the workshop (building the image, deploying JupyterHub, hosting the assets)? That is
> the organizer/developer job - see **[ORGANIZER.md](ORGANIZER.md)**. This page is for attendees.

## 1. Start your environment

On the AUP Learning Cloud spawn page pick **ROSCON 2026 → "Fine-tuning on GPUs"** and start the
server. It lands you in `/ryzers/notebooks` with the two notebooks below. (Optional) sanity-check
the GPU and the MolmoAct2 stack from a terminal:

```bash
/ryzers/test_torch.sh        # ROCm torch sees the GPU
/ryzers/test_molmoact2.sh    # MolmoAct2 + LIBERO training stack imports
```

## 2. Load the workshop assets (once, no download)

The base checkpoint, the LIBERO dataset and our fine-tuned checkpoint are large (~65 GB total),
so instead of downloading them at the venue your instructor stages them on shared storage. Copy
them into your pod once - a few minutes - with:

```bash
scripts/fetch_assets.sh
```

That places everything in your `~/.cache/huggingface` and `~/checkpoints`. The notebooks then
**detect the cache and skip every download automatically** - you do not toggle anything.

> If assets were not staged, skip this step: the notebooks fall back to downloading the base
> model and dataset from the Hugging Face Hub at run time (needs network access and is slow).

## 3. Run the notebooks

| Notebook | What it does |
|---|---|
| `finetune_molmoact2_libero.ipynb` | ROCm smoke-test → load the base checkpoint → open-loop DROID validation → **LoRA fine-tune** a few steps → closed-loop LIBERO eval of the fine-tuned policy. |
| `interactive_sim_molmoact2_libero.ipynb` | Standalone: drives the fine-tuned policy **live in LIBERO**, embedded right in the notebook. Type an instruction, watch the arm act. No dependency on notebook 1. |

Open each and **Run All**. The interactive sim needs **no extra port**: it is served inside your
session and proxied through your existing JupyterHub route via
[`jupyter-server-proxy`][jsp], then embedded with an `IFrame`.

[jsp]: https://github.com/jupyterhub/jupyter-server-proxy

## Which weights run in the demo?

Both the closed-loop eval (notebook 1, Step 5) and the interactive sim default to our
**fine-tuned reference checkpoint** at `~/checkpoints/reference/pretrained_model` (staged by
`fetch_assets.sh` in step 2). A short in-notebook LoRA run proves the training loop but will not
fully converge, so the reference checkpoint is what gives a compelling demo.

Override the policy from a cell or the pod env:

| Var | Default | Meaning |
|---|---|---|
| `POLICY_PATH` | (unset) | Explicit checkpoint: a local dir **or** a Hugging Face repo id. Wins over the default. |
| `PREFER_TRAINED` | `0` | `1` = eval the checkpoint *this* notebook run just produced instead of the reference. |
| `SUITE` | `libero_object` | LIBERO task suite. |
| `TASK_ID` | `3` | Task within the suite. |
| `STEPS` | `10` | LoRA steps in notebook 1 (raise for real training). |
| `FT_MODE` | `lora_vlm` | `lora_vlm` / `action_expert_only` / `full`. |

Training outputs land under `~/outputs` and `~/checkpoints`; small eval artifacts under
`~/outputs`. Model weights and large videos stay on the machine (never re-hosted here).
