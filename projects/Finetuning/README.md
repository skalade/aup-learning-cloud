# ROSCon 2026: VLA Finetuning

LoRA fine-tune the **MolmoAct2** vision-language-action model on the **LIBERO** simulator and
demo the result interactively - all on AMD hardware (a single Strix Halo iGPU, or a multi-GPU
AMD Instinct node). Everything runs inside the course container; no extra ports are required.

Environment checks:

```bash
/ryzers/test_torch.sh
/ryzers/test_molmoact2.sh
```

Models are cached under `~/.cache/huggingface`. Training outputs and checkpoints are written
under `~/outputs` and `~/checkpoints`.

## Notebooks

| Notebook | What it does |
|---|---|
| `finetune_molmoact2_libero.ipynb` | End-to-end training path: model/ROCm smoke-test, base checkpoint + dataset prefetch, open-loop DROID validation, LoRA fine-tune, and closed-loop LIBERO eval. |
| `interactive_sim_molmoact2_libero.ipynb` | Standalone, self-contained interactive LIBERO demo. Runs the fine-tuned policy in the simulator (**synchronous** plan-execute-replan loop) and embeds the live viewport inline in the notebook. No dependency on the training notebook. |

## Interactive sim server

`scripts/interactive_server_ft.py` is the small HTTP server the interactive notebook launches.
It loads a LeRobot-format policy, runs `lerobot_eval.rollout` (the same closed-loop used by the
eval), and serves a minimal web UI (single-frame `/frame` polling + a command box).

There is **no extra port to forward**: the server binds an internal port and is reached through
each JupyterHub user's existing notebook route via [`jupyter-server-proxy`][jsp] at
`{JUPYTERHUB_SERVICE_PREFIX}/proxy/8080/`. The course Dockerfile installs `jupyter-server-proxy`.

[jsp]: https://github.com/jupyterhub/jupyter-server-proxy

## Choosing the policy weights

Both the closed-loop eval and the interactive sim default to a **fine-tuned checkpoint**:

- `POLICY_PATH` - explicit override; a local LeRobot checkpoint dir **or** a Hugging Face repo id.
- `REFERENCE_POLICY` - default location, `~/checkpoints/reference/pretrained_model`.

A short in-notebook LoRA run demonstrates the pipeline but will not fully converge; for a strong
demo, stage a longer-trained fine-tuned checkpoint at `REFERENCE_POLICY` (or set `POLICY_PATH`).
Large checkpoints and datasets are **not** baked into the image - they are downloaded from the
Hugging Face Hub at run time (base model / dataset) or staged onto the user's persistent storage
(the fine-tuned checkpoint), and cached there for reuse.

## Useful environment variables

| Var | Default | Meaning |
|---|---|---|
| `POLICY_PATH` | (unset) | Explicit policy checkpoint (dir or Hub repo id); overrides the default. |
| `REFERENCE_POLICY` | `~/checkpoints/reference/pretrained_model` | Default fine-tuned checkpoint location. |
| `SUITE` | `libero_object` | LIBERO task suite for the interactive sim. |
| `TASK_ID` | `3` | Task within the suite. |
| `RT_PORT` | `8080` | Internal port for the sim server (proxied, never exposed directly). |
| `BASE_CKPT` | `allenai/MolmoAct2-DROID` | Base checkpoint to LoRA-adapt from. |
| `DATASET_REPO` | `allenai/MolmoAct2-LIBERO-Dataset` | Fine-tuning dataset. |

The training command runs single-GPU on Strix Halo and multi-GPU on an AMD Instinct node
automatically (it launches `N_GPUS` processes).
