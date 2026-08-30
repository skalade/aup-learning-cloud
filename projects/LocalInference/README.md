# ROSCon 2026: Local Robot Inference

Complete the notebooks in order:

0. `0_overview.ipynb`
1. `1_local_inference.ipynb`
2. `2_robot_agents.ipynb`
3. `3_code_as_policy.ipynb`
4. `4_robot_harness_optimization.ipynb`
5. `temp_evolving_rai.ipynb`

## Notebook 4 — structure and time budget

Notebook 4 is built for a 15–20 minute guided session:

| Section | Content | Compute |
| --- | --- | --- |
| The experiment | Editable surface, `helix.toml`, evaluator boundary | pre-rendered |
| One generation, live | A single mutation end to end | opt-in, several minutes |
| Recorded run | Two-generation, two-task Qwen3-Coder evolution | pre-rendered |
| Start the robot services | OWLv2, SAM2, Contact-GraspNet, PyRoKi | live, ~10 s |
| Watch both policies run | Seed vs deployed, one layout per task | **live, ~55 s** |

The seed-vs-deployed replay is the only section that runs live by default, and
it executes frozen Python against the robot services with no model in the
control loop. The recorded run supplies the search-progress plot, the lineage
and gate decisions, and the deployed diff split per file.

Every score the notebook shows comes from the six validation layouts the search
optimized against, which keeps the session short at the cost of saying nothing
about generalization. The notebook states that limit where it reports the
numbers and points at the deployed repository for anyone who wants to replay it
on layouts the search never saw.

The appendix is off by default (`RUN_APPENDIX = False`). Its `helix.toml`
objective names the defect so Gemma E2B can land one accepted mutation inside a
live session, so it demonstrates the loop's mechanics rather than the agent's
ability to diagnose. `rho_demo.DEFAULT_GUIDANCE_DISCLOSURE` carries that caveat
and the notebook prints it before the run.

## Generating notebook 4's recorded assets

**`recorded_results/` is not committed.** Produce it before building the course
image, or the recorded half of notebook 4 has nothing to read. Run this inside
the LocalInference image on a machine with the GPU:

```bash
python /ryzers/notebooks/scripts/rho_multitask_study.py \
  --output-dir /ryzers/notebooks/recorded_results \
  --model user.Qwen3-Coder-30B-A3B-Instruct-Q4_K_M \
  --generations 2 \
  --proposals 4 \
  --minibatch-size 2 \
  --hidden-repeats 3
```

`--generations 2` is short because a six-generation version of this experiment
was run first and everything it achieved landed in generation 1; the extra
generations rejected most of their proposals and deployed a candidate that
differed from the generation-1 winner by a `try/except` that re-raises.

Notebook 4 needs only two things out of that run:

- `rho_multitask_report.json` — schema `rho-multitask-helix-report/v2`
- `repos/seed/` and `repos/selected/` — the two policy repositories the live
  replay cell runs

The study also writes `videos/` and replays a set of layouts the search never
sampled, controlled by `--hidden-repeats`. The notebook renders neither, because
it replays both policies live instead, so delete `videos/` before committing:
it is the only large thing the study produces, and this repository has no Git
LFS configuration. Without it the recorded tree is about 230 KB.

Copy the resulting tree back to `projects/LocalInference/recorded_results/` so
`build.sh` bakes it into the image. Verify it from the notebook's preflight cell
or directly:

```bash
python -c "import sys; sys.path.insert(0, 'scripts'); import rho_report; \
print(rho_report.format_preflight(rho_report.preflight('recorded_results')))"
```

Media paths are stored relative to the recorded root, so a study produced on one
host replays correctly from `/ryzers/notebooks/recorded_results` in the image.

## Recorded study evidence

`recorded_results/overnight_study_summary.json` is the compact entry point for
the controlled overnight study. It links to:

- `capx_matrix_analysis.json`: 60 same-image model rollouts, the 4-task oracle,
  task metrics, latency, code signals, and failure taxonomy.
- `rho_study_analysis.json`: three single-policy mutation-agent preflights,
  Qwen surface comparisons, the two-generation multi-task result, and the
  bounded cube-restack depth study.
- `video_validation.json`: recording counts, report-path checks, container
  validation, and ffprobe results.
- `capx_selected/provenance.json`: selected local Gemma programs and their
  exact source trials.

The raw per-trial CaP-X and RHO reports remain below `recorded_results/`.
Regenerate the compact analyses after adding new shards with:

```bash
python scripts/capx_analyze.py \
  recorded_results/capx_primary_oracle/results.json \
  recorded_results/capx_primary_qwen/results.json \
  recorded_results/capx_local_gemma_shards/*/results.json \
  --output recorded_results/capx_matrix_analysis.json
python scripts/rho_analyze.py \
  --recorded-root recorded_results \
  --capx-analysis recorded_results/capx_matrix_analysis.json \
  --output recorded_results/rho_study_analysis.json
```

The temporary Evolving RAI notebook applies the same one-generation
repository-evolution pattern to a live RAI tool-calling agent. A deterministic
in-memory tabletop replaces the full O3DE benchmark so it fits the workshop:
HELIX edits both `prompt.py` and `tools.py`, the notebook displays the selected
files, and RAI reruns one held-out manipulation test. It does not replay or
claim to reproduce paper results.

## Measured workshop runtime

Measured on the workshop Strix Halo GPU:

- CaP-X: 4.5 seconds for model setup, 11.4 seconds for perception/control
  setup, 14.2 seconds for one LLM call, and 11.8 seconds for one rollout.
- RHO service startup: the `ensure_services` cell sits immediately above the
  live replay, so nothing loads OWLv2, SAM2, Contact-GraspNet or PyRoKi until
  the notebook needs them. Run it early if you would rather pay the cost before
  the session; it is idempotent and the later cell will reuse what is running.
- RHO seed-vs-deployed replay: four rollouts, two per task, 10 to 16 seconds
  each. This is the only live compute in the guided path. The seed crashes every
  time; measured five times on each of the six validation layouts the deployed
  policy solved 26 of 30, so budget for the occasional miss rather than treating
  it as a failure. `scripts/rho_replay_scan.py` reproduces those pass rates, and
  `rho_report.validation_trial` uses them to show the layouts that ran 5 for 5.
- RHO recorded run: reads `recorded_results/` and renders immediately.
- RHO one-generation-live cell: allow 4–8 minutes for four simulator
  evaluations and one bounded OpenCode mutation. Off by default, and it starts
  the services itself when enabled.

Environment checks:

```bash
/ryzers/test_ros.sh
/ryzers/test_o3de.sh
/ryzers/test_rai.sh
/ryzers/test_lemonade-sdk.sh
/ryzers/test_capx.sh
/ryzers/test_rho.sh
/ryzers/test_rho_multitask.sh
/ryzers/test_rai_toy_evolution.sh
```

The toy RAI evolution test is static/mock by default. Inside the
LocalInference image, opt into a real RAI seed-versus-repaired agent check:

```bash
RAI_TOY_RUN_LIVE=1 /ryzers/test_rai_toy_evolution.sh
```

The workshop's Gemma E2B, Gemma E4B, and Qwen3-Coder Q4_K_M GGUFs are baked
under `/opt/lemonade-cache`, outside the JupyterHub home-volume mount. Notebook
4 uses Gemma E2B for the fast live mutation and the 17.3 GB Qwen checkpoint for
the recorded multi-task evolution. SAM2.1 Large and OWLv2 Large are likewise
baked under `/opt/capx-cache`; neither runtime path needs a Hugging Face token
or a first-run model download. Their checkpoints are staged through FP16 before
an on-device FP32 conversion to avoid a multi-minute ROCm transfer while
retaining FP32 execution. Matching `*_fast.yaml` CaP-X configs retain SAM2.1
Small and OWLv2 Base for comparisons.
