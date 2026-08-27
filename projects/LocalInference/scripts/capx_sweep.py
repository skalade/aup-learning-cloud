#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Run a quick compact-model sweep over CaP-X's single-arm tasks."""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


SERVER_URL = "http://localhost:13305/api/v1/chat/completions"

FAST_SCENARIOS = {
    "cube lift": "env_configs/cube_lifting/franka_robosuite_cube_lifting.yaml",
    "cube stack": "env_configs/cube_stack/franka_robosuite_cube_stack.yaml",
    "cube restack": "env_configs/cube_restack/franka_robosuite_cube_restack.yaml",
    "spill wipe": "env_configs/spill_wipe/franka_robosuite_spill_wipe.yaml",
}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model: str
    temperature: float
    size_gb: float
    temperature_source: str
    checkpoint: str | None = None


MODELS = [
    ModelSpec(
        "gemma-e2b",
        "Gemma-4-E2B-it-GGUF",
        1.0,
        4.09,
        "google/gemma-4-E2B-it generation_config.json",
    ),
    ModelSpec(
        "gpt-oss-20b",
        "gpt-oss-20b-mxfp4-GGUF",
        1.0,
        12.1,
        "openai/gpt-oss README recommended sampling parameters",
    ),
    ModelSpec(
        "gemma-e4b",
        "Gemma-4-E4B-it-GGUF",
        1.0,
        5.97,
        "google/gemma-4-E4B-it generation_config.json",
    ),
    ModelSpec(
        "gemma-12b",
        "Gemma-4-12B-it-GGUF",
        1.0,
        7.29,
        "google/gemma-4-12B-it generation_config.json",
    ),
    ModelSpec(
        "qwen3.5-9b",
        "Qwen3.5-9B-GGUF",
        0.6,
        6.88,
        "Qwen3.5 model card: thinking mode for precise coding",
    ),
    ModelSpec(
        "qwen3-coder-30b-a3b",
        "user.Qwen3-Coder-30B-A3B-Instruct-Q4_K_M",
        0.7,
        17.31,
        "Qwen/Qwen3-Coder-30B-A3B-Instruct model card",
        (
            "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:"
            "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"
        ),
    ),
    ModelSpec(
        "qwen3.8-27b",
        "user.Qwen3.8-27B-UD-Q4_K_M",
        1.0,
        15.33,
        "Qwen/Qwen3.8-27B generation_config.json",
        "unsloth/Qwen3.8-27B-GGUF:Qwen3.8-27B-UD-Q4_K_M.gguf",
    ),
    ModelSpec(
        "ministral-8b",
        "user.Ministral-3-8B-Instruct-2512-Q5_K_M",
        0.15,
        6.06,
        "Ministral 3 Instruct model card example",
        (
            "mistralai/Ministral-3-8B-Instruct-2512-GGUF:"
            "Ministral-3-8B-Instruct-2512-Q5_K_M.gguf"
        ),
    ),
    ModelSpec(
        "ministral-14b",
        "user.Ministral-3-14B-Instruct-2512-Q5_K_M",
        0.15,
        9.62,
        "Ministral 3 Instruct model card example",
        (
            "mistralai/Ministral-3-14B-Instruct-2512-GGUF:"
            "Ministral-3-14B-Instruct-2512-Q5_K_M.gguf"
        ),
    ),
    ModelSpec(
        "qwen2.5-coder-7b",
        "user.Qwen2.5-Coder-7B-Instruct-Q6_K",
        0.7,
        6.25,
        "Qwen/Qwen2.5-Coder-7B-Instruct generation_config.json",
        (
            "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF:"
            "qwen2.5-coder-7b-instruct-q6_k.gguf"
        ),
    ),
    ModelSpec(
        "qwen2.5-coder-14b",
        "user.Qwen2.5-Coder-14B-Instruct-Q4_K_M",
        0.7,
        8.99,
        "Qwen/Qwen2.5-Coder-14B-Instruct generation_config.json",
        (
            "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF:"
            "qwen2.5-coder-14b-instruct-q4_k_m.gguf"
        ),
    ),
    ModelSpec(
        "granite-4.1-3b",
        "user.granite-4.1-3b-Q8_0",
        0.0,
        3.62,
        "No sampling recommendation in generation_config.json; greedy fallback",
        "ibm-granite/granite-4.1-3b-GGUF:granite-4.1-3b-Q8_0.gguf",
    ),
]


class Logger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = path.open("a", encoding="utf-8", buffering=1)

    def __call__(self, message: object = "") -> None:
        text = str(message)
        print(text, flush=True)
        self._stream.write(text + "\n")

    def close(self) -> None:
        self._stream.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("CAPX_SWEEP_OUTPUT", "/tmp/capx_sweep")),
    )
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--models",
        nargs="*",
        metavar="KEY",
        help="Model keys to run; defaults to all configured models",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        metavar="LABEL",
        choices=tuple(FAST_SCENARIOS),
        help="Task labels to run; defaults to all established tasks",
    )
    parser.add_argument("--skip-oracle", action="store_true")
    parser.add_argument(
        "--oracle-only",
        action="store_true",
        help="Run the controlled oracle and write its report without loading an LLM",
    )
    parser.add_argument("--skip-pull", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    return parser.parse_args()


def select_models(keys: list[str] | None) -> list[ModelSpec]:
    if not keys:
        return MODELS
    by_key = {spec.key: spec for spec in MODELS}
    unknown = sorted(set(keys) - set(by_key))
    if unknown:
        raise SystemExit(
            f"unknown model key(s): {', '.join(unknown)}; "
            f"choose from {', '.join(by_key)}"
        )
    return [by_key[key] for key in keys]


def pull_custom_model(spec: ModelSpec, log: Logger) -> None:
    if spec.checkpoint is None:
        return
    log(f"Registering/downloading {spec.model} ({spec.checkpoint})")
    started = time.monotonic()
    proc = subprocess.run(
        [
            "lemonade",
            "pull",
            spec.model,
            "--checkpoint",
            "main",
            spec.checkpoint,
            "--recipe",
            "llamacpp",
            "--label",
            "coding",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    for line in proc.stdout.splitlines():
        log(f"  {line}")
    if proc.returncode:
        raise RuntimeError(f"lemonade pull failed with exit code {proc.returncode}")
    log(f"Model preparation took {(time.monotonic() - started) / 60:.1f} minutes")


def serializable(row: dict) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in row.items()
    }


def model_summary(spec: ModelSpec, rows: list[dict]) -> dict:
    solved = sum(bool(row["solved"]) for row in rows)
    errors = sum(bool(row["error"]) for row in rows)
    rewards = [float(row["reward"]) for row in rows]
    elapsed = [float(row.get("elapsed_seconds", 0.0)) for row in rows]
    per_task = {}
    for label in FAST_SCENARIOS:
        task_rows = [row for row in rows if row["label"] == label]
        per_task[label] = {
            "solved": sum(bool(row["solved"]) for row in task_rows),
            "rollouts": len(task_rows),
            "mean_reward": (
                statistics.fmean(float(row["reward"]) for row in task_rows)
                if task_rows
                else 0.0
            ),
        }
    return {
        **asdict(spec),
        "solved": solved,
        "rollouts": len(rows),
        "success_rate": solved / len(rows) if rows else 0.0,
        "sandbox_errors": errors,
        "mean_reward": statistics.fmean(rewards) if rewards else 0.0,
        "mean_rollout_seconds": statistics.fmean(elapsed) if elapsed else 0.0,
        "per_task": per_task,
    }


def write_reports(
    output_dir: Path,
    metadata: dict,
    summaries: list[dict],
    rows: list[dict],
) -> None:
    payload = {
        "metadata": metadata,
        "models": summaries,
        "rollouts": [serializable(row) for row in rows],
    }
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "rollouts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fieldnames = [
            "model_key",
            "model",
            "temperature",
            "label",
            "trial",
            "solved",
            "reward",
            "error",
            "elapsed_seconds",
            "dir",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serializable(row).get(key) for key in fieldnames})


def stop_servers(servers: list[object], log: Logger) -> None:
    for server in servers:
        terminate = getattr(server, "terminate", None)
        if terminate is not None:
            terminate()
    for server in servers:
        wait = getattr(server, "wait", None)
        if wait is not None:
            try:
                wait(timeout=10)
            except subprocess.TimeoutExpired:
                kill = getattr(server, "kill", None)
                if kill is not None:
                    kill()
    log("Perception services stopped")


def main() -> int:
    args = parse_args()
    if args.list_models:
        for spec in MODELS:
            print(
                f"{spec.key:<22} {spec.model:<52} "
                f"temp={spec.temperature:<4} {spec.size_gb:>5.2f} GB"
            )
        return 0
    if args.trials < 1:
        raise SystemExit("--trials must be at least 1")
    if args.oracle_only and args.skip_oracle:
        raise SystemExit("--oracle-only cannot be combined with --skip-oracle")

    selected = select_models(args.models)
    scenarios_requested = (
        {label: FAST_SCENARIOS[label] for label in args.tasks}
        if args.tasks
        else FAST_SCENARIOS
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CAPX_WORK"] = str(output_dir / "artifacts")

    # Imports are intentionally delayed until CAPX_WORK is fixed; capx_demo also
    # moves into CAPX_ROOT so upstream relative config paths resolve correctly.
    from capx.envs.launch import LaunchArgs
    from capx.envs.runner import _start_api_servers
    from capx.utils.launch_utils import _load_config
    from capx_demo import (
        DEFAULT_MODEL,
        benchmark_scenarios,
        ensure_lemonade,
        lemonade_alive,
        quiet_output as quiet,
    )

    scenarios = scenarios_requested

    log = Logger(output_dir / "sweep.log")
    metadata = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "image_id": os.environ.get("EXPERIMENT_IMAGE_ID"),
        "source_revision": os.environ.get("EXPERIMENT_SOURCE_REVISION"),
        "trials_per_task": args.trials,
        "max_tokens": args.max_tokens,
        "server_url": SERVER_URL,
        # The image bakes in the OWLv2 + SAM2 path; kept so reports stay
        # comparable with earlier sweeps that selected it at runtime.
        "perception": "open",
        "scenarios": scenarios,
        "models": [] if args.oracle_only else [asdict(spec) for spec in selected],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    rows: list[dict] = []
    summaries: list[dict] = []
    servers: list[object] = []
    try:
        log("Starting shared OWLv2, SAM2, Contact-GraspNet, and PyRoKi services")
        setup_args = LaunchArgs(
            config_path=next(iter(scenarios.values())),
            model=selected[0].model,
            server_url=SERVER_URL,
            temperature=selected[0].temperature,
            max_tokens=args.max_tokens,
        )
        with quiet(output_dir / "services.log"):
            _, _, api_servers = _load_config(setup_args)
            servers = _start_api_servers(api_servers, 900.0)
        log("Perception services are ready")

        if not args.skip_oracle:
            log("\n===== oracle smoke test: one rollout per task =====")
            oracle_rows = benchmark_scenarios(
                model="oracle",
                server_url=SERVER_URL,
                scenarios=scenarios,
                temperature=0.0,
                max_tokens=args.max_tokens,
                trials=1,
                oracle=True,
                verbose=args.verbose,
                progress=log,
            )
            for row in oracle_rows:
                row["model_key"] = "oracle"
                row["model"] = "oracle"
                row["temperature"] = 0.0
            rows.extend(oracle_rows)
            write_reports(output_dir, metadata, summaries, rows)
            if args.oracle_only:
                return 0

        for index, spec in enumerate(selected, start=1):
            log(
                f"\n===== model {index}/{len(selected)}: {spec.key} "
                f"({spec.model}, temperature={spec.temperature}) ====="
            )
            model_started = time.monotonic()
            try:
                if spec.checkpoint is not None and not args.skip_pull:
                    if not lemonade_alive():
                        log("Starting Lemonade before custom model registration")
                        ensure_lemonade(DEFAULT_MODEL, progress=log)
                    pull_custom_model(spec, log)
                ensure_lemonade(spec.model, progress=log)
                model_rows = benchmark_scenarios(
                    model=spec.model,
                    server_url=SERVER_URL,
                    scenarios=scenarios,
                    temperature=spec.temperature,
                    max_tokens=args.max_tokens,
                    trials=args.trials,
                    verbose=args.verbose,
                    progress=log,
                )
                for row in model_rows:
                    row["model_key"] = spec.key
                    row["model"] = spec.model
                    row["temperature"] = spec.temperature
                rows.extend(model_rows)
                summary = model_summary(spec, model_rows)
                summary["total_model_minutes"] = (
                    time.monotonic() - model_started
                ) / 60
                summaries.append(summary)
                log(
                    f"{spec.key}: {summary['solved']}/{summary['rollouts']} solved, "
                    f"mean reward {summary['mean_reward']:.3f}, "
                    f"{summary['total_model_minutes']:.1f} minutes"
                )
            except Exception as exc:
                log(f"{spec.key} FAILED: {type(exc).__name__}: {exc}")
                summaries.append(
                    {
                        **asdict(spec),
                        "failed": True,
                        "failure": f"{type(exc).__name__}: {exc}",
                        "total_model_minutes": (
                            time.monotonic() - model_started
                        ) / 60,
                    }
                )
            write_reports(output_dir, metadata, summaries, rows)
    finally:
        if servers:
            stop_servers(servers, log)
        subprocess.run(
            ["lemonade", "unload"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        log.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
