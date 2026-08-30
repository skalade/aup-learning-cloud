#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Replay the seed and deployed repositories over a study's validation layouts.

A single rollout moves enough that one pass settles nothing: grasp sampling is
stochastic, so the same frozen policy can solve a layout once and miss it the
next time. This replays every validation layout ``--reps`` times and reports a
pass rate per layout, which is what ``rho_report.validation_trial`` uses to pick
the layout shown in the notebook's live comparison.

    python scripts/rho_replay_scan.py --reps 5
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rho_demo
import rho_multitask_demo
import rho_report

NOISY_PREFIXES = ("Running SAM2", "Object ", "Grasp sample", "Approaching", "Lifting")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recorded-root",
        type=Path,
        default=rho_report.DEFAULT_RECORDED_ROOT,
        help="study directory holding the report and the seed/selected repositories",
    )
    parser.add_argument(
        "--reps", type=int, default=5, help="replays per layout per repository"
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=Path("/tmp/rho_replay_scan"),
        help="where to write clips, if --capture is set",
    )
    parser.add_argument(
        "--capture", action="store_true", help="record video for every replay"
    )
    return parser.parse_args(argv)


def validation_layouts(report: dict) -> dict[str, list[int]]:
    """Trial ids per task, in report order."""
    layouts: dict[str, list[int]] = defaultdict(list)
    for entry in report.get("baseline_validation", []):
        layouts[str(entry["task"])].append(int(entry["trial"]))
    return dict(layouts)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.recorded_root
    repos = {"seed": root / "repos" / "seed", "deployed": root / "repos" / "selected"}

    rho_demo.ensure_services(model=rho_demo.DEFAULT_RHO_MODEL)
    rho_demo.VIDEO_ROOT = args.video_root

    report = rho_report.load_report(root)
    tally: dict[tuple[str, int, str], dict[str, float]] = defaultdict(
        lambda: {"solved": 0, "reward": 0.0}
    )

    for task, trials in validation_layouts(report).items():
        for trial in trials:
            for label, repo in repos.items():
                for _ in range(args.reps):
                    result = rho_multitask_demo.replay_trials(
                        repo, trials={task: [trial]}, capture=args.capture
                    )[0]
                    bucket = tally[(task, trial, label)]
                    bucket["solved"] += int(bool(result["task_completed"]))
                    bucket["reward"] += float(result["reward"])
            print(
                f"{task:<11} trial {trial}: "
                + "  ".join(
                    f"{label} {tally[(task, trial, label)]['solved']}/{args.reps}"
                    f" (mean {tally[(task, trial, label)]['reward'] / args.reps:.3f})"
                    for label in repos
                ),
                flush=True,
            )

    totals = defaultdict(int)
    for (_, _, label), bucket in tally.items():
        totals[label] += int(bucket["solved"])
    attempts = len(tally) // len(repos) * args.reps
    print()
    for label in repos:
        print(f"{label:<9} solved {totals[label]} of {attempts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
