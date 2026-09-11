# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Infrastructure helpers for the humanoid locomotion workshop notebooks.

The notebook narrative intentionally stays focused on locomotion, terrain, rewards,
and training. Runner construction, batched evaluation, table rendering, and
inline playback live here instead.
"""

from __future__ import annotations

import gc
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from io import StringIO
from pathlib import Path

import torch

from gslab.envs import GenesisManagerBasedRlEnv
from gslab.rl import RslRlVecEnvWrapper
from gslab.rl.rsl_rl.runner import GslabOnPolicyRunner
from gslab.tasks.registry import load_env_cfg, load_rl_cfg
from gslab.vis import NotebookViewCfg, NotebookViewer


def run_environment_check() -> None:
    """Verify the workshop wheel and execute one Genesis step on AMDGPU."""
    from importlib.metadata import version

    import genesis as gs
    import gslab
    import gslab.tasks  # verify task registration before changing the default device

    print(f"gslab  : {version('gslab')} ({gslab.SRC_PATH})")
    print(f"Genesis: {gs.__version__}")
    print(f"PyTorch: {torch.__version__}")

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot see the AMD GPU.")
    if torch.version.hip is None:
        raise RuntimeError(
            "PyTorch is not a ROCm build. Install it from the rocm7.2 wheel index."
        )

    if not getattr(gs, "_initialized", False):
        gs.init(backend=gs.amdgpu, seed=0, performance_mode=True)
    if gs.backend != gs.amdgpu:
        raise RuntimeError(f"Genesis selected {gs.backend!r}, expected gs.amdgpu.")

    probe = torch.arange(8, device="cuda", dtype=torch.float32)
    if probe.square().sum().item() != 140.0:
        raise RuntimeError("PyTorch ROCm compute probe returned an unexpected result.")

    smoke_scene = None
    smoke_box = None
    try:
        smoke_scene = gs.Scene(
            show_viewer=False,
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        )
        smoke_scene.add_entity(gs.morphs.Plane())
        smoke_box = smoke_scene.add_entity(
            gs.morphs.Box(size=(0.10, 0.10, 0.10), pos=(0.0, 0.0, 0.20))
        )
        smoke_scene.build(n_envs=1)
        smoke_scene.step()
        if not torch.isfinite(smoke_box.get_pos()).all().item():
            raise RuntimeError("Genesis produced a non-finite physics state.")
    finally:
        if smoke_scene is not None:
            smoke_scene.destroy()
        smoke_scene = smoke_box = None
        clear_device_cache()

    print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"ROCm    : {torch.version.hip}")
    print("PASS: PyTorch sees the GPU and Genesis completed a step on gs.amdgpu.")


def clear_device_cache() -> None:
    """Collect released scenes before the next large GPU allocation."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def require_checkpoint(path: Path) -> Path:
    """Return a usable checkpoint or raise an actionable workshop error."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not ready: {path}. Restore the checkpoint or wait for training."
        )
    if path.stat().st_size < 1_000_000:
        raise RuntimeError(
            f"{path.name} is unexpectedly small and is not a usable model checkpoint."
        )
    return path


def _quiet_call(function, *args, **kwargs):
    """Suppress verbose model architecture output while preserving exceptions."""
    sink = StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        return function(*args, **kwargs)


def make_runner(
    env,
    task: str,
    checkpoint: Path | None = None,
    log_dir: str | Path | None = None,
    agent_cfg=None,
    device: str = "cuda",
):
    """Construct an rsl_rl runner quietly and optionally load actor weights."""
    if agent_cfg is None:
        agent_cfg = load_rl_cfg(task)
        agent_cfg.logger = "tensorboard"

    def build():
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = GslabOnPolicyRunner(
            wrapped,
            asdict(agent_cfg),
            str(log_dir) if log_dir is not None else None,
            device,
        )
        if checkpoint is not None:
            runner.load(
                str(require_checkpoint(Path(checkpoint))),
                load_cfg={"actor": True},
                strict=True,
            )
        return runner

    return _quiet_call(build), agent_cfg


def load_policy(runner, checkpoint: Path, device: str = "cuda"):
    """Load actor weights without printing the full model architecture."""
    checkpoint = require_checkpoint(Path(checkpoint))
    _quiet_call(
        runner.load,
        str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
    )
    return runner.get_inference_policy(device=device)


def assign_environments(
    env,
    levels: int | torch.Tensor,
    terrain_types: int | torch.Tensor,
) -> None:
    """Assign each robot to an explicit terrain cell."""
    levels = torch.as_tensor(levels, device=env.device, dtype=torch.long)
    terrain_types = torch.as_tensor(
        terrain_types, device=env.device, dtype=torch.long
    )
    if levels.ndim == 0:
        levels = levels.expand(env.num_envs)
    if terrain_types.ndim == 0:
        terrain_types = terrain_types.expand(env.num_envs)
    if levels.numel() != env.num_envs or terrain_types.numel() != env.num_envs:
        raise ValueError("Terrain assignments must contain one value per robot.")

    terrain = env.terrain
    terrain.terrain_levels[:] = levels
    terrain.terrain_types[:] = terrain_types
    env_ids = torch.arange(env.num_envs, device=env.device)
    terrain.env_origins[:] = terrain._origins_for(env_ids)
    env.refresh_terrain_spawn()


def hold_command(env, command: torch.Tensor) -> None:
    """Prevent velocity commands from resampling during a comparison."""
    env.commands[:] = command
    for name in env.command_manager.active_terms:
        term = env.command_manager.get_term(name)
        if term is not None:
            term.hold()
            term.command[:] = command


def _reset_assignment(env, levels, terrain_types) -> None:
    assign_environments(env, levels, terrain_types)
    env.seed(0)
    env.reset()


def _rollout(
    env,
    policy,
    duration_s: float,
    command: tuple[float, float, float],
):
    command_tensor = torch.tensor(
        command, device=env.device, dtype=torch.float32
    ).expand(env.num_envs, -1)
    start_x = env.robot.get_pos()[:, 0].clone()
    fell = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    travelled = torch.zeros(env.num_envs, device=env.device)

    for _ in range(round(duration_s / env.step_dt)):
        hold_command(env, command_tensor)
        env.obs_buf = env._compute_observations()
        with torch.inference_mode():
            actions = policy(env.get_observations())
        _, _, terminated, _, _ = env.step(actions)

        newly_fallen = terminated & ~fell
        if newly_fallen.any():
            travelled[newly_fallen] = (
                env.robot.get_pos()[newly_fallen, 0] - start_x[newly_fallen]
            )
        fell |= terminated

    survived = ~fell
    travelled[survived] = env.robot.get_pos()[survived, 0] - start_x[survived]
    return fell, travelled


def _evaluate_batched_env(
    env,
    runner,
    checkpoint: Path,
    num_levels: int,
    num_types: int,
    duration_s: float,
    command: tuple[float, float, float],
    device: str,
):
    """Cover every difficulty with a small, reusable robot batch."""
    if env.num_envs < num_types:
        raise ValueError(
            f"Use at least {num_types} robots so every terrain type is represented."
        )

    policy = load_policy(runner, checkpoint, device=device)
    terrain_types = torch.arange(env.num_envs, device=env.device) % num_types
    rows = []
    all_falls = []
    all_travel = []

    for level in range(num_levels):
        levels = torch.full(
            (env.num_envs,), level, device=env.device, dtype=torch.long
        )
        _reset_assignment(env, levels, terrain_types)
        fell, travelled = _rollout(env, policy, duration_s, command)
        rows.append(
            (
                level,
                100.0 * fell.float().mean().item(),
                travelled.mean().item(),
            )
        )
        all_falls.append(fell)
        all_travel.append(travelled)

    falls = torch.cat(all_falls)
    travel = torch.cat(all_travel)
    overall = (100.0 * falls.float().mean().item(), travel.mean().item())
    return rows, overall


def show_results(rows, overall, title: str) -> None:
    """Render metrics as rows and difficulty levels as columns."""
    from IPython.display import Markdown, display

    headings = [f"level {level}" for level, _, _ in rows] + ["all"]
    falls = [f"{value:.0f}%" for _, value, _ in rows] + [f"{overall[0]:.0f}%"]
    travel = [f"{value:.2f} m" for _, _, value in rows] + [f"{overall[1]:.2f} m"]
    table = [
        f"**{title} ({len(rows)} difficulty levels)**",
        "",
        "| metric | " + " | ".join(headings) + " |",
        "|---|" + "---:|" * len(headings),
        "| falls | " + " | ".join(falls) + " |",
        "| travel | " + " | ".join(travel) + " |",
    ]
    display(Markdown("\n".join(table)))


def evaluate_task(
    task: str,
    checkpoint: Path,
    num_envs: int = 8,
    duration_s: float = 6.0,
    command: tuple[float, float, float] = (0.8, 0.0, 0.0),
    device: str = "cuda",
):
    """Evaluate every difficulty using a reusable batch of robots."""
    env = runner = None
    try:
        cfg = load_env_cfg(task, play=True)
        cfg.curriculum = {}
        cfg.scene.terrain.spawn_jitter = 0.0
        cfg.episode_length_s = duration_s
        cfg.seed = 0
        generator = cfg.scene.terrain.terrain_generator
        num_levels = generator.num_rows
        num_types = len(generator.sub_terrains)
        cfg.scene.num_envs = num_envs

        env = GenesisManagerBasedRlEnv(cfg=cfg)
        runner, _ = make_runner(env, task, device=device)
        return _evaluate_batched_env(
            env,
            runner,
            checkpoint,
            num_levels,
            num_types,
            duration_s,
            command,
            device,
        )
    finally:
        if env is not None:
            env.close()
        runner = env = None
        clear_device_cache()


class EvaluationSession:
    """An eight-robot evaluation scene with a fixed overview camera."""

    def __init__(
        self,
        task: str,
        num_envs: int = 8,
        device: str = "cuda",
        width: int = 900,
        height: int = 540,
    ) -> None:
        self.task = task
        self.num_envs = num_envs
        self.device = device

        cfg = load_env_cfg(task, play=True)
        cfg.curriculum = {}
        cfg.scene.terrain.spawn_jitter = 0.0
        cfg.episode_length_s = 30.0
        cfg.seed = 0
        cfg.scene.terrain.debug_vis = False

        generator = cfg.scene.terrain.terrain_generator
        self.terrain_names = list(generator.sub_terrains)
        self.num_levels = generator.num_rows
        self.num_types = len(self.terrain_names)
        if num_envs < self.num_types:
            raise ValueError(
                f"num_envs must be at least {self.num_types} to cover every terrain type."
            )
        cfg.scene.num_envs = num_envs

        self.view_level = self.num_levels - 1
        self.view_type = self.terrain_names.index("real_stairs")

        grid_x = generator.size[0] * self.num_levels
        grid_y = generator.size[1] * self.num_types
        terrain_span = max(grid_x, grid_y)
        camera_distance = terrain_span / 3.0
        self.camera_pos = (
            -camera_distance,
            -camera_distance,
            0.9 * camera_distance,
        )
        self.camera_lookat = (0.0, 0.0, 0.0)

        self.viewer = NotebookViewer(
            NotebookViewCfg(
                width=width,
                height=height,
                fov=55,
                pos=self.camera_pos,
                lookat=self.camera_lookat,
                follow=False,
                follow_column=None,
                follow_env=0,
                every=4,
                far=max(150.0, 4.0 * terrain_span),
            )
        )
        self.env = GenesisManagerBasedRlEnv(cfg=cfg, pre_build_hooks=[self.viewer])
        self.runner, _ = make_runner(self.env, task, device=device)

    def summary(self) -> str:
        return (
            f"evaluation scene: {self.num_envs} robots; "
            f"{self.num_levels} levels × {self.num_types} types run in batches\n"
            f"camera: fixed overview at {self.camera_pos}, looking at "
            f"{self.camera_lookat}\n"
            f"playback start: level {self.view_level}, "
            f"{self.terrain_names[self.view_type]}"
        )

    def evaluate(
        self,
        checkpoint: Path,
        duration_s: float = 6.0,
        command: tuple[float, float, float] = (0.8, 0.0, 0.0),
    ):
        return _evaluate_batched_env(
            self.env,
            self.runner,
            checkpoint,
            self.num_levels,
            self.num_types,
            duration_s,
            command,
            self.device,
        )

    def show_results(self, rows, overall, title: str) -> None:
        show_results(rows, overall, title)

    def watch(
        self,
        checkpoint: Path,
        label: str,
        duration_s: float = 30.0,
        command: tuple[float, float, float] = (0.8, 0.0, 0.0),
    ) -> None:
        policy = load_policy(self.runner, checkpoint, device=self.device)
        _reset_assignment(self.env, self.view_level, self.view_type)
        steps = round(duration_s / self.env.step_dt)
        print(
            f"{label} — {duration_s:.0f} s, difficulty {self.view_level}, "
            f"terrain {self.terrain_names[self.view_type]}, fixed overview camera"
        )
        self.viewer.widget = None
        self.viewer.show()
        self.viewer.play(policy, steps=steps, command=command)

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
        self.runner = self.env = self.viewer = None
        clear_device_cache()
