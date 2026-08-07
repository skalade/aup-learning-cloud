# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Real-time (RTC-style) MolmoAct2 x LIBERO demo for FINE-TUNED checkpoints.

This is the REAL-TIME sibling of interactive_server_ft.py. The clean (chunk-replay) ft
demo freezes the world during each model forward and fast-replays the chunk; great for a
clean rollout, but it hides planner latency. Here we decouple planning from the simulator
so the robot moves continuously (RTC-style "plan ahead while executing"):

  * a SIM thread owns the MuJoCo env and steps it at wall-clock RT_HZ (LIBERO's control
    rate). Each tick it consumes one action from the ActivePlan; if the plan is empty (the
    planner is still thinking) it applies a HOLD action so the robot stays put.
  * a PLANNER thread runs the policy forward to produce the next action chunk. It never
    touches the env (MuJoCo is not thread-safe), only the model + tensors.

Two things differ from the shipped-checkpoint real-time server (interactive_server_rt.py):
  1. Policy loading: we play a LeRobot-format FINE-TUNED checkpoint via `--policy.path`
     (processor + normalization stats restored from the checkpoint, no norm_tag needed),
     on the trainable allenai/lerobot@molmoact2-policy stack in /opt/train-venv. The
     bf16 inference dtype is set on the config (model_dtype=bfloat16), so no dtype monkey
     patch is needed on this path. If POLICY_PATH is unset we fall back to a released HF
     checkpoint via --policy.checkpoint_path (+ optional apply_dtype_patch for bf16 speed).
  2. RT_STITCH defaults to `blend` here (this workshop showcases the RTC "real-time
     chunking" motion), whereas the research server defaults to `hold`.

RTC blending (RT_STITCH=blend): the planner starts the next chunk while the current one is
still executing (RT_REPLAN_AT), then we freeze the steps reality already executed during
inference and ramp-blend the overlap (RT_BLEND_STEPS) onto the new chunk so motion stays
continuous instead of pausing to think. The gripper (absolute) switches with hysteresis
(RT_GRIPPER_HYST). RT_STITCH=hold keeps the plain stop-and-decide demo.

HOLD is safe because LIBERO uses an OSC_POSE controller with control_delta=True: the 7-D
action is [dx,dy,dz, droll,dpitch,dyaw, gripper]. Zero on the 6 delta dims => target ==
current pose => the impedance controller holds position. The gripper dim is absolute, so
HOLD keeps the LAST gripper command.

Env: POLICY_PATH (LeRobot fine-tuned ckpt dir/Hub repo) OR CKPT (released HF repo),
NORM_TAG (libero, only for the CKPT fallback), SUITE, TASK_ID, SEED, PORT (8080),
VIEW_RES (720), VIDEO_RES (600), NUM_STEPS, RT_HZ (20), RT_STITCH (blend),
RT_REPLAN_AT (-1=auto chunk//2), RT_BLEND_STEPS (4), RT_GRIPPER_HYST (0.4),
RT_LOOKAHEAD (0, hold mode only), RT_MAX_STEPS (1200), OUT_DIR (/outputs).
Open http://localhost:PORT (remote box: ssh -L PORT:localhost:PORT <host>).
"""
import io
import json
import os
import random
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import collections as _collections


def _find_action_queue(policy):
    """Return the deque holding the remaining action chunk after select_action(), or None.

    lerobot branches name it differently (_action_queue / _action_queues[i] / ...), so we
    probe known names first, then fall back to scanning the policy's instance dict for any
    non-empty deque. Robust across the 0.5.x line used by the AUP finetuning image."""
    for name in ("_action_queue", "_action_queues", "_queues", "action_queue", "_queue"):
        q = getattr(policy, name, None)
        if q is None:
            continue
        if isinstance(q, (list, tuple)):
            if q and hasattr(q[0], "popleft"):
                return q[0]
        elif hasattr(q, "popleft"):
            return q
    for v in vars(policy).values():
        if isinstance(v, _collections.deque):
            return v
        if isinstance(v, (list, tuple)) and v and isinstance(v[0], _collections.deque):
            return v[0]
    return None


SUITE = os.environ.get("SUITE", "libero_object")
TASK_ID = int(os.environ.get("TASK_ID") or "3")
SEED = int(os.environ.get("SEED") or "1000")
POLICY_PATH = os.environ.get("POLICY_PATH", "").strip()
CKPT = os.environ.get("CKPT", "allenai/MolmoAct2-LIBERO").strip()  # fallback (non-Think)
NORM_TAG = os.environ.get("NORM_TAG", "libero")
PORT = int(os.environ.get("PORT") or "8080")
VIEW_RES = int(os.environ.get("VIEW_RES") or "720")    # live viewport (RT favors smoothness)
VIDEO_RES = int(os.environ.get("VIDEO_RES") or "600")  # saved debug video (kept small)
RT_HZ = float(os.environ.get("RT_HZ") or "20")         # wall-clock control rate
RT_LOOKAHEAD = int(os.environ.get("RT_LOOKAHEAD") or "0")   # hold mode: replan when buffer <= this
RT_MAX_STEPS = int(os.environ.get("RT_MAX_STEPS") or "1200")  # RT runs in wall time; holds burn budget
# --- RTC-style "plan ahead while executing" knobs (default = blend for this workshop) ---
RT_STITCH = os.environ.get("RT_STITCH", "blend").strip().lower()   # blend | hold
RT_REPLAN_AT = int(os.environ.get("RT_REPLAN_AT") or "-1")  # blend: replan when remaining <= this (-1 => auto chunk//2)
RT_BLEND_STEPS = int(os.environ.get("RT_BLEND_STEPS") or "4")  # blend: ramp window W over delta dims
RT_GRIPPER_HYST = float(os.environ.get("RT_GRIPPER_HYST") or "0.4")  # blend: min change to flip absolute gripper
OUT_DIR = os.environ.get("OUT_DIR", "/outputs")
SUITES = ["libero_object", "libero_goal", "libero_spatial", "libero_10"]
DT = 1.0 / RT_HZ

# ---- shared state ------------------------------------------------------------
STATE = {
    "mode": "loading",        # loading | idle | running
    "instruction": "",
    "scene_task": "",
    "suite": SUITE, "task_id": TASK_ID,
    "objects": [],
    "step": 0, "infer_ms": 0.0, "success": False,
    "holding": False, "buffer": 0, "hold_pct": 0.0,
    "stitch": RT_STITCH, "mean_accel": 0.0, "max_jerk": 0.0,
    "status": "starting", "frame": None, "video_url": "",
}
LOCK = threading.Lock()
PENDING = {"action": None, "instruction": ""}
EVENT = threading.Event()
STOP = {"flag": False}


def _font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _encode_jpeg(rgb, quality=85):
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb)).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _set_frame(rgb):
    with LOCK:
        STATE["frame"] = _encode_jpeg(rgb)


def _banner_frame(rgb, text, size, tag=""):
    """Downscale to `size`; top banner shows the command + an optional status tag
    (THINKING while holding, BLENDING while stitching a fresh chunk in)."""
    img = Image.fromarray(np.ascontiguousarray(rgb)).resize((size, size), Image.BILINEAR)
    bh = max(40, size // 12)
    bh += bh % 2
    canvas = Image.new("RGB", (size, size + bh), (15, 15, 18))
    canvas.paste(img, (0, bh))
    d = ImageDraw.Draw(canvas)
    f = _font(max(14, size // 30))
    cap = 44 if tag else 60  # leave room on the right for the status tag
    msg = text if len(text) <= cap else text[:cap - 3] + "..."
    d.text((10, bh // 2), msg, fill=(240, 240, 240), font=f, anchor="lm")
    if tag:
        color = (255, 180, 80) if tag == "THINKING" else (120, 210, 140)
        d.text((size - 10, bh // 2), tag, fill=color, font=f, anchor="rm")
    return np.asarray(canvas)


# ---- RTC-style chunk stitching (action level) --------------------------------
# Actions are env-space 7-D [dx,dy,dz, droll,dpitch,dyaw, gripper]: the 6 motion
# dims are OSC deltas (relative), the gripper is absolute. We index every plan by an
# absolute "motion step" counter (number of real, non-hold actions executed so far),
# so two overlapping chunks can be aligned: a chunk planned from the obs at motion
# step `base` has actions[j] intended for motion step base+j.
class ActivePlan:
    """A list of env-space actions plus a consume cursor, indexed by motion step.

    Invariant maintained by the SIM loop: base + cursor == current motion-step counter,
    i.e. actions[cursor] is the action for the step about to execute.
    """

    def __init__(self, actions, base):
        self.actions = [np.ascontiguousarray(a, dtype=np.float32) for a in actions]
        self.base = int(base)
        self.cursor = 0

    def remaining(self):
        return len(self.actions) - self.cursor

    def next_action(self):
        if self.cursor < len(self.actions):
            a = self.actions[self.cursor]
            self.cursor += 1
            return a
        return None


def _blend_gripper(old_g, new_g, hyst):
    """Absolute gripper with hysteresis: only flip when the command moves enough."""
    if abs(float(new_g) - float(old_g)) < hyst:
        return float(old_g)
    return float(new_g)


def splice(old, new_actions, drop, blend_steps, gripper_hyst):
    """Freeze-and-inpaint at the action level (RTC-style, no model changes).

    `new_actions[drop + k]` aligns with the old plan's `actions[old.cursor + k]` (both are
    motion step `old.base + old.cursor + k`). We discard the `drop` steps of the new chunk
    that reality already executed since it was planned, then ramp from the old plan to the
    new one over `blend_steps` on the delta dims so there is no velocity jump at the splice.
    The gripper (absolute) switches with hysteresis.
    """
    new_actions = [np.ascontiguousarray(a, dtype=np.float32) for a in new_actions]
    nnew = len(new_actions)
    drop = int(max(0, min(drop, nnew - 1)))  # always keep at least the last action
    base = old.base + old.cursor
    last_g = float(old.actions[old.cursor][0, 6]) if old.remaining() > 0 else None
    merged = []
    for k in range(nnew - drop):
        new_a = new_actions[drop + k].copy()
        oi = old.cursor + k
        old_a = old.actions[oi] if oi < len(old.actions) else None
        if blend_steps > 0 and k < blend_steps and old_a is not None:
            alpha = (k + 1) / (blend_steps + 1)  # ramps old->new, never a hard jump
            new_a[:, :6] = (1.0 - alpha) * old_a[:, :6] + alpha * new_a[:, :6]
        if last_g is not None:
            g = _blend_gripper(last_g, float(new_a[0, 6]), gripper_hyst)
            new_a[:, 6] = g
            last_g = g
        merged.append(new_a)
    return ActivePlan(merged, base)


def smoothness_metrics(applied):
    """Finite-difference accel/jerk of the executed per-step deltas (motion dims).

    `applied` is a list of executed 7-D actions. delta ~ per-step displacement (velocity
    proxy), so accel = diff(delta), jerk = diff(accel). Returns mean |accel| and max |jerk|
    over the run (chunk-boundary spikes dominate).
    """
    if len(applied) < 3:
        return 0.0, 0.0
    arr = np.asarray(applied, dtype=np.float64)[:, :6]
    accel = np.diff(arr, axis=0)
    jerk = np.diff(accel, axis=0)
    mean_accel = float(np.mean(np.linalg.norm(accel, axis=1)))
    max_jerk = float(np.max(np.linalg.norm(jerk, axis=1)))
    return mean_accel, max_jerk


class Scene:
    def __init__(self, suite, task_id, env, scene_task, objects):
        self.suite, self.task_id = suite, task_id
        self.env, self.scene_task, self.objects = env, scene_task, objects

    def hi_res(self, size):
        try:
            le = self.env.envs[0]
            img = le._env.sim.render(width=size, height=size, camera_name="agentview")
            return np.asarray(img)[::-1, ::-1]
        except Exception:
            return np.asarray(self.env.call("render")[0])

    def idle_frame(self, size):
        try:
            self.env.reset(seed=SEED)
        except Exception:
            pass
        return self.hi_res(size)


# Runtime overrides applied to the loaded checkpoint config when using --policy.path.
# (lerobot's path-field filter strips inline --policy.* overrides, so we set these on the
# config object after parsing instead.) Matches interactive_server_ft.py / eval_ckpt.sh.
_PATH_OVERRIDES = {
    "device": "cuda",
    "inference_action_mode": "continuous",
    "model_dtype": "bfloat16",
    "use_amp": True,
    "enable_inference_cuda_graph": False,
}


def _fallback_policy_cli():
    """Policy argv for the FALLBACK case: a released HF checkpoint via
    --policy.checkpoint_path (draccus parses these natively). Used only when POLICY_PATH is
    unset. bf16 speed on this path relies on apply_dtype_patch (called by the launcher)."""
    args = [
        "--policy.type=molmoact2",
        f"--policy.checkpoint_path={CKPT}",
        f"--policy.norm_tag={NORM_TAG}",
        "--policy.inference_action_mode=continuous",
        "--policy.model_dtype=bfloat16",
        "--policy.use_amp=true",
        "--policy.enable_inference_cuda_graph=false",
        "--policy.device=cuda",
    ]
    if os.environ.get("NUM_STEPS"):
        args.append(f"--policy.num_inference_steps={os.environ['NUM_STEPS']}")
    return args


def engine_thread():
    import torch
    import draccus
    from lerobot.configs import parser as lrparser
    from lerobot.configs.eval import EvalPipelineConfig
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    import lerobot.scripts.lerobot_eval as ev

    ACTION = ev.ACTION
    preprocess_observation = ev.preprocess_observation

    common = [
        "--env.type=libero",
        "--env.camera_name_mapping={\"agentview_image\":\"image\",\"robot0_eye_in_hand_image\":\"wrist_image\"}",
        "--eval.batch_size=1", "--eval.n_episodes=1",
        f"--seed={SEED}", "--output_dir=/tmp/interactive_rt_ft_eval",
    ]

    def parse_eval(extra):
        """Build an EvalPipelineConfig. For a fine-tuned LeRobot checkpoint we register its
        path in lerobot's path-field registry (what `@parser.wrap()` does for --policy.path)
        so draccus loads the pretrained policy config, then apply runtime overrides on the
        config object. For the released-HF fallback we pass --policy.checkpoint_path natively."""
        if POLICY_PATH:
            lrparser._config_path_args["policy"] = POLICY_PATH  # noqa: SLF001
            cfg = draccus.parse(EvalPipelineConfig, args=list(common) + list(extra))
            for k, v in _PATH_OVERRIDES.items():
                if hasattr(cfg.policy, k):
                    setattr(cfg.policy, k, v)
            if os.environ.get("NUM_STEPS") and hasattr(cfg.policy, "num_inference_steps"):
                cfg.policy.num_inference_steps = int(os.environ["NUM_STEPS"])
        else:
            cfg = draccus.parse(EvalPipelineConfig, args=_fallback_policy_cli() + list(common) + list(extra))
        return cfg

    src = POLICY_PATH if POLICY_PATH else CKPT
    with LOCK:
        STATE["status"] = f"loading policy from {src} (first run JIT-compiles kernels) ..."
    base_cfg = parse_eval([f"--env.task={SUITE}", f"--env.task_ids=[{TASK_ID}]"])
    policy = make_policy(cfg=base_cfg.policy, env_cfg=base_cfg.env, rename_map=base_cfg.rename_map)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=base_cfg.policy, pretrained_path=base_cfg.policy.pretrained_path,
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": base_cfg.rename_map},
        },
    )
    env_pre, env_post = make_env_pre_post_processors(env_cfg=base_cfg.env, policy_cfg=base_cfg.policy)

    def make_generator():
        """The molmoact2 flow-matching head samples noise; a per-rollout generator makes it
        reproducible. lerobot_eval exposes a helper on some branches; fall back gracefully."""
        mk = getattr(ev, "_make_rollout_action_generator", None)
        if mk is not None:
            try:
                return mk(policy, [SEED])
            except Exception:  # noqa: BLE001
                pass
        try:
            dev = str(policy.config.device)
            return torch.Generator(device=dev if dev.startswith("cuda") else "cpu").manual_seed(SEED)
        except Exception:  # noqa: BLE001
            return None

    def build_scene(suite, task_id):
        cfg = parse_eval([f"--env.task={suite}", f"--env.task_ids=[{task_id}]"])
        envs = make_env(cfg.env, n_envs=1, use_async_envs=False, trust_remote_code=cfg.trust_remote_code)
        env = envs[suite][task_id]
        for e in env.envs:
            try:
                e._max_episode_steps = RT_MAX_STEPS  # RT runs in wall-clock time; give it room
            except Exception:
                pass
        env.reset(seed=SEED)
        le = env.envs[0]
        scene_task = getattr(le, "task_description", "") or ""
        try:
            objs = [getattr(o, "name", str(o)).replace("_1", "").replace("_", " ")
                    for o in le._env.env.objects]
        except Exception:
            objs = list(getattr(le._env, "obj_of_interest", []))
        return Scene(suite, task_id, env, scene_task, objs)

    def show_idle(sc):
        with LOCK:
            STATE.update(mode="idle", status="idle - send an instruction", step=0,
                         success=False, suite=sc.suite, task_id=sc.task_id,
                         scene_task=sc.scene_task, objects=sc.objects,
                         instruction="", video_url="", holding=False, buffer=0, hold_pct=0.0)
        STATE.pop("_video_path", None)
        _set_frame(sc.idle_frame(VIEW_RES))

    os.makedirs(os.path.join(OUT_DIR, "interactive_rt_ft"), exist_ok=True)
    scene = build_scene(SUITE, TASK_ID)

    # ---- one planning step: model forward -> full env-space chunk -------------
    def plan_chunk(proc_obs, generator):
        t0 = time.perf_counter()
        with torch.inference_mode():
            try:
                a0 = policy.select_action(proc_obs, generator=generator)
            except TypeError:
                a0 = policy.select_action(proc_obs)  # branch without a generator kwarg
            try:
                torch.cuda.synchronize()  # ensure the forward finished before we time it
            except Exception:  # noqa: BLE001
                pass
        infer_wall = time.perf_counter() - t0
        raw = [a0]
        # Drain the rest of the chunk the policy just queued (attribute name varies by
        # lerobot version: _action_queues[i] on some, a single _action_queue on others).
        q = _find_action_queue(policy)
        while q:
            a = q.popleft()
            if a.ndim == 1:
                a = a.unsqueeze(0)
            raw.append(a.to(device=a0.device, dtype=torch.float32))
        out = []
        for a in raw:
            pa = postprocessor(a)
            pa = env_post({ACTION: pa})[ACTION]
            out.append(np.asarray(pa.to("cpu").numpy()))
        # molmoact2-policy select_action doesn't always populate _last_model_inference_s,
        # so fall back to the measured wall-clock forward time (keeps the RT latency readout
        # meaningful across lerobot branches).
        infer_s = float(getattr(policy, "_last_model_inference_s", 0.0)) or infer_wall
        print(f"[plan] infer={infer_s:.2f}s chunk={len(out)}", flush=True)
        return out, infer_s

    def run_command(sc, instruction):
        blend = RT_STITCH == "blend"
        with LOCK:
            STATE.update(mode="running", instruction=instruction, step=0, success=False,
                         status=f"running: {instruction}", video_url="", holding=False,
                         buffer=0, hold_pct=0.0, stitch=RT_STITCH, mean_accel=0.0, max_jerk=0.0)
        STOP["flag"] = False
        policy.reset()
        obs, _ = sc.env.reset(seed=SEED)
        generator = make_generator()
        max_steps = sc.env.call("_max_episode_steps")[0]

        buf_lock = threading.Lock()
        shared = {"obs": obs, "done": False, "last_gripper": 0.0,
                  "hold_steps": 0, "total_steps": 0, "motion_steps": 0,
                  "plan": None, "chunk_len": 0, "blend_until": -1}

        def replan_threshold():
            if not blend:
                return RT_LOOKAHEAD  # hold mode: refill only when (nearly) empty
            if RT_REPLAN_AT >= 0:
                return RT_REPLAN_AT
            return max(1, shared["chunk_len"] // 2) if shared["chunk_len"] else 6

        def planner():
            while not STOP["flag"] and not shared["done"]:
                with buf_lock:
                    plan = shared["plan"]
                    rem = plan.remaining() if plan is not None else 0
                if rem > replan_threshold():
                    time.sleep(0.005)
                    continue
                with LOCK:
                    cur = shared["obs"]
                plan_base_motion = shared["motion_steps"]
                try:
                    proc = preprocess_observation(cur)
                    proc["task"] = [instruction for _ in range(sc.env.num_envs)]
                    proc = env_pre(proc)
                    proc = preprocessor(proc)
                    chunk, infer_s = plan_chunk(proc, generator)
                except Exception as e:  # noqa: BLE001
                    import traceback; traceback.print_exc()
                    print("plan error:", e, flush=True)
                    time.sleep(0.02)
                    continue
                with buf_lock:
                    shared["chunk_len"] = len(chunk)
                    cur_motion = shared["motion_steps"]
                    old = shared["plan"]
                    if blend and old is not None and old.remaining() > 0:
                        drop = cur_motion - plan_base_motion  # steps reality already did
                        shared["plan"] = splice(old, chunk, drop, RT_BLEND_STEPS, RT_GRIPPER_HYST)
                        shared["blend_until"] = shared["total_steps"] + RT_BLEND_STEPS
                    else:
                        drop = cur_motion - plan_base_motion if blend else 0
                        d = int(max(0, min(drop, len(chunk) - 1)))
                        shared["plan"] = ActivePlan(chunk[d:], cur_motion)
                with LOCK:
                    if infer_s > 0:
                        STATE["infer_ms"] = round(infer_s * 1000.0, 0)

        pth = threading.Thread(target=planner, daemon=True)
        pth.start()

        frames = []
        applied = []
        success = False
        next_t = time.perf_counter()
        step = 0
        while not STOP["flag"] and not shared["done"] and step < max_steps:
            with buf_lock:
                plan = shared["plan"]
                action = plan.next_action() if plan is not None else None
                if action is not None:
                    shared["motion_steps"] += 1
                depth_buf = plan.remaining() if plan is not None else 0
            holding = action is None
            if holding:
                action = np.zeros((sc.env.num_envs, 7), dtype=np.float32)
                action[:, 6] = shared["last_gripper"]
                shared["hold_steps"] += 1
            else:
                shared["last_gripper"] = float(action[0, 6])
            shared["total_steps"] += 1
            applied.append(np.asarray(action[0], dtype=np.float32).copy())

            obs, reward, terminated, truncated, info = sc.env.step(action)
            with LOCK:
                shared["obs"] = obs
            if "final_info" in info:
                try:
                    success = bool(np.asarray(info["final_info"]["is_success"]).any())
                except Exception:
                    pass
            done = bool(np.any(terminated) or np.any(truncated))

            blending = (not holding) and shared["total_steps"] <= shared["blend_until"]
            tag = "THINKING" if holding else ("BLENDING" if blending else "")
            rgb = sc.hi_res(VIEW_RES)
            _set_frame(rgb)
            frames.append(_banner_frame(rgb, instruction, VIDEO_RES, tag))
            step += 1
            hp = 100.0 * shared["hold_steps"] / max(1, shared["total_steps"])
            ma, mj = smoothness_metrics(applied[-60:])  # rolling live readout
            with LOCK:
                STATE.update(step=step, holding=holding, buffer=depth_buf, hold_pct=round(hp, 0),
                             mean_accel=round(ma, 4), max_jerk=round(mj, 4),
                             status=("thinking (holding pose)" if holding
                                     else ("blending chunk" if blending else f"running: {instruction}")))
            if done:
                shared["done"] = True

            next_t += DT
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.perf_counter()  # fell behind real-time; resync

        shared["done"] = True
        STOP["flag"] = True
        pth.join(timeout=5.0)

        ma, mj = smoothness_metrics(applied)  # final whole-run metrics
        url = ""
        if frames:
            ts = datetime.now().strftime("%H%M%S")
            name = f"interactive_rt_ft/{ts}_{sc.suite}_{sc.task_id}_{RT_STITCH}_{'ok' if success else 'run'}.mp4"
            path = os.path.join(OUT_DIR, name)
            try:
                import imageio
                with imageio.get_writer(path, fps=int(RT_HZ), codec="libx264", quality=8,
                                        macro_block_size=1, output_params=["-pix_fmt", "yuv420p"]) as w:
                    for fr in frames:
                        w.append_data(fr)
                url = "video?ts=" + ts
                with LOCK:
                    STATE["_video_path"] = path
            except Exception as e:  # noqa: BLE001
                print("video save failed:", e, flush=True)
            try:
                np.save(os.path.splitext(path)[0] + "_actions.npy", np.asarray(applied, dtype=np.float32))
            except Exception as e:  # noqa: BLE001
                print("action dump failed:", e, flush=True)

        hp = 100.0 * shared["hold_steps"] / max(1, shared["total_steps"])
        print(f"[run] stitch={RT_STITCH} steps={shared['total_steps']} hold%={hp:.0f} "
              f"mean_accel={ma:.4f} max_jerk={mj:.4f} success={success}", flush=True)
        with LOCK:
            STATE.update(mode="idle", success=success, video_url=url, holding=False,
                         mean_accel=round(ma, 4), max_jerk=round(mj, 4),
                         status=("success" if success else ("stopped" if STOP["flag"] else "done")))
        _set_frame(sc.idle_frame(VIEW_RES))

    # Warm up flash/JIT kernels NOW (the first model forward compiles them and can take
    # ~20s); otherwise that one-time stall would burn the first real-time episode.
    with LOCK:
        STATE["status"] = "warming up GPU kernels (one-time JIT) ..."
    try:
        wobs, _ = scene.env.reset(seed=SEED)
        wproc = preprocess_observation(wobs)
        wproc["task"] = [scene.scene_task or "pick up the object" for _ in range(scene.env.num_envs)]
        wproc = env_pre(wproc)
        wproc = preprocessor(wproc)
        _t = time.perf_counter()
        plan_chunk(wproc, make_generator())
        policy.reset()
        print(f"[warmup] done in {time.perf_counter()-_t:.1f}s", flush=True)
    except Exception as e:  # noqa: BLE001
        import traceback; traceback.print_exc()
        print("warmup failed:", e, flush=True)
    show_idle(scene)

    # main control loop
    while True:
        EVENT.wait()
        EVENT.clear()
        with LOCK:
            action, instruction = PENDING["action"], PENDING["instruction"]
            PENDING["action"] = None
        if action == "randomize":
            STOP["flag"] = True
            suite = random.choice(SUITES)
            tid = random.randint(0, 9)
            with LOCK:
                STATE["status"] = f"loading scene {suite}/{tid} ..."
            try:
                new_scene = build_scene(suite, tid)
            except Exception as e:  # noqa: BLE001
                with LOCK:
                    STATE["status"] = f"scene build failed: {e}"
                continue
            try:
                scene.env.close()
            except Exception:
                pass
            scene = new_scene
            show_idle(scene)
        elif action == "run":
            run_command(scene, instruction)


# ---------------------------------------------------------------------------
PAGE = b"""<!doctype html><html><head><meta charset=utf-8>
<title>MolmoAct2 x LIBERO (fine-tuned, REAL-TIME)</title>
<style>
 body{background:#0f1012;color:#e8e8ea;font-family:system-ui,sans-serif;margin:0;padding:20px}
 .wrap{max-width:760px;margin:0 auto}
 h1{font-size:18px;font-weight:600;margin:0 0 4px}
 .sub{font-size:12px;color:#8b94a0;margin:0 0 12px}
 img{width:640px;height:auto;border-radius:10px;background:#000;display:block}
 .row{display:flex;gap:8px;margin-top:14px}
 input{flex:1;padding:11px;border-radius:8px;border:1px solid #333;background:#1b1b1f;color:#eee;font-size:15px}
 button{padding:11px 16px;border-radius:8px;border:0;color:#fff;font-size:15px;cursor:pointer}
 .send{background:#3b82f6}.stop{background:#ef4444}.rand{background:#8b5cf6}
 .meta{margin-top:12px;font-size:13px;color:#9aa3ad}
 .think{color:#ffb450;font-weight:600}
 .panel{margin-top:12px;background:#16171b;border:1px solid #26272c;border-radius:10px;padding:12px;font-size:13px}
 .chip{display:inline-block;background:#23252b;border-radius:14px;padding:3px 10px;margin:3px 4px 0 0;color:#cdd3da}
 a{color:#7aa2ff} video{width:640px;border-radius:10px;margin-top:10px;background:#000}
</style></head><body><div class=wrap>
<h1>MolmoAct2 x LIBERO - fine-tuned checkpoint, REAL-TIME</h1>
<p class=sub>The simulator runs at wall-clock speed. In <b>blend</b> mode the planner runs ahead and new chunks are stitched in (RTC-style) so motion stays continuous; in <b>hold</b> mode the robot pauses while the model thinks.</p>
<img id="sim" alt="sim">
<div class=row>
 <input id=cmd placeholder="type an instruction, then Send (resets the scene and runs it in real time)">
 <button class=send onclick=send()>Send</button>
 <button class=stop onclick=stop()>Stop</button>
 <button class=rand onclick=rnd()>Randomize</button>
</div>
<div class=meta id=meta>status: loading...</div>
<div class=panel><b id=scene>scene</b><div id=objs></div></div>
<div id=vidwrap></div>
</div>
<script>
async function send(){const v=document.getElementById('cmd').value;if(!v)return;
 await fetch('command',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:'instruction='+encodeURIComponent(v)});}
async function stop(){await fetch('stop',{method:'POST'});}
async function rnd(){await fetch('randomize',{method:'POST'});}
document.getElementById('cmd').addEventListener('keydown',e=>{if(e.key==='Enter')send();});
let lastVid='';
async function poll(){
 try{const s=await(await fetch('status')).json();
  let t='['+s.mode+'] '+s.status+' | '+s.stitch+' | step '+s.step+' | last infer '+s.infer_ms+' ms | buffer '+s.buffer+' | hold '+s.hold_pct+'% | accel '+s.mean_accel+' | jerk '+s.max_jerk;
  const m=document.getElementById('meta'); m.textContent=t; m.className='meta'+(s.holding?' think':'');
  document.getElementById('scene').textContent='Scene: '+s.suite+' / task '+s.task_id+' - "'+s.scene_task+'"';
  document.getElementById('objs').innerHTML='Objects in scene: '+(s.objects||[]).map(o=>'<span class=chip>'+o+'</span>').join('');
  const box=document.getElementById('cmd');
  if(s.mode==='idle'&&!box.value&&s.scene_task)box.value=s.scene_task;
  if(s.video_url&&s.video_url!==lastVid){lastVid=s.video_url;
   document.getElementById('vidwrap').innerHTML='<div style=\"margin-top:8px;font-size:13px;color:#9aa3ad\">last run video:</div><video controls autoplay loop src=\"'+s.video_url+'\"></video>';}
 }catch(e){}
 setTimeout(poll,500);
}
function frameTick(){var n=new Image();n.onload=function(){var im=document.getElementById('sim');if(im)im.src=n.src;requestAnimationFrame(frameTick);};n.onerror=function(){setTimeout(frameTick,200);};n.src='frame?t='+Date.now();}
frameTick();
poll();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE)
        elif path == "/status":
            with LOCK:
                s = {k: STATE[k] for k in ("mode", "status", "instruction", "scene_task",
                                           "suite", "task_id", "objects", "step", "infer_ms",
                                           "success", "holding", "buffer", "hold_pct", "video_url",
                                           "stitch", "mean_accel", "max_jerk")}
            self._send(200, "application/json", json.dumps(s).encode())
        elif path == "/video":
            with LOCK:
                p = STATE.get("_video_path")
            if p and os.path.exists(p):
                with open(p, "rb") as f:
                    self._send(200, "video/mp4", f.read())
            else:
                self._send(404, "text/plain", b"no video")
        elif path == "/frame":
            with LOCK:
                frame = STATE.get("frame")
            if frame:
                self._send(200, "image/jpeg", frame)
            else:
                self._send(404, "text/plain", b"no frame")
        elif path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with LOCK:
                        frame = STATE["frame"]
                    if frame:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.04)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/command":
            n = int(self.headers.get("Content-Length", "0"))
            instr = parse_qs(self.rfile.read(n).decode()).get("instruction", [""])[0].strip()
            if instr:
                STOP["flag"] = True
                with LOCK:
                    PENDING["action"], PENDING["instruction"] = "run", instr
                EVENT.set()
            self.send_response(204)
            self.end_headers()
        elif path == "/stop":
            STOP["flag"] = True
            self.send_response(204)
            self.end_headers()
        elif path == "/randomize":
            STOP["flag"] = True
            with LOCK:
                PENDING["action"] = "randomize"
            EVENT.set()
            self.send_response(204)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()



def _engine_guard():
    """Run the engine thread; on any unhandled error surface it in STATE so the UI shows
    'error' instead of hanging on 'loading' forever (important for the workshop)."""
    try:
        engine_thread()
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        with LOCK:
            STATE["mode"] = "error"
            STATE["status"] = f"engine failed: {e}"


def main():
    threading.Thread(target=_engine_guard, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"real-time (fine-tuned) demo on http://localhost:{PORT}  "
          f"(remote box: ssh -L {PORT}:localhost:{PORT} <host>)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
