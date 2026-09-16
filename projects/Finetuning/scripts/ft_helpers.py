# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Plumbing for the MolmoAct2 LoRA fine-tuning notebook.

The notebook keeps the *teaching* code inline (env-quieting, the exact train/eval
commands, the flow-matching action-head call). The bulky, non-teaching machinery lives
here so the cells stay readable:

  * subprocess streamers  - stream a child process into the notebook, preserving '\\r'
    so tqdm bars redraw in place; scrape (step, loss) points; hide chatty lines.
  * HF download machinery  - cache-aware prefetch with a GB heartbeat, local-asset
    staging, an input preflight, and the optional full-LIBERO route.
  * DROID video decode     - open-loop rollout on a few real DROID episodes, decoding
    only the frames we replan on and rendering GT-vs-predicted actions inline.
"""
import json
import os
import random
import re
import subprocess
import sys
import threading
import time

import numpy as np
import torch
from huggingface_hub import snapshot_download

# ---------------------------------------------------------------------------------------
# Subprocess streamers
# ---------------------------------------------------------------------------------------
# Backstop line filter: even with the notebook's warning-suppression flags, drop any stray
# ROCm/torch warning lines so the streamed subprocess output stays readable.
_NOISE = (
    "UserWarning",
    "HIPBLAS_STATUS",
    "hipblasLtMatmul",
    "Triggered internally",
    "scaled_dot_product_attention",
    "run_backward",
    "aotriton",
    "AOTRITON",
)

# LeRobot logs one metrics line per `log_freq` steps, e.g. "step:10 ... loss:1.684 grdn:.. lr:..".
# We scrape those (step, loss) points so Step 4 can draw a small loss curve. Handles the
# human-formatted step number (10, 1.5K, 2M) that LeRobot prints for large runs.
_STEP_RE = re.compile(r"\bstep:\s*([0-9.]+)\s*([KMB]?)")
_LOSS_RE = re.compile(r"\bloss:\s*([0-9.]+)")
_SUF = {"": 1.0, "K": 1e3, "M": 1e6, "B": 1e9}


def _popen(cmd, env=None, cwd=None):
    print("$ " + (cmd if isinstance(cmd, str) else " ".join(map(str, cmd))) + "\n", flush=True)
    # Binary pipe (no text=True): universal-newline mode would rewrite every '\r' to '\n' and turn
    # a single self-updating tqdm bar into one new line per tick. We decode + split ourselves.
    return subprocess.Popen(
        cmd,
        shell=isinstance(cmd, str),
        env=env,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )


def _pump(p, quiet=True, scrape=None, drop=None):
    """Stream a child process into the notebook, PRESERVING carriage returns so tqdm/progress bars
    redraw in place on ONE line (Jupyter honors '\\r'). `scrape(line)` collects data from complete
    '\\n' lines; `drop(line)` hides a complete line from the display (it is still scraped)."""
    import codecs

    dec = codecs.getincrementaldecoder("utf-8")("replace")
    buf = ""
    while True:
        chunk = p.stdout.read(4096)
        if not chunk:
            break
        buf += dec.decode(chunk)
        while True:
            i_n, i_r = buf.find("\n"), buf.find("\r")
            idxs = [i for i in (i_n, i_r) if i != -1]
            if not idxs:
                break
            cut = min(idxs) + 1
            seg, buf = buf[:cut], buf[cut:]
            if quiet and any(tok in seg for tok in _NOISE):
                continue
            full_line = seg.endswith("\n")
            if full_line and scrape:
                scrape(seg)
            if not (full_line and drop and drop(seg)):
                sys.stdout.write(seg)
                sys.stdout.flush()
    if buf and not (quiet and any(tok in buf for tok in _NOISE)):
        if scrape:
            scrape(buf)
        if not (drop and drop(buf)):
            sys.stdout.write(buf)
            sys.stdout.flush()
    p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"command failed (exit {p.returncode})")


def stream_cmd(cmd, env=None, cwd=None, quiet=True, scrape=None, drop=None):
    """Stream a subprocess into the notebook; tqdm/progress bars stay on ONE self-updating line.
    Optional `scrape`/`drop` callbacks collect from / hide complete lines (see `_pump`)."""
    _pump(_popen(cmd, env=env, cwd=cwd), quiet=quiet, scrape=scrape, drop=drop)


def run_cmd(cmd, env=None, cwd=None, quiet_warnings=True):
    """Stream a subprocess into the notebook; tqdm/progress bars stay on ONE self-updating line."""
    _pump(_popen(cmd, env=env, cwd=cwd), quiet=quiet_warnings)


def run_train(cmd, env=None, cwd=None):
    """Stream a training subprocess and return captured [(step, loss), ...]. The tqdm progress bar
    stays on one line; the chatty per-step "step:.. loss:.." metric lines are hidden from the
    display (still scraped, so the loss curve below is unaffected)."""
    pts = []

    def _scrape(line):
        ms, ml = _STEP_RE.search(line), _LOSS_RE.search(line)
        if ms and ml:
            pts.append((float(ms.group(1)) * _SUF[ms.group(2)], float(ml.group(1))))

    def _drop(line):
        return bool(_STEP_RE.search(line) and _LOSS_RE.search(line))

    _pump(_popen(cmd, env=env, cwd=cwd), quiet=True, scrape=_scrape, drop=_drop)
    return pts


# ---------------------------------------------------------------------------------------
# HF download machinery
# ---------------------------------------------------------------------------------------
def _hub():
    return os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")


def _repo_dir(repo_id, repo_type):
    pfx = "datasets--" if repo_type == "dataset" else "models--"
    return os.path.join(_hub(), pfx + repo_id.replace("/", "--"))


def _dir_gb(path):
    try:
        return int(subprocess.check_output(["du", "-sb", path], stderr=subprocess.DEVNULL).split()[0]) / 1e9
    except Exception:
        return 0.0


def _fully_cached(repo_id, repo_type):
    # True only if every file is already present (no network) -> instant, no re-download.
    try:
        snapshot_download(repo_id=repo_id, repo_type=repo_type, local_files_only=True)
        return True
    except Exception:
        return False


def _heartbeat(repo_id, repo_type, stop):
    d, last = _repo_dir(repo_id, repo_type), _dir_gb(_repo_dir(repo_id, repo_type))
    while not stop.wait(3):
        cur = _dir_gb(d)
        print(f"      ...{repo_id}: {cur:.2f} GB on disk (+{max(cur - last, 0):.2f} GB/3s)", flush=True)
        last = cur


def prefetch(repo_id, repo_type, tries=5):
    """Download `repo_id` into the HF cache (idempotent). Skips instantly if fully cached; shows a
    plain GB heartbeat while pulling (VERBOSE_DOWNLOAD=1); retries through network hiccups."""
    verbose_dl = os.environ.get("VERBOSE_DOWNLOAD", "1") == "1"
    if _fully_cached(repo_id, repo_type):
        print(f"    CACHED  [{repo_type}] {repo_id}  ({_dir_gb(_repo_dir(repo_id, repo_type)):.2f} GB) - skipping")
        return
    print(f"    FETCH   [{repo_type}] {repo_id}  - downloading into HF cache", flush=True)
    for n in range(1, tries + 1):
        stop, th = threading.Event(), None
        if verbose_dl:
            th = threading.Thread(target=_heartbeat, args=(repo_id, repo_type, stop), daemon=True)
            th.start()
        try:
            t0 = time.time()
            snapshot_download(repo_id=repo_id, repo_type=repo_type)
            stop.set()
            if th:
                th.join(timeout=1)
            print(f"    DONE    [{repo_type}] {repo_id}  ({_dir_gb(_repo_dir(repo_id, repo_type)):.2f} GB in {time.time() - t0:.0f}s)")
            return
        except Exception as e:
            stop.set()
            if th:
                th.join(timeout=1)
            print(f"      (network hiccup on {repo_id}, retry {n}/{tries}: {str(e)[:80]}; resuming cached bytes)")
            if n == tries:
                raise
            time.sleep(5)


def _stage_from_dir(assets_dir):
    """Copy a local resources dir (base ckpt + dataset under hf_hub/, plus the reference checkpoint)
    into the HF cache + REFERENCE_POLICY so prefetch() finds everything cached. Idempotent."""
    import shutil

    if not assets_dir or not os.path.isdir(assets_dir):
        return
    hub = _hub()
    hub_src = os.path.join(assets_dir, "hf_hub")
    if os.path.isdir(hub_src):
        os.makedirs(hub, exist_ok=True)
        for _name in sorted(os.listdir(hub_src)):
            _s, _d = os.path.join(hub_src, _name), os.path.join(hub, _name)
            if os.path.isdir(_s) and not os.path.exists(_d):
                print(f"    STAGE   {_name} -> HF cache", flush=True)
                shutil.copytree(_s, _d, symlinks=True)
    _ref_src = os.path.join(assets_dir, "checkpoints", "reference", "pretrained_model")
    _ref_dst = os.environ.get("REFERENCE_POLICY", os.path.expanduser("~/checkpoints/reference/pretrained_model"))
    if os.path.isdir(_ref_src) and not os.path.isdir(_ref_dst):
        print(f"    STAGE   reference checkpoint -> {_ref_dst}", flush=True)
        os.makedirs(os.path.dirname(_ref_dst), exist_ok=True)
        shutil.copytree(_ref_src, _ref_dst, symlinks=True)


def stage_assets():
    """If ASSETS_DIR is set, stage the base checkpoint + dataset + fine-tuned checkpoint from local
    storage into the HF cache (no Hub re-download). Otherwise the image-baked cache is used as-is."""
    assets_dir = os.environ.get("ASSETS_DIR", "").strip()
    if assets_dir:
        print(f"== staging assets from ASSETS_DIR={assets_dir} (no Hub re-download) ==")
        _stage_from_dir(assets_dir)
        print("== asset staging done ==\n")
    else:
        print(f"== assets read from the image-baked cache: HF_HOME={os.environ.get('HF_HOME', '')} (no staging needed) ==\n")


def preflight_inputs(base_ckpt, dataset_repo):
    """Confirm every input (base checkpoint, dataset, optional policy) is found BEFORE the long
    model load / training, so a missing asset fails fast with a clear message."""
    print("== preflight: inputs the notebook needs ==")
    _needed = [(base_ckpt, "model"), (dataset_repo, "dataset")]
    _pol = os.environ.get("POLICY_PATH") or os.environ.get("REFERENCE_POLICY", "")
    for _rid, _rt in _needed:
        _d = _repo_dir(_rid, _rt)
        if _fully_cached(_rid, _rt):
            print(f"  [ok]      {_rt:7s} {_rid}  CACHED ({_dir_gb(_d):.2f} GB)")
        else:
            print(f"  [missing] {_rt:7s} {_rid}  will download")
    if _pol:
        _ok = os.path.isdir(_pol) and os.path.exists(os.path.join(_pol, "config.json"))
        print(f"  [{'ok' if _ok else 'n/a'}]      policy  {_pol}  {'found' if _ok else '(not staged yet)'}")
    print("== end preflight ==\n")


def fetch_full_libero(dataset_repo):
    """EXTENDED ROUTE (USE_FULL_LIBERO=1): pull the COMPLETE LIBERO dataset from the Hub (needs
    internet; ~33 GB) and repoint the LeRobot dataset home at it, replacing the staged subset.
    Because the subset reuses each file's real content hash, only missing files download."""
    import shutil
    import huggingface_hub.constants as _hc

    print("USE_FULL_LIBERO=1 -> fetching the COMPLETE LIBERO dataset from the Hub (large; needs internet) ...", flush=True)
    _prev_off = _hc.HF_HUB_OFFLINE
    _hc.HF_HUB_OFFLINE = False  # this flag is captured at import time; flip it to actually reach the Hub
    os.environ["HF_HUB_OFFLINE"] = "0"
    try:
        _full = snapshot_download(repo_id=dataset_repo, repo_type="dataset", local_files_only=False)
    finally:
        _hc.HF_HUB_OFFLINE = _prev_off
        os.environ["HF_HUB_OFFLINE"] = "1" if _prev_off else "0"
    # Point the LeRobot dataset home at the full snapshot (replaces the staged-subset link) so Step 4
    # trains on the complete dataset.
    _lr_home = os.environ.get(
        "HF_LEROBOT_HOME",
        os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "lerobot"),
    )
    _lr = os.path.join(_lr_home, *dataset_repo.split("/"))
    os.makedirs(os.path.dirname(_lr), exist_ok=True)
    if os.path.islink(_lr) or os.path.isfile(_lr):
        os.remove(_lr)
    elif os.path.isdir(_lr):
        shutil.rmtree(_lr)
    os.symlink(_full, _lr)
    print(f"    full LIBERO dataset ready ({_dir_gb(_full):.1f} GB) -> {_lr}")


# ---------------------------------------------------------------------------------------
# DROID video decode + open-loop rollout
# ---------------------------------------------------------------------------------------
def _grab_frames(path, indices):
    """Decode the specific source frames we replan on (nearest-match fallback)."""
    import av

    want = sorted({int(i) for i in indices})
    if not want:
        return {}
    container = av.open(path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    rate, tb = float(stream.average_rate), stream.time_base
    try:
        container.seek(max(int((want[0] / rate) / tb), 0), stream=stream, backward=True, any_frame=False)
    except Exception:
        container.seek(0)
    out, remaining = {}, set(want)
    for frame in container.decode(stream):
        if frame.pts is None:
            continue
        idx = int(round(float(frame.pts * tb) * rate))
        if idx in remaining:
            out[idx] = frame.to_ndarray(format="rgb24")
            remaining.discard(idx)
        elif idx > want[-1]:
            break
        if not remaining:
            break
    container.close()
    if remaining and out:
        got = sorted(out)
        for idx in list(remaining):
            out[idx] = out[min(got, key=lambda g: abs(g - idx))]
    return out


def _decode_clip(path, start_idx, n_frames):
    """Decode n contiguous frames from start_idx for the episode video."""
    import av

    container = av.open(path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    rate, tb = float(stream.average_rate), stream.time_base
    try:
        container.seek(max(int((start_idx / rate) / tb), 0), stream=stream, backward=True, any_frame=False)
    except Exception:
        container.seek(0)
    frames = []
    for frame in container.decode(stream):
        if frame.pts is None:
            continue
        if int(round(float(frame.pts * tb) * rate)) < start_idx:
            continue
        frames.append(frame.to_ndarray(format="rgb24"))
        if len(frames) >= n_frames:
            break
    container.close()
    return frames


def _run_openloop_episode(base_policy, predict_chunk, norm_tag, num_steps, out_dir):
    """Replay one random DROID episode open-loop (replan every STRIDE) and show GT-vs-pred."""
    import matplotlib.pyplot as plt  # inline backend -> plots render in the notebook, no Agg/subprocess
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from IPython.display import Video, display
    from PIL import Image

    eval_repo = os.environ.get("EVAL_REPO", "allenai/MolmoAct2-DROID-Dataset")
    stride = int(os.environ.get("STRIDE", "15"))
    view_cam = "observation.images." + os.environ.get("CAM", "exterior_1_left")
    cams = ["observation.images.exterior_1_left", "observation.images.exterior_2_left", "observation.images.wrist_left"]
    dim_names = ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]

    info_path = hf_hub_download(eval_repo, "meta/info.json", repo_type="dataset")
    info = json.load(open(info_path))
    fps, data_tmpl, video_tmpl = info["fps"], info["data_path"], info["video_path"]
    meta_ep = pq.read_table(
        hf_hub_download(eval_repo, "meta/episodes/chunk-000/file-000.parquet", repo_type="dataset")
    ).to_pydict()
    episodes = list(meta_ep["episode_index"])
    # The workshop pre-stages only a subset of the DROID dataset (kept offline, zero-copy), so
    # restrict the random pick to episodes whose data parquet AND all camera videos are present
    # in the local cache. This avoids reaching for a file that was not bundled.
    snap_root = os.path.dirname(os.path.dirname(info_path))
    _present = lambda rel: os.path.exists(os.path.join(snap_root, rel))
    available = [
        ep
        for r, ep in enumerate(episodes)
        if _present(data_tmpl.format(chunk_index=meta_ep["data/chunk_index"][r], file_index=meta_ep["data/file_index"][r]))
        and all(
            _present(
                video_tmpl.format(
                    video_key=c,
                    chunk_index=meta_ep[f"videos/{c}/chunk_index"][r],
                    file_index=meta_ep[f"videos/{c}/file_index"][r],
                )
            )
            for c in cams
        )
    ]
    if not available:
        raise RuntimeError("No pre-staged DROID episodes found in the local cache; check the workshop assets.")
    ep_env = os.environ.get("EPISODE", "")
    episode = int(ep_env) if ep_env else random.choice(available)
    mr = episodes.index(episode)
    d_chunk, d_file = meta_ep["data/chunk_index"][mr], meta_ep["data/file_index"][mr]
    length = meta_ep["length"][mr]
    task = meta_ep["tasks"][mr]
    task = task[0] if isinstance(task, (list, tuple)) else task
    print(f"episode {episode}  task={task!r}  length={length}")

    # dataset_from/to_index are GLOBAL rows; filter the per-file table by episode_index.
    dt = pq.read_table(
        hf_hub_download(eval_repo, data_tmpl.format(chunk_index=d_chunk, file_index=d_file), repo_type="dataset")
    )
    ep_table = dt.filter(pc.equal(dt.column("episode_index"), episode))
    col = lambda name: ep_table.column(name).to_pylist()
    states = np.asarray(col("observation.state"), np.float32)
    gt = np.asarray(col("action"), np.float32)
    timestamps = np.asarray(col("timestamp"), np.float64)

    replan_pts = list(range(0, length, stride))
    cam_meta = {
        c: {
            "chunk": meta_ep[f"videos/{c}/chunk_index"][mr],
            "file": meta_ep[f"videos/{c}/file_index"][mr],
            "from_ts": meta_ep[f"videos/{c}/from_timestamp"][mr],
        }
        for c in cams
    }
    frames_by_cam = {}
    for c in cams:
        cm = cam_meta[c]
        vp = hf_hub_download(
            eval_repo,
            video_tmpl.format(video_key=c, chunk_index=cm["chunk"], file_index=cm["file"]),
            repo_type="dataset",
        )
        idxs = [int(round((cm["from_ts"] + timestamps[t]) * fps)) for t in replan_pts]
        grabbed = _grab_frames(vp, idxs)
        frames_by_cam[c] = {t: grabbed[int(round((cm["from_ts"] + timestamps[t]) * fps))] for t in replan_pts}

    pred = np.full_like(gt, np.nan)
    lat = []
    for t in replan_pts:
        pics = [Image.fromarray(frames_by_cam[c][t]) for c in cams]
        torch.cuda.synchronize()
        ts = time.perf_counter()
        chunk = predict_chunk(base_policy, norm_tag, pics, task, states[t], num_steps)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - ts) * 1000)
        n = min(stride, length - t, chunk.shape[0])
        pred[t : t + n] = chunk[:n]

    valid = ~np.isnan(pred).any(axis=1)
    l1 = float(np.abs(pred[valid] - gt[valid]).mean())
    mse = float(((pred[valid] - gt[valid]) ** 2).mean())
    print(f"open-loop        : L1={l1:.4f} MSE={mse:.4f}  ({np.mean(lat):.0f} ms/infer)")

    # Episode exterior view (the scene the model predicts on).
    cm = cam_meta[view_cam]
    vpath = hf_hub_download(
        eval_repo,
        video_tmpl.format(video_key=view_cam, chunk_index=cm["chunk"], file_index=cm["file"]),
        repo_type="dataset",
    )
    frames = _decode_clip(vpath, int(round((cm["from_ts"] + timestamps[0]) * fps)), length)
    vid = os.path.join(out_dir, f"droid_ep{episode}.mp4")
    if frames:
        import imageio

        imageio.mimsave(vid, frames, fps=int(round(fps)), codec="libx264", quality=7)

    # GT (solid) vs predicted (dashed), overlaid per dim (workspace rule 2.a) - inline + saved.
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), squeeze=False)
    x = np.arange(gt.shape[0])
    for d in range(8):
        ax = axes[d // 4][d % 4]
        ax.plot(x, gt[:, d], color="tab:blue", lw=1.4, label="GT (teleop)")
        ax.plot(x, pred[:, d], color="tab:red", lw=1.2, ls="--", label="MolmoAct2 (pred)")
        ax.set_title(dim_names[d], fontsize=10)
        ax.tick_params(labelsize=7)
        if d == 0:
            ax.legend(fontsize=8, loc="best")
    fig.suptitle(
        f"MolmoAct2-DROID open-loop (ROCm), ep {episode}\n{task}\nL1={l1:.4f}  MSE={mse:.4f}  (GT solid, pred dashed)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(os.path.join(out_dir, f"droid_ep{episode}_actions.png"), dpi=110)
    plt.show()
    if frames:
        display(Video(vid, embed=True, width=480))


def run_openloop_episodes(base_policy, predict_chunk, norm_tag, num_steps, out_dir):
    """Replay N_DROID_EPISODES random DROID episodes open-loop, rendering GT-vs-predicted actions
    (and the exterior-cam video) inline for each. `predict_chunk` is the notebook's flow-matching
    action-head call, passed in so the model-call semantics stay visible in the notebook."""
    n_droid = int(os.environ.get("N_DROID_EPISODES", "2"))
    for _ in range(n_droid):
        _run_openloop_episode(base_policy, predict_chunk, norm_tag, num_steps, out_dir)
