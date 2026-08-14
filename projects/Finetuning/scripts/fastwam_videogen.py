# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""FastWAM video imagination (joint path) for notebook 3.

Runs FastWAM's joint video+action denoising route (`FastWAM.infer_joint`) from a single
ground-truth start frame + proprio + language prompt, then decodes the imagined future clip and
writes a two-column MP4: ground-truth future (left) vs FastWAM imagined (right) - the "what the
model dreams" view (workshop rule 2.a: GT left, prediction right). Reuses the upstream
`RobotVideoDataset` + `FastWAMProcessor` so cam-concat / resize / [-1,1] normalization match
training, and the exact planning-path caches shipped in the image are used (fastest faithful
route). Reports joint-path latency per clip.

This is a subprocess backend for the notebook: it runs in the isolated /opt/fastwam-venv and
prints one `VIDEOGEN_MP4=<path>` line per finished clip plus a final `VIDEOGEN_SUMMARY=<json>`
line the notebook parses to embed the clips inline.

Env (all have baked-image defaults):
  FASTWAM_REPO           /repos/fastwam
  CONFIG_NAME            sim_libero
  CKPT                   $FASTWAM_RELEASE_DIR/libero_uncond_2cam224.pt
  DATASET_STATS          $FASTWAM_RELEASE_DIR/libero_uncond_2cam224_dataset_stats.json
  DATASET_DIR            $FASTWAM_DATA_DIR/libero_object_no_noops_lerobot
  NUM_VIDEOS (2)  NUM_STEPS (20)  SEED (0)  FPS (6)  OUT_DIR (/outputs)  TAG (libero_object)
"""
import json
import os
import sys
import time

import numpy as np
import torch
import imageio
from PIL import Image, ImageDraw

FASTWAM_REPO = os.environ.get("FASTWAM_REPO", "/repos/fastwam")
CONFIG_NAME = os.environ.get("CONFIG_NAME", "sim_libero")
RELEASE_DIR = os.environ.get("FASTWAM_RELEASE_DIR", "/opt/fastwam-assets/fastwam_release")
DATA_DIR_BASE = os.environ.get("FASTWAM_DATA_DIR", "/opt/fastwam-assets/data")
CKPT = os.environ.get("CKPT") or os.path.join(RELEASE_DIR, "libero_uncond_2cam224.pt")
DATASET_STATS = os.environ.get("DATASET_STATS") or os.path.join(
    RELEASE_DIR, "libero_uncond_2cam224_dataset_stats.json")
DATASET_DIR = os.environ.get("DATASET_DIR") or os.path.join(
    DATA_DIR_BASE, "libero_object_no_noops_lerobot")
NUM_VIDEOS = int(os.environ.get("NUM_VIDEOS") or "2")
NUM_STEPS = int(os.environ.get("NUM_STEPS") or "20")
SEED = int(os.environ.get("SEED") or "0")
FPS = int(os.environ.get("FPS") or "6")
OUT_DIR = os.environ.get("OUT_DIR", "/outputs")
TAG = os.environ.get("TAG", "libero_object")


def _compose_cfg():
    from omegaconf import OmegaConf
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    for name, fn in (("eval", eval), ("max", lambda x: max(x)),
                     ("split", lambda s, idx: s.split("/")[int(idx)])):
        try:
            OmegaConf.register_new_resolver(name, fn, replace=True)
        except Exception:
            pass
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.path.join(FASTWAM_REPO, "configs"), version_base="1.3"):
        return compose(config_name=CONFIG_NAME, overrides=[f"ckpt={CKPT}"])


def _build_dataset(cfg):
    from hydra.utils import instantiate
    import fastwam.datasets.lerobot.robot_video_dataset as rvd
    from fastwam.utils import misc

    def _stub_text_context(self, prompt):
        return torch.zeros(self.context_len, 8), torch.ones(self.context_len, dtype=torch.bool)
    rvd.RobotVideoDataset._get_cached_text_context = _stub_text_context
    try:
        misc.get_work_dir = lambda *a, **k: "/tmp"
    except Exception:
        pass

    return instantiate(
        cfg.data.train,
        dataset_dirs=[DATASET_DIR],
        is_training_set=False,
        val_set_proportion=0.0,
        pretrained_norm_stats=DATASET_STATS,
        skip_padding_as_possible=False,
    )


def _video_tensor_to_frames(video):
    """[C, T, H, W] in [-1,1] -> list of uint8 HxWx3 numpy frames."""
    v = video.detach().float().clamp(-1, 1)
    v = ((v + 1.0) * 127.5).to(torch.uint8).cpu().numpy()   # [C, T, H, W]
    return [np.ascontiguousarray(v[:, t].transpose(1, 2, 0)) for t in range(v.shape[1])]


def _to_rgb(frame):
    if isinstance(frame, Image.Image):
        return np.array(frame.convert("RGB"))
    return np.asarray(frame)[..., :3]


def _label(img, text):
    pil = Image.fromarray(img.astype(np.uint8))
    ImageDraw.Draw(pil).text((6, 6), text, fill=(255, 255, 0))
    return np.array(pil)


def main() -> int:
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__}  hip={torch.version.hip}  device={dev}", flush=True)
    print(f"config={CONFIG_NAME}  ckpt={os.path.basename(CKPT)}  videos={NUM_VIDEOS}  steps={NUM_STEPS}",
          flush=True)
    for p, what in ((CKPT, "checkpoint"), (DATASET_STATS, "dataset stats"), (DATASET_DIR, "GT dataset")):
        if not os.path.exists(p):
            print(f"FAIL: missing {what}: {p}", file=sys.stderr)
            return 1

    if FASTWAM_REPO not in sys.path:
        sys.path.insert(0, FASTWAM_REPO)
    from hydra.utils import instantiate
    cfg = _compose_cfg()
    num_video_frames = int(cfg.data.train.num_frames)
    action_horizon = num_video_frames - 1
    print(f"num_video_frames={num_video_frames}  action_horizon={action_horizon}", flush=True)

    t0 = time.time()
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda")
    model.load_checkpoint(str(CKPT))
    model = model.to("cuda").eval()
    print(f"model loaded: {time.time()-t0:.1f}s  proprio_dim={model.proprio_dim}", flush=True)

    ds = _build_dataset(cfg)
    starts = ds.lerobot_dataset.episode_data_index["from"].tolist()
    n = min(NUM_VIDEOS, len(starts))
    out_dir = os.path.join(OUT_DIR, f"videogen_{TAG}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"episodes={len(starts)}  generating {n} imagined clips -> {out_dir}\n", flush=True)

    per = []
    for k in range(n):
        idx = int(starts[k])
        sample = ds[idx]
        video = sample["video"]                                    # [C, T, H, W] in [-1,1]
        input_image = video[:, 0].unsqueeze(0).to("cuda", dtype=model.torch_dtype)
        proprio = sample["proprio"][0:1].to("cuda", dtype=model.torch_dtype)
        prompt = sample["prompt"]

        t1 = time.time()
        with torch.no_grad():
            out = model.infer_joint(
                prompt=prompt, input_image=input_image,
                num_video_frames=num_video_frames, action_horizon=action_horizon,
                proprio=proprio, num_inference_steps=NUM_STEPS, seed=SEED,
                rand_device="cpu", test_action_with_infer_action=False,
            )
        latency = time.time() - t1

        gen_frames = [_to_rgb(f) for f in out["video"]]            # list of HxWx3 uint8
        gt_frames = _video_tensor_to_frames(video)                 # aligned GT future
        tt = min(len(gen_frames), len(gt_frames))

        stitched = []
        for gt, gen in zip(gt_frames[:tt], gen_frames[:tt]):
            if gt.shape[:2] != gen.shape[:2]:
                gt = np.array(Image.fromarray(gt).resize((gen.shape[1], gen.shape[0]), Image.BILINEAR))
            left = _label(gt, "ground truth")
            right = _label(gen, "FastWAM imagined")
            stitched.append(np.concatenate([left, right], axis=1))

        mp4 = os.path.join(out_dir, f"clip{k:02d}_gt_vs_imagined.mp4")
        imageio.mimwrite(mp4, stitched, fps=FPS, quality=8, macro_block_size=1)

        gt_a = sample["action"].float().cpu().numpy()
        pr_a = out["action"].float().cpu().numpy()
        T = min(gt_a.shape[0], pr_a.shape[0])
        norm_mae = float(np.abs(pr_a[:T] - gt_a[:T]).mean())
        per.append({"clip": k, "frame_idx": idx, "frames": tt,
                    "joint_latency_s": round(latency, 3),
                    "action_norm_mae": round(norm_mae, 4), "prompt": prompt[:80], "mp4": mp4})
        print(f"clip{k:02d} idx={idx:7d} frames={tt} joint_latency={latency:.2f}s "
              f"actMAE={norm_mae:.4f}", flush=True)
        # Machine-parseable line the notebook picks up to embed the clip inline.
        print(f"VIDEOGEN_MP4={mp4}", flush=True)

    lat = np.array([p["joint_latency_s"] for p in per]) if per else np.array([0.0])
    steady = lat[1:] if len(lat) > 1 else lat
    summary = {
        "tag": TAG, "config": CONFIG_NAME, "num_videos": n,
        "num_inference_steps": NUM_STEPS, "num_video_frames": num_video_frames, "fps": FPS,
        "joint_latency_s_first_warmup": round(float(lat[0]), 3),
        "joint_latency_s_mean_steady": round(float(steady.mean()), 3),
        "clips": per,
    }
    with open(os.path.join(out_dir, "video_latency.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n== {TAG} imagined-video generation ==", flush=True)
    print(f"joint-path latency ({NUM_STEPS} steps): warmup={lat[0]:.2f}s  "
          f"steady-mean={steady.mean():.2f}s", flush=True)
    print("VIDEOGEN_SUMMARY=" + json.dumps(summary), flush=True)
    print("PASS: FastWAM video imagination complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
