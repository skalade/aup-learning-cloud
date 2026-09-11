# ROSCon Humanoid Stairs Workshop

Prepared for the workshop hosted by AMD and Robotec.ai on AMD Strix Halo mini-PCs.

## Contents

```text
.
├── INSTALL.md
├── checkpoints/
│   ├── g1_flat_baseline.pt
│   └── g1_stairs_easy_trained.pt
├── 00_workshop_intro.ipynb
├── humanoid_locomotion_train.ipynb
├── humanoid_locomotion_eval.ipynb
├── scripts/
│   └── workshop_helpers.py
└── wheels/
    └── gslab-0.1.0-py3-none-any.whl
```

Start with `00_workshop_intro.ipynb` to verify the environment. The two hands-on notebooks are
`humanoid_locomotion_train.ipynb` and `humanoid_locomotion_eval.ipynb`.

Training always writes the final checkpoint to:

```text
logs/roscon_stairs/model_stairs.pt
```

All paths are relative to this folder, so no source checkout or timestamp lookup is needed.
Start with [INSTALL.md](INSTALL.md).
