# Infrastructure installation guide

This handoff is intended for the infrastructure owner preparing the Jupyter environment for the
AMD and Robotec.ai ROSCon workshop.

## Required platform

| Component | Requirement |
|---|---|
| Host | Ubuntu 24.04 |
| Hardware | AMD Strix Halo mini-PC|
| ROCm | **ROCm 7.2** |
| Python | 3.13 recommended; the wheel supports Python 3.12 and 3.13 |
| PyTorch | `2.12.0+rocm7.2` |
| Triton | `triton-rocm==3.7.0`  |
| Genesis | `genesis-world==1.4.0` (installed through the gslab wheel) |
| gslab | Bundled `wheels/gslab-0.1.0-py3-none-any.whl` |


## 1. Prepare GPU access

Install ROCm 7.2 according to AMD's host installation guide. 

## 2. Create the Python environment

Run these commands from the course root directory (the folder containing `INSTALL.md`,
`checkpoints/`, and `notebooks/`). They intentionally install the ROCm build of PyTorch first,
before resolving the gslab wheel dependencies.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.13 .venv
source .venv/bin/activate

uv pip install \
  --index https://download.pytorch.org/whl/rocm7.2 \
  "torch==2.12.0+rocm7.2" "triton-rocm==3.7.0"

uv pip install \
  wheels/gslab-0.1.0-py3-none-any.whl \
  jupyterlab ipykernel ipywidgets
```

The wheel includes the full `gslab` Python package, Unitree G1 MJCF, and robot meshes. No editable
source checkout is needed.

## 3. Register the workshop kernel

```bash
python -m ipykernel install --user \
  --name gslab-roscon \
  --display-name "gslab ROSCon (ROCm 7.2)"
```

All packaged notebooks request the kernel named `gslab-roscon`.

## 4. Launch JupyterLab

Launch Jupyter from the course root directory so the notebooks can discover the bundled
checkpoints and fixed log directory:

```bash
jupyter lab --notebook-dir .
```

For a remote service, add the site's usual bind address, authentication, TLS, and proxy options.
Do not expose an unauthenticated Jupyter server.

## 5. Acceptance test

Open `00_workshop_intro.ipynb` with the **gslab ROSCon (ROCm 7.2)** kernel and run
its final environment-check cell. The final line should be:

```text
PASS: PyTorch sees the GPU and Genesis completed a step on gs.amdgpu.
```

A CPU fallback is treated as a failure. Do not continue to the training scene until this cell
passes.

