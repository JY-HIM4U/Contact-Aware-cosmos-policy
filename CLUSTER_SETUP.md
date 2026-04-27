# Cluster Setup (bare-metal, no Docker)

For HPC clusters where Docker isn't available. The official path is Docker
(see [SETUP.md](SETUP.md)) — this guide is for environments where you have
to install dependencies directly on a login/compute node.

The right path depends on the node architecture and what's already on disk.

| Scenario | Recommended path |
|---|---|
| x86_64 cluster with CUDA 12.8 toolkit available | **Path A — uv-native** |
| aarch64 (e.g. NVIDIA GH200 Grace+Hopper) | **Path B — conda for CUDA + uv pip for Python**, or **Path C — pure conda** |
| You'll be source-building flash-attn / TE / custom CUDA kernels | **Path B or C** (need a unified CUDA toolkit layout) |
| You want to mirror local dev exactly | **Path A** (uv is the project's source of truth) |

---

## Path A — uv-native (matches local x86 dev)

Closest to local dev. Works cleanly on x86_64 nodes with a recent CUDA
toolkit on the system.

```bash
# Install uv if needed (single binary)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync the env (always pass --extra cu128 — see project memory)
cd Contact-Aware-cosmos-policy
uv sync --extra cu128 --group libero

# Activate
source .venv/bin/activate

# Verify CUDA wired up
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# Expected: 2.7.0+cu128 12.8 True
```

**Known gotchas with Path A:**
- On aarch64, `uv sync` fails because `flash-attn==2.7.3+cu128.torch27` only has
  x86_64 wheels. uv refuses to skip it. Use Path B or C instead.
- If `torch.cuda.is_available()` returns `False`, your venv ended up with
  CPU-only torch. Re-install with the CUDA index:
  ```bash
  uv pip install --index-url https://download.pytorch.org/whl/cu128 \
      torch==2.7.0 torchvision torchaudio
  ```

---

## Path B — conda for CUDA toolkit + uv pip for Python deps (recommended for aarch64)

Best of both worlds: conda gives you a unified CUDA toolkit layout
(bin/include/lib64 in one place) that source-builds find without manual
env-var dancing, while `uv pip` keeps the Python-side fast and
`pyproject.toml`-driven.

```bash
# 1. Bootstrap a conda env with Python and CUDA toolkit
#    (use mamba/micromamba for speed if available)
conda create -n cosmos-policy -c nvidia -c conda-forge \
    python=3.10 cuda-toolkit=12.8 cudnn

conda activate cosmos-policy

# 2. Verify CUDA toolkit is in PATH and CUDA_HOME is set
nvcc --version           # should report 12.8
echo $CUDA_HOME          # should point inside the conda env
which nvcc

# 3. Install Python deps via uv pip (no resolver, just installs into the active env)
cd Contact-Aware-cosmos-policy
uv pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.7.0 torchvision torchaudio
uv pip install transformer-engine-cu12     # prebuilt, no compile needed
uv pip install -e .                        # install cosmos-policy itself
uv pip install libero bddl easydict draccus 'mujoco==3.3.2' \
               cloudpickle gym 'imageio[ffmpeg]'

# 4. flash-attn from source (only x86 has prebuilt; aarch64 must build)
#    Needs the CUDA toolkit from step 1 plus build deps.
uv pip install ninja pybind11
uv pip install --no-build-isolation flash-attn==2.7.3
# Build takes 10–30 min. If MAX_JOBS is unset it defaults to all cores
# (can OOM on shared nodes); set MAX_JOBS=4 to throttle.

# 5. Sanity check
python -c "import torch, transformer_engine.pytorch as te, flash_attn; print('OK')"
```

**Why this works on aarch64:** conda's `cuda-toolkit=12.8` package has
proper aarch64 builds, and `uv pip install` skips the lockfile resolver
entirely — so you bypass the x86-only `flash-attn` pin in `uv.lock` and
build the matching aarch64 version yourself.

---

## Path C — pure conda env (most isolated)

If you don't want uv on the cluster at all (some cluster admins have
strong opinions). Everything via conda + pip.

```bash
conda create -n cosmos-policy -c nvidia -c conda-forge \
    python=3.10 cuda-toolkit=12.8 cudnn ninja pybind11
conda activate cosmos-policy

cd Contact-Aware-cosmos-policy
pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.7.0 torchvision torchaudio
pip install transformer-engine-cu12
pip install -e .
pip install libero bddl easydict draccus 'mujoco==3.3.2' \
            cloudpickle gym 'imageio[ffmpeg]'
pip install --no-build-isolation flash-attn==2.7.3
```

Diverges most from local dev — `pyproject.toml` is no longer the source
of truth — but is the simplest single-tool flow.

---

## Common gotchas (any path)

- **`$HOME` on compute nodes is sometimes tmpfs** (e.g. `/tmp/home/$USER`).
  Don't put datasets, checkpoints, or conda envs there — they'll vanish
  between jobs. Use `/data/...` or whatever persistent scratch the cluster
  exposes. Set `BASE_DATASETS_DIR` and `LIBERO_DATA_ROOT` explicitly.
- **HF gated repos** (e.g. `nvidia/Cosmos-Predict2-2B-Video2World`) need
  `huggingface-cli login` once with a token that has been granted access
  via the model card.
- **flash-attn, TE, megatron-core**: when source-building any of these,
  `nvcc --version` must match `torch.version.cuda` (e.g. cu128 torch ↔
  CUDA 12.8 toolkit). A 12.8 ↔ 13.0 mismatch silently produces broken
  binaries that crash at runtime. Conda's `cuda-toolkit=12.8` keeps this
  consistent.
- **GH200 specifically**: ARM CPU. `transformer-engine-cu12` has aarch64
  prebuilt wheels. `flash-attn==2.7.3` does NOT — must compile from source
  (see Path B step 4).

---

## Data prep (any path)

After the env is set up:

```bash
# Pick a writable, persistent location
export BASE_DATASETS_DIR=/data/<your-cluster-scratch>/cosmos-policy-data
export LIBERO_DATA_ROOT=$BASE_DATASETS_DIR/libero_raw/libero_90

# Download libero_90 demos (one-time, ~few GB)
mkdir -p $BASE_DATASETS_DIR/libero_raw && cd $BASE_DATASETS_DIR/libero_raw
huggingface-cli download yifengzhu-hf/LIBERO-datasets --repo-type dataset \
    --include "libero_90/*" --local-dir .

# Single-task prep (winedrawer experiment)
cd <repo>
bash contact_aware_wm/prepare_libero90_winedrawer.sh
```
