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

## Path C — pure conda env (most isolated, recommended for first GH200 run)

Single-tool flow: conda manages Python, CUDA, and Python packages; uv is
not used on the cluster at all. Diverges most from local dev (`pyproject.toml`
is no longer the source of truth here), but is the most predictable on
HPC clusters where you don't control the system CUDA layout.

The walkthrough below assumes a typical SLURM HPC cluster with persistent
shared scratch and tmpfs `$HOME`. Replace `<...>` placeholders.

### C.1 Prereqs

```bash
# Are conda/mamba already on the cluster?
which conda mamba micromamba
module avail anaconda 2>&1 | head     # SLURM clusters often module-load it
```

If a cluster-managed conda is available (`module load anaconda`), use it.
Otherwise install miniforge to **persistent scratch** (NOT `$HOME` — that's
likely tmpfs and will vanish between jobs):

```bash
PERSIST=/data/clear/robot-simulation     # personal scratch root          # e.g. /data/clear/<user>
CONDA_DIR=$PERSIST/miniforge3
curl -L -o /tmp/miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash /tmp/miniforge.sh -b -p $CONDA_DIR
source $CONDA_DIR/etc/profile.d/conda.sh
# Persist activation in shell profile (or your sbatch script):
echo "source $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.bashrc
```

### C.2 Create env on persistent scratch

Don't let conda put the env under `$HOME` if `$HOME` is tmpfs — use
`--prefix` to land it in shared scratch.

```bash
PERSIST=/data/clear/robot-simulation     # personal scratch root
ENV_PREFIX=$PERSIST/envs/cosmos-policy

mamba create --prefix $ENV_PREFIX -c nvidia -c conda-forge -y \
    python=3.10 cuda-toolkit=12.8 cudnn ninja pybind11 git

conda activate $ENV_PREFIX

# Verify
nvcc --version            # should report 12.8
echo $CUDA_HOME           # should point inside $ENV_PREFIX
which python              # should be $ENV_PREFIX/bin/python
```

If you prefer named envs over `--prefix`, configure conda's env path first:
```bash
conda config --add envs_dirs $PERSIST/envs
mamba create -n cosmos-policy ...     # then conda activate cosmos-policy
```

### C.3 Install Python dependencies

```bash
cd $PERSIST/Contact-Aware-cosmos-policy

# 1. PyTorch with CUDA 12.8 (aarch64 wheels available for torch >= 2.4)
pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.7.0 torchvision torchaudio

python -c "import torch; assert torch.cuda.is_available(); \
           print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"

# 2. TransformerEngine — try the prebuilt cu12 wheel first
pip install transformer-engine-cu12 || {
    echo "Prebuilt TE wheel not available for this platform; building from source"
    pip install --no-build-isolation transformer_engine[pytorch]
}
python -c "import transformer_engine.pytorch as te; print('TE OK')"

# 3. cosmos_policy itself + LIBERO sim deps
pip install -e .
pip install libero bddl easydict draccus 'mujoco==3.3.2' \
            cloudpickle gym 'imageio[ffmpeg]'

# 4. flash-attn (no aarch64 prebuilt; build from source against conda CUDA)
#    MAX_JOBS throttles parallel compilation — set to half your core count
#    to avoid OOM on shared login nodes. Build takes 10–30 min.
MAX_JOBS=4 pip install --no-build-isolation flash-attn==2.7.3

# 5. Other commonly needed packages cosmos-policy imports
pip install megatron-core wandb hydra-core omegaconf
```

After every step you should be able to import without errors. If an
import surfaces a missing module, install it with `pip install <pkg>` —
cosmos-policy's `pyproject.toml` is the canonical list to consult.

### C.4 HF auth (one-time per user)

The `nvidia/Cosmos-Predict2-2B-Video2World` checkpoint is gated. Run
`huggingface-cli login` once on a login node with internet:

```bash
huggingface-cli login         # paste a read token from https://huggingface.co/settings/tokens
# Pre-cache the gated checkpoint in persistent scratch to avoid mid-job downloads:
export HF_HOME=$PERSIST/hf_cache
huggingface-cli download nvidia/Cosmos-Predict2-2B-Video2World model-480p-16fps.pt
huggingface-cli download nvidia/Cosmos-Policy-LIBERO-Predict2-2B
```

Persist `HF_HOME` in your shell profile / sbatch script — by default HF
caches into `$HOME/.cache` which on tmpfs vanishes between jobs.

### C.5 wandb (optional but training configs reference it)

```bash
wandb login                   # paste API key from https://wandb.ai/authorize
# Or disable: export WANDB_MODE=disabled in your sbatch script
```

### C.6 Data prep (winedrawer experiment)

```bash
export BASE_DATASETS_DIR=$PERSIST/cosmos-policy-data
export LIBERO_DATA_ROOT=$BASE_DATASETS_DIR/libero_raw/libero_90

# One-time: download libero_90 demos
mkdir -p $BASE_DATASETS_DIR/libero_raw && cd $BASE_DATASETS_DIR/libero_raw
huggingface-cli download yifengzhu-hf/LIBERO-datasets --repo-type dataset \
    --include "libero_90/*" --local-dir .

# Single-task prep
cd $PERSIST/Contact-Aware-cosmos-policy
PY=python bash contact_aware_wm/prepare_libero90_winedrawer.sh
```

### C.7 Smoke test before launching long training

```bash
# Minimal: import path resolution + 1-step training
cd $PERSIST/Contact-Aware-cosmos-policy
BASE_DATASETS_DIR=$PERSIST/cosmos-policy-data \
LIBERO_DATA_ROOT=$LIBERO_DATA_ROOT \
torchrun --nproc_per_node=1 -m cosmos_policy.scripts.train \
    --config=cosmos_policy/config/config.py \
    -- experiment=cosmos_predict2_2b_480p_libero90_winedrawer_v_full \
    trainer.max_iter=2 trainer.callbacks.compile_tokenizer.enabled=False
```

If that completes 2 iterations, you're good. Then bump `max_iter` back to 20000.

### C.8 Sample SLURM sbatch script

For interactive testing, the working srun pattern on this cluster is:
```bash
srun --account=clear --partition=quanta-gh200 --qos=quanta-main \
     --time=48:00:00 --cpus-per-task=10 --gpus-per-node=1 --mem=30GB \
     --pty /bin/bash
```

For batch training, save as `slurm_train_winedrawer.sbatch` in the repo:

```bash
#!/bin/bash
#SBATCH --job-name=cosmos-winedrawer
#SBATCH --account=clear
#SBATCH --partition=quanta-gh200
#SBATCH --qos=quanta-main
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G                 # 2B model FT needs more than the 30G interactive value
#SBATCH --time=48:00:00            # max on quanta-main; 20k iters at batch=8 may not finish in one job — see C.10
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Activate conda env
PERSIST=/data/clear/robot-simulation
source $PERSIST/miniforge3/etc/profile.d/conda.sh
conda activate $PERSIST/envs/cosmos-policy

# Persistent caches (avoid tmpfs $HOME on compute node)
export HF_HOME=$PERSIST/hf_cache
export TRITON_CACHE_DIR=$PERSIST/triton_cache
export TORCH_EXTENSIONS_DIR=$PERSIST/torch_ext_cache
mkdir -p $HF_HOME $TRITON_CACHE_DIR $TORCH_EXTENSIONS_DIR

# Project paths
cd $PERSIST/Contact-Aware-cosmos-policy
export BASE_DATASETS_DIR=$PERSIST/cosmos-policy-data
export LIBERO_DATA_ROOT=$BASE_DATASETS_DIR/libero_raw/libero_90

# Pick V or V+F:
EXPERIMENT=cosmos_predict2_2b_480p_libero90_winedrawer_v_full
# EXPERIMENT=cosmos_predict2_2b_480p_libero90_winedrawer_vf_full

torchrun --nproc_per_node=1 \
    -m cosmos_policy.scripts.train \
    --config=cosmos_policy/config/config.py \
    -- experiment=$EXPERIMENT
```

Submit:
```bash
mkdir -p logs && sbatch slurm_train_winedrawer.sbatch
squeue -u $USER
tail -f logs/cosmos-winedrawer-<jobid>.out
```

### C.10 Resuming across the 48h time-limit

A 20k-iter run at batch=8 may not finish in 48h on a single GH200 — and
`quanta-main`'s max walltime is 48h. Cosmos-policy checkpoints every
`save_iter=1000` steps to `/tmp/imaginaire4-output/.../checkpoints/`, but
`/tmp` on a compute node is ephemeral. Two options:

**Option A — redirect checkpoint output to persistent scratch.** Add to
the sbatch script before `torchrun`:
```bash
export IMAGINAIRE4_OUTPUT_DIR=$PERSIST/cosmos-policy-runs   # if the trainer respects this
# Or use a Hydra override:
TRAIN_OUTPUT_OVERRIDE="job.path_local_root=$PERSIST/cosmos-policy-runs"
torchrun ... -- experiment=$EXPERIMENT $TRAIN_OUTPUT_OVERRIDE
```
Then on resume, the trainer auto-detects `latest_checkpoint.txt` and continues.

**Option B — chain jobs with `--dependency=afterok`.** Submit a
self-resubmitting wrapper that re-launches itself if `latest_checkpoint.txt`
shows < 20000 iters:
```bash
sbatch slurm_train_winedrawer.sbatch
JOBID=$(squeue -u $USER -h -o %i | tail -1)
sbatch --dependency=afterok:$JOBID slurm_train_winedrawer.sbatch
```

### C.9 Path-C-specific gotchas

- **Don't use `pip install` from outside the activated env** — it'll silently
  install into a different python. Always check `which pip` reports the
  conda env path first.
- **mamba/conda channel priority**: if you see "package not found" errors,
  prepend `-c nvidia -c conda-forge` to the create/install commands so
  the NVIDIA channel is checked first for CUDA components.
- **`pip install -e .` and a stale `.venv/` from a prior uv attempt**:
  delete `.venv/` and `uv.lock`-related caches before running pip in the
  conda env, or pip may resolve against the wrong site-packages.
- **`flash-attn` build OOM**: cap `MAX_JOBS=4` (or fewer); the default
  spawns one compile job per core which can OOM on shared login nodes.
- **No internet on compute nodes**: many clusters firewall outbound from
  compute nodes. Pre-stage HF checkpoints (C.4) and any pip packages
  on a login node before submitting jobs. Set `HF_HUB_OFFLINE=1` and
  `WANDB_MODE=offline` in the sbatch script if applicable.

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
export BASE_DATASETS_DIR=/data/clear/robot-simulation/cosmos-policy-data
export LIBERO_DATA_ROOT=$BASE_DATASETS_DIR/libero_raw/libero_90

# Download libero_90 demos (one-time, ~few GB)
mkdir -p $BASE_DATASETS_DIR/libero_raw && cd $BASE_DATASETS_DIR/libero_raw
huggingface-cli download yifengzhu-hf/LIBERO-datasets --repo-type dataset \
    --include "libero_90/*" --local-dir .

# Single-task prep (winedrawer experiment)
cd $PERSIST/Contact-Aware-cosmos-policy
bash contact_aware_wm/prepare_libero90_winedrawer.sh
```
