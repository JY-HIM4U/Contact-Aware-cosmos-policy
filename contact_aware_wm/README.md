# contact_aware_wm

Contact-aware world-model experiments built on top of NVIDIA's [Cosmos Policy](https://github.com/NVlabs/cosmos-policy) (LIBERO). Scripts here:

- Run Cosmos Policy as an **autoregressive world model** on LIBERO tasks (chaining chunk-by-chunk predictions).
- Record per-step **force/torque (F/T)** signals from the MuJoCo sim.
- Extract F/T from LIBERO demos, train an F/T-conditioned world model, and compare rollouts.

> This folder lives inside a fork of `NVlabs/cosmos-policy`. The upstream package is used as a library via `from cosmos_policy...` imports.

## Quickstart

```bash
# 1. clone the fork (this repo) and install cosmos-policy (editable)
git clone <this-fork-url> cosmos-policy
cd cosmos-policy
conda env create -n cosmos python=3.10   # or use your existing env
conda activate cosmos
pip install -e .                         # installs cosmos_policy
pip install -r contact_aware_wm/requirements.txt

# 2. set paths (see "Data you need to download" below)
export CONTACT_AWARE_WM="$PWD/contact_aware_wm"
export LIBERO_DATA_ROOT=/path/to/LIBERO/datasets/libero_90/libero_90
export DATA_ROOT=/path/to/libero_90_with_ft       # F/T-augmented dataset
export CHECKPOINTS_ROOT=/path/to/checkpoints      # our trained ckpts
export RESULTS_ROOT="$CONTACT_AWARE_WM/results"   # outputs land here

# 3. run the autoregressive world-model demo
python contact_aware_wm/run_cosmos_ar_with_ft_plot.py \
    --all_tasks --num_autoreg_steps 15 \
    --proprio_mode predicted \
    --out_dir "$RESULTS_ROOT/cosmos_ar_contact"
```

Tip: put the `export` lines in a file (e.g. `~/.contact_aware_wm.env`) and `source` it.

## Environment variables

All scripts resolve paths via `contact_aware_wm/paths.py`, which reads these env vars (with sensible defaults):

| Variable | Purpose | Default |
|---|---|---|
| `CONTACT_AWARE_WM` | This folder's path | parent dir of `paths.py` |
| `LIBERO_DATA_ROOT` | Raw LIBERO `libero_90/` HDF5 dir | `~/LIBERO/datasets/libero_90/libero_90` |
| `DATA_ROOT` | F/T-augmented dataset dir | `$CONTACT_AWARE_WM/data/libero_90_with_ft` |
| `CHECKPOINTS_ROOT` | Trained model checkpoints | `$CONTACT_AWARE_WM/checkpoints` |
| `RESULTS_ROOT` | Videos / plots / logs output dir | `$CONTACT_AWARE_WM/results` |

## Data you need to download

These are **not** in the git repo (too large). Acquire them separately:

### 1. LIBERO dataset (public) — required

Follow the official LIBERO setup:
<https://github.com/Lifelong-Robot-Learning/LIBERO>

You need the `libero_90` HDF5 demos. After download, point `LIBERO_DATA_ROOT` at the dir containing files like `KITCHEN_SCENE2_*.hdf5`.

Size: ~20 GB.

### 2. Cosmos Policy LIBERO checkpoint (public) — required

Auto-downloaded by `huggingface_hub` on first run:
- `nvidia/Cosmos-Policy-LIBERO-Predict2-2B`

No manual step needed if HF cache is writable.

### 3. F/T-augmented dataset (`libero_90_with_ft`) — required for F/T experiments

This is our derived dataset: LIBERO demos re-played through the sim with F/T sensors attached, saved as NPZ. To regenerate it from scratch:

```bash
python contact_aware_wm/extract_ft_libero.py \
    --libero_dir "$LIBERO_DATA_ROOT" \
    --out_dir    "$DATA_ROOT"
```

Size: ~60 GB. This takes many hours on a single machine.

**Shortcut:** ask the project owner (Jaeyoun) for a copy via shared drive / rsync — it's much faster than regenerating.

### 4. Trained checkpoints — optional

Needed only for `evaluate_libero.py`, `train_libero.py` resume, and the `libero_improved_*` baselines in `results/`. Not needed to run the Cosmos-AR world-model demo. Obtain from the project owner.

## What each script does

| Script | Purpose |
|---|---|
| `run_cosmos_ar_with_ft_plot.py` | Roll out Cosmos Policy as an AR world model; record F/T; side-by-side GT vs predicted video + plot. **Supports `--proprio_mode {initial,predicted,real}`.** |
| `run_cosmos_policy_autoregressive.py` | Earlier version of above, no F/T. |
| `run_cosmos_policy_libero.py` | Single-episode Cosmos Policy inference on LIBERO. |
| `run_cosmos_libero.py` | Our F/T-aware model's inference on LIBERO. |
| `run_cosmos_ar_with_ft_plot.py` | Main demo — contact + autoregression. |
| `run_contact_tasks.{py,sh}` | Batch driver over contact-heavy LIBERO tasks. |
| `extract_ft_libero.py` | Build the F/T-augmented dataset from raw LIBERO demos. |
| `train_libero.py` / `train_fm.py` | Train the contact-aware world model. |
| `evaluate_libero.py` / `evaluate_fm.py` | Evaluate trained checkpoints. |
| `regenerate_ft_plots.py` | Rebuild combined F/T + drift plots from saved rollouts. |
| `cosmos_ft/` | Dataset + F/T utilities (subclass of Cosmos's LIBERODataset). |

## Known quirks

- **Proprio feedback in AR rollouts**: Cosmos Policy was trained as a single-chunk predictor (§4.2 of the paper: "we do not use input history nor predict future frames across multiple subsequent timesteps"). When chaining chunks into a long AR rollout (as our scripts do), you choose how to update proprio: `--proprio_mode initial` (freeze at t=0), `predicted` (use `future_proprio` head, experimental), or `real` (oracle, sim-only). `initial` is the default for backward compatibility.
- **Editable install dangling after reboot**: if you installed `cosmos_policy` with `pip install -e .` from `/tmp/...`, a reboot wipes `/tmp` and the env breaks. Clone to a persistent path.
