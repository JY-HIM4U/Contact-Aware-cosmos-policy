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

## F/T fine-tuning (paper method — latent injection)

Goal: fine-tune `nvidia/Cosmos-Policy-LIBERO-Predict2-2B` so it conditions on — and predicts — wrist-frame 6-D Force/Torque alongside the existing image/proprio/action/value latents. This follows **Cosmos Policy Appendix A.1**: F/T is injected by normalizing to [−1, +1], duplicating the 6-D vector to fill a latent-frame volume, and overwriting a blank latent frame.

### Two implementations in this tree (use the first)

1. **Upstream latent injection (paper-method, primary).** Fully plumbed in the cosmos_policy package:
   - `cosmos_policy/datasets/libero_dataset_ft.py` — `LIBERODatasetFT`, inserts `current_ft` at latent idx 2 and `future_ft` after `future_proprio`.
   - `cosmos_policy/models/policy_video2world_model.py` — routes `current_ft_latent_idx` / `future_ft_latent_idx` through `replace_latent_with_proprio` (same kernel as proprio).
   - `cosmos_policy/utils/utils.py::duplicate_array` — the paper's duplication op.
   - Training entrypoint: `contact_aware_wm/train_ft_demo.py`.
2. **Side-encoder baseline (not paper-method).** `contact_aware_wm/cosmos_ft/{ft_modules,libero_dataset_ft}.py` — `FTEncoder` (Conv1D) + `FTDecoder` MLP. Kept as an ablation.

### Frame layout after F/T insertion (11 latent frames, 1 + 10·`n_dup` = 41 raw images when `n_dup=4`)

```
0: blank | 1: proprio | 2: current_ft | 3: wrist | 4: primary
5: action | 6: future_proprio | 7: future_ft | 8: future_wrist | 9: future_primary
10: value
```

### F/T data

- Source: `extract_ft_libero.py` → `$DATA_ROOT/libero_90_with_ft/{TASK}/demo_{i}.npz` with key `ft_wrist`, shape `(T, 6)`.
- Stats: `$DATA_ROOT/libero_90_with_ft/dataset_stats_all.json` with `ft_min`, `ft_max` used to rescale to [−1, +1].

### Prerequisites

1. T5 embeddings for libero_90 commands — NOT in HF-shipped `libero_t5_embeddings.pkl` (which covers libero_spatial/goal/object only, 40 entries). Extend via `contact_aware_wm/precompute_t5_libero90.py`.
2. `t5_embeddings.pkl` written to `$DATA_ROOT/libero_90_with_ft/t5_embeddings.pkl`.
3. After precompute, `models--google-t5--t5-11b` in HF cache (~85 G) can be deleted.

### Workflow

```bash
conda activate cosmos
export CONTACT_AWARE_WM=/home/jaeyoun/cosmos-policy/contact_aware_wm
export LIBERO_DATA_ROOT=/home/jaeyoun/LIBERO/datasets/libero_90/libero_90
export DATA_ROOT=/home/jaeyoun/contact-aware-wm/data

# 1. T5 precompute (libero_90 + libero_10), extends existing HF cache
python contact_aware_wm/precompute_t5_libero90.py \
    --out_pkl "$DATA_ROOT/libero_90_with_ft/t5_embeddings.pkl"

# 2. Data pipeline smoke test (CPU-only, no GPU)
python contact_aware_wm/train_ft_demo.py --dry_run \
    --libero_data_dir "$LIBERO_DATA_ROOT" \
    --ft_data_dir     "$DATA_ROOT/libero_90_with_ft" \
    --t5_path         "$DATA_ROOT/libero_90_with_ft/t5_embeddings.pkl"

# 3. VRAM fit (single A4000, bf16, grad-ckpt)
python contact_aware_wm/test_vram_fit.py

# 4. Short smoke training (100 steps, save every 25)
python contact_aware_wm/train_ft_demo.py --num_steps 100

# 5. Full fine-tune across 4× A4000
torchrun --nproc_per_node=4 contact_aware_wm/train_ft_demo.py --num_steps 5000
```

### Crash history (Apr 2026)

- Two hard crashes Apr 17 (12:14, 14:04 EDT); prior boots ended mid-Docker-netns log (abrupt, not clean). System stable since Apr 17 15:12.
- Likely contributors: (a) `/` at 94% full pre-cleanup, (b) simultaneous container churn and heavy diffusion training on 4× A4000 (16 GB each = 64 GB total VRAM — tight for a 2B latent-diffusion backbone).
- Mitigations in current flow: checkpoint every 25 steps, bf16 + grad-checkpointing, cleared 61 G of cache (see `~/.cache/*` deletions), delete `t5-11b` (85 G) once embeddings are saved.

### Known issues / gotchas

- `LIBERODatasetFT` filters episodes to those whose `command` appears in the T5 pkl — if the pkl is incomplete you'll see `Removed N episodes without T5 match` and the dataset may shrink unexpectedly.
- The paper's latent-injection duplication rounds up when `(H′·W′·C′) / dact` is non-integral — `duplicate_array` handles this with a trailing partial block. For 6-D F/T this divides cleanly into most latent-frame volumes, but verify for new latent shapes.
- `has_ft` flag per sample lets you mask the F/T loss on rollout samples / demos without extracted F/T.

### Progress log

| Date | Step | Status | Notes |
|---|---|---|---|
| 2026-04-15 | F/T extraction (`libero_90_with_ft`) | done | 184 M NPZs + `dataset_stats_all.json` |
| 2026-04-15 | `libero_improved_ft_ctx4_anchor` checkpoint (side-encoder baseline) | done | `checkpoints/libero_improved_ft_ctx4_anchor/best_val.pt` (868 M, 62 epochs) |
| 2026-04-17 | Attempted cosmos-libero F/T fine-tune | **crashed** | Two hard crashes (12:14, 14:04 EDT). Disk was at 94%, 4× A4000 likely OOM under concurrent container load |
| 2026-04-19 | Disk cleanup | done | Freed 61 G (caches + `RH20T_cfg3.tar.gz`); `/` now 92% |
| 2026-04-19 | README: paper-method F/T section | done | This document |
| 2026-04-19 | `precompute_t5_libero90.py` written | done | Extends HF-shipped pkl with 100 missing commands (90 + 10) |
| 2026-04-19 | T5 precompute v1 | done, invalid | Keyed by `benchmark.get_task_names()` (with `KITCHEN_SCENE…` prefix). Dry-run later revealed dataset uses HDF5 `language_instruction` — only 3/90 tasks matched (150 episodes kept vs. 4500). Pkl discarded. |
| 2026-04-20 | Delete `models--google-t5--t5-11b` (85 G) | done | Freed 85 G (disk 90%, 377 G avail). Had to re-download for v2 fix. |
| 2026-04-20 | `train_ft_demo.py --dry_run` | done | Latent layout verified: cur_ft=2, fut_ft=7; video `(3, 41, 224, 224)`; but dataset filter caught the T5-key bug (150/4500 episodes). |
| 2026-04-20 | Fix: key by HDF5 `language_instruction`, drop scene prefix | done | `precompute_t5_libero90.py` rewritten; reads `data.attrs["problem_info"]`; 72 unique commands missing (libero_10 10/10 and 2/74 libero_90 already in HF pkl). |
| 2026-04-20 | T5 precompute v2 attempt, batch=8 | failed | `CUDA error: mapping of buffer object failed` on cross-GPU transfer with `device_map=auto` at batch size 8. Reverted to batch=1. |
| 2026-04-20 → 21 | T5 precompute v2, batch=1 (CPU) | done | 10 h 14 min on CPU (~520 s/prompt × 72) after CUDA runtime failed to init. 112 entries total. Coverage: libero_90 74/74, libero_10 10/10. Output: `$DATA_ROOT/libero_90_with_ft/t5_embeddings.pkl` (113 M). |
| 2026-04-21 | Fix CUDA driver (uvm reload) | pending | Now safe — no process holds `/dev/nvidia-uvm`. Run `! sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm` to recover GPU for training. |
| — | Re-run `train_ft_demo.py --dry_run` with corrected pkl | pending | Expect ≈ full episode match (was 150/4500 with the bad pkl). |
| — | `train_ft_demo.py --dry_run` | pending | Validates dataset shape `(C, 41, H, W)` and latent-idx alignment |
| — | `test_vram_fit.py` (single GPU, bf16, grad-ckpt) | pending | Find largest feasible batch/seq |
| — | Smoke training: 100 steps, ckpt every 25 | pending | Must survive overnight without crash before full run |
| — | Full fine-tune: `torchrun --nproc_per_node=4`, 5 k steps | pending | Target: beat `libero_improved_ft_ctx4_anchor` baseline on contact-heavy LIBERO tasks |
| — | Eval on contact tasks via `run_cosmos_libero.py` | pending | Compare vs `libero_improved_noft_ctx1` |

Update this table as each step moves; keep one row per step, no prose.


260423 Talk
- Existing model already reaches 100% success (Libero Spatial)
    - Recursive Test?
    - Check 100% one more time. (Test set was the subset of the training set?)
    - Fine-tuning reduces the performance
        - Gripper (6 dim) concat to proprioception
- Spiky F/T/Contact


Future plan;
Consider Robomimic
Jaeyoun: latent injection -> F/T -> Fine-tuning 5090 for libero 90
Minyoung: Base line performance check (100%?) -> Reading paper.

