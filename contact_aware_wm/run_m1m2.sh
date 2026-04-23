#!/usr/bin/env bash
# Orchestrator for M1/M2 of the contact-aware Cosmos Policy experiments
# on the matrix machine. Idempotent — safe to re-run any step.
#
# Usage:
#   bash contact_aware_wm/run_m1m2.sh <command>
#
# Commands:
#   check         Show what is / isn't ready (env, data, GPU).
#   env           Install the Python venv via uv (cu128 + libero group).
#   data          Download LIBERO-Cosmos-Policy spatial subset + t5.
#   m1a           Validate the dataset pipeline end-to-end (no training).
#   m1b           100-step smoke-test fine-tune on libero_spatial (V, no F/T).
#   m1b-lora      100-step smoke-test with LoRA adapters (requires peft).
#   m2-v          Full V baseline fine-tune (no F/T) on libero_spatial.
#   m2-vf         Full V+F baseline fine-tune (requires F/T NPZ).
#   m2-eval       Autoregressive eval on LIBERO-10; logs PSNR + FP rate.
#   all-m1        env + data + m1a + m1b in sequence.
#
# Environment overrides (defaults in parens):
#   REPO_ROOT              ($HOME/Projects/Contact-Aware-cosmos-policy)
#   BASE_DATASETS_DIR      ($HOME/data)
#   FT_DATA_DIR            ($BASE_DATASETS_DIR/libero_90_with_ft)
#   WANDB_MODE             (offline)
#   NUM_GPUS               (1)
#   MAX_ITER_SMOKE         (100)
#   MAX_ITER_FULL          (20000)
#   BATCH_SIZE             (1)
#   SUITE_SUBSET           (libero_spatial_regen)

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/Projects/Contact-Aware-cosmos-policy}"
BASE_DATASETS_DIR="${BASE_DATASETS_DIR:-$HOME/data}"
FT_DATA_DIR="${FT_DATA_DIR:-$BASE_DATASETS_DIR/libero_90_with_ft}"
WANDB_MODE="${WANDB_MODE:-offline}"
NUM_GPUS="${NUM_GPUS:-1}"
MAX_ITER_SMOKE="${MAX_ITER_SMOKE:-100}"
MAX_ITER_FULL="${MAX_ITER_FULL:-20000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SUITE_SUBSET="${SUITE_SUBSET:-libero_spatial_regen}"

LIBERO_DIR="$BASE_DATASETS_DIR/LIBERO-Cosmos-Policy"
LOG_DIR="$REPO_ROOT/contact_aware_wm/logs"
mkdir -p "$LOG_DIR"

# ---- helpers ---------------------------------------------------------------

log()  { printf '\033[36m[m1m2]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[m1m2]\033[0m %s\n' "$*" 1>&2; }
die()  { printf '\033[31m[m1m2]\033[0m %s\n' "$*" 1>&2; exit 1; }

have_uv()        { command -v uv >/dev/null 2>&1; }
have_venv()      { [ -x "$REPO_ROOT/.venv/bin/python" ]; }
have_libero()    { [ -f "$LIBERO_DIR/success_only/t5_embeddings.pkl" ]; }
have_ft_stats()  { [ -f "$FT_DATA_DIR/dataset_stats_all.json" ]; }

# uv run wrapper — always runs inside the Contact-Aware venv.
UV="uv run --extra cu128 --group libero --python 3.10"

run_uv() {
  cd "$REPO_ROOT"
  BASE_DATASETS_DIR="$BASE_DATASETS_DIR" HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" \
  WANDB_MODE="$WANDB_MODE" TOKENIZERS_PARALLELISM=false \
  $UV "$@"
}

notify_slack() {
  # Send a completion ping back to the Slack thread if research-bot is present.
  local msg="$1" logfile="${2:-}"
  local notify="$HOME/Projects/research-bot/notify.py"
  if [ -f "$notify" ]; then
    python "$notify" "$msg" ${logfile:+--log "$logfile"} \
      --channel C0AUKLVEG65 --thread 1776740072.985249 || true
  fi
}

# ---- commands --------------------------------------------------------------

cmd_check() {
  log "repo:            $REPO_ROOT"
  log "datasets:        $BASE_DATASETS_DIR"
  log "ft npz:          $FT_DATA_DIR"
  log "log dir:         $LOG_DIR"
  printf '\n'
  have_uv       && log "✓ uv present ($(uv --version))" || warn "✗ uv not installed"
  have_venv     && log "✓ .venv ready" || warn "✗ .venv missing — run: bash $0 env"
  have_libero   && log "✓ LIBERO spatial data present" \
                || warn "✗ LIBERO-Cosmos-Policy missing — run: bash $0 data"
  have_ft_stats && log "✓ F/T stats present" \
                || warn "✗ F/T dataset_stats_all.json missing (needed for use_ft=True)"
  printf '\n'
  if command -v nvidia-smi >/dev/null 2>&1; then
    log "GPU:"
    nvidia-smi --query-gpu=index,name,memory.free,memory.used --format=csv \
      | sed 's/^/    /'
  fi
}

cmd_env() {
  have_uv || die "uv is not installed. See https://docs.astral.sh/uv/"
  cd "$REPO_ROOT"
  log "Running uv sync (cu128 + libero). This may take 10+ min the first time."
  uv sync --extra cu128 --group libero --python 3.10 2>&1 \
    | tee "$LOG_DIR/env_uv_sync.log"
  log "uv sync complete."
  notify_slack "✅ Contact-Aware uv sync done" "$LOG_DIR/env_uv_sync.log"
}

cmd_data() {
  mkdir -p "$LIBERO_DIR"
  log "Downloading LIBERO-Cosmos-Policy spatial subset + stats + t5."
  # success_only/ is tiny (~2 GB); all_episodes/ is ~12 GB per suite and only
  # needed if you want failure-rollout supervision.
  hf download nvidia/LIBERO-Cosmos-Policy --repo-type dataset \
    --include 'success_only/t5_embeddings.pkl' \
    --include 'success_only/dataset_statistics*.json' \
    --include "success_only/${SUITE_SUBSET}/*" \
    --local-dir "$LIBERO_DIR" 2>&1 \
    | tee "$LOG_DIR/data_hf_download.log"
  log "Data download complete. Listing:"
  ls -la "$LIBERO_DIR/success_only" | sed 's/^/    /'
  notify_slack "✅ LIBERO-spatial data ready" "$LOG_DIR/data_hf_download.log"
}

cmd_m1a() {
  have_venv || die "Run 'bash $0 env' first."
  have_libero || die "Run 'bash $0 data' first."
  log "M1a — dataset dry run (no training; validates LIBERODataset)"
  # With F/T if stats exist, without F/T otherwise. Drop --no_ft once F/T
  # NPZs + stats are generated by extract_ft_libero.py.
  local ft_flag=""
  have_ft_stats || ft_flag="--no_ft"
  run_uv python contact_aware_wm/train_ft_demo.py --dry_run $ft_flag \
      --libero_data_dir "$LIBERO_DIR/success_only" \
      --ft_data_dir "$FT_DATA_DIR" \
      --t5_path "$LIBERO_DIR/success_only/t5_embeddings.pkl" \
      2>&1 | tee "$LOG_DIR/m1a_dryrun.log"
  log "M1a done — see $LOG_DIR/m1a_dryrun.log"
  notify_slack "✅ M1a dry-run passed" "$LOG_DIR/m1a_dryrun.log"
}

# Core single-GPU / few-GPU fine-tune via Cosmos's own trainer.
# All experiment-specific knobs are CLI overrides so nothing in the upstream
# config needs to be touched.
_train_one() {
  local tag="$1" ; shift
  local extra=( "$@" )

  have_venv   || die "Run 'bash $0 env' first."
  have_libero || die "Run 'bash $0 data' first."

  local logf="$LOG_DIR/${tag}.log"
  log "Training [$tag] — $NUM_GPUS GPU(s), batch=$BATCH_SIZE, max_iter=$MAX_ITER_SMOKE"
  log "Log: $logf"

  run_uv torchrun --nproc_per_node="$NUM_GPUS" --master_port=12341 \
    -m cosmos_policy.scripts.train \
    --config=cosmos_policy/config/config.py -- \
    experiment=cosmos_predict2_2b_480p_libero \
    trainer.max_iter="$MAX_ITER_SMOKE" \
    trainer.logging_iter=1 \
    trainer.grad_accum_iter=1 \
    trainer.run_validation=False \
    dataloader_train.batch_size="$BATCH_SIZE" \
    dataloader_train.num_workers=2 \
    dataloader_train.persistent_workers=False \
    checkpoint.save_iter="$MAX_ITER_SMOKE" \
    dataloader_train.dataset.data_dir="$LIBERO_DIR/success_only" \
    dataloader_train.dataset.rollout_data_dir="" \
    dataloader_train.dataset.demonstration_sampling_prob=1.0 \
    dataloader_train.dataset.t5_text_embeddings_path="$LIBERO_DIR/success_only/t5_embeddings.pkl" \
    "${extra[@]}" \
    2>&1 | tee "$logf"

  notify_slack "✅ [$tag] training run finished" "$logf"
}

cmd_m1b() {
  # Plain full fine-tune (V) at smoke-test scale. Expect OOM risk on a single
  # 24 GB 4090 — if so, retry with m1b-lora.
  _train_one m1b_smoke
}

cmd_m1b_lora() {
  # Writes a tiny config-patch shim that wraps the diffusion transformer with
  # peft LoRA adapters before training. Requires peft>=0.17.1 (already in deps).
  have_venv || die "Run 'bash $0 env' first."
  log "M1b-LoRA — 100 steps of LoRA fine-tune"
  local logf="$LOG_DIR/m1b_lora.log"
  # See contact_aware_wm/lora_patch.py for the monkey-patch that adds LoRA.
  export CAWM_ENABLE_LORA=1
  run_uv torchrun --nproc_per_node="$NUM_GPUS" --master_port=12341 \
    -m cosmos_policy.scripts.train \
    --config=cosmos_policy/config/config.py -- \
    experiment=cosmos_predict2_2b_480p_libero \
    trainer.max_iter="$MAX_ITER_SMOKE" \
    trainer.logging_iter=1 \
    trainer.grad_accum_iter=1 \
    trainer.run_validation=False \
    dataloader_train.batch_size="$BATCH_SIZE" \
    dataloader_train.num_workers=2 \
    dataloader_train.persistent_workers=False \
    checkpoint.save_iter="$MAX_ITER_SMOKE" \
    dataloader_train.dataset.data_dir="$LIBERO_DIR/success_only" \
    dataloader_train.dataset.rollout_data_dir="" \
    dataloader_train.dataset.demonstration_sampling_prob=1.0 \
    dataloader_train.dataset.t5_text_embeddings_path="$LIBERO_DIR/success_only/t5_embeddings.pkl" \
    2>&1 | tee "$logf"
  notify_slack "✅ M1b-LoRA finished" "$logf"
}

cmd_m2_v() {
  # Full V baseline (no F/T): longer run to collect real fidelity numbers.
  local prev=$MAX_ITER_SMOKE
  MAX_ITER_SMOKE="$MAX_ITER_FULL" _train_one m2_v
  MAX_ITER_SMOKE="$prev"
}

cmd_m2_vf() {
  # V+F: requires F/T NPZ + dataset_stats_all.json. Enables use_ft=True and
  # bumps state_t to 11 via dataset kwargs. If F/T NPZs are missing the
  # dataset will fill zeros and training becomes equivalent to V.
  have_ft_stats \
    || warn "No F/T stats at $FT_DATA_DIR — training will use zero F/T (not V+F)."
  local prev=$MAX_ITER_SMOKE
  MAX_ITER_SMOKE="$MAX_ITER_FULL" _train_one m2_vf \
    dataloader_train.dataset.use_ft=True \
    dataloader_train.dataset.ft_data_dir="$FT_DATA_DIR" \
    dataloader_train.dataset.ft_stats_path="$FT_DATA_DIR/dataset_stats_all.json" \
    model.config.state_t=11
  MAX_ITER_SMOKE="$prev"
}

cmd_m2_eval() {
  have_venv || die "Run 'bash $0 env' first."
  local logf="$LOG_DIR/m2_eval.log"
  log "M2-eval — autoregressive rollout on LIBERO-10 contact tasks"
  run_uv python contact_aware_wm/run_cosmos_ar_with_ft_plot.py \
      --all_tasks --num_autoreg_steps 15 \
      --proprio_mode predicted \
      --out_dir "$LOG_DIR/m2_eval_rollouts" \
      2>&1 | tee "$logf"
  notify_slack "✅ M2 eval done" "$logf"
}

cmd_all_m1() {
  cmd_check
  have_venv   || cmd_env
  have_libero || cmd_data
  cmd_m1a
  cmd_m1b
}

# ---- dispatch --------------------------------------------------------------

cmd="${1:-check}"; shift || true

case "$cmd" in
  check)     cmd_check ;;
  env)       cmd_env ;;
  data)      cmd_data ;;
  m1a)       cmd_m1a ;;
  m1b)       cmd_m1b ;;
  m1b-lora)  cmd_m1b_lora ;;
  m2-v)      cmd_m2_v ;;
  m2-vf)     cmd_m2_vf ;;
  m2-eval)   cmd_m2_eval ;;
  all-m1)    cmd_all_m1 ;;
  *)         die "Unknown command: $cmd. See header of this file for usage." ;;
esac
