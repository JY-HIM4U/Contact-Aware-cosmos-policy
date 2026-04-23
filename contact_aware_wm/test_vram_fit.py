#!/usr/bin/env python3
"""Test whether Cosmos Policy 2B fits in VRAM for fine-tuning on a single GPU.

This script:
  1. Loads the pretrained checkpoint
  2. Builds a synthetic F/T-augmented batch (state_t=11, chunk_duration=41)
  3. Runs a single forward + backward pass
  4. Reports peak VRAM usage

Usage:
    cd /home/jaeyoun/cosmos-policy
    conda activate cosmos
    CUDA_VISIBLE_DEVICES=0 python contact_aware_wm/test_vram_fit.py
    CUDA_VISIBLE_DEVICES=0 python contact_aware_wm/test_vram_fit.py --grad_checkpoint
"""
import os
import sys
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn


def report_vram(label=""):
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"  [{label}] allocated={allocated:.2f}GB reserved={reserved:.2f}GB peak={peak:.2f}GB")


def make_fake_batch(batch_size=1, state_t=11, chunk_size=16, image_size=224, device="cuda"):
    """Build a minimal synthetic batch matching the F/T-augmented layout."""
    n_dup = 4
    T_raw = 1 + (state_t - 1) * n_dup  # 41

    batch = {
        "video": torch.randint(0, 256, (batch_size, 3, T_raw, image_size, image_size),
                               dtype=torch.uint8, device=device),
        "actions": torch.randn(batch_size, chunk_size, 7, device=device),
        "proprio": torch.randn(batch_size, 9, device=device),
        "future_proprio": torch.randn(batch_size, 9, device=device),
        "current_ft": torch.randn(batch_size, 6, device=device),
        "future_ft": torch.randn(batch_size, 6, device=device),
        "t5_text_embeddings": torch.randn(batch_size, 512, 4096, device=device),
        "t5_text_mask": torch.ones(batch_size, 512, dtype=torch.int64, device=device),
        "fps": 16 * torch.ones(batch_size, device=device),
        "padding_mask": torch.zeros(batch_size, 1, image_size, image_size, device=device),
        "image_size": image_size * torch.ones(batch_size, 4, device=device),
        "rollout_data_mask": torch.zeros(batch_size, dtype=torch.long, device=device),
        "world_model_sample_mask": torch.zeros(batch_size, dtype=torch.long, device=device),
        "value_function_sample_mask": torch.zeros(batch_size, dtype=torch.long, device=device),
        "value_function_return": torch.rand(batch_size, device=device),
        # Latent indices (state_t=11 layout)
        "current_proprio_latent_idx": torch.full((batch_size,), 1, dtype=torch.long, device=device),
        "current_ft_latent_idx": torch.full((batch_size,), 2, dtype=torch.long, device=device),
        "current_wrist_image_latent_idx": torch.full((batch_size,), 3, dtype=torch.long, device=device),
        "current_image_latent_idx": torch.full((batch_size,), 4, dtype=torch.long, device=device),
        "action_latent_idx": torch.full((batch_size,), 5, dtype=torch.long, device=device),
        "future_proprio_latent_idx": torch.full((batch_size,), 6, dtype=torch.long, device=device),
        "future_ft_latent_idx": torch.full((batch_size,), 7, dtype=torch.long, device=device),
        "future_wrist_image_latent_idx": torch.full((batch_size,), 8, dtype=torch.long, device=device),
        "future_image_latent_idx": torch.full((batch_size,), 9, dtype=torch.long, device=device),
        "value_latent_idx": torch.full((batch_size,), 10, dtype=torch.long, device=device),
    }
    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_checkpoint", action="store_true",
                        help="Apply activation checkpointing to save VRAM")
    parser.add_argument("--fp16", action="store_true", default=True)
    args = parser.parse_args()

    device = "cuda"
    torch.cuda.reset_peak_memory_stats()

    # --- 1. Load model ---
    print("[vram] Loading Cosmos Policy model...")
    from cosmos_policy.experiments.robot.cosmos_utils import get_model

    class ModelCfg:
        config = "cosmos_predict2_2b_480p_libero__inference_only"
        ckpt_path = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B"
        config_file = "cosmos_policy/config/config.py"

    model, cosmos_config = get_model(ModelCfg())
    report_vram("after model load")

    # Override state_t for the new layout
    if hasattr(model, 'config'):
        model.config.state_t = 11
        model.config.min_num_conditional_frames = 5
        model.config.max_num_conditional_frames = 5
    # Override video_noise_multiplier for new state_t
    import math
    if hasattr(model, 'video_noise_multiplier'):
        model.video_noise_multiplier = math.sqrt(11)

    model.train()

    if args.fp16:
        model = model.half()
        print("[vram] Model converted to fp16")
    report_vram("after model.train()")

    # --- 2. Optionally apply activation checkpointing ---
    if args.grad_checkpoint:
        print("[vram] Applying activation checkpointing...")
        from cosmos_policy._src.imaginaire.utils.fsdp_helper import apply_fsdp_checkpointing
        # Find transformer block classes
        from cosmos_policy._src.predict2.networks.wan2pt1 import WanAttentionBlock
        apply_fsdp_checkpointing(model, [WanAttentionBlock])
        report_vram("after activation checkpointing")

    # --- 3. Build fake batch ---
    print(f"[vram] Building fake batch (batch_size={args.batch_size})...")
    batch = make_fake_batch(batch_size=args.batch_size, device=device)
    if args.fp16:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point() and k != "video":
                batch[k] = v.half()
    report_vram("after batch creation")

    # --- 4. Forward + backward ---
    print("[vram] Running forward pass...")
    try:
        with torch.cuda.amp.autocast(enabled=args.fp16):
            output_batch, loss = model.training_step(batch, 0)
        report_vram("after forward")

        print(f"[vram] Loss: {loss.item():.4f}")
        print("[vram] Running backward pass...")
        loss.backward()
        report_vram("after backward")

        print(f"\n[vram] SUCCESS — peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        total_gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"[vram] GPU total: {total_gpu_mem:.1f} GB")
        headroom = total_gpu_mem - torch.cuda.max_memory_allocated() / 1e9
        print(f"[vram] Headroom: {headroom:.2f} GB")
        if headroom < 1.0:
            print("[vram] WARNING: Very tight! Consider --grad_checkpoint or smaller batch.")

    except torch.cuda.OutOfMemoryError as e:
        print(f"\n[vram] OOM! Peak before crash: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        print(f"[vram] GPU total: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
        if not args.grad_checkpoint:
            print("[vram] Try: python test_vram_fit.py --grad_checkpoint")
        else:
            print("[vram] Even with checkpointing, doesn't fit. Need FSDP multi-GPU or LoRA.")
        sys.exit(1)


if __name__ == "__main__":
    main()
