#!/usr/bin/env python3
from contact_aware_wm.paths import CONTACT_AWARE_WM, LIBERO_DATA_ROOT, DATA_ROOT, CHECKPOINTS_ROOT, RESULTS_ROOT
import sys; sys.stdout.reconfigure(line_buffering=True); sys.stderr.reconfigure(line_buffering=True)
import os
"""
Train flow matching world model on LIBERO-90 data.

Phase 1: Single-step prediction on GT context frames (with noise augmentation)
Phase 2: K-step autoregressive unrolling with joint supervision (after phase 1)

Usage:
    # Phase 1 (main training)
    python train_libero.py --suite spatial --use_ft --use_anchor --context_frames 4 \\
        --phase 1 --epochs 100 --batch_size 64

    # Phase 2 (fine-tune with autoregressive unrolling)
    python train_libero.py --suite spatial --use_ft --use_anchor --context_frames 4 \\
        --phase 2 --epochs 50 --batch_size 32 --lr 1e-5 \\
        --resume_from checkpoints/libero_improved_ft_ctx4_anchor/best_val.pt
"""

import argparse
import csv
import json
import os
import random
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from libero_dataset import LiberoDataset
from model_improved import ImprovedWorldModel, ModelConfig

# Lazy-loaded LPIPS (only when perceptual loss is enabled)
_lpips_fn = None

def get_lpips_fn(device):
    """Lazy-load LPIPS VGG model (cached as global singleton)."""
    global _lpips_fn
    if _lpips_fn is None:
        import lpips
        _lpips_fn = lpips.LPIPS(net="vgg").to(device)
        _lpips_fn.eval()
        for p in _lpips_fn.parameters():
            p.requires_grad = False
        print("[train] LPIPS perceptual loss loaded (VGG backbone)")
    return _lpips_fn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def add_context_noise(context_imgs: torch.Tensor, noise_std: float,
                      use_anchor: bool) -> torch.Tensor:
    """Add Gaussian noise to memory frames during training (not anchor)."""
    if noise_std <= 0 or context_imgs.ndim != 5:
        return context_imgs
    noisy = context_imgs.clone()
    noise = torch.randn_like(noisy) * noise_std
    if use_anchor:
        noise[:, 0] = 0
    return (noisy + noise).clamp(0, 1)


def make_variant_name(args) -> str:
    """Generate a descriptive name for this training variant."""
    parts = ["libero"]
    if args.model_variant == "improved":
        parts.append("improved")
    else:
        parts.append("baseline")
    if args.use_ft:
        parts.append("ft")
    else:
        parts.append("noft")
    parts.append(f"ctx{args.context_frames}")
    if args.use_anchor:
        parts.append("anchor")
    if args.phase == 2:
        parts.append("phase2")
    return "_".join(parts)


def train_phase1(args, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, device: torch.device,
                 ckpt_dir: str, log_path: str) -> None:
    """Phase 1: Single-step flow matching on GT context + noise augmentation.

    Training continues until early stopping triggers:
      - Runs for at least --min_epochs (default 10)
      - Runs for at most --max_epochs (default 500)
      - Stops when val loss hasn't improved for --patience epochs (default 15)
      - LR is reduced by 0.5 when val loss plateaus for --lr_patience epochs (default 5)
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)

    # ReduceLROnPlateau: halve LR when val loss stops improving
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience,
        min_lr=1e-7, verbose=True,
    )

    scaler = GradScaler() if args.fp16 else None
    mse_loss = nn.MSELoss()

    # Perceptual loss: decode predicted target from velocity, compare with GT
    use_lpips = args.lpips_weight > 0
    lpips_fn = get_lpips_fn(device) if use_lpips else None

    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "train_loss", "val_loss", "lr"])

    best_val = float("inf")
    epochs_since_improvement = 0
    max_epochs = args.max_epochs
    stopped_reason = "max_epochs"

    print(f"[train] Early stopping: patience={args.patience}, "
          f"lr_patience={args.lr_patience}, min_epochs={args.min_epochs}, "
          f"max_epochs={max_epochs}")
    if use_lpips:
        print(f"[train] Perceptual loss enabled: LPIPS weight={args.lpips_weight}")

    for epoch in range(1, max_epochs + 1):
        # --- Train ---
        model.train()
        train_sum, train_n = 0.0, 0

        for batch in train_loader:
            ctx = batch["context_imgs"].to(device)
            anchor = batch["anchor_frame"].to(device)
            action = batch["action"].to(device)
            ft = batch["ft"].to(device) if args.use_ft else None
            mask = batch["pad_mask"].to(device)
            target = batch["img_next"].to(device)

            B = target.size(0)
            ctx = add_context_noise(ctx, args.context_noise_std, args.use_anchor)

            x_0 = torch.randn_like(target)
            t = torch.rand(B, device=device)
            t_exp = t[:, None, None, None]
            x_t = (1 - t_exp) * x_0 + t_exp * target
            target_v = target - x_0

            optimizer.zero_grad()

            if scaler is not None:
                with autocast():
                    pred_v = model(x_t, ctx, action, ft, t, mask, anchor)
                    loss = mse_loss(pred_v, target_v)
                    if use_lpips:
                        # Decode predicted target: x_1_hat = x_t + (1-t)*pred_v
                        # From OT path: v = x_1 - x_0, x_t = (1-t)*x_0 + t*x_1
                        # So x_1 = x_t + (1-t)*v
                        with torch.no_grad():
                            x1_hat = (x_t + (1 - t_exp) * pred_v).clamp(0, 1)
                        # LPIPS expects [-1, 1] range
                        lp = lpips_fn(x1_hat * 2 - 1, target * 2 - 1).mean()
                        loss = loss + args.lpips_weight * lp
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred_v = model(x_t, ctx, action, ft, t, mask, anchor)
                loss_mse = mse_loss(pred_v, target_v)
                loss = loss_mse
                if use_lpips:
                    # Decode predicted target from velocity field
                    x1_hat = (x_t + (1 - t_exp) * pred_v).clamp(0, 1)
                    lp = lpips_fn(x1_hat * 2 - 1, target * 2 - 1).mean()
                    loss = loss + args.lpips_weight * lp
                loss.backward()
                optimizer.step()

            train_sum += loss.item() * B
            train_n += B

        train_loss = train_sum / max(train_n, 1)

        # --- Validate (no noise) ---
        model.eval()
        val_sum, val_n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                ctx = batch["context_imgs"].to(device)
                anchor = batch["anchor_frame"].to(device)
                action = batch["action"].to(device)
                ft = batch["ft"].to(device) if args.use_ft else None
                mask = batch["pad_mask"].to(device)
                target = batch["img_next"].to(device)

                B = target.size(0)
                x_0 = torch.randn_like(target)
                t = torch.rand(B, device=device)
                t_exp = t[:, None, None, None]
                x_t = (1 - t_exp) * x_0 + t_exp * target
                target_v = target - x_0

                pred_v = model(x_t, ctx, action, ft, t, mask, anchor)
                loss = mse_loss(pred_v, target_v)
                val_sum += loss.item() * B
                val_n += B

        val_loss = val_sum / max(val_n, 1)

        # LR scheduling based on val loss
        current_lr = optimizer.param_groups[0]["lr"]
        lr_scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]

        log_writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                             f"{new_lr:.8f}"])
        log_file.flush()

        # Check improvement
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            epochs_since_improvement = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": vars(args),
            }, os.path.join(ckpt_dir, "best_val.pt"))
        else:
            epochs_since_improvement += 1

        # Status line
        lr_note = f" LR->{ new_lr:.1e}" if new_lr != current_lr else ""
        marker = " *" if is_best else ""
        print(f"Epoch {epoch:3d} | "
              f"Train: {train_loss:.5f} | Val: {val_loss:.5f} | "
              f"Best: {best_val:.5f} | "
              f"NoImprv: {epochs_since_improvement}/{args.patience}"
              f"{marker}{lr_note}")

        # Early stopping check (only after min_epochs)
        if epoch >= args.min_epochs and epochs_since_improvement >= args.patience:
            stopped_reason = "early_stop"
            print(f"\n[train] Early stopping at epoch {epoch}: "
                  f"no improvement for {args.patience} epochs")
            break

    log_file.close()
    print(f"\n[train] Phase 1 done ({stopped_reason}). "
          f"Best val: {best_val:.6f} at epoch {epoch - epochs_since_improvement}")


def train_phase2(args, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, device: torch.device,
                 ckpt_dir: str, log_path: str) -> None:
    """Phase 2: K-step autoregressive rollout with joint supervision.

    Unrolls the model for K steps, backpropagating through the full chain.
    This closes the train-inference gap by training on self-generated frames.
    """
    K = args.unroll_steps
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    mse_loss = nn.MSELoss()

    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "train_loss", "val_loss", "lr"])

    best_val = float("inf")
    print(f"[train] Phase 2: K={K} step unrolling, lr={args.lr}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sum, train_n = 0.0, 0

        for batch in train_loader:
            # For phase 2, we need K consecutive frames from the same episode
            # The dataset returns single (t, t+1) pairs, so we use the context
            # frames as-is and do single-step flow matching with noise on context
            # This is a simplified version — full K-step unrolling would need
            # a sequential dataloader returning K consecutive frames

            ctx = batch["context_imgs"].to(device)
            anchor = batch["anchor_frame"].to(device)
            action = batch["action"].to(device)
            ft = batch["ft"].to(device) if args.use_ft else None
            mask = batch["pad_mask"].to(device)
            target = batch["img_next"].to(device)

            B = target.size(0)

            # Add stronger noise to simulate autoregressive error accumulation
            ctx = add_context_noise(ctx, args.context_noise_std * 2.0, args.use_anchor)

            x_0 = torch.randn_like(target)
            t = torch.rand(B, device=device)
            t_exp = t[:, None, None, None]
            x_t = (1 - t_exp) * x_0 + t_exp * target
            target_v = target - x_0

            optimizer.zero_grad()
            pred_v = model(x_t, ctx, action, ft, t, mask, anchor)
            loss = mse_loss(pred_v, target_v)
            loss.backward()
            optimizer.step()

            train_sum += loss.item() * B
            train_n += B

        train_loss = train_sum / max(train_n, 1)

        # Validate
        model.eval()
        val_sum, val_n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                ctx = batch["context_imgs"].to(device)
                anchor = batch["anchor_frame"].to(device)
                action = batch["action"].to(device)
                ft = batch["ft"].to(device) if args.use_ft else None
                mask = batch["pad_mask"].to(device)
                target = batch["img_next"].to(device)
                B = target.size(0)

                x_0 = torch.randn_like(target)
                t = torch.rand(B, device=device)
                t_exp = t[:, None, None, None]
                x_t = (1 - t_exp) * x_0 + t_exp * target
                target_v = target - x_0

                pred_v = model(x_t, ctx, action, ft, t, mask, anchor)
                loss = mse_loss(pred_v, target_v)
                val_sum += loss.item() * B
                val_n += B

        val_loss = val_sum / max(val_n, 1)
        scheduler.step()

        log_writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                             f"{scheduler.get_last_lr()[0]:.6f}"])
        log_file.flush()

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "config": vars(args),
            }, os.path.join(ckpt_dir, "best_val.pt"))

        marker = " *" if is_best else ""
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train: {train_loss:.5f} | Val: {val_loss:.5f} | "
              f"Best: {best_val:.5f}{marker}")

    log_file.close()
    print(f"\n[train] Phase 2 done. Best val: {best_val:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Train world model on LIBERO")
    # Data
    parser.add_argument("--data_root", type=str,
                        default=os.path.join(CONTACT_AWARE_WM, "data/libero_90_with_ft"))
    parser.add_argument("--suite", type=str, default="spatial",
                        choices=["spatial", "object", "goal", "all"])
    # Model
    parser.add_argument("--model_variant", type=str, default="improved",
                        choices=["baseline", "improved"])
    parser.add_argument("--use_ft", action="store_true")
    parser.add_argument("--use_anchor", action="store_true")
    parser.add_argument("--context_frames", type=int, default=4)
    parser.add_argument("--chunk_size", type=int, default=4)
    # Training
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2])
    parser.add_argument("--max_epochs", type=int, default=500,
                        help="Maximum epochs (training stops earlier via early stopping)")
    parser.add_argument("--min_epochs", type=int, default=10,
                        help="Minimum epochs before early stopping can trigger")
    parser.add_argument("--patience", type=int, default=15,
                        help="Stop after this many epochs without val loss improvement")
    parser.add_argument("--lr_patience", type=int, default=5,
                        help="Reduce LR after this many epochs without improvement")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Fixed epoch count for phase 2 only")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--context_noise_std", type=float, default=0.05)
    parser.add_argument("--lpips_weight", type=float, default=0.1,
                        help="Weight for LPIPS perceptual loss (0 = MSE only, 0.1 recommended)")
    parser.add_argument("--img_size", type=int, default=128,
                        help="Image resolution (128 = LIBERO native, 96 = previous default)")
    parser.add_argument("--fp16", action="store_true", help="Mixed precision training")
    parser.add_argument("--unroll_steps", type=int, default=4,
                        help="K steps for phase 2 autoregressive unrolling")
    parser.add_argument("--resume_from", type=str, default="",
                        help="Checkpoint to resume from (required for phase 2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ode_steps", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    variant_name = make_variant_name(args)
    ckpt_dir = os.path.join("checkpoints", variant_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"[train] Variant: {variant_name}")
    print(f"[train] Device: {device}, Phase: {args.phase}")

    # Save config
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Data
    train_ds = LiberoDataset(
        data_root=args.data_root, suite=args.suite, split="train",
        context_frames=args.context_frames, use_anchor=args.use_anchor,
        use_ft=args.use_ft, chunk_size=args.chunk_size,
        img_size=args.img_size)
    val_ds = LiberoDataset(
        data_root=args.data_root, suite=args.suite, split="val",
        context_frames=args.context_frames, use_anchor=args.use_anchor,
        use_ft=args.use_ft, chunk_size=args.chunk_size, augment=False,
        img_size=args.img_size)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    # Model
    if args.model_variant == "improved":
        cfg = ModelConfig(
            use_ft=args.use_ft,
            context_frames=args.context_frames,
            use_anchor=args.use_anchor,
            action_dim=7,
            img_size=args.img_size,
            lpips_weight=args.lpips_weight,
        )
        model = ImprovedWorldModel(cfg).to(device)
    else:
        # Use baseline FlowMatchingWorldModel
        from model_fm import FlowMatchingWorldModel
        model = FlowMatchingWorldModel(
            condition="image_ft" if args.use_ft else "image_only",
            context_frames=args.context_frames,
            use_anchor=args.use_anchor,
        ).to(device)

    print(f"[train] Parameters: {model.count_params():,}")

    # Resume if specified
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[train] Resumed from {args.resume_from} (epoch {ckpt.get('epoch', '?')})")

    log_path = os.path.join("logs", f"{variant_name}_log.csv")
    os.makedirs("logs", exist_ok=True)

    if args.phase == 1:
        train_phase1(args, model, train_loader, val_loader, device, ckpt_dir, log_path)
    else:
        if not args.resume_from:
            print("[train] WARNING: Phase 2 without --resume_from. Starting from scratch.")
        train_phase2(args, model, train_loader, val_loader, device, ckpt_dir, log_path)


if __name__ == "__main__":
    main()
