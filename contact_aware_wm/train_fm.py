"""
Training loop for the flow matching world model.

Uses Conditional Optimal Transport (OT) flow matching:
  x_0 ~ N(0, I)              (noise)
  x_1 = target image         (ground truth next frame)
  t ~ U[0, 1]                (random timestep)
  x_t = (1 - t) * x_0 + t * x_1    (interpolant)
  target velocity = x_1 - x_0       (constant-speed OT path)

Loss: MSE(v_theta(x_t, t, cond), target_velocity)

Supports:
  --context_frames N   Multi-frame context window (default 1 = backward compat)
  --use_anchor         First-frame anchoring (slot 0 = episode's first GT frame)
  --context_noise_std  Gaussian noise on memory frames during training (not anchor)
                       to close train-inference gap (WoVR / World4RL motivation)
  --chunk_size H       Temporal chunk size for action/F/T history

Usage:
    # Baseline (backward compat):
    python train_fm.py --condition image_ft --context_frames 1

    # Improved (multi-frame + anchor + noise):
    python train_fm.py --condition image_ft --context_frames 4 --use_anchor \\
        --context_noise_std 0.05 --epochs 100
"""

import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset import RH20TWorldModelDataset
from model_fm import FlowMatchingWorldModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def add_context_noise(context_imgs, noise_std, use_anchor):
    """Add Gaussian noise to memory frames during training.

    Noise is added to all context frames EXCEPT the anchor (slot 0)
    when use_anchor=True. This simulates the imperfect predicted frames
    the model will receive during autoregressive rollout, closing the
    train-inference gap.

    Args:
        context_imgs: (B, N, 3, 96, 96) context frames
        noise_std: standard deviation of Gaussian noise
        use_anchor: if True, skip noise on slot 0

    Returns:
        noisy_context: (B, N, 3, 96, 96)
    """
    if noise_std <= 0 or context_imgs.ndim != 5:
        return context_imgs

    noisy = context_imgs.clone()
    noise = torch.randn_like(noisy) * noise_std

    if use_anchor:
        # Zero out noise on anchor slot (slot 0) — keep it clean
        noise[:, 0] = 0

    noisy = (noisy + noise).clamp(0, 1)
    return noisy


def train(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_fm] Device: {device}")
    print(f"[train_fm] Condition: {args.condition}")
    print(f"[train_fm] Context frames: {args.context_frames}, "
          f"anchor: {args.use_anchor}, noise_std: {args.context_noise_std}")
    print(f"[train_fm] Chunk size: {args.chunk_size}, ODE steps: {args.ode_steps}")

    # Data
    train_dataset = RH20TWorldModelDataset(
        split="train", augment=True, chunk_size=args.chunk_size,
        context_frames=args.context_frames, use_anchor=args.use_anchor)
    val_dataset = RH20TWorldModelDataset(
        split="val", augment=False, chunk_size=args.chunk_size,
        context_frames=args.context_frames, use_anchor=args.use_anchor)

    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    # Model
    model = FlowMatchingWorldModel(
        condition=args.condition,
        context_frames=args.context_frames,
        use_anchor=args.use_anchor,
    ).to(device)
    print(f"[train_fm] Parameters: {model.count_params():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    mse_loss = nn.MSELoss()

    # Checkpoint and log dirs
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Include context config in checkpoint name for clarity
    tag = f"fm_{args.condition}"
    if args.context_frames > 1:
        tag += f"_ctx{args.context_frames}"
        if args.use_anchor:
            tag += "_anchor"
    ckpt_path = os.path.join("checkpoints", f"{tag}_best.pt")
    log_path = os.path.join("logs", f"{tag}_log.csv")

    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "train_loss", "val_loss", "val_mse_recon", "lr"])

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for batch in train_loader:
            context_imgs = batch["context_imgs"].to(device)  # (B, N, 3, 96, 96) or (B, 3, 96, 96)
            action = batch["action"].to(device)
            ft = batch["ft"].to(device)
            pad_mask = batch["pad_mask"].to(device)
            img_next = batch["img_next"].to(device)

            B = img_next.size(0)

            # Noisy context augmentation: add noise to memory frames during training
            # to close the train-inference gap (model sees imperfect inputs at rollout)
            context_imgs = add_context_noise(
                context_imgs, args.context_noise_std, args.use_anchor)

            # Flow matching
            x_0 = torch.randn_like(img_next)
            t = torch.rand(B, device=device)
            t_expand = t[:, None, None, None]
            x_t = (1 - t_expand) * x_0 + t_expand * img_next
            target_velocity = img_next - x_0

            ft_in = ft if args.condition == "image_ft" else None
            pred_velocity = model(x_t, context_imgs, action, ft_in, t, pad_mask)

            loss = mse_loss(pred_velocity, target_velocity)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * B
            train_count += B

        train_loss = train_loss_sum / max(train_count, 1)

        # ---- Validate (no noise augmentation) ----
        model.eval()
        val_loss_sum = 0.0
        val_mse_recon_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for batch in val_loader:
                context_imgs = batch["context_imgs"].to(device)
                action = batch["action"].to(device)
                ft = batch["ft"].to(device)
                pad_mask = batch["pad_mask"].to(device)
                img_next = batch["img_next"].to(device)

                B = img_next.size(0)

                # Flow matching loss (clean context, no noise)
                x_0 = torch.randn_like(img_next)
                t = torch.rand(B, device=device)
                t_expand = t[:, None, None, None]
                x_t = (1 - t_expand) * x_0 + t_expand * img_next
                target_velocity = img_next - x_0

                ft_in = ft if args.condition == "image_ft" else None
                pred_velocity = model(x_t, context_imgs, action, ft_in, t, pad_mask)
                loss = mse_loss(pred_velocity, target_velocity)
                val_loss_sum += loss.item() * B

                # Reconstruction MSE (periodically)
                if epoch % args.recon_every == 0 or epoch == args.epochs:
                    pred_img = model.sample(
                        context_imgs, action, ft_in,
                        num_steps=args.ode_steps, pad_mask=pad_mask)
                    recon_mse = ((pred_img - img_next) ** 2).mean().item()
                    val_mse_recon_sum += recon_mse * B

                val_count += B

        val_loss = val_loss_sum / max(val_count, 1)
        do_recon = epoch % args.recon_every == 0 or epoch == args.epochs
        val_mse_recon = val_mse_recon_sum / max(val_count, 1) if do_recon else -1.0

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        log_writer.writerow([
            epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
            f"{val_mse_recon:.6f}", f"{current_lr:.6f}",
        ])
        log_file.flush()

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "condition": args.condition,
                "ode_steps": args.ode_steps,
                "chunk_size": args.chunk_size,
                "context_frames": args.context_frames,
                "use_anchor": args.use_anchor,
            }, ckpt_path)

        best_marker = " *" if is_best else ""
        recon_str = f" | Recon MSE: {val_mse_recon:.5f}" if val_mse_recon >= 0 else ""
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train: {train_loss:.5f} | Val: {val_loss:.5f} | "
            f"Best: {best_val_loss:.5f}{best_marker}{recon_str}"
        )

    log_file.close()
    total_time = time.time() - start_time

    print(f"\n[train_fm] Done in {total_time:.1f}s")
    print(f"[train_fm] Best val loss: {best_val_loss:.6f}")
    print(f"[train_fm] Checkpoint: {ckpt_path}")
    print(f"[train_fm] Log: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train flow matching world model")
    parser.add_argument("--condition", type=str, default="image_only",
                        choices=["image_only", "image_ft"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ode_steps", type=int, default=20)
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument("--recon_every", type=int, default=10)
    # New: multi-frame context + anchor + noise
    parser.add_argument("--context_frames", type=int, default=1,
                        help="Number of context frames (1 = single frame, >1 = multi-frame)")
    parser.add_argument("--use_anchor", action="store_true",
                        help="Use first-frame anchoring (slot 0 = episode first GT frame)")
    parser.add_argument("--context_noise_std", type=float, default=0.0,
                        help="Gaussian noise std on memory frames during training (0 = off)")
    args = parser.parse_args()
    train(args)
