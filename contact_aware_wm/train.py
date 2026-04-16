"""
Training loop for the RH20T world model.

Usage:
    python train.py --condition image_only --epochs 50 --seed 42
    python train.py --condition image_ft   --epochs 50 --seed 42
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
from model import WorldModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")
    print(f"[train] Condition: {args.condition}")
    print(f"[train] Seed: {args.seed}")

    # Data
    train_dataset = RH20TWorldModelDataset(split="train", augment=True)
    val_dataset = RH20TWorldModelDataset(split="val", augment=False)

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
    model = WorldModel(condition=args.condition).to(device)
    print(f"[train] Parameters: {model.count_params():,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    # Checkpoint and log dirs
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    ckpt_path = os.path.join("checkpoints", f"{args.condition}_best.pt")
    log_path = os.path.join("logs", f"{args.condition}_log.csv")

    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "train_loss", "val_loss", "lr"])

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for batch in train_loader:
            img_t = batch["img_t"].to(device)
            action = batch["action"].to(device)
            ft = batch["ft"].to(device)
            img_next = batch["img_next"].to(device)

            if args.condition == "image_only":
                pred = model(img_t, action)
            else:
                pred = model(img_t, action, ft)

            loss = mse_loss(pred, img_next) + 0.1 * l1_loss(pred, img_next)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * img_t.size(0)
            train_count += img_t.size(0)

        train_loss = train_loss_sum / max(train_count, 1)

        # Validate
        model.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for batch in val_loader:
                img_t = batch["img_t"].to(device)
                action = batch["action"].to(device)
                ft = batch["ft"].to(device)
                img_next = batch["img_next"].to(device)

                if args.condition == "image_only":
                    pred = model(img_t, action)
                else:
                    pred = model(img_t, action, ft)

                loss = mse_loss(pred, img_next) + 0.1 * l1_loss(pred, img_next)
                val_loss_sum += loss.item() * img_t.size(0)
                val_count += img_t.size(0)

        val_loss = val_loss_sum / max(val_count, 1)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Log
        log_writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{current_lr:.6f}"])
        log_file.flush()

        # Save best
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "condition": args.condition,
            }, ckpt_path)

        best_marker = " *" if is_best else ""
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"Best: {best_val_loss:.4f}{best_marker}"
        )

    log_file.close()
    total_time = time.time() - start_time

    print(f"\n[train] Done in {total_time:.1f}s")
    print(f"[train] Best val loss: {best_val_loss:.6f}")
    print(f"[train] Checkpoint: {ckpt_path}")
    print(f"[train] Log: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RH20T world model")
    parser.add_argument("--condition", type=str, default="image_only",
                        choices=["image_only", "image_ft"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)
