"""
Evaluate both world models on the test set.

Computes MSE, SSIM, LPIPS (mean ± std).
Generates:
  - results/metrics.txt         — comparison table
  - results/comparison_grid.png — visual comparison grid
  - results/ft_uncertainty_plot.png — F/T magnitude vs prediction improvement
"""

import os
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
import lpips
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import RH20TWorldModelDataset
from model import WorldModel


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(condition, device):
    """Load best checkpoint for a given condition."""
    ckpt_path = os.path.join("checkpoints", f"{condition}_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = WorldModel(condition=condition).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[eval] Loaded {condition} (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f})")
    return model


def compute_metrics(model, dataloader, condition, device, lpips_fn):
    """Compute per-sample MSE, SSIM, LPIPS. Returns arrays + visualization data."""
    all_mse = []
    all_ssim = []
    all_lpips = []
    all_ft_mag = []

    vis_inputs = []
    vis_preds = []
    vis_targets = []

    with torch.no_grad():
        for batch in dataloader:
            img_t = batch["img_t"].to(device)
            action = batch["action"].to(device)
            ft = batch["ft"].to(device)
            img_next = batch["img_next"].to(device)

            if condition == "image_only":
                pred = model(img_t, action)
            else:
                pred = model(img_t, action, ft)

            bs = img_t.size(0)

            # Per-sample MSE
            mse = ((pred - img_next) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
            all_mse.extend(mse.tolist())

            # LPIPS (expects [-1, 1])
            lp = lpips_fn(pred * 2 - 1, img_next * 2 - 1).squeeze().cpu().numpy()
            if lp.ndim == 0:
                lp = np.array([lp.item()])
            all_lpips.extend(lp.tolist())

            # F/T magnitude (L2 norm of force components, first 3 dims)
            ft_mag = torch.norm(ft[:, :3], dim=1).cpu().numpy()
            all_ft_mag.extend(ft_mag.tolist())

            # SSIM (per-image on CPU)
            pred_np = pred.cpu().numpy().transpose(0, 2, 3, 1)
            target_np = img_next.cpu().numpy().transpose(0, 2, 3, 1)
            for i in range(bs):
                s = ssim(target_np[i], pred_np[i], data_range=1.0, channel_axis=2)
                all_ssim.append(s)

            # Save for visualization
            if len(vis_inputs) * dataloader.batch_size < 64:
                vis_inputs.append(img_t.cpu())
                vis_preds.append(pred.cpu())
                vis_targets.append(img_next.cpu())

    metrics = {
        "mse": np.array(all_mse),
        "ssim": np.array(all_ssim),
        "lpips": np.array(all_lpips),
    }

    inputs = torch.cat(vis_inputs, dim=0) if vis_inputs else None
    preds = torch.cat(vis_preds, dim=0) if vis_preds else None
    targets = torch.cat(vis_targets, dim=0) if vis_targets else None

    return metrics, np.array(all_ft_mag), inputs, preds, targets


def save_comparison_grid(inputs, preds_baseline, preds_ft, targets, save_path, n=8):
    """Save grid: [input | pred_baseline | pred_ft | ground truth]."""
    n = min(n, len(inputs))
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    titles = ["Input (t)", "Pred (image only)", "Pred (image + F/T)", "Ground Truth (t+1)"]
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=13, fontweight="bold")

    for i in range(n):
        imgs = [
            inputs[i].permute(1, 2, 0).numpy(),
            preds_baseline[i].permute(1, 2, 0).numpy(),
            preds_ft[i].permute(1, 2, 0).numpy(),
            targets[i].permute(1, 2, 0).numpy(),
        ]
        for j, img in enumerate(imgs):
            axes[i, j].imshow(np.clip(img, 0, 1))
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[eval] Saved comparison grid: {save_path}")


def save_ft_plot(ft_mag, mse_baseline, mse_ft, save_path):
    """Plot F/T magnitude vs prediction improvement (delta MSE).

    This is the key plot testing whether F/T helps more when forces are larger.
    """
    delta_mse = mse_baseline - mse_ft  # positive = F/T helps

    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot
    ax.scatter(ft_mag, delta_mse, alpha=0.4, s=20, c="steelblue", edgecolors="none")

    # Trend line
    if len(ft_mag) > 5:
        z = np.polyfit(ft_mag, delta_mse, 1)
        p = np.poly1d(z)
        x_range = np.linspace(ft_mag.min(), ft_mag.max(), 100)
        ax.plot(x_range, p(x_range), "r-", linewidth=2, label=f"Trend (slope={z[0]:.4f})")
        ax.legend(fontsize=12)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("F/T signal magnitude (N)", fontsize=14)
    ax.set_ylabel("Prediction improvement with F/T (ΔMSE)", fontsize=14)
    ax.set_title("Does F/T conditioning help more during contact?", fontsize=14)
    ax.tick_params(labelsize=11)

    # Annotate regions
    ax.text(0.02, 0.98, "↑ F/T helps", transform=ax.transAxes,
            fontsize=10, va="top", color="green", fontweight="bold")
    ax.text(0.02, 0.02, "↓ F/T hurts", transform=ax.transAxes,
            fontsize=10, va="bottom", color="red", fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[eval] Saved F/T uncertainty plot: {save_path}")


def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] Device: {device}")

    # Load test data
    test_dataset = RH20TWorldModelDataset(split="test", augment=False)
    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=(device.type == "cuda"),
    )

    # Load LPIPS
    lpips_fn = lpips.LPIPS(net="vgg").to(device)
    lpips_fn.eval()

    # Load models
    model_baseline = load_model("image_only", device)
    model_ft = load_model("image_ft", device)

    # Evaluate
    print("\n[eval] Evaluating image_only (baseline)...")
    metrics_b, ft_mag_b, inputs_b, preds_b, targets_b = compute_metrics(
        model_baseline, test_loader, "image_only", device, lpips_fn
    )

    print("[eval] Evaluating image_ft (ours)...")
    metrics_f, ft_mag_f, inputs_f, preds_f, targets_f = compute_metrics(
        model_ft, test_loader, "image_ft", device, lpips_fn
    )

    # Print results table
    print("\n" + "┌" + "─" * 65 + "┐")
    print(f"│ {'Model':<20} │ {'MSE':>12} │ {'SSIM':>12} │ {'LPIPS':>12} │")
    print("├" + "─" * 22 + "┼" + "─" * 14 + "┼" + "─" * 14 + "┼" + "─" * 14 + "┤")

    for name, m in [("Image only", metrics_b), ("Image + F/T", metrics_f)]:
        mse_str = f"{m['mse'].mean():.5f}±{m['mse'].std():.5f}"
        ssim_str = f"{m['ssim'].mean():.4f}±{m['ssim'].std():.4f}"
        lpips_str = f"{m['lpips'].mean():.5f}±{m['lpips'].std():.5f}"
        print(f"│ {name:<20} │ {mse_str:>12} │ {ssim_str:>12} │ {lpips_str:>12} │")

    print("└" + "─" * 22 + "┴" + "─" * 14 + "┴" + "─" * 14 + "┴" + "─" * 14 + "┘")

    # Improvement
    mse_imp = (metrics_b["mse"].mean() - metrics_f["mse"].mean()) / metrics_b["mse"].mean() * 100
    ssim_imp = (metrics_f["ssim"].mean() - metrics_b["ssim"].mean()) / metrics_b["ssim"].mean() * 100
    lpips_imp = (metrics_b["lpips"].mean() - metrics_f["lpips"].mean()) / metrics_b["lpips"].mean() * 100

    print(f"\nF/T improvement: MSE {mse_imp:+.2f}%, SSIM {ssim_imp:+.2f}%, LPIPS {lpips_imp:+.2f}%")

    # Save results
    os.makedirs("results", exist_ok=True)

    # Save metrics text
    with open("results/metrics.txt", "w") as f:
        f.write("RH20T World Model — F/T Conditioning Experiment\n")
        f.write("=" * 65 + "\n\n")
        for name, m in [("Image only", metrics_b), ("Image + F/T", metrics_f)]:
            f.write(f"{name}:\n")
            f.write(f"  MSE:   {m['mse'].mean():.6f} ± {m['mse'].std():.6f}\n")
            f.write(f"  SSIM:  {m['ssim'].mean():.6f} ± {m['ssim'].std():.6f}\n")
            f.write(f"  LPIPS: {m['lpips'].mean():.6f} ± {m['lpips'].std():.6f}\n\n")
        f.write(f"F/T improvement: MSE {mse_imp:+.2f}%, SSIM {ssim_imp:+.2f}%, LPIPS {lpips_imp:+.2f}%\n")
    print("[eval] Saved metrics to results/metrics.txt")

    # Save comparison grid
    if preds_b is not None and preds_f is not None:
        save_comparison_grid(
            inputs_b, preds_b, preds_f, targets_b,
            save_path="results/comparison_grid.png",
            n=8,
        )

    # Save F/T uncertainty plot (key output!)
    save_ft_plot(
        ft_mag_b, metrics_b["mse"], metrics_f["mse"],
        save_path="results/ft_uncertainty_plot.png",
    )

    # Save raw per-sample metrics for further analysis
    np.savez(
        "results/per_sample_metrics.npz",
        mse_baseline=metrics_b["mse"],
        mse_ft=metrics_f["mse"],
        ssim_baseline=metrics_b["ssim"],
        ssim_ft=metrics_f["ssim"],
        lpips_baseline=metrics_b["lpips"],
        lpips_ft=metrics_f["lpips"],
        ft_magnitude=ft_mag_b,
    )
    print("[eval] Saved per-sample metrics to results/per_sample_metrics.npz")


if __name__ == "__main__":
    main()
