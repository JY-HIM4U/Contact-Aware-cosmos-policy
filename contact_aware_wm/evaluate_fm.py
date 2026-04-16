"""
Evaluate flow matching world models on the test set.

Supports both single-frame and multi-frame context models.
Computes MSE, SSIM, LPIPS (mean +/- std).
"""

import os
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
from model_fm import FlowMatchingWorldModel


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_baseline_model(condition, device):
    ckpt_path = os.path.join("checkpoints", f"{condition}_best.pt")
    if not os.path.exists(ckpt_path):
        return None
    model = WorldModel(condition=condition).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[eval] Loaded baseline {condition} (epoch {ckpt['epoch']})")
    return model


def load_fm_model(tag, device):
    """Load FM checkpoint by tag name (e.g. 'fm_image_ft_ctx4_anchor')."""
    ckpt_path = os.path.join("checkpoints", f"{tag}_best.pt")
    if not os.path.exists(ckpt_path):
        return None, {}

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    condition = ckpt.get("condition", "image_ft")
    ctx = ckpt.get("context_frames", 1)
    anc = ckpt.get("use_anchor", False)

    model = FlowMatchingWorldModel(
        condition=condition, context_frames=ctx, use_anchor=anc).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    meta = {
        "condition": condition,
        "context_frames": ctx,
        "use_anchor": anc,
        "ode_steps": ckpt.get("ode_steps", 20),
        "chunk_size": ckpt.get("chunk_size", 4),
        "epoch": ckpt["epoch"],
        "val_loss": ckpt.get("val_loss", -1),
    }
    print(f"[eval] Loaded {tag} (epoch {meta['epoch']}, ctx={ctx}, anchor={anc})")
    return model, meta


def compute_metrics(predict_fn, dataloader, device, lpips_fn):
    all_mse, all_ssim, all_lpips, all_ft_mag = [], [], [], []
    vis_inputs, vis_preds, vis_targets = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            context_imgs = batch["context_imgs"].to(device)
            action = batch["action"].to(device)
            ft = batch["ft"].to(device)
            pad_mask = batch["pad_mask"].to(device)
            img_next = batch["img_next"].to(device)

            pred = predict_fn(context_imgs, action, ft, pad_mask)
            bs = pred.size(0)

            mse = ((pred - img_next) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
            all_mse.extend(mse.tolist())

            lp = lpips_fn(pred * 2 - 1, img_next * 2 - 1).squeeze().cpu().numpy()
            if lp.ndim == 0:
                lp = np.array([lp.item()])
            all_lpips.extend(lp.tolist())

            if ft.ndim == 3:
                ft_current = ft[:, -1, :3]
            else:
                ft_current = ft[:, :3]
            ft_mag = torch.norm(ft_current, dim=1).cpu().numpy()
            all_ft_mag.extend(ft_mag.tolist())

            pred_np = pred.cpu().numpy().transpose(0, 2, 3, 1)
            target_np = img_next.cpu().numpy().transpose(0, 2, 3, 1)
            for i in range(bs):
                s = ssim(target_np[i], pred_np[i], data_range=1.0, channel_axis=2)
                all_ssim.append(s)

            if len(vis_inputs) * dataloader.batch_size < 64:
                vis_inputs.append(batch["img_t"].cpu())
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


def save_comparison_grid(inputs, preds_dict, targets, save_path, n=8):
    n = min(n, len(inputs))
    names = list(preds_dict.keys())
    ncols = 2 + len(names)
    fig, axes = plt.subplots(n, ncols, figsize=(3.5 * ncols, 3.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    titles = ["Input (t)"] + names + ["Ground Truth (t+1)"]
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold")
    for i in range(n):
        imgs = [inputs[i].permute(1, 2, 0).numpy()]
        for name in names:
            imgs.append(preds_dict[name][i].permute(1, 2, 0).numpy())
        imgs.append(targets[i].permute(1, 2, 0).numpy())
        for j, img in enumerate(imgs):
            axes[i, j].imshow(np.clip(img, 0, 1))
            axes[i, j].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[eval] Saved comparison grid: {save_path}")


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] Device: {device}")

    lpips_fn = lpips.LPIPS(net="vgg").to(device)
    lpips_fn.eval()

    # Discover all FM checkpoints
    ckpt_dir = "checkpoints"
    fm_tags = []
    for f in sorted(os.listdir(ckpt_dir)):
        if f.startswith("fm_") and f.endswith("_best.pt"):
            tag = f.replace("_best.pt", "")
            fm_tags.append(tag)

    print(f"[eval] Found FM checkpoints: {fm_tags}")

    # Build eval tasks: (name, predict_fn, dataloader)
    eval_tasks = []

    # Baseline regression
    baseline = load_baseline_model("image_only", device)
    if baseline is not None:
        ds_base = RH20TWorldModelDataset(split="test", augment=False, chunk_size=1)
        dl_base = DataLoader(ds_base, batch_size=64, shuffle=False,
                             num_workers=min(4, os.cpu_count() or 1),
                             pin_memory=(device.type == "cuda"))

        def make_baseline_fn(m):
            def fn(ctx, a, f, pm):
                return m(ctx, a)  # ctx is img_t when context_frames=1
            return fn
        eval_tasks.append(("Baseline (regression)", make_baseline_fn(baseline), dl_base))

    # FM models
    for tag in fm_tags:
        model, meta = load_fm_model(tag, device)
        if model is None:
            continue

        ctx = meta["context_frames"]
        anc = meta["use_anchor"]
        cond = meta["condition"]
        ode_s = meta["ode_steps"]
        chunk = meta["chunk_size"]

        ds = RH20TWorldModelDataset(
            split="test", augment=False, chunk_size=chunk,
            context_frames=ctx, use_anchor=anc)
        dl = DataLoader(ds, batch_size=64, shuffle=False,
                        num_workers=min(4, os.cpu_count() or 1),
                        pin_memory=(device.type == "cuda"))

        # Build display name
        parts = [f"FM-{ctx}f"]
        if anc:
            parts.append("anchor")
        parts.append(f"({cond.replace('image_', '')})")
        name = " ".join(parts)

        def make_fm_fn(m, c, s):
            def fn(ctx_imgs, a, f, pm):
                ft_in = f if c == "image_ft" else None
                return m.sample(ctx_imgs, a, ft_in, num_steps=s, pad_mask=pm)
            return fn
        eval_tasks.append((name, make_fm_fn(model, cond, ode_s), dl))

    # Evaluate
    all_metrics = {}
    all_preds = {}
    shared_inputs = None
    shared_targets = None

    for name, pred_fn, loader in eval_tasks:
        print(f"\n[eval] Evaluating: {name}")
        metrics, ft_mag, inputs, preds, targets = compute_metrics(
            pred_fn, loader, device, lpips_fn)
        all_metrics[name] = metrics
        all_preds[name] = preds
        if shared_inputs is None:
            shared_inputs = inputs
            shared_targets = targets

    # Print results
    print("\n" + "=" * 85)
    print(f"{'Model':<30} | {'MSE':>16} | {'SSIM':>14} | {'LPIPS':>16}")
    print("-" * 85)
    for name, m in all_metrics.items():
        mse_str = f"{m['mse'].mean():.5f}+/-{m['mse'].std():.5f}"
        ssim_str = f"{m['ssim'].mean():.4f}+/-{m['ssim'].std():.4f}"
        lpips_str = f"{m['lpips'].mean():.5f}+/-{m['lpips'].std():.5f}"
        print(f"{name:<30} | {mse_str:>16} | {ssim_str:>14} | {lpips_str:>16}")
    print("=" * 85)

    # Save
    os.makedirs("results", exist_ok=True)
    with open("results/fm_metrics.txt", "w") as f:
        f.write("RH20T World Model — Flow Matching Evaluation\n")
        f.write("=" * 65 + "\n\n")
        for name, m in all_metrics.items():
            f.write(f"{name}:\n")
            f.write(f"  MSE:   {m['mse'].mean():.6f} +/- {m['mse'].std():.6f}\n")
            f.write(f"  SSIM:  {m['ssim'].mean():.6f} +/- {m['ssim'].std():.6f}\n")
            f.write(f"  LPIPS: {m['lpips'].mean():.6f} +/- {m['lpips'].std():.6f}\n\n")
    print("[eval] Saved metrics to results/fm_metrics.txt")

    if shared_inputs is not None and len(all_preds) > 0:
        save_comparison_grid(
            shared_inputs, all_preds, shared_targets,
            save_path="results/fm_comparison_grid.png", n=8)

    print("[eval] Done.")


if __name__ == "__main__":
    main()
