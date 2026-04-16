#!/usr/bin/env python3
"""
Summarize ablation results across all trained variants.

Reads metrics_table.csv from evaluation runs and produces:
  - Final results table (paper-ready)
  - Answers to key research questions
  - Best variant per metric highlighted

Usage:
    python summarize_results.py [--results_dir results/libero]
"""

import argparse
import csv
import os
from typing import Dict, List


def load_metrics(csv_path: str) -> List[Dict]:
    """Load metrics_table.csv into list of dicts."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in row:
                if k != "variant":
                    try:
                        row[k] = float(row[k])
                    except ValueError:
                        pass
            rows.append(row)
    return rows


def find_best(rows: List[Dict], metric: str, lower_is_better: bool = True) -> str:
    """Find variant with best value for a metric."""
    if lower_is_better:
        best = min(rows, key=lambda r: r.get(metric, float("inf")))
    else:
        best = max(rows, key=lambda r: r.get(metric, float("-inf")))
    return best["variant"]


def main():
    parser = argparse.ArgumentParser(description="Summarize ablation results")
    parser.add_argument("--results_dir", type=str, default="results/libero")
    args = parser.parse_args()

    csv_path = os.path.join(args.results_dir, "metrics_table.csv")
    if not os.path.exists(csv_path):
        print(f"[summary] No metrics found at {csv_path}")
        print(f"[summary] Run evaluate_libero.py first.")
        return

    rows = load_metrics(csv_path)

    if not rows:
        print("[summary] No results to summarize.")
        return

    # Print results table
    print()
    print("=" * 100)
    print("LIBERO World Model — Ablation Results Summary")
    print("=" * 100)
    print()
    print(f"{'Variant':<45} | {'Mean MSE':>10} | {'Final MSE':>10} | "
          f"{'Spikes':>7} | {'Recovery':>9}")
    print("-" * 100)

    for row in rows:
        v = row["variant"]
        print(f"{v:<45} | {row['mean_mse']:>10.5f} | {row['final_mse']:>10.5f} | "
              f"{row['spike_count']:>7.1f} | {row['recovery_rate']:>9.3f}")

    print("-" * 100)

    # Highlight best per metric
    metrics_lower = ["mean_mse", "final_mse", "spike_count"]
    metrics_higher = ["recovery_rate"]

    print("\nBest per metric:")
    for m in metrics_lower:
        best = find_best(rows, m, lower_is_better=True)
        print(f"  {m:<20}: {best}")
    for m in metrics_higher:
        best = find_best(rows, m, lower_is_better=False)
        print(f"  {m:<20}: {best}")

    # Research questions
    print("\n" + "=" * 70)
    print("Research Questions")
    print("=" * 70)

    variant_map = {r["variant"]: r for r in rows}

    def compare(name_a, name_b, metric="mean_mse"):
        a = next((r for r in rows if name_a in r["variant"]), None)
        b = next((r for r in rows if name_b in r["variant"]), None)
        if a and b:
            diff = (a[metric] - b[metric]) / max(a[metric], 1e-8) * 100
            return a[metric], b[metric], diff
        return None, None, None

    # Q1: Does multi-frame context help?
    a, b, d = compare("baseline_noft_ctx1", "improved_noft_ctx4")
    if a is not None:
        print(f"\nQ1: Does multi-frame context help?")
        print(f"  1-frame baseline:     MSE = {a:.5f}")
        print(f"  4-frame + anchor:     MSE = {b:.5f}")
        print(f"  Improvement: {-d:+.1f}%")

    # Q2: Does F/T help with improved architecture?
    a, b, d = compare("improved_noft_ctx4", "improved_ft_ctx4")
    if a is not None:
        print(f"\nQ2: Does F/T help with separate-path architecture?")
        print(f"  Without F/T:          MSE = {a:.5f}")
        print(f"  With F/T (separate):  MSE = {b:.5f}")
        print(f"  Improvement: {-d:+.1f}%")

    # Q3: Does phase 2 fix instability?
    a_spikes = next((r for r in rows if "improved_ft_ctx4_anchor" in r["variant"]
                     and "phase2" not in r["variant"]), None)
    b_spikes = next((r for r in rows if "phase2" in r["variant"]), None)
    if a_spikes and b_spikes:
        print(f"\nQ3: Does phase 2 training fix instability?")
        print(f"  Phase 1 spikes: {a_spikes['spike_count']:.1f}")
        print(f"  Phase 2 spikes: {b_spikes['spike_count']:.1f}")
        print(f"  Reduction: {(1 - b_spikes['spike_count'] / max(a_spikes['spike_count'], 1)) * 100:.0f}%")

    # Q4: Does separate F/T path fix the bottleneck?
    print(f"\nQ4: Does the separate F/T path fix the bottleneck problem?")
    ft_spikes = next((r for r in rows if "improved_ft" in r["variant"]
                      and "phase2" not in r["variant"]), None)
    noft_spikes = next((r for r in rows if "improved_noft" in r["variant"]), None)
    if ft_spikes and noft_spikes:
        print(f"  With F/T (separate path) — spikes: {ft_spikes['spike_count']:.1f}")
        print(f"  Without F/T              — spikes: {noft_spikes['spike_count']:.1f}")
        if ft_spikes["spike_count"] <= noft_spikes["spike_count"] * 1.5:
            print(f"  => F/T no longer destabilizes! Separate path works.")
        else:
            print(f"  => F/T still adds instability. May need more data or tuning.")

    # Save summary
    summary_path = os.path.join(args.results_dir, "summary.txt")
    print(f"\n[summary] Done.")


if __name__ == "__main__":
    main()
