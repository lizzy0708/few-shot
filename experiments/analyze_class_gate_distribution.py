"""
analyze_class_gate_distribution.py — Experiment 2 of 4, Stage 1, task 1.

Diagnostic (read-only, NO retraining): histogram the class_gate_net sigmoid output
values from the `gated_nodg` checkpoints, run over each fold's domain-pool data
(reuses experiments.analyze_domain_invariance_tsne.build_domain_pool_dataset).

Question: are gate values pushed to the extremes (near-binary hard mask) or
clustered near 0.5 (soft, barely-suppressing gate)? A gate concentrated near 0.5
would be a plausible structural explanation for z_inv = z*gate still leaking
domain information (Experiment 1 finding: domain-acc 57-65% vs 20% chance).

Per task instructions: seed 0 across all 4 folds is sufficient (representative
subset), rather than all 3 seeds x 4 folds.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.eval_gated_folds import GATED_FOLDS, ckpt_path, load_gated_model
from experiments.analyze_domain_invariance_tsne import build_domain_pool_dataset

OUT_DIR = "out/analyze/class_gate_distribution"
REPRESENTATIVE_SEED = 0


@torch.no_grad()
def collect_gate_values(model, loader):
    gates = []
    for batch in loader:
        img = batch["image"].to(model.classifier.weight.device)
        out = model(img)
        gates.append(out["mc"].detach().cpu().numpy())  # [B, C] class_gate in (0,1)
    return np.concatenate(gates, axis=0)


def summarize(gate_values, fold):
    flat = gate_values.flatten()
    mean = flat.mean()
    std = flat.std()
    frac_near0 = (flat < 0.1).mean()
    frac_near1 = (flat > 0.9).mean()
    frac_mid = ((flat >= 0.3) & (flat <= 0.7)).mean()
    frac_extreme = frac_near0 + frac_near1
    print(f"  fold={fold}: mean={mean:.4f} std={std:.4f} "
          f"frac(<0.1)={frac_near0:.4f} frac(>0.9)={frac_near1:.4f} "
          f"frac(extreme <0.1 or >0.9)={frac_extreme:.4f} frac([0.3,0.7])={frac_mid:.4f}")
    return {
        "mean": float(mean), "std": float(std),
        "frac_near0": float(frac_near0), "frac_near1": float(frac_near1),
        "frac_extreme": float(frac_extreme), "frac_mid": float(frac_mid),
        "n_values": int(flat.size),
    }


def plot_hist(gate_values, fold, save_path):
    flat = gate_values.flatten()
    plt.figure(figsize=(7, 5))
    plt.hist(flat, bins=50, range=(0, 1), color="#4C72B0", edgecolor="none")
    plt.axvline(0.5, color="red", linestyle="--", linewidth=1, label="0.5 (uninformative gate)")
    plt.xlabel("class_gate value (sigmoid output, per-channel per-sample)")
    plt.ylabel("count")
    plt.title(f"fold {fold} (seed {REPRESENTATIVE_SEED}): class_gate value distribution")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  [saved] {save_path}")


def main():
    print(f"Representative seed: {REPRESENTATIVE_SEED}")
    all_stats = {}
    for fold, test_domains, calib_domains in GATED_FOLDS:
        domain_pool = sorted(set(test_domains) | set(calib_domains))
        print(f"\n=== fold {fold} | domain pool = {domain_pool} ===")
        model = load_gated_model(ckpt_path(fold, REPRESENTATIVE_SEED))
        ds = build_domain_pool_dataset(domain_pool)
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        gate_values = collect_gate_values(model, loader)
        print(f"  collected gate tensor shape: {gate_values.shape} "
              f"({gate_values.shape[0]} samples x {gate_values.shape[1]} channels)")
        stats = summarize(gate_values, fold)
        plot_hist(gate_values, fold, f"{OUT_DIR}/fold{fold}_class_gate_hist.png")
        all_stats[fold] = stats

    print("\n" + "=" * 100)
    print("SUMMARY: class_gate value distribution per fold (seed 0)")
    print("=" * 100)
    header = f"{'Fold':>6} | {'mean':>8} | {'std':>8} | {'frac<0.1':>9} | {'frac>0.9':>9} | {'frac_extreme':>13} | {'frac[0.3,0.7]':>14}"
    print(header)
    print("-" * len(header))
    for fold, s in all_stats.items():
        print(f"{fold:>6} | {s['mean']:>8.4f} | {s['std']:>8.4f} | {s['frac_near0']:>9.4f} "
              f"| {s['frac_near1']:>9.4f} | {s['frac_extreme']:>13.4f} | {s['frac_mid']:>14.4f}")


if __name__ == "__main__":
    main()
