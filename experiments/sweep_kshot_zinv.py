"""
sweep_kshot_zinv.py — Experiment B: K-shot sensitivity for the Table 2 pipeline
(z_inv, eval_gated_folds.py / GatedMaskModel, checkpoints/coarse5_fold*_gated_nodg_s{0,1,2}.pth).

Support-set size K in {1,2,4,8,16}; n_sigma stays fixed at the currently-adopted value
(2.0, beta=0.5) so only the shot-count effect is isolated. Does NOT retrain and does
NOT modify eval_gated_folds.py -- monkey-patches the module-level SHOT constant that
build_support_query() reads at call time (Python late-binds a bare name inside a
function body to whatever the enclosing module's namespace holds at CALL time, not at
def time, so `base.SHOT = k` before calling base.run_fold() changes what shot count
build_support_query() draws without editing the file), and reuses run_fold() unchanged
for everything else (leakage-fix exclude_paths, disjoint asserts, 3-train-seed score
ensemble, threshold/metric computation).

K=4 is the existing Table 2 result (AUROC 0.9539/Acc 0.9054/F1 0.9430, see
docs/generated/results.md) and is NOT rerun here -- reuse that number when reporting.

Usage:
  conda run -n torch python experiments/sweep_kshot_zinv.py --beta 0.5 --n_sigma 2.0 --k_values 1 2 8 16
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np

import experiments.eval_gated_folds as base

# Existing Table 2 numbers (K=4) -- NOT rerun, kept here for the combined report/plot.
K4_REFERENCE = {
    "500": {"auroc_mean": 0.9829, "acc_mean": 0.9533, "f1_mean": 0.9731},
    "600": {"auroc_mean": 0.9500, "acc_mean": 0.9034, "f1_mean": 0.9429},
    "700": {"auroc_mean": 0.9820, "acc_mean": 0.9323, "f1_mean": 0.9611},
    "800": {"auroc_mean": 0.9008, "acc_mean": 0.8325, "f1_mean": 0.8948},
    "avg_auroc": 0.9539, "avg_acc": 0.9054, "avg_f1": 0.9430,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--n_sigma", type=float, default=2.0,
                     help="held fixed across K (current adopted value) so only shot count varies")
    ap.add_argument("--k_values", type=int, nargs="+", default=[1, 2, 8, 16],
                     help="K=4 is the existing Table 2 result and is skipped by default")
    ap.add_argument("--eval_seeds", type=int, nargs="+", default=base.EVAL_SEEDS)
    ap.add_argument("--train_seeds", type=int, nargs="+", default=base.TRAIN_SEEDS)
    ap.add_argument("--out_dir", type=str, default="docs/generated/expB_kshot")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Device      : {base.device}")
    print(f"beta        : {args.beta}")
    print(f"n_sigma     : {args.n_sigma} (fixed across K)")
    print(f"k_values    : {args.k_values} (K=4 = existing Table 2 result, not rerun)")
    print(f"eval_seeds  : {args.eval_seeds}\n")

    original_shot = base.SHOT
    all_results = {"4": K4_REFERENCE}
    try:
        for k in args.k_values:
            base.SHOT = k
            print(f"=== K={k} (base.SHOT patched from {original_shot}) ===")
            fold_rows = {}
            for fold, test_domains, calib_domains in base.GATED_FOLDS:
                res = base.run_fold(fold, test_domains, calib_domains, args.beta, args.n_sigma,
                                     args.train_seeds, args.eval_seeds)
                fold_rows[fold] = res
                print(f"  K={k} fold={fold} AUROC={res['auroc_mean']:.4f} "
                      f"Acc={res['acc_mean']:.4f} F1={res['f1_mean']:.4f}")
            avg_auroc = float(np.mean([r["auroc_mean"] for r in fold_rows.values()]))
            avg_acc = float(np.mean([r["acc_mean"] for r in fold_rows.values()]))
            avg_f1 = float(np.mean([r["f1_mean"] for r in fold_rows.values()]))
            all_results[str(k)] = {**fold_rows, "avg_auroc": avg_auroc, "avg_acc": avg_acc, "avg_f1": avg_f1}
            print(f"  K={k} Avg AUROC={avg_auroc:.4f} Acc={avg_acc:.4f} F1={avg_f1:.4f}\n")
    finally:
        base.SHOT = original_shot  # restore, in case this module stays imported

    out_json = os.path.join(args.out_dir, "kshot_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Wrote {out_json}")

    # ---- plots: fold-level AUROC/Acc curves vs K ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks_sorted = sorted(int(k) for k in all_results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for fold, _, _ in base.GATED_FOLDS:
        aurocs = [all_results[str(k)][fold]["auroc_mean"] for k in ks_sorted]
        accs = [all_results[str(k)][fold]["acc_mean"] for k in ks_sorted]
        axes[0].plot(ks_sorted, aurocs, marker="o", label=f"fold {fold}")
        axes[1].plot(ks_sorted, accs, marker="o", label=f"fold {fold}")
    for ax, title, ylab in zip(axes, ["AUROC vs K", "Accuracy vs K"], ["AUROC", "Accuracy"]):
        ax.set_xlabel("K (support shots)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks_sorted)
        ax.set_xticklabels([str(k) for k in ks_sorted])
        ax.axvline(4, color="crimson", ls="--", lw=1, label="current K=4")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Experiment B: K-shot sensitivity (z_inv, Table 2 pipeline, n_sigma=2.0 fixed)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(args.out_dir, "kshot_auroc_acc.png"), dpi=150)
    print(f"Wrote {args.out_dir}/kshot_auroc_acc.png")


if __name__ == "__main__":
    main()
