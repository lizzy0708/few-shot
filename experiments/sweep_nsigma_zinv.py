"""
sweep_nsigma_zinv.py — Experiment A: FPR-Recall (Specificity-Recall) trade-off sweep
over n_sigma for the Table 2 pipeline (z_inv, eval_gated_folds.py / GatedMaskModel,
checkpoints/coarse5_fold*_gated_nodg_s{0,1,2}.pth).

Does NOT retrain and does NOT modify eval_gated_folds.py -- imports and reuses
load_gated_model/build_support_query/build_calib_normal/score_one_model unchanged.
The (expensive) model forward pass runs ONCE per (fold, eval_seed) -- exactly like the
original Table 2 evaluation -- and n_sigma is then swept over the resulting ensembled
scores via cheap threshold arithmetic (no extra forward passes per n_sigma candidate).

For each fold, reports Specificity (TNR) / Recall (TPR) / Balanced Accuracy / AUROC per
n_sigma candidate (mean +/- std over eval seeds; AUROC is threshold-independent so it's
identical across candidates), plus a pooled-scores ROC curve with each n_sigma's
realized (FPR, TPR) operating point marked (using each candidate's mean threshold
across eval seeds, applied to the pooled query scores).

target(test)-domain labels ARE used here (final-evaluation stage, same as the existing
Table 2 protocol -- n_sigma itself is not selected using them, only measured).

Usage:
  conda run -n torch python experiments/sweep_nsigma_zinv.py --beta 0.5
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, recall_score

import experiments.eval_gated_folds as base

N_SIGMA_CANDIDATES = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
CURRENT_NSIGMA = 2.0
# For reference only -- these were LOCO-tuned on RAW z (Exp.3b, experiments/tune_nsigma_rawz.py),
# a DIFFERENT feature space from z_inv used here. Cross-checked for directional consistency only.
LOCO_RAWZ_NSIGMA = {"500": 1.5, "600": 2.5, "700": 2.0, "800": 2.5}


def specificity_score(y_true, y_pred):
    """TNR = TN / (TN+FP). Label convention: 0=normal, 1=anomaly (positive=anomaly)."""
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    return tn / (tn + fp) if (tn + fp) > 0 else float("nan")


def run_fold_sweep(fold, test_domain, calib_domains, beta, eval_seeds, train_seeds, verbose=True):
    models = {s: base.load_gated_model(base.ckpt_path(fold, s)) for s in train_seeds}
    calib_normal_ds = base.build_calib_normal(calib_domains)

    per_ns_vals = {ns: [] for ns in N_SIGMA_CANDIDATES}  # list of (spec, recall, bal_acc, threshold)
    per_seed_auroc = []
    pooled_scores, pooled_labels = [], []

    for eval_seed in eval_seeds:
        support_ds, query_ds = base.build_support_query(test_domain, eval_seed)

        per_model_query, per_model_support, per_model_calib, ref_labels = [], [], [], None
        for s in train_seeds:
            q, sc, c, ql = base.score_one_model(models[s], support_ds, calib_normal_ds, query_ds, beta, 0.0)
            if ref_labels is None:
                ref_labels = ql
            per_model_query.append(q)
            per_model_support.append(sc)
            per_model_calib.append(c)

        ens_query = np.mean(np.stack(per_model_query, axis=0), axis=0)
        ens_support = np.mean(np.stack(per_model_support, axis=0), axis=0)
        ens_calib = np.mean(np.stack(per_model_calib, axis=0), axis=0)

        auroc = roc_auc_score(ref_labels, ens_query)
        per_seed_auroc.append(auroc)
        pooled_scores.append(ens_query)
        pooled_labels.append(ref_labels)

        for ns in N_SIGMA_CANDIDATES:
            threshold = ens_support.mean() + ns * ens_calib.std()
            preds = (ens_query > threshold).astype(int)
            recall = recall_score(ref_labels, preds, zero_division=0)
            spec = specificity_score(ref_labels, preds)
            bal_acc = (recall + spec) / 2.0
            per_ns_vals[ns].append((spec, recall, bal_acc, threshold))

        if verbose:
            print(f"    fold={fold} eval_seed={eval_seed} scored, AUROC={auroc:.4f}")

    pooled_scores = np.concatenate(pooled_scores)
    pooled_labels = np.concatenate(pooled_labels)

    summary = {}
    for ns in N_SIGMA_CANDIDATES:
        specs = [v[0] for v in per_ns_vals[ns]]
        recs = [v[1] for v in per_ns_vals[ns]]
        baccs = [v[2] for v in per_ns_vals[ns]]
        thrs = [v[3] for v in per_ns_vals[ns]]
        summary[ns] = {
            "spec_mean": float(np.mean(specs)), "spec_std": float(np.std(specs)),
            "recall_mean": float(np.mean(recs)), "recall_std": float(np.std(recs)),
            "bal_acc_mean": float(np.mean(baccs)), "bal_acc_std": float(np.std(baccs)),
            "mean_threshold": float(np.mean(thrs)),
        }
    auroc_mean, auroc_std = float(np.mean(per_seed_auroc)), float(np.std(per_seed_auroc))
    return summary, auroc_mean, auroc_std, pooled_scores, pooled_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--eval_seeds", type=int, nargs="+", default=base.EVAL_SEEDS)
    ap.add_argument("--train_seeds", type=int, nargs="+", default=base.TRAIN_SEEDS)
    ap.add_argument("--out_dir", type=str, default="docs/generated/expA_fpr_recall")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Device : {base.device}")
    print(f"beta   : {args.beta}")
    print(f"n_sigma candidates: {N_SIGMA_CANDIDATES}")
    print(f"eval_seeds: {args.eval_seeds}\n")

    all_fold_data = {}
    for fold, test_domains, calib_domains in base.GATED_FOLDS:
        test_domain = test_domains[0]
        print(f"=== fold={fold} (test_domain={test_domain}, calib_domains={calib_domains}) ===")
        summary, auroc_mean, auroc_std, pooled_scores, pooled_labels = run_fold_sweep(
            fold, test_domain, calib_domains, args.beta, args.eval_seeds, args.train_seeds
        )
        all_fold_data[fold] = {
            "summary": summary, "auroc_mean": auroc_mean, "auroc_std": auroc_std,
            "pooled_scores": pooled_scores, "pooled_labels": pooled_labels,
        }
        print(f"  fold={fold} AUROC={auroc_mean:.4f}±{auroc_std:.4f} (threshold-independent)")
        header = f"  {'n_sigma':>8} | {'Specificity':>14} | {'Recall':>14} | {'BalAcc':>14}"
        print(header)
        for ns in N_SIGMA_CANDIDATES:
            s = summary[ns]
            marker = "  <== current" if ns == CURRENT_NSIGMA else ""
            print(f"  {ns:>+8.1f} | {s['spec_mean']:.4f}±{s['spec_std']:.4f} "
                  f"| {s['recall_mean']:.4f}±{s['recall_std']:.4f} "
                  f"| {s['bal_acc_mean']:.4f}±{s['bal_acc_std']:.4f}{marker}")
        print()

    # ---- save JSON summary (metrics only, not raw pooled arrays) ----
    json_out = {
        fold: {
            "summary": d["summary"], "auroc_mean": d["auroc_mean"], "auroc_std": d["auroc_std"],
        } for fold, d in all_fold_data.items()
    }
    json_out["n_sigma_candidates"] = N_SIGMA_CANDIDATES
    json_out["current_nsigma"] = CURRENT_NSIGMA
    json_out["loco_rawz_nsigma_for_reference"] = LOCO_RAWZ_NSIGMA
    with open(os.path.join(args.out_dir, "nsigma_sweep.json"), "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"Wrote {args.out_dir}/nsigma_sweep.json")

    # ---- ROC curve plots with n_sigma operating points ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, (fold, test_domains, calib_domains) in zip(axes.flat, base.GATED_FOLDS):
        d = all_fold_data[fold]
        fpr, tpr, _ = roc_curve(d["pooled_labels"], d["pooled_scores"])
        ax.plot(fpr, tpr, color="steelblue", lw=1.5, label=f"ROC (pooled, AUROC={d['auroc_mean']:.3f})")
        ax.plot([0, 1], [0, 1], color="gray", lw=0.8, ls="--")
        for ns in N_SIGMA_CANDIDATES:
            s = d["summary"][ns]
            op_fpr, op_tpr = 1.0 - s["spec_mean"], s["recall_mean"]
            is_current = (ns == CURRENT_NSIGMA)
            ax.scatter([op_fpr], [op_tpr], s=80 if is_current else 40,
                       color="crimson" if is_current else "darkorange",
                       zorder=5, edgecolor="black" if is_current else "none", linewidth=1.2)
            ax.annotate(f"{ns:+.1f}", (op_fpr, op_tpr), textcoords="offset points",
                        xytext=(5, 3), fontsize=7)
        loco_ns = LOCO_RAWZ_NSIGMA.get(fold)
        if loco_ns in d["summary"]:
            s = d["summary"][loco_ns]
            ax.scatter([1.0 - s["spec_mean"]], [s["recall_mean"]], s=100, marker="*",
                       color="forestgreen", zorder=6, label=f"Exp.3b LOCO ref (raw z, n_sigma={loco_ns:+.1f})")
        ax.set_title(f"Fold {fold} (test_domain={test_domains[0]})")
        ax.set_xlabel("FPR (1 - Specificity)")
        ax.set_ylabel("TPR (Recall)")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)
    fig.suptitle("Experiment A: n_sigma operating points on ROC curve (z_inv, Table 2 pipeline)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(args.out_dir, "roc_nsigma_operating_points.png"), dpi=150)
    print(f"Wrote {args.out_dir}/roc_nsigma_operating_points.png")

    # ---- balanced accuracy vs n_sigma line plot (all folds) ----
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    for fold, _, _ in base.GATED_FOLDS:
        d = all_fold_data[fold]
        ys = [d["summary"][ns]["bal_acc_mean"] for ns in N_SIGMA_CANDIDATES]
        ax2.plot(N_SIGMA_CANDIDATES, ys, marker="o", label=f"fold {fold}")
    ax2.axvline(CURRENT_NSIGMA, color="crimson", ls="--", lw=1, label=f"current n_sigma={CURRENT_NSIGMA}")
    ax2.set_xlabel("n_sigma")
    ax2.set_ylabel("Balanced Accuracy")
    ax2.set_title("Balanced Accuracy vs n_sigma (z_inv, Table 2 pipeline)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(args.out_dir, "balanced_acc_vs_nsigma.png"), dpi=150)
    print(f"Wrote {args.out_dir}/balanced_acc_vs_nsigma.png")


if __name__ == "__main__":
    main()
