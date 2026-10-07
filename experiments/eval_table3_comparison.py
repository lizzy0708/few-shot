"""
eval_table3_comparison.py — 2026-10-07. Does NOT modify any existing script. New
file only. No retraining -- reuses existing checkpoints unchanged (and the fixed
untrained ImageNet encoder for the "pretrained" row).

Evaluates three of Table 3's candidate rows -- pretrained (untrained ImageNet
ResNet50), raw z (trained GatedMaskModel, pre-gate feature), gradient mask
(OriginalMaskModel) -- under the EXACT SAME protocol: 3-seed score ensemble
(pretrained has no seed axis, see below), beta=0.5, n_sigma=2.0, leakage-fixed
support/query, 4-fold x 5-eval-seed, PLUS a balanced-query condition (anomaly
undersampled to match normal count, stratified by fault_type, same algorithm as
eval_gated_folds_balanced.py's build_balanced_query, imported unchanged here).

The 4th row (z_inv/gate, Table 2's own setting) is NOT recomputed here -- it was
already fully measured (both current and balanced conditions, all 6 metrics,
fold-level) by eval_gated_folds_balanced.py in the previous turn
(docs/generated/balanced_eval/eval.txt) and is simply reused in the final report
to avoid ~120 redundant combinations' worth of compute.

Model-specific feature extraction (everything else -- support/calib/query scoring,
beta-blend, LedoitWolf, threshold, balanced resampling, metrics -- is identical
across rows, implemented once in run_row()):
  - pretrained: experiments.eval_pretrained_fewshot.load_pretrained_model (fixed,
    never-trained ImageNet ResNet50 layer3+GAP), feature_key='z'. No train-seed
    axis (deterministic encoder, see that module's docstring) -- train_seeds=[0].
  - raw_z: the SAME checkpoints/coarse5_fold*_gated_nodg_s{0,1,2}.pth as z_inv,
    just feature_key='z' (pre-gate) instead of 'z_c_notd' -- identical to
    eval_gated_folds_rawz.py's approach.
  - gradient_mask: experiments.eval_gated_folds_origmask's load_origmask_model +
    get_features_origmask (class_label=0 for support/calib, None for query --
    see that module's docstring for why this differs from plain get_features).

Usage:
  conda run -n torch python experiments/eval_table3_comparison.py --beta 0.5 --n_sigma 2.0
"""
import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score, recall_score, confusion_matrix
)

from experiments.eval_all_folds import get_features, mahalanobis_score, fit_normal_distribution
from experiments.eval_gated_folds_balanced import build_balanced_query
from experiments.eval_pretrained_fewshot import load_pretrained_model
import experiments.eval_gated_folds_origmask as om
import experiments.eval_gated_folds as base


def _extract_z(model, loader):
    return get_features(model, loader, feature_key="z")


ROW_CONFIGS = {
    "pretrained": dict(
        train_seeds=[0],
        load_fn=lambda fold, seed: load_pretrained_model(None),
        extract_sc=_extract_z,
        extract_q=_extract_z,
    ),
    "raw_z": dict(
        train_seeds=base.TRAIN_SEEDS,
        load_fn=lambda fold, seed: base.load_gated_model(base.ckpt_path(fold, seed)),
        extract_sc=_extract_z,
        extract_q=_extract_z,
    ),
    "gradient_mask": dict(
        train_seeds=base.TRAIN_SEEDS,
        load_fn=lambda fold, seed: om.load_origmask_model(om.ckpt_path_origmask(fold, seed)),
        extract_sc=lambda m, l: om.get_features_origmask(m, l, class_label_fixed=0),
        extract_q=lambda m, l: om.get_features_origmask(m, l, class_label_fixed=None),
    ),
}


def run_row(cfg, folds, beta, n_sigma, eval_seeds):
    results = {}
    for fold, test_domains, calib_domains in folds:
        test_domain = test_domains[0]
        models = {s: cfg["load_fn"](fold, s) for s in cfg["train_seeds"]}
        calib_normal_ds = base.build_calib_normal(calib_domains)

        calib_stats = {}
        for s in cfg["train_seeds"]:
            calib_loader = DataLoader(calib_normal_ds, batch_size=32, shuffle=False)
            calib_feats, _ = cfg["extract_sc"](models[s], calib_loader)
            calib_np = calib_feats.cpu().numpy()
            _, prec_np = fit_normal_distribution(calib_np)
            calib_stats[s] = (calib_np, calib_np.mean(axis=0), prec_np)

        for cond in ("current", "balanced"):
            rows = []
            for eval_seed in eval_seeds:
                support_ds, query_ds_full = base.build_support_query(test_domain, eval_seed)
                if cond == "balanced":
                    query_ds, _, _ = build_balanced_query(query_ds_full, eval_seed)
                else:
                    query_ds = query_ds_full

                per_model = []
                ref_labels = None
                for s in cfg["train_seeds"]:
                    calib_np, calib_centroid, prec_np = calib_stats[s]
                    support_loader = DataLoader(support_ds, batch_size=len(support_ds), shuffle=False)
                    support_feats, _ = cfg["extract_sc"](models[s], support_loader)
                    support_np = support_feats.cpu().numpy()
                    support_proto = support_np.mean(axis=0)
                    blended = beta * support_proto + (1.0 - beta) * calib_centroid
                    support_scores = mahalanobis_score(support_np, blended, prec_np)
                    calib_scores = mahalanobis_score(calib_np, blended, prec_np)

                    query_loader = DataLoader(query_ds, batch_size=32, shuffle=False)
                    query_feats, labels = cfg["extract_q"](models[s], query_loader)
                    query_scores = mahalanobis_score(query_feats.cpu().numpy(), blended, prec_np)

                    if ref_labels is None:
                        ref_labels = labels
                    per_model.append((query_scores, support_scores, calib_scores))

                ens_q = np.mean(np.stack([t[0] for t in per_model], axis=0), axis=0)
                ens_s = np.mean(np.stack([t[1] for t in per_model], axis=0), axis=0)
                ens_c = np.mean(np.stack([t[2] for t in per_model], axis=0), axis=0)
                threshold = ens_s.mean() + n_sigma * ens_c.std()
                preds = (ens_q > threshold).astype(int)
                tn, fp, fn, tp = confusion_matrix(ref_labels, preds, labels=[0, 1]).ravel()

                m = dict(
                    eval_seed=eval_seed,
                    auroc=roc_auc_score(ref_labels, ens_q),
                    acc=accuracy_score(ref_labels, preds),
                    f1=f1_score(ref_labels, preds, zero_division=0),
                    bal_acc=balanced_accuracy_score(ref_labels, preds),
                    specificity=tn / (tn + fp) if (tn + fp) > 0 else float("nan"),
                    recall=recall_score(ref_labels, preds, zero_division=0),
                    tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
                )
                rows.append(m)
                print(f"    [{cond}] fold={fold} eval_seed={eval_seed} AUROC={m['auroc']:.4f} "
                      f"Acc={m['acc']:.4f} F1={m['f1']:.4f} BalAcc={m['bal_acc']:.4f} "
                      f"Spec={m['specificity']:.4f} Recall={m['recall']:.4f}")
            results[(fold, cond)] = rows
    return results


def summarize(results, fold_list):
    summary = {}
    for (fold, cond), rows in results.items():
        summary[(fold, cond)] = {
            k: (np.mean([r[k] for r in rows]), np.std([r[k] for r in rows]))
            for k in ("auroc", "acc", "f1", "bal_acc", "specificity", "recall")
        }
    for cond in ("current", "balanced"):
        for k in ("auroc", "acc", "f1", "bal_acc", "specificity", "recall"):
            vals = [summary[(fold, cond)][k][0] for fold in fold_list]
            summary[("Avg", cond)] = summary.get(("Avg", cond), {})
            summary[("Avg", cond)][k] = (np.mean(vals), np.nan)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--n_sigma", type=float, default=2.0)
    ap.add_argument("--eval_seeds", type=int, nargs="+", default=base.EVAL_SEEDS)
    ap.add_argument("--rows", type=str, nargs="+", default=list(ROW_CONFIGS.keys()),
                     choices=list(ROW_CONFIGS.keys()))
    args = ap.parse_args()

    for row_name in args.rows:
        print(f"\n{'#' * 100}\nROW: {row_name}\n{'#' * 100}")
        cfg = ROW_CONFIGS[row_name]
        results = run_row(cfg, base.GATED_FOLDS, args.beta, args.n_sigma, args.eval_seeds)
        summary = summarize(results, [f[0] for f in base.GATED_FOLDS])
        print(f"\n--- SUMMARY for {row_name} ---")
        for (fold, cond), metrics in summary.items():
            line = f"row={row_name} fold={fold:>4} cond={cond:>9} | "
            line += " ".join(f"{k}={v[0]:.4f}" for k, v in metrics.items())
            print(line)

    print("\nDONE")


if __name__ == "__main__":
    main()
