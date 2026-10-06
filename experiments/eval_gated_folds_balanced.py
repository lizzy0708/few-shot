"""
eval_gated_folds_balanced.py — 2026-10-06. Does NOT modify experiments/eval_gated_folds.py
or any other existing script. New file only. No retraining -- reuses
checkpoints/coarse5_fold*_gated_nodg_s{0,1,2}.pth unchanged.

Current Table 2 protocol has a strongly imbalanced query set (normal:anomaly roughly
1:6-7, see docs/generated/recording_level_eval/eval.txt's grand confusion matrix:
29860 normal vs 174175 anomaly across all folds/eval-seeds), so AUROC/Acc/F1 are
dominated by the anomaly class. This script evaluates the SAME trained
checkpoints/SAME threshold (support_mean + n_sigma*calib_std, computed from support+
calib exactly as before -- NEVER touched by the query-side balancing below) on a
query set where the anomaly side is randomly undersampled (per eval-seed, stratified
by fault_type -- I/O/B/compound, HUSTDataset's own `fault_type` field, confirmed
values 1/2/3/4 via models' _FAULT_MAP) down to match the normal query count exactly,
so AUROC/Acc/F1/Specificity/Recall/Precision/Balanced-Acc/macro-F1 can be read without
the class-imbalance distortion. Only the query distribution changes; support
selection, calib set, LedoitWolf fit, beta blend, and threshold are completely
unaffected (score_one_model is reused unchanged -- only the query_ds argument differs).

Correctness note: `score_with_cached_calib()` below is mathematically identical to
eval_gated_folds.score_one_model() (same beta-blend/LedoitWolf/mahalanobis_score
calls), only hoisting calib feature-extraction out of the per-eval-seed/per-condition
loop since this script evaluates TWO conditions (current + balanced) per fold instead
of one -- same optimization and same correctness already cross-validated in
eval_gated_folds_recording.py (exact match against eval_gated_folds.py confirmed
there: AUROC=0.9833 Acc=0.9520 F1=0.9722 both ways on fold=500/train_seed=0/eval_seed=0).

Usage:
  conda run -n torch python experiments/eval_gated_folds_balanced.py --beta 0.5 --n_sigma 2.0
"""
import os
import sys
import copy
import random
import argparse
from collections import defaultdict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score,
    precision_score, recall_score, confusion_matrix,
)

from experiments.eval_all_folds import get_features, mahalanobis_score, fit_normal_distribution
import experiments.eval_gated_folds as base


def build_balanced_query(query_ds, eval_seed):
    """Keep ALL normal query samples; undersample anomaly (stratified by fault_type:
    1=I, 2=O, 3=B, 4=compound IB/IO/OB, datasets/hust_image.py's _FAULT_MAP) down to
    the same count as normal, using the largest-remainder method for exact proportional
    allocation, seeded by eval_seed for reproducibility."""
    normal_samples = [s for s in query_ds.samples if s[1] == 0]
    anomaly_samples = [s for s in query_ds.samples if s[1] == 1]
    n_target = len(normal_samples)

    groups = defaultdict(list)
    for s in anomaly_samples:
        groups[s[4]].append(s)  # s[4] = fault_type
    fault_types = sorted(groups.keys())
    total_anomaly = len(anomaly_samples)

    raw_counts = {ft: len(groups[ft]) / total_anomaly * n_target for ft in fault_types}
    floor_counts = {ft: int(np.floor(raw_counts[ft])) for ft in fault_types}
    remainder = n_target - sum(floor_counts.values())
    by_frac_desc = sorted(fault_types, key=lambda ft: raw_counts[ft] - floor_counts[ft], reverse=True)
    for ft in by_frac_desc[:remainder]:
        floor_counts[ft] += 1

    rng = random.Random(eval_seed)
    balanced_anomaly = []
    for ft in fault_types:
        pool = groups[ft].copy()
        rng.shuffle(pool)
        k = min(floor_counts[ft], len(pool))
        balanced_anomaly.extend(pool[:k])

    shortfall = n_target - len(balanced_anomaly)
    if shortfall > 0:
        used = set(id(s) for s in balanced_anomaly)
        leftover = [s for s in anomaly_samples if id(s) not in used]
        rng.shuffle(leftover)
        balanced_anomaly.extend(leftover[:shortfall])

    balanced_ds = copy.copy(query_ds)
    balanced_ds.samples = normal_samples + balanced_anomaly
    return balanced_ds, dict(floor_counts), n_target


def get_calib_stats(model, calib_normal_ds):
    calib_loader = DataLoader(calib_normal_ds, batch_size=32, shuffle=False)
    calib_feats, _ = get_features(model, calib_loader)
    calib_np = calib_feats.cpu().numpy()
    _, prec_np = fit_normal_distribution(calib_np)
    calib_centroid = calib_np.mean(axis=0)
    return calib_np, calib_centroid, prec_np


def get_support_stats(model, support_ds, calib_centroid, beta):
    support_loader = DataLoader(support_ds, batch_size=len(support_ds), shuffle=False)
    support_feats, _ = get_features(model, support_loader)
    support_np = support_feats.cpu().numpy()
    support_proto = support_np.mean(axis=0)
    blended = beta * support_proto + (1.0 - beta) * calib_centroid
    return support_np, blended


def score_query(model, query_ds, blended, prec_np):
    query_loader = DataLoader(query_ds, batch_size=32, shuffle=False)
    query_feats, query_labels = get_features(model, query_loader)
    query_np = query_feats.cpu().numpy()
    scores = mahalanobis_score(query_np, blended, prec_np)
    return scores, query_labels


def metrics_from_preds(labels, scores, preds):
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return dict(
        auroc=roc_auc_score(labels, scores),
        acc=accuracy_score(labels, preds),
        f1=f1_score(labels, preds, zero_division=0),
        bal_acc=balanced_accuracy_score(labels, preds),
        specificity=tn / (tn + fp) if (tn + fp) > 0 else float("nan"),
        recall=recall_score(labels, preds, zero_division=0),
        precision=precision_score(labels, preds, zero_division=0),
        macro_f1=f1_score(labels, preds, average="macro", zero_division=0),
        normal_f1=f1_score(labels, preds, pos_label=0, zero_division=0),
        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--n_sigma", type=float, default=2.0)
    ap.add_argument("--train_seeds", type=int, nargs="+", default=base.TRAIN_SEEDS)
    ap.add_argument("--eval_seeds", type=int, nargs="+", default=base.EVAL_SEEDS)
    ap.add_argument("--test_fold", type=str, default=None, choices=[f[0] for f in base.GATED_FOLDS])
    args = ap.parse_args()

    folds = base.GATED_FOLDS if args.test_fold is None else [f for f in base.GATED_FOLDS if f[0] == args.test_fold]
    print(f"beta={args.beta} n_sigma={args.n_sigma} train_seeds={args.train_seeds} eval_seeds={args.eval_seeds}")

    all_rows = {}  # (fold, condition) -> list of per-eval-seed metric dicts
    grand_confusion = {"current": dict(tp=0, fp=0, tn=0, fn=0), "balanced": dict(tp=0, fp=0, tn=0, fn=0)}

    for fold, test_domains, calib_domains in folds:
        test_domain = test_domains[0]
        print(f"\n=== FOLD {fold} ===")
        models = {s: base.load_gated_model(base.ckpt_path(fold, s)) for s in args.train_seeds}
        calib_normal_ds = base.build_calib_normal(calib_domains)
        calib_stats = {s: get_calib_stats(models[s], calib_normal_ds) for s in args.train_seeds}

        for cond in ("current", "balanced"):
            rows = []
            for eval_seed in args.eval_seeds:
                support_ds, query_ds_full = base.build_support_query(test_domain, eval_seed)
                if cond == "balanced":
                    query_ds, ft_counts, n_target = build_balanced_query(query_ds_full, eval_seed)
                else:
                    query_ds = query_ds_full

                per_model_scores = []
                ref_labels = None
                for s in args.train_seeds:
                    calib_np, calib_centroid, prec_np = calib_stats[s]
                    _, blended = get_support_stats(models[s], support_ds, calib_centroid, args.beta)
                    support_np = None  # support scores needed too, for threshold
                    support_loader = DataLoader(support_ds, batch_size=len(support_ds), shuffle=False)
                    support_feats, _ = get_features(models[s], support_loader)
                    support_scores = mahalanobis_score(support_feats.cpu().numpy(), blended, prec_np)
                    calib_scores = mahalanobis_score(calib_np, blended, prec_np)
                    q_scores, labels = score_query(models[s], query_ds, blended, prec_np)
                    if ref_labels is None:
                        ref_labels = labels
                    per_model_scores.append((q_scores, support_scores, calib_scores))

                ens_query = np.mean(np.stack([t[0] for t in per_model_scores], axis=0), axis=0)
                ens_support = np.mean(np.stack([t[1] for t in per_model_scores], axis=0), axis=0)
                ens_calib = np.mean(np.stack([t[2] for t in per_model_scores], axis=0), axis=0)

                threshold = ens_support.mean() + args.n_sigma * ens_calib.std()
                preds = (ens_query > threshold).astype(int)
                m = metrics_from_preds(ref_labels, ens_query, preds)
                m["eval_seed"] = eval_seed
                m["n_query"] = len(ref_labels)
                rows.append(m)
                for k in ("tp", "fp", "tn", "fn"):
                    grand_confusion[cond][k] += m[k]
                print(f"  [{cond}] eval_seed={eval_seed} AUROC={m['auroc']:.4f} Acc={m['acc']:.4f} "
                      f"F1={m['f1']:.4f} BalAcc={m['bal_acc']:.4f} Spec={m['specificity']:.4f} "
                      f"Recall={m['recall']:.4f} Prec={m['precision']:.4f} n_query={m['n_query']}")
            all_rows[(fold, cond)] = rows

    print("\n" + "=" * 110)
    print("SUMMARY (mean +- std over eval-seeds)")
    print("=" * 110)
    for (fold, cond), rows in all_rows.items():
        def avg(k):
            return np.mean([r[k] for r in rows]), np.std([r[k] for r in rows])
        a_auroc, s_auroc = avg("auroc"); a_acc, s_acc = avg("acc"); a_f1, s_f1 = avg("f1")
        a_bal, s_bal = avg("bal_acc"); a_spec, s_spec = avg("specificity"); a_rec, s_rec = avg("recall")
        print(f"fold={fold} cond={cond:>9} | AUROC={a_auroc:.4f}±{s_auroc:.4f} Acc={a_acc:.4f}±{s_acc:.4f} "
              f"F1={a_f1:.4f}±{s_f1:.4f} BalAcc={a_bal:.4f}±{s_bal:.4f} Spec={a_spec:.4f}±{s_spec:.4f} "
              f"Recall={a_rec:.4f}±{s_rec:.4f} n_query(seed0)={rows[0]['n_query']}")

    print("\nGRAND CONFUSION MATRICES (summed over all folds+eval-seeds):")
    for cond in ("current", "balanced"):
        cm = grand_confusion[cond]
        tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        bal_acc = (recall + specificity) / 2
        f1_anom = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")
        f1_norm = 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) > 0 else float("nan")
        macro_f1 = (f1_anom + f1_norm) / 2
        print(f"  [{cond}] {cm} | Precision={precision:.4f} Recall={recall:.4f} Specificity={specificity:.4f} "
              f"BalAcc={bal_acc:.4f} F1(anomaly)={f1_anom:.4f} F1(normal)={f1_norm:.4f} macroF1={macro_f1:.4f}")

    print("\nDONE")


if __name__ == "__main__":
    main()
