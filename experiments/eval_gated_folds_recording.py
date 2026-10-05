"""
eval_gated_folds_recording.py — 2026-10-05. Does NOT modify experiments/eval_gated_folds.py
or any other existing script. New file only. No retraining -- reuses
checkpoints/coarse5_fold*_gated_nodg_s{0,1,2}.pth unchanged.

Two things in one script (both need the same per-sample confusion-matrix machinery,
which the existing run_fold()/score_one_model() don't expose -- they only return
aggregated AUROC/Acc/F1):

1. "current" condition: exact reproduction of eval_gated_folds.py's protocol
   (support 4-shot drawn from ALL 3 recordings of the held-out domain pooled together,
   query = everything else via the existing window-level exclude_paths leakage fix).
   Recomputed here (not read from cached results.md numbers) because we need
   TP/FP/TN/FN/Specificity, which were never computed/saved anywhere. This also
   answers task 5 (confusion-matrix recomputation for Fig.4) directly.

2. "recording_<R_i>" conditions: a STRONGER leakage control. The held-out domain's
   normal windows come from 3 separate source recordings (.mat files) -- e.g. domain
   500's normal windows are all from N500.mat/N502.mat/N504.mat (confirmed via
   filename prefixes, `ls processed/500/normal | sed -E 's/_[0-9]+\\.png$//' | sort -u`
   -> N500 N502 N504, 499 windows each = 1497). The *current* protocol's leakage fix
   only excludes the exact 4 support windows from query (datasets/hust_image.py
   exclude_paths, 2026-09-17) -- it does NOT prevent query from containing OTHER
   windows from the SAME recording as a support window, which could be temporally
   close/correlated to a support window. This script instead draws support ONLY from
   one recording R_i and excludes that recording's ENTIRE normal window set from
   query (not just the 4 selected windows) -- a strictly stronger, recording-disjoint
   split. 3 recordings x 5 eval-seeds x 4 folds.

Correctness note: `score_with_cached_calib()` below is mathematically IDENTICAL to
eval_gated_folds.score_one_model() (same beta-blend, same LedoitWolf precision, same
mahalanobis_score call) -- it only hoists the calib feature-extraction +
LedoitWolf-fit out of the per-eval-seed loop (calib_domains/calib_normal_ds does not
depend on eval_seed or on which recording-condition is being run, only on fold+train_seed,
so recomputing it 20x per fold as score_one_model's calling pattern would otherwise do
is pure waste given this script's 4x-larger combination count). This is verified
below: the "current" condition's aggregate AUROC/Acc/F1 computed here is cross-checked
against the already-published Table 2 numbers (docs/generated/results.md, "세션 최고
2026-08-17") for exact agreement before trusting the recording-level numbers.

Usage:
  conda run -n torch python experiments/eval_gated_folds_recording.py --beta 0.5 --n_sigma 2.0
"""
import os
import re
import sys
import random
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix

from datasets.hust_image import HUSTDataset
from experiments.eval_all_folds import transform, get_features, mahalanobis_score, fit_normal_distribution
import experiments.eval_gated_folds as base


def recording_id(path):
    """Recording (source .mat file) prefix from a window filename, e.g.
    'processed/500/normal/N502_419840.png' -> 'N502'. Confirmed pattern:
    processed/make_gadf.py:84 saves windows as f"{file[:-4]}_{i}.png"."""
    m = re.match(r'^([A-Za-z]+\d+)_', os.path.basename(path))
    assert m, f"cannot parse recording id from path: {path}"
    return m.group(1)


def get_recordings_for_domain(domain):
    normal_dir = os.path.join(base.ROOT, domain, "normal")
    ids = set()
    for fname in os.listdir(normal_dir):
        if fname.lower().endswith(".png"):
            ids.add(recording_id(fname))
    return sorted(ids)


def build_support_query_recording(test_domain, eval_seed, R_i):
    """support: SHOT normal windows drawn only from recording R_i (seeded shuffle,
    same random.seed(eval_seed)+random.shuffle convention as HUSTDataset's own
    shot-selection, datasets/hust_image.py:110-116, applied here to the R_i-only
    subset). query: normal windows from the OTHER recordings only (R_i's entire
    normal set excluded, not just the 4 drawn) + all anomaly windows unchanged."""
    all_normal_ds = HUSTDataset(root=base.ROOT, domain=test_domain, only_normal=True,
                                 transform=transform, all_domains=base.ALL_DOMAINS)
    ri_samples = [s for s in all_normal_ds.samples if recording_id(s[0]) == R_i]
    assert len(ri_samples) >= base.SHOT, (
        f"domain={test_domain} recording={R_i} has only {len(ri_samples)} normal "
        f"samples, need >= {base.SHOT}"
    )

    random.seed(eval_seed)
    shuffled = ri_samples.copy()
    random.shuffle(shuffled)
    support_samples = shuffled[:base.SHOT]

    support_ds = HUSTDataset(root=base.ROOT, domain=test_domain, only_normal=True,
                              transform=transform, all_domains=base.ALL_DOMAINS)
    support_ds.samples = support_samples

    query_ds = HUSTDataset(root=base.ROOT, domain=test_domain, only_normal=False,
                            transform=transform, all_domains=base.ALL_DOMAINS)
    query_ds.samples = [s for s in query_ds.samples
                         if not (s[1] == 0 and recording_id(s[0]) == R_i)]

    support_paths = set(s[0] for s in support_ds.samples)
    query_paths = set(s[0] for s in query_ds.samples)
    ok = support_paths.isdisjoint(query_paths)
    assert ok, f"recording-level leakage: {support_paths & query_paths}"
    n_normal_query = sum(1 for s in query_ds.samples if s[1] == 0)
    return support_ds, query_ds, n_normal_query


def get_calib_stats(model, calib_normal_ds):
    """Hoisted out of the per-eval-seed/per-condition loop -- see module docstring."""
    calib_loader = DataLoader(calib_normal_ds, batch_size=32, shuffle=False)
    calib_feats, _ = get_features(model, calib_loader)
    calib_np = calib_feats.cpu().numpy()
    _, prec_np = fit_normal_distribution(calib_np)
    calib_centroid = calib_np.mean(axis=0)
    return calib_np, calib_centroid, prec_np


def score_with_cached_calib(model, support_ds, query_ds, calib_np, calib_centroid, prec_np, beta, n_sigma):
    """Mathematically identical to base.score_one_model (see module docstring)."""
    support_loader = DataLoader(support_ds, batch_size=len(support_ds), shuffle=False)
    query_loader = DataLoader(query_ds, batch_size=32, shuffle=False)
    support_feats, _ = get_features(model, support_loader)
    query_feats, query_labels = get_features(model, query_loader)
    support_np = support_feats.cpu().numpy()
    query_np = query_feats.cpu().numpy()
    support_proto = support_np.mean(axis=0)
    blended = beta * support_proto + (1.0 - beta) * calib_centroid
    query_scores = mahalanobis_score(query_np, blended, prec_np)
    support_scores = mahalanobis_score(support_np, blended, prec_np)
    calib_scores = mahalanobis_score(calib_np, blended, prec_np)
    return query_scores, support_scores, calib_scores, query_labels


def run_eval_seed(models, train_seeds, calib_stats_by_seed, support_ds, query_ds, beta, n_sigma):
    per_model_query, per_model_support, per_model_calib, ref_labels = [], [], [], None
    for s in train_seeds:
        calib_np, calib_centroid, prec_np = calib_stats_by_seed[s]
        q, sc, c, labels = score_with_cached_calib(
            models[s], support_ds, query_ds, calib_np, calib_centroid, prec_np, beta, n_sigma
        )
        if ref_labels is None:
            ref_labels = labels
        else:
            assert np.array_equal(ref_labels, labels), "query label order mismatch across train seeds"
        per_model_query.append(q)
        per_model_support.append(sc)
        per_model_calib.append(c)

    ens_query = np.mean(np.stack(per_model_query, axis=0), axis=0)
    ens_support = np.mean(np.stack(per_model_support, axis=0), axis=0)
    ens_calib = np.mean(np.stack(per_model_calib, axis=0), axis=0)

    threshold = ens_support.mean() + n_sigma * ens_calib.std()
    preds = (ens_query > threshold).astype(int)

    auroc = roc_auc_score(ref_labels, ens_query)
    acc = accuracy_score(ref_labels, preds)
    f1 = f1_score(ref_labels, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(ref_labels, preds, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return dict(auroc=auroc, acc=acc, f1=f1, specificity=specificity, recall=recall,
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))


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

    all_results = {}  # (fold, condition) -> list of per-eval-seed dicts
    grand_confusion_current = dict(tp=0, fp=0, tn=0, fn=0)

    for fold, test_domains, calib_domains in folds:
        test_domain = test_domains[0]
        print(f"\n=== FOLD {fold} (test_domain={test_domain}, calib_domains={calib_domains}) ===")

        models = {s: base.load_gated_model(base.ckpt_path(fold, s)) for s in args.train_seeds}
        calib_normal_ds = base.build_calib_normal(calib_domains)
        calib_stats_by_seed = {s: get_calib_stats(models[s], calib_normal_ds) for s in args.train_seeds}

        # --- current condition ---
        rows = []
        fold_confusion = dict(tp=0, fp=0, tn=0, fn=0)
        for eval_seed in args.eval_seeds:
            support_ds, query_ds = base.build_support_query(test_domain, eval_seed)
            n_normal_query = sum(1 for s in query_ds.samples if s[1] == 0)
            res = run_eval_seed(models, args.train_seeds, calib_stats_by_seed, support_ds, query_ds,
                                 args.beta, args.n_sigma)
            res["n_normal_query"] = n_normal_query
            rows.append(res)
            for k in ("tp", "fp", "tn", "fn"):
                fold_confusion[k] += res[k]
                grand_confusion_current[k] += res[k]
            print(f"  [current] eval_seed={eval_seed} AUROC={res['auroc']:.4f} Acc={res['acc']:.4f} "
                  f"F1={res['f1']:.4f} Spec={res['specificity']:.4f} "
                  f"TP={res['tp']} FP={res['fp']} TN={res['tn']} FN={res['fn']} n_normal_query={n_normal_query}")
        all_results[(fold, "current")] = rows
        print(f"  [current] FOLD CONFUSION (summed over {len(args.eval_seeds)} eval-seeds): {fold_confusion}")

        # --- recording-level conditions ---
        recordings = get_recordings_for_domain(test_domain)
        print(f"  recordings for domain {test_domain}: {recordings}")
        for R_i in recordings:
            rows = []
            for eval_seed in args.eval_seeds:
                support_ds, query_ds, n_normal_query = build_support_query_recording(
                    test_domain, eval_seed, R_i
                )
                res = run_eval_seed(models, args.train_seeds, calib_stats_by_seed, support_ds, query_ds,
                                     args.beta, args.n_sigma)
                res["n_normal_query"] = n_normal_query
                rows.append(res)
                print(f"  [recording={R_i}] eval_seed={eval_seed} AUROC={res['auroc']:.4f} Acc={res['acc']:.4f} "
                      f"F1={res['f1']:.4f} Spec={res['specificity']:.4f} n_normal_query={n_normal_query}")
            all_results[(fold, f"recording_{R_i}")] = rows

    # --- summary ---
    print("\n" + "=" * 100)
    print("SUMMARY (mean +- std over eval-seeds)")
    print("=" * 100)
    header = f"{'Fold':>6} | {'Condition':>16} | {'AUROC':>16} | {'Acc':>16} | {'F1':>16} | {'Spec':>16} | {'n_normal_q':>10}"
    print(header)
    print("-" * len(header))
    for (fold, cond), rows in all_results.items():
        aurocs = [r["auroc"] for r in rows]
        accs = [r["acc"] for r in rows]
        f1s = [r["f1"] for r in rows]
        specs = [r["specificity"] for r in rows]
        nq = rows[0]["n_normal_query"]
        print(f"{fold:>6} | {cond:>16} | {np.mean(aurocs):.4f}±{np.std(aurocs):.4f} "
              f"| {np.mean(accs):.4f}±{np.std(accs):.4f} | {np.mean(f1s):.4f}±{np.std(f1s):.4f} "
              f"| {np.mean(specs):.4f}±{np.std(specs):.4f} | {nq:>10}")

    print("\nGRAND CONFUSION MATRIX, current condition, all folds summed:")
    print(grand_confusion_current)
    tp, fp, tn, fn = (grand_confusion_current[k] for k in ("tp", "fp", "tn", "fn"))
    print(f"Recall(TPR)={tp/(tp+fn):.4f} Specificity(TNR)={tn/(tn+fp):.4f}")

    print("\nDONE")


if __name__ == "__main__":
    main()
