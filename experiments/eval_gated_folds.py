"""
eval_gated_folds.py — leakage-fixed, dead-flag-free evaluation for the coarse5
gated-model (GatedMaskModel, `use_domain_gate=False` / "nodg") track.

Written 2026-09-19 to replace the LOST `scratchpad/ensemble_all_folds_beta05.py`
(referenced in docs/exec-plans/completed/2026-08-coarse5-gated-model-pipeline.md
and docs/generated/results.md, but no longer present anywhere on disk). This is a
NEW, from-scratch script — it does not try to byte-for-byte recreate the missing
file, only to reproduce (or show divergence from) the documented protocol using
checkpoints/coarse5_fold*_gated_nodg_s{0,1,2}.pth, which already exist and are
NOT retrained here.

Protocol (per ARCHITECTURE.md "트랙 3", docs/design-docs/proto-beta-blending.md,
docs/design-docs/threshold-protocol.md, AGENTS.md protocol table):
  - Model: GatedMaskModel, use_domain_gate=False (class_gate only), encoder_layer=layer3
  - Data root: processed/ (coarse5), 5 domains {400,500,600,700,800}, 400 never held out
  - 4-fold LOO over {500,600,700,800}; calib_domains = the other 4 domains (incl. 400)
  - Feature: z_c_notd == z_inv = z * class_gate  (GAP-pooled)
  - Covariance: LedoitWolf FULL matrix (sklearn.covariance.LedoitWolf), fit on pooled
    calib-domain NORMAL-ONLY features (reuses eval_all_folds.fit_normal_distribution)
  - Prototype blend: blended = beta*support_proto + (1-beta)*calib_centroid
      support_proto  = mean of the 4-shot support set's z_inv (this model's feature space)
      calib_centroid = mean of ALL calib-domain NORMAL z_inv, pooled across domains
        (docs/design-docs/proto-beta-blending.md: "calib 도메인들의 (uncentered) 전체
        정상 z_inv 평균" — the same pooled set the LedoitWolf fit uses, not a
        per-domain-then-averaged mean)
  - Threshold: threshold = mean(support scores) + n_sigma * std(calib scores), scores
    = Mahalanobis distance to the blended prototype under the fitted precision matrix
    (docs/design-docs/threshold-protocol.md: normal-only, leakage-free n_sigma protocol)
  - 3 TRAINING-seed ensemble (checkpoints s0/s1/s2): per RELIABILITY.md §1, "3개 학습
    seed의 이상 점수를 평균하는 점수 앙상블" — a SCORE ensemble. Implemented here as:
    for each of the 3 checkpoints, independently compute (query_scores, support_scores,
    calib_scores) in that checkpoint's own feature space, then elementwise-average the
    three parallel score arrays before computing AUROC / the threshold / Acc / F1. This
    is the natural single-mechanism generalization of "averaging the anomaly score" to
    both the ranking metric (AUROC) and the threshold-derived metrics (Acc/F1) — the
    docs do not spell out the threshold side explicitly, so this is a documented design
    choice, not a quoted fact.
  - 5 EVAL-seed loop (support/query resampling), separate axis from the 3 training seeds.
  - Leakage fix: HUSTDataset(..., exclude_paths=support_paths) for query_ds construction,
    identical to eval_all_folds.py's 2026-09-17 fix, plus an explicit disjointness assert
    per (fold, eval_seed).

Usage:
  conda run -n torch python experiments/eval_gated_folds.py
  conda run -n torch python experiments/eval_gated_folds.py --sanity_check
  conda run -n torch python experiments/eval_gated_folds.py --test_fold 500
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

from datasets.hust_image import HUSTDataset
from models.gated_mask_model import GatedMaskModel
from experiments.eval_all_folds import (
    transform, device, get_features, mahalanobis_score, fit_normal_distribution,
)

ALL_DOMAINS = ["400", "500", "600", "700", "800"]
ROOT = "processed"  # AGENTS.md protocol table: coarse5 gated track uses processed/

# (test_fold, test_domains, calib_domains) — 4-fold LOO over {500,600,700,800};
# "400" is never held out (matches train_gated_mask_model.py's --all_domains and
# the fact that only 4 fold checkpoints exist, not 5) — mirrors eval_all_folds.py's
# FOLDS_COARSE structure exactly.
GATED_FOLDS = [
    ("500", ["500"], ["400", "600", "700", "800"]),
    ("600", ["600"], ["400", "500", "700", "800"]),
    ("700", ["700"], ["400", "500", "600", "800"]),
    ("800", ["800"], ["400", "500", "600", "700"]),
]

TRAIN_SEEDS = [0, 1, 2]
EVAL_SEEDS = [0, 1, 2, 3, 4]
SHOT = 4


def ckpt_path(fold, train_seed):
    return f"checkpoints/coarse5_fold{fold}_gated_nodg_s{train_seed}.pth"


def load_gated_model(path):
    sd = torch.load(path, map_location=device)
    assert "domain_classifier_disc.weight" not in sd and "domain_gate_net.net.0.weight" not in sd, (
        f"{path}: checkpoint has domain_gate keys — this is not a 'nodg' checkpoint, refusing to load"
    )
    num_domains = sd["domain_classifier.weight"].shape[0]
    in_dim = sd["classifier.weight"].shape[1]
    encoder_layer = "layer3" if in_dim == 1024 else "layer4"
    model = GatedMaskModel(
        num_classes=2, num_domains=num_domains,
        encoder_layer=encoder_layer, use_domain_gate=False,
    ).to(device)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, f"{path}: state_dict mismatch missing={missing} unexpected={unexpected}"
    model.eval()
    return model


def build_support_query(test_domain, eval_seed):
    """Leakage-fixed support/query construction (mirrors eval_all_folds.py post-09-17-fix)."""
    support_ds = HUSTDataset(
        root=ROOT, domain=test_domain, only_normal=True,
        shot=SHOT, transform=transform, seed=eval_seed, all_domains=ALL_DOMAINS,
    )
    support_paths = set(s[0] for s in support_ds.samples)
    query_ds = HUSTDataset(
        root=ROOT, domain=test_domain, only_normal=False,
        transform=transform, all_domains=ALL_DOMAINS,
        exclude_paths=support_paths,
    )
    query_paths = set(s[0] for s in query_ds.samples)
    ok = support_paths.isdisjoint(query_paths)
    assert ok, (
        f"support/query leakage in domain={test_domain} eval_seed={eval_seed}: "
        f"{support_paths & query_paths}"
    )
    print(f"    [leakage-check] fold_domain={test_domain} eval_seed={eval_seed} "
          f"support={len(support_paths)} query={len(query_paths)} disjoint={ok}")
    return support_ds, query_ds


def build_calib_normal(calib_domains):
    calib_sets = [
        HUSTDataset(root=ROOT, domain=d, only_normal=True, transform=transform, all_domains=ALL_DOMAINS)
        for d in calib_domains
    ]
    return ConcatDataset(calib_sets)


def score_one_model(model, support_ds, calib_normal_ds, query_ds, beta, n_sigma):
    """Score a single (fold, train_seed) model. Returns per-sample arrays + query_labels."""
    support_loader = DataLoader(support_ds, batch_size=len(support_ds), shuffle=False)
    query_loader = DataLoader(query_ds, batch_size=32, shuffle=False)
    calib_loader = DataLoader(calib_normal_ds, batch_size=32, shuffle=False)

    support_feats, _ = get_features(model, support_loader)
    query_feats, query_labels = get_features(model, query_loader)
    calib_feats, _ = get_features(model, calib_loader)

    support_np = support_feats.cpu().numpy()
    query_np = query_feats.cpu().numpy()
    calib_np = calib_feats.cpu().numpy()

    # LedoitWolf shrinkage, FULL covariance matrix (not diagonal) — reused as-is from
    # eval_all_folds.py's fit_normal_distribution (task requirement #5).
    _, prec_np = fit_normal_distribution(calib_np)

    support_proto = support_np.mean(axis=0)
    calib_centroid = calib_np.mean(axis=0)  # pooled mean over ALL calib-domain normals
    blended = beta * support_proto + (1.0 - beta) * calib_centroid

    query_scores = mahalanobis_score(query_np, blended, prec_np)
    support_scores = mahalanobis_score(support_np, blended, prec_np)
    calib_scores = mahalanobis_score(calib_np, blended, prec_np)

    return query_scores, support_scores, calib_scores, query_labels


def run_fold(fold, test_domains, calib_domains, beta, n_sigma, train_seeds, eval_seeds, verbose=True):
    assert len(test_domains) == 1, "coarse5 gated folds are single-domain"
    test_domain = test_domains[0]

    models = {s: load_gated_model(ckpt_path(fold, s)) for s in train_seeds}
    calib_normal_ds = build_calib_normal(calib_domains)

    fold_aurocs, fold_accs, fold_f1s, fold_precs, fold_recs = [], [], [], [], []

    for eval_seed in eval_seeds:
        support_ds, query_ds = build_support_query(test_domain, eval_seed)

        per_model_query, per_model_support, per_model_calib, ref_labels = [], [], [], None
        for s in train_seeds:
            q_scores, s_scores, c_scores, q_labels = score_one_model(
                models[s], support_ds, calib_normal_ds, query_ds, beta, n_sigma
            )
            if ref_labels is None:
                ref_labels = q_labels
            else:
                assert np.array_equal(ref_labels, q_labels), (
                    f"query label order mismatch across train seeds for fold={fold} eval_seed={eval_seed}"
                )
            per_model_query.append(q_scores)
            per_model_support.append(s_scores)
            per_model_calib.append(c_scores)

        # 3-train-seed SCORE ensemble: elementwise average of the per-model score arrays
        # (RELIABILITY.md §1: "3개 학습 seed의 이상 점수를 평균하는 점수 앙상블").
        ens_query = np.mean(np.stack(per_model_query, axis=0), axis=0)
        ens_support = np.mean(np.stack(per_model_support, axis=0), axis=0)
        ens_calib = np.mean(np.stack(per_model_calib, axis=0), axis=0)

        threshold = ens_support.mean() + n_sigma * ens_calib.std()
        preds = (ens_query > threshold).astype(int)

        auroc = roc_auc_score(ref_labels, ens_query)
        acc = accuracy_score(ref_labels, preds)
        f1 = f1_score(ref_labels, preds, zero_division=0)
        prec = precision_score(ref_labels, preds, zero_division=0)
        rec = recall_score(ref_labels, preds, zero_division=0)

        fold_aurocs.append(auroc)
        fold_accs.append(acc)
        fold_f1s.append(f1)
        fold_precs.append(prec)
        fold_recs.append(rec)

        if verbose:
            print(f"    fold={fold} eval_seed={eval_seed} beta={beta} n_sigma={n_sigma} "
                  f"| AUROC={auroc:.4f} Acc={acc:.4f} F1={f1:.4f}")

    return {
        "auroc_mean": float(np.mean(fold_aurocs)), "auroc_std": float(np.std(fold_aurocs)),
        "acc_mean": float(np.mean(fold_accs)), "acc_std": float(np.std(fold_accs)),
        "f1_mean": float(np.mean(fold_f1s)), "f1_std": float(np.std(fold_f1s)),
        "prec_mean": float(np.mean(fold_precs)), "rec_mean": float(np.mean(fold_recs)),
    }


def sanity_check(beta_a=1.0, beta_b=0.5, n_sigma=2.0, fold="500", train_seed=0, eval_seed=0):
    """Prove proto_beta is NOT dead: compare beta=1.0 (no blend) vs beta=0.5 on a single
    (fold, train_seed, eval_seed) triple, single-model (no ensemble) for isolation."""
    fold_row = [f for f in GATED_FOLDS if f[0] == fold][0]
    _, test_domains, calib_domains = fold_row
    test_domain = test_domains[0]

    model = load_gated_model(ckpt_path(fold, train_seed))
    calib_normal_ds = build_calib_normal(calib_domains)
    support_ds, query_ds = build_support_query(test_domain, eval_seed)

    results = {}
    for beta in (beta_a, beta_b):
        q_scores, s_scores, c_scores, q_labels = score_one_model(
            model, support_ds, calib_normal_ds, query_ds, beta, n_sigma
        )
        threshold = s_scores.mean() + n_sigma * c_scores.std()
        preds = (q_scores > threshold).astype(int)
        auroc = roc_auc_score(q_labels, q_scores)
        acc = accuracy_score(q_labels, preds)
        f1 = f1_score(q_labels, preds, zero_division=0)
        results[beta] = (auroc, acc, f1, q_scores.copy())
        print(f"  [sanity] fold={fold} train_seed={train_seed} eval_seed={eval_seed} beta={beta} "
              f"| AUROC={auroc:.4f} Acc={acc:.4f} F1={f1:.4f}")

    a = results[beta_a]
    b = results[beta_b]
    identical_scores = np.array_equal(a[3], b[3])
    print(f"  [sanity] score arrays identical beta={beta_a} vs beta={beta_b}? {identical_scores}")
    print(f"  [sanity] AUROC delta = {a[0]-b[0]:+.4f} | Acc delta = {a[1]-b[1]:+.4f} | F1 delta = {a[2]-b[2]:+.4f}")
    if identical_scores or (a[:3] == b[:3]):
        print("  [sanity] WARNING: beta had NO effect — proto_beta may still be dead in this implementation!")
    else:
        print("  [sanity] OK: beta changes the score/metrics — proto_beta is live.")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=0.5, help="proto_beta (paper value: 0.5)")
    ap.add_argument("--n_sigma", type=float, default=2.0, help="threshold n_sigma (paper value: 2.0)")
    ap.add_argument("--test_fold", type=str, default=None, choices=[f[0] for f in GATED_FOLDS])
    ap.add_argument("--train_seeds", type=int, nargs="+", default=TRAIN_SEEDS)
    ap.add_argument("--eval_seeds", type=int, nargs="+", default=EVAL_SEEDS)
    ap.add_argument("--sanity_check", action="store_true",
                     help="Run only the beta=1.0 vs beta=0.5 single-fold/seed sanity check and exit.")
    args = ap.parse_args()

    if args.sanity_check:
        sanity_check(n_sigma=args.n_sigma)
        return

    folds = GATED_FOLDS if args.test_fold is None else [f for f in GATED_FOLDS if f[0] == args.test_fold]

    print(f"Device      : {device}")
    print(f"Root        : {ROOT}")
    print(f"beta        : {args.beta}")
    print(f"n_sigma     : {args.n_sigma}")
    print(f"train_seeds : {args.train_seeds}")
    print(f"eval_seeds  : {args.eval_seeds}")
    print()

    header = f"{'Fold':>6} | {'AUROC':>16} | {'Acc':>16} | {'F1':>16}"
    print(header)
    print("-" * len(header))

    all_rows = {}
    for fold, test_domains, calib_domains in folds:
        res = run_fold(fold, test_domains, calib_domains, args.beta, args.n_sigma,
                        args.train_seeds, args.eval_seeds)
        all_rows[fold] = res
        print(f"{fold:>6} | {res['auroc_mean']:.4f}±{res['auroc_std']:.4f} "
              f"| {res['acc_mean']:.4f}±{res['acc_std']:.4f} "
              f"| {res['f1_mean']:.4f}±{res['f1_std']:.4f}")

    print("-" * len(header))
    if len(all_rows) == 4:
        avg_auroc = np.mean([r["auroc_mean"] for r in all_rows.values()])
        avg_acc = np.mean([r["acc_mean"] for r in all_rows.values()])
        avg_f1 = np.mean([r["f1_mean"] for r in all_rows.values()])
        print(f"{'Avg':>6} | {avg_auroc:>16.4f} | {avg_acc:>16.4f} | {avg_f1:>16.4f}")


if __name__ == "__main__":
    main()
