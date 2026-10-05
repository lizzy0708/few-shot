"""
eval_gated_folds_origmask.py — 2026-10-05. Re-evaluates the gradient-threshold
checkpoints (checkpoints/coarse5_fold*_origmask_s{0,1,2}.pth, OriginalMaskModel)
under the EXACT current Table 2 protocol (z_inv, 3-seed score ensemble, beta=0.5,
n_sigma=2.0, LedoitWolf, leakage-fix + disjoint assert, 4-fold x 5 eval-seeds), so
it can be directly compared against the gate (GatedMaskModel) numbers on equal
footing. Does NOT modify experiments/eval_gated_folds.py, experiments/eval_all_folds.py,
or any other existing script -- new file only, no retraining.

Why a new script instead of reusing eval_all_folds.get_features() for this model:
get_features(model, loader, feature_key, md) never passes `class_label` to the
model -- it always calls `model(img)` with the default class_label=None, which
makes OriginalMaskModel's gradient-based mc use max-logit (`class_logits.max(dim=1)
[0].sum()`, models/original_mask_model.py:71-73) for EVERY sample, support/calib
included. But docs/design-docs/gate-vs-gradient-threshold.md's "평가 스크립트 메모"
section documents that this is wrong for support/calib (already only_normal=True,
true label=0 known) and caused worse threshold collapse when done that way ("처음엔
calib/support까지 class_label=None으로 잘못 평가해서 fold 800/600에 훨씬 심한 threshold
붕괴가 나왔었다") -- the correct protocol passes class_label=0 explicitly for
support/calib, and leaves class_label=None (max-logit) only for query (true label
genuinely unknown at inference time). Confirmed via source read: get_features's
signature (experiments/eval_all_folds.py:93) has no class_label parameter at all,
so the current codebase cannot reproduce that corrected protocol without a new
function -- get_features_origmask() below adds exactly that, nothing else.

Grad-context check (explicitly requested): get_features()/get_features_origmask()
are never wrapped in torch.no_grad() (confirmed: `grep -n "no_grad" experiments/
eval_all_folds.py` only matches inside compute_md_from_calib and the deterministic-
algorithms setup, never inside get_features). This is required for OriginalMaskModel,
whose forward() internally calls `torch.autograd.grad(class_score, z, create_graph=True,
retain_graph=True)` (models/original_mask_model.py:74-77, 84-87) to build mc/md --
it needs its OWN `with torch.enable_grad():` block (models/original_mask_model.py:62),
which re-enables grad locally regardless of any OUTER no_grad context, so this would
actually be safe either way; the important fact is simply that nothing in the call
chain used here disables it.

Usage:
  conda run -n torch python experiments/eval_gated_folds_origmask.py --beta 0.5 --n_sigma 2.0
"""
import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from models.original_mask_model import OriginalMaskModel
from experiments.eval_all_folds import device, transform, mahalanobis_score, fit_normal_distribution
import experiments.eval_gated_folds as base


def ckpt_path_origmask(fold, train_seed):
    return f"checkpoints/coarse5_fold{fold}_origmask_s{train_seed}.pth"


def load_origmask_model(path):
    sd = torch.load(path, map_location=device)
    num_domains = sd["domain_classifier.weight"].shape[0]
    in_dim = sd["classifier.weight"].shape[1]
    encoder_layer = "layer3" if in_dim == 1024 else "layer4"
    model = OriginalMaskModel(num_classes=2, num_domains=num_domains, encoder_layer=encoder_layer).to(device)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, f"{path}: state_dict mismatch missing={missing} unexpected={unexpected}"
    model.eval()
    return model


def get_features_origmask(model, loader, class_label_fixed=None):
    """Like experiments.eval_all_folds.get_features(feature_key='z_c_notd') but adds the
    class_label handling OriginalMaskModel needs (see module docstring): if
    class_label_fixed is not None, every sample in this loader is given that fixed label
    (use 0 for support/calib, whose true label is already known via only_normal=True);
    if None, the model falls back to its own max-logit default (use for query, whose
    label is genuinely unknown at inference time)."""
    feats, labels = [], []
    for batch in loader:
        img = batch["image"].to(device)
        if class_label_fixed is not None:
            cl = torch.full((img.size(0),), class_label_fixed, dtype=torch.long, device=device)
        else:
            cl = None
        out = model(img, class_label=cl)
        z = out["z_c_notd"]
        z = F.adaptive_avg_pool2d(z, 1).view(z.size(0), -1)
        feats.append(z.detach())
        labels.extend(batch["label"].numpy().tolist())
    return torch.cat(feats, dim=0), np.array(labels)


def score_one_model_origmask(model, support_ds, calib_normal_ds, query_ds, beta, n_sigma):
    support_loader = DataLoader(support_ds, batch_size=len(support_ds), shuffle=False)
    query_loader = DataLoader(query_ds, batch_size=32, shuffle=False)
    calib_loader = DataLoader(calib_normal_ds, batch_size=32, shuffle=False)

    # support/calib: true label is 0 (only_normal=True) -- class_label=0 fixed.
    support_feats, _ = get_features_origmask(model, support_loader, class_label_fixed=0)
    calib_feats, _ = get_features_origmask(model, calib_loader, class_label_fixed=0)
    # query: true label unknown at inference -- class_label=None (model's own max-logit).
    query_feats, query_labels = get_features_origmask(model, query_loader, class_label_fixed=None)

    support_np = support_feats.cpu().numpy()
    query_np = query_feats.cpu().numpy()
    calib_np = calib_feats.cpu().numpy()

    _, prec_np = fit_normal_distribution(calib_np)
    support_proto = support_np.mean(axis=0)
    calib_centroid = calib_np.mean(axis=0)
    blended = beta * support_proto + (1.0 - beta) * calib_centroid

    query_scores = mahalanobis_score(query_np, blended, prec_np)
    support_scores = mahalanobis_score(support_np, blended, prec_np)
    calib_scores = mahalanobis_score(calib_np, blended, prec_np)
    return query_scores, support_scores, calib_scores, query_labels


def run_fold(fold, test_domains, calib_domains, beta, n_sigma, train_seeds, eval_seeds):
    assert len(test_domains) == 1
    test_domain = test_domains[0]
    models = {s: load_origmask_model(ckpt_path_origmask(fold, s)) for s in train_seeds}
    calib_normal_ds = base.build_calib_normal(calib_domains)

    ens_rows = []
    per_seed_rows = {s: [] for s in train_seeds}

    for eval_seed in eval_seeds:
        support_ds, query_ds = base.build_support_query(test_domain, eval_seed)

        per_model_query, per_model_support, per_model_calib, ref_labels = [], [], [], None
        for s in train_seeds:
            q, sc, c, labels = score_one_model_origmask(models[s], support_ds, calib_normal_ds, query_ds, beta, n_sigma)
            if ref_labels is None:
                ref_labels = labels
            else:
                assert np.array_equal(ref_labels, labels), "query label order mismatch across train seeds"
            per_model_query.append(q)
            per_model_support.append(sc)
            per_model_calib.append(c)

            # per-seed (no ensemble) metrics for this (eval_seed, train_seed)
            thr_s = sc.mean() + n_sigma * c.std()
            preds_s = (q > thr_s).astype(int)
            per_seed_rows[s].append(dict(
                eval_seed=eval_seed,
                auroc=roc_auc_score(labels, q),
                acc=accuracy_score(labels, preds_s),
                f1=f1_score(labels, preds_s, zero_division=0),
            ))

        ens_query = np.mean(np.stack(per_model_query, axis=0), axis=0)
        ens_support = np.mean(np.stack(per_model_support, axis=0), axis=0)
        ens_calib = np.mean(np.stack(per_model_calib, axis=0), axis=0)
        threshold = ens_support.mean() + n_sigma * ens_calib.std()
        preds = (ens_query > threshold).astype(int)

        ens_rows.append(dict(
            eval_seed=eval_seed,
            auroc=roc_auc_score(ref_labels, ens_query),
            acc=accuracy_score(ref_labels, preds),
            f1=f1_score(ref_labels, preds, zero_division=0),
        ))
        print(f"  [origmask ensemble] fold={fold} eval_seed={eval_seed} beta={beta} n_sigma={n_sigma} "
              f"| AUROC={ens_rows[-1]['auroc']:.4f} Acc={ens_rows[-1]['acc']:.4f} F1={ens_rows[-1]['f1']:.4f}")

    return ens_rows, per_seed_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--n_sigma", type=float, default=2.0)
    ap.add_argument("--train_seeds", type=int, nargs="+", default=base.TRAIN_SEEDS)
    ap.add_argument("--eval_seeds", type=int, nargs="+", default=base.EVAL_SEEDS)
    ap.add_argument("--test_fold", type=str, default=None, choices=[f[0] for f in base.GATED_FOLDS])
    args = ap.parse_args()

    folds = base.GATED_FOLDS if args.test_fold is None else [f for f in base.GATED_FOLDS if f[0] == args.test_fold]
    print(f"Device={device} beta={args.beta} n_sigma={args.n_sigma} train_seeds={args.train_seeds} eval_seeds={args.eval_seeds}")

    all_ens = {}
    all_per_seed = {}
    for fold, test_domains, calib_domains in folds:
        ens_rows, per_seed_rows = run_fold(fold, test_domains, calib_domains, args.beta, args.n_sigma,
                                            args.train_seeds, args.eval_seeds)
        all_ens[fold] = ens_rows
        all_per_seed[fold] = per_seed_rows

    print("\n" + "=" * 90)
    print("ENSEMBLE SUMMARY (3-seed score ensemble, mean+-std over eval-seeds)")
    print("=" * 90)
    for fold, rows in all_ens.items():
        aurocs = [r["auroc"] for r in rows]
        accs = [r["acc"] for r in rows]
        f1s = [r["f1"] for r in rows]
        print(f"{fold:>6} | {np.mean(aurocs):.4f}±{np.std(aurocs):.4f} | "
              f"{np.mean(accs):.4f}±{np.std(accs):.4f} | {np.mean(f1s):.4f}±{np.std(f1s):.4f}")
    if len(all_ens) == 4:
        avg_auroc = np.mean([np.mean([r["auroc"] for r in rows]) for rows in all_ens.values()])
        avg_acc = np.mean([np.mean([r["acc"] for r in rows]) for rows in all_ens.values()])
        avg_f1 = np.mean([np.mean([r["f1"] for r in rows]) for rows in all_ens.values()])
        print(f"{'Avg':>6} | {avg_auroc:.4f} | {avg_acc:.4f} | {avg_f1:.4f}")

    print("\n" + "=" * 90)
    print("PER-SEED SUMMARY (no ensemble, mean over eval-seeds, then mean+-std over train-seeds)")
    print("=" * 90)
    for fold, per_seed in all_per_seed.items():
        for s, rows in per_seed.items():
            aurocs = [r["auroc"] for r in rows]
            accs = [r["acc"] for r in rows]
            f1s = [r["f1"] for r in rows]
            print(f"fold={fold} seed={s} | AUROC={np.mean(aurocs):.4f} Acc={np.mean(accs):.4f} F1={np.mean(f1s):.4f}")

    if len(all_per_seed) == 4:
        seeds = args.train_seeds
        for s in seeds:
            seed_aurocs = [np.mean([r["auroc"] for r in all_per_seed[fold][s]]) for fold in all_per_seed]
            seed_accs = [np.mean([r["acc"] for r in all_per_seed[fold][s]]) for fold in all_per_seed]
            seed_f1s = [np.mean([r["f1"] for r in all_per_seed[fold][s]]) for fold in all_per_seed]
            print(f"seed={s} 4-fold avg | AUROC={np.mean(seed_aurocs):.4f} Acc={np.mean(seed_accs):.4f} F1={np.mean(seed_f1s):.4f}")

    print("\nDONE")


if __name__ == "__main__":
    main()
