"""
measure_prediction_invariance.py — 2026-10-07. Does NOT modify any existing
script. New file only. No retraining -- reuses existing checkpoints unchanged.

Metric 2 (prediction invariance): per-domain risk = 1 - BalancedAcc under the
Table 2 few-shot pipeline (support 4-shot + calib-domain LedoitWolf covariance +
beta=0.5 blend + n_sigma=2.0 threshold, leakage-fixed), then the VARIANCE of risk
across domains, for each model/feature. Models/features: same four model
variants as measure_representation_invariance.py -- (a) pretrained, (b) GRL-on
3-seed, (c) GRL-off seed0, (d) domain-label-shuffle seed0 -- x {raw z, z_inv}.

Methodology fixed before results (mirrors the request's pre-registration):

1. PRIMARY risk set (required): for each model(/seed), the 4 OFFICIAL held-out
   domains {500,600,700,800}, each scored using THAT domain's own fold-specific
   checkpoint (exactly Table 2's LOO design) -- 4 risk values -> variance across
   them is the headline Metric 2 number. For GRL-on, computed separately per
   seed (not the 3-seed ensemble) so seed-to-seed std of this variance can be
   reported; for GRL-off/shuffle (1 seed) no seed std exists; for pretrained (one
   fixed encoder, no seed axis) no seed std exists either.
2. SECONDARY/bonus risk set ("가능하면 calib 도메인도"): for every fold-specific
   checkpoint, ALSO score every domain OTHER than its official held-out domain as
   if it were the query domain, with calib dynamically redefined as
   ALL_DOMAINS - {query_domain} (so the official held-out domain is folded back
   into calib for these bonus queries). This yields, per checkpoint, 5 domain-as-
   query risk values instead of 1; domain "400" (never officially held out) is
   covered this way using whichever fold-checkpoint happens to be evaluated --
   reported as its own row, not pooled into the primary variance.
3. Domain-count caveat (stated explicitly, not glossed over): both risk sets have
   only 4-5 data points, so their variance is a highly unstable estimate --
   flagged in every reported table, not just prose.

Protocol per domain-as-query evaluation: SHOT=4 support (seeded by eval_seed,
5 eval-seeds averaged), calib = ALL_DOMAINS - {query_domain} normal-only,
LedoitWolf full covariance, beta=0.5 blend of support-proto/calib-centroid,
threshold = support_mean + 2*sigma(calib), BalancedAcc from the resulting
confusion matrix. Reuses experiments.eval_gated_folds's build_calib_normal,
and experiments.eval_all_folds's get_features/mahalanobis_score/
fit_normal_distribution unchanged; adds its own generalized support/query
builder (arbitrary query domain, not restricted to GATED_FOLDS' 4 official
test domains) and its own class_label-aware extractor passthrough for the
gradient-mask case is NOT needed here (gradient mask is not one of this
metric's 4 model variants).

Usage:
  conda run -n torch python experiments/measure_prediction_invariance.py
"""
import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from datasets.hust_image import HUSTDataset
from experiments.eval_all_folds import transform, device, get_features, mahalanobis_score, fit_normal_distribution
from experiments.eval_pretrained_fewshot import load_pretrained_model
import experiments.eval_gated_folds as base

ALL_DOMAINS = ["400", "500", "600", "700", "800"]
SHOT = 4
EVAL_SEEDS = base.EVAL_SEEDS
BETA = 0.5
N_SIGMA = 2.0

FOLD_CHECKPOINTS = {
    "GRL-off": lambda fold: f"checkpoints/coarse5_fold{fold}_gated_nodg_nogrl_s0.pth",
    "shuffle": lambda fold: f"checkpoints/coarse5_fold{fold}_gated_nodg_shuffled_s0.pth",
}
OFFICIAL_HELDOUT = ["500", "600", "700", "800"]


def build_support_query_generic(query_domain, calib_domains, eval_seed):
    support_ds = HUSTDataset(root=base.ROOT, domain=query_domain, only_normal=True,
                              shot=SHOT, transform=transform, seed=eval_seed, all_domains=ALL_DOMAINS)
    support_paths = set(s[0] for s in support_ds.samples)
    query_ds = HUSTDataset(root=base.ROOT, domain=query_domain, only_normal=False,
                            transform=transform, all_domains=ALL_DOMAINS, exclude_paths=support_paths)
    return support_ds, query_ds


def score_domain(model, query_domain, calib_domains, feature_key, eval_seed):
    support_ds, query_ds = build_support_query_generic(query_domain, calib_domains, eval_seed)
    calib_ds = base.build_calib_normal(calib_domains)

    support_loader = DataLoader(support_ds, batch_size=len(support_ds), shuffle=False)
    query_loader = DataLoader(query_ds, batch_size=32, shuffle=False)
    calib_loader = DataLoader(calib_ds, batch_size=32, shuffle=False)

    support_feats, _ = get_features(model, support_loader, feature_key=feature_key)
    calib_feats, _ = get_features(model, calib_loader, feature_key=feature_key)
    query_feats, query_labels = get_features(model, query_loader, feature_key=feature_key)

    support_np, calib_np, query_np = (t.cpu().numpy() for t in (support_feats, calib_feats, query_feats))
    _, prec_np = fit_normal_distribution(calib_np)
    support_proto = support_np.mean(axis=0)
    calib_centroid = calib_np.mean(axis=0)
    blended = BETA * support_proto + (1.0 - BETA) * calib_centroid

    support_scores = mahalanobis_score(support_np, blended, prec_np)
    calib_scores = mahalanobis_score(calib_np, blended, prec_np)
    query_scores = mahalanobis_score(query_np, blended, prec_np)
    threshold = support_scores.mean() + N_SIGMA * calib_scores.std()
    preds = (query_scores > threshold).astype(int)
    bal_acc = balanced_accuracy_score(query_labels, preds)
    return bal_acc


def risk_for_domain_ensemble(models_by_seed, query_domain, calib_domains, feature_key):
    """3-seed score ensemble (or 1-model if len(models_by_seed)==1), averaged over
    EVAL_SEEDS, mirroring Table 2's ensembling convention."""
    seeds = list(models_by_seed.keys())
    per_eval_seed_bal = []
    for eval_seed in EVAL_SEEDS:
        support_ds, query_ds = build_support_query_generic(query_domain, calib_domains, eval_seed)
        calib_ds = base.build_calib_normal(calib_domains)
        per_model_scores = []
        ref_labels = None
        for s in seeds:
            model = models_by_seed[s]
            support_loader = DataLoader(support_ds, batch_size=len(support_ds), shuffle=False)
            query_loader = DataLoader(query_ds, batch_size=32, shuffle=False)
            calib_loader = DataLoader(calib_ds, batch_size=32, shuffle=False)
            support_feats, _ = get_features(model, support_loader, feature_key=feature_key)
            calib_feats, _ = get_features(model, calib_loader, feature_key=feature_key)
            query_feats, labels = get_features(model, query_loader, feature_key=feature_key)
            support_np, calib_np, query_np = (t.cpu().numpy() for t in (support_feats, calib_feats, query_feats))
            _, prec_np = fit_normal_distribution(calib_np)
            blended = BETA * support_np.mean(axis=0) + (1.0 - BETA) * calib_np.mean(axis=0)
            sc_s = mahalanobis_score(support_np, blended, prec_np)
            sc_c = mahalanobis_score(calib_np, blended, prec_np)
            sc_q = mahalanobis_score(query_np, blended, prec_np)
            if ref_labels is None:
                ref_labels = labels
            per_model_scores.append((sc_q, sc_s, sc_c))
        ens_q = np.mean(np.stack([t[0] for t in per_model_scores]), axis=0)
        ens_s = np.mean(np.stack([t[1] for t in per_model_scores]), axis=0)
        ens_c = np.mean(np.stack([t[2] for t in per_model_scores]), axis=0)
        threshold = ens_s.mean() + N_SIGMA * ens_c.std()
        preds = (ens_q > threshold).astype(int)
        per_eval_seed_bal.append(balanced_accuracy_score(ref_labels, preds))
    return float(np.mean(per_eval_seed_bal))


def run_variant(model_name, models_by_fold_seed, feature_key):
    """models_by_fold_seed: dict[fold] -> dict[seed] -> model (seed key arbitrary
    for single-seed variants, e.g. {0: model})."""
    primary_risk = {}  # fold -> risk (ensembled over available seeds)
    bonus_risk = {}    # (checkpoint_fold, query_domain) -> risk (single ensemble call per combo)
    per_seed_primary = {s: {} for s in next(iter(models_by_fold_seed.values())).keys()}

    for fold in OFFICIAL_HELDOUT:
        models_by_seed = models_by_fold_seed[fold]
        calib = [d for d in ALL_DOMAINS if d != fold]
        bal = risk_for_domain_ensemble(models_by_seed, fold, calib, feature_key)
        primary_risk[fold] = 1.0 - bal
        print(f"  [{model_name}/{feature_key}] PRIMARY fold={fold} query={fold} BalAcc={bal:.4f} risk={1-bal:.4f}")

        for s, model in models_by_seed.items():
            bal_s = risk_for_domain_ensemble({s: model}, fold, calib, feature_key)
            per_seed_primary[s][fold] = 1.0 - bal_s

        for d in ALL_DOMAINS:
            if d == fold:
                continue
            calib_bonus = [x for x in ALL_DOMAINS if x != d]
            bal_b = risk_for_domain_ensemble(models_by_seed, d, calib_bonus, feature_key)
            bonus_risk[(fold, d)] = 1.0 - bal_b
            print(f"  [{model_name}/{feature_key}] BONUS checkpoint_fold={fold} query={d} "
                  f"BalAcc={bal_b:.4f} risk={1-bal_b:.4f}")

    primary_values = list(primary_risk.values())
    primary_var = float(np.var(primary_values))
    per_seed_vars = [float(np.var(list(per_seed_primary[s].values()))) for s in per_seed_primary]

    print(f"\n  [{model_name}/{feature_key}] PRIMARY risk values: {primary_risk}")
    print(f"  [{model_name}/{feature_key}] PRIMARY variance (ensemble): {primary_var:.6g}")
    print(f"  [{model_name}/{feature_key}] PRIMARY variance per-seed: {per_seed_vars} "
          f"mean={np.mean(per_seed_vars):.6g} std={np.std(per_seed_vars):.6g}")
    print(f"  [{model_name}/{feature_key}] BONUS risk values: {bonus_risk}")

    return dict(model=model_name, feature=feature_key, primary_risk=primary_risk,
                primary_var=primary_var, per_seed_vars=per_seed_vars, bonus_risk=bonus_risk)


def main():
    all_results = []

    print("\n### MODEL: pretrained ###")
    pm = load_pretrained_model(None)
    models_by_fold_seed = {fold: {0: pm} for fold in OFFICIAL_HELDOUT}
    all_results.append(run_variant("pretrained", models_by_fold_seed, "z"))

    print("\n### MODEL: GRL-on (3 seed) ###")
    models_by_fold_seed_on = {}
    for fold in OFFICIAL_HELDOUT:
        models_by_fold_seed_on[fold] = {s: base.load_gated_model(base.ckpt_path(fold, s)) for s in base.TRAIN_SEEDS}
    all_results.append(run_variant("GRL-on", models_by_fold_seed_on, "z"))
    all_results.append(run_variant("GRL-on", models_by_fold_seed_on, "z_c_notd"))

    print("\n### MODEL: GRL-off (seed0) ###")
    models_by_fold_seed_off = {fold: {0: base.load_gated_model(FOLD_CHECKPOINTS["GRL-off"](fold))} for fold in OFFICIAL_HELDOUT}
    all_results.append(run_variant("GRL-off", models_by_fold_seed_off, "z"))
    all_results.append(run_variant("GRL-off", models_by_fold_seed_off, "z_c_notd"))

    print("\n### MODEL: domain-label-shuffle (seed0) ###")
    models_by_fold_seed_sh = {fold: {0: base.load_gated_model(FOLD_CHECKPOINTS["shuffle"](fold))} for fold in OFFICIAL_HELDOUT}
    all_results.append(run_variant("shuffle", models_by_fold_seed_sh, "z"))
    all_results.append(run_variant("shuffle", models_by_fold_seed_sh, "z_c_notd"))

    print("\n" + "=" * 100)
    print("FINAL SUMMARY (domain count=4/5 -> variance estimate is unstable, see docstring)")
    print("=" * 100)
    for r in all_results:
        print(f"model={r['model']:>10} feature={r['feature']:>10} primary_var={r['primary_var']:.6g} "
              f"per_seed_var_mean={np.mean(r['per_seed_vars']) if r['per_seed_vars'] else float('nan'):.6g} "
              f"per_seed_var_std={np.std(r['per_seed_vars']) if len(r['per_seed_vars'])>1 else float('nan'):.6g}")

    print("\nDONE")


if __name__ == "__main__":
    main()
