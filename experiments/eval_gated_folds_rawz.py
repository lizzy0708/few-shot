"""
eval_gated_folds_rawz.py — Experiment 3: isolate whether z_inv's decomposition
(z_inv = z * class_gate) contributes anything to the actual Mahalanobis scoring
pipeline behind Table 2, by rerunning that EXACT pipeline on raw z instead.

Does NOT modify experiments/eval_gated_folds.py, and does NOT retrain anything —
same checkpoints/coarse5_fold*_gated_nodg_s{0,1,2}.pth as Table 2.

The ONLY change from eval_gated_folds.py: `get_features(..., feature_key="z")`
instead of its default `feature_key="z_c_notd"` (="z_inv"=z*class_gate). GatedMaskModel's
"z" output is the raw 4D backbone feature [B,C,H,W] straight out of FeatureExtractor,
before class_gate ever multiplies it -- get_features's own internal
F.adaptive_avg_pool2d(z,1) then produces exactly the same pooled vector as
GatedMaskModel's own "z_pool" key (z_pool = GAP(z) is computed identically inside the
model's forward), so "z" here matches Experiment 1's "raw z" exactly.

Everything else is byte-for-byte identical to eval_gated_folds.py: same checkpoints,
same 3 train-seed x 4 fold x 5 eval-seed structure, same score-ensemble averaging,
same calib_centroid definition (pooled mean of ALL calib-domain normal features in
the chosen feature space), same beta=0.5 blend, same LedoitWolf full-covariance fit,
same n_sigma=2.0 threshold, same exclude_paths leakage fix + disjointness asserts --
achieved by importing and reusing run_fold()/build_support_query()/build_calib_normal()
unchanged, and monkey-patching only the module-level `score_one_model` function that
run_fold() calls internally.

Usage:
  conda run -n torch python experiments/eval_gated_folds_rawz.py --beta 0.5 --n_sigma 2.0
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader

from experiments.eval_all_folds import get_features, mahalanobis_score, fit_normal_distribution
import experiments.eval_gated_folds as base


def score_one_model_rawz(model, support_ds, calib_normal_ds, query_ds, beta, n_sigma):
    """Identical to base.score_one_model except feature_key='z' (raw, pre-gate) instead
    of the default 'z_c_notd' (=z_inv, post-gate)."""
    support_loader = DataLoader(support_ds, batch_size=len(support_ds), shuffle=False)
    query_loader = DataLoader(query_ds, batch_size=32, shuffle=False)
    calib_loader = DataLoader(calib_normal_ds, batch_size=32, shuffle=False)

    support_feats, _ = get_features(model, support_loader, feature_key="z")
    query_feats, query_labels = get_features(model, query_loader, feature_key="z")
    calib_feats, _ = get_features(model, calib_loader, feature_key="z")

    support_np = support_feats.cpu().numpy()
    query_np = query_feats.cpu().numpy()
    calib_np = calib_feats.cpu().numpy()

    # LedoitWolf shrinkage, FULL covariance matrix -- same as base.score_one_model.
    _, prec_np = fit_normal_distribution(calib_np)

    support_proto = support_np.mean(axis=0)
    calib_centroid = calib_np.mean(axis=0)  # pooled mean over ALL calib-domain normals
    blended = beta * support_proto + (1.0 - beta) * calib_centroid

    query_scores = mahalanobis_score(query_np, blended, prec_np)
    support_scores = mahalanobis_score(support_np, blended, prec_np)
    calib_scores = mahalanobis_score(calib_np, blended, prec_np)

    return query_scores, support_scores, calib_scores, query_labels


# Monkey-patch: base.run_fold() looks up `score_one_model` as a module-level global at
# call time, so patching the module attribute redirects it without touching
# eval_gated_folds.py's source. Everything else in run_fold (ensembling, leakage-fixed
# support/query construction, disjointness asserts, threshold/metric computation)
# is reused completely unchanged.
base.score_one_model = score_one_model_rawz


if __name__ == "__main__":
    base.main()
