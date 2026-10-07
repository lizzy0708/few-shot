"""
measure_representation_invariance.py — 2026-10-07. Does NOT modify any existing
script. New file only. No retraining -- reuses existing checkpoints unchanged.

Re-measures representation invariance per the XDomainMix (IJCAI 2024, Sec 4.2
"Model Invariance") style class/domain-covariance-distance metric, for:
  features: raw z (pre-gate GAP) and z_inv (post class_gate GAP)
  models:   (a) pretrained (untrained ImageNet ResNet50 layer3)
            (b) GRL-on gate, 3 seeds (checkpoints/coarse5_fold*_gated_nodg_s{0,1,2}.pth)
            (c) GRL-off seed0 (checkpoints/coarse5_fold*_gated_nodg_nogrl_s0.pth)
            (d) domain-label-shuffle seed0 (checkpoints/coarse5_fold*_gated_nodg_shuffled_s0.pth)

ALL methodology choices below are fixed BEFORE any result is produced (per the
task's pre-registration requirement) and are not changed afterward:

1. Per (domain, class) covariance C_y^(i): plain empirical covariance
   (np.cov(X, rowvar=False), ddof=1) of the GAP-pooled feature, class y in
   {0=normal, 1=anomaly}, domain i in {400,500,600,700,800}.
2. Sample size: N=1497 per (domain,class) group, EVERY group (normal is already
   exactly 1497/domain; anomaly is uniformly randomly subsampled down from
   ~8000-9000/domain -- NOT fault-type-stratified, unlike the separate balanced-
   eval task; this is a plain random subsample, seed=42, chosen ONCE independent
   of model/feature so every model/feature compares the identical underlying
   images). Confirmed group sizes: `ls processed/<d>/normal | wc -l` = 1497 for
   all d in {400,500,600,700,800}.
3. Metric 1 (raw), scope (i) calib-calib (mirrors XDomainMix: invariance measured
   across the N training/calib domains of a given split):
     M1_i = (1/(|Y|*|D_calib|)) * sum_{y in Y} sum_{unordered pairs i<i' in D_calib}
            ||C_y^(i) - C_y^(i')||_F^2
   |Y|=2, |D_calib|=4 (so denominator=8, same as scope (ii) below -- NOT the
   number of pairs C(4,2)=6; this literal denominator is as specified in the task
   request, kept exactly as given rather than re-derived).
4. Metric 1 (raw), scope (ii) held-out vs calib (every calib domain compared
   against the one domain this fold's checkpoint never trained on):
     M1_ii = (1/(|Y|*|D_calib|)) * sum_{y in Y} sum_{d in D_calib}
             ||C_y^(held) - C_y^(d)||_F^2
5. Normalization (primary reported metric): M1_normalized = M1_raw / mean(||C_y^(i)||_F^2)
   where the mean is taken over exactly the same (domain,class) covariances that
   appear in that specific M1 sum (so scope (i)'s normalizer uses only the 4 calib
   domains' covariances x 2 classes = 8 matrices; scope (ii)'s normalizer uses the
   held-out domain's 2 covariances plus the 4 calib domains' 8 = 10 matrices).
6. Noise floor: for each (domain,class) group appearing in a given scope's domain
   set, split its N=1497 samples into two random halves (seed=123, fixed, same
   split reused across all models/features for comparability), compute their two
   covariances, take ||C_half1-C_half2||_F^2 normalized by mean(||C_half1||_F^2,
   ||C_half2||_F^2) (that group's own scale), then AVERAGE this normalized value
   across all (domain,class) groups in the scope's domain set -- giving one
   noise-floor number directly on the same normalized scale as M1_normalized,
   per scope/model/feature.
7. Fold-specificity: pretrained is model-independent of fold (one fixed encoder),
   so its features are extracted ONCE per (domain,class) and reused across all 4
   folds' scope computations. GRL-on/off/shuffle use a genuinely different
   checkpoint per fold (coarse5_fold<fold>_..._s<seed>.pth), so features are
   extracted separately for each (fold, seed) combination.

Feature-mixing/augmentation check (requested, answered directly): grep across
experiments/train_gated_mask_model.py, models/gated_mask_model.py and
datasets/hust_image.py for mixup/mixing/cutmix/augment/lam/Beta-sampling finds
ZERO matches -- confirmed NONE present (datasets/hust_image.py's default
transform is Resize((224,224))+ToTensor() only, no augmentation).

Usage:
  conda run -n torch python experiments/measure_representation_invariance.py
"""
import os
import sys
import random
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.hust_image import HUSTDataset
from experiments.eval_all_folds import transform, device
from experiments.eval_pretrained_fewshot import load_pretrained_model
import experiments.eval_gated_folds as base

ALL_DOMAINS = ["400", "500", "600", "700", "800"]
N_PER_GROUP = 1497
SUBSAMPLE_SEED = 42
SPLIT_SEED = 123

GATED_FOLDS_ALL_AS_HELDOUT = {
    "500": ["400", "600", "700", "800"],
    "600": ["400", "500", "700", "800"],
    "700": ["400", "500", "600", "800"],
    "800": ["400", "500", "600", "700"],
}


def get_domain_class_samples(domain, label):
    ds = HUSTDataset(root=base.ROOT, domain=domain, only_normal=(label == 0),
                      transform=transform, all_domains=ALL_DOMAINS)
    if label == 0:
        return ds.samples
    return [s for s in ds.samples if s[1] == 1]


def subsample(samples, n, seed):
    rng = random.Random(seed)
    pool = samples.copy()
    rng.shuffle(pool)
    assert len(pool) >= n, f"only {len(pool)} samples available, need {n}"
    return pool[:n]


def build_group_dataset(domain, label):
    samples = subsample(get_domain_class_samples(domain, label), N_PER_GROUP, SUBSAMPLE_SEED)
    ds = HUSTDataset(root=base.ROOT, domain=domain, only_normal=(label == 0),
                      transform=transform, all_domains=ALL_DOMAINS)
    ds.samples = samples
    return ds


def extract_both_features(model, loader, is_pretrained=False):
    raw_feats, inv_feats = [], []
    for batch in loader:
        img = batch["image"].to(device)
        out = model(img)
        z = F.adaptive_avg_pool2d(out["z"], 1).flatten(1)
        zinv = F.adaptive_avg_pool2d(out["z_inv"], 1).flatten(1)
        raw_feats.append(z.detach())
        inv_feats.append(zinv.detach())
    raw = torch.cat(raw_feats, dim=0).cpu().numpy()
    inv = torch.cat(inv_feats, dim=0).cpu().numpy()
    return raw, inv


def covariance(X):
    return np.cov(X, rowvar=False, ddof=1)


def frob_sq(A, B):
    return float(np.sum((A - B) ** 2))


def noise_floor_for_group(X, split_seed=SPLIT_SEED):
    n = X.shape[0]
    idx = list(range(n))
    random.Random(split_seed).shuffle(idx)
    half = n // 2
    X1, X2 = X[idx[:half]], X[idx[half:2 * half]]
    C1, C2 = covariance(X1), covariance(X2)
    dist = frob_sq(C1, C2)
    scale = (np.sum(C1 ** 2) + np.sum(C2 ** 2)) / 2.0
    return dist / scale if scale > 0 else float("nan")


def metric1_scope_i(covs, calib_domains, classes=(0, 1)):
    total = 0.0
    for y in classes:
        for a in range(len(calib_domains)):
            for b in range(a + 1, len(calib_domains)):
                total += frob_sq(covs[(calib_domains[a], y)], covs[(calib_domains[b], y)])
    denom = len(classes) * len(calib_domains)
    raw = total / denom
    norm_set = [covs[(d, y)] for d in calib_domains for y in classes]
    norm = np.mean([np.sum(c ** 2) for c in norm_set])
    return raw, raw / norm if norm > 0 else float("nan")


def metric1_scope_ii(covs, held_domain, calib_domains, classes=(0, 1)):
    total = 0.0
    for y in classes:
        for d in calib_domains:
            total += frob_sq(covs[(held_domain, y)], covs[(d, y)])
    denom = len(classes) * len(calib_domains)
    raw = total / denom
    norm_set = [covs[(held_domain, y)] for y in classes] + [covs[(d, y)] for d in calib_domains for y in classes]
    norm = np.mean([np.sum(c ** 2) for c in norm_set])
    return raw, raw / norm if norm > 0 else float("nan")


def extract_all_domain_class_features(model):
    """Returns dict[(domain,label)] -> (raw_feats [N,1024], inv_feats [N,1024])."""
    out = {}
    for domain in ALL_DOMAINS:
        for label in (0, 1):
            ds = build_group_dataset(domain, label)
            loader = DataLoader(ds, batch_size=32, shuffle=False)
            raw, inv = extract_both_features(model, loader)
            out[(domain, label)] = (raw, inv)
            print(f"    extracted domain={domain} label={label} n={raw.shape[0]}")
    return out


def compute_covs_and_noise(feats_dict, feature_name):
    """feature_name: 'raw' or 'inv'. Returns covs dict[(domain,label)]->cov, and
    noise_floor dict[(domain,label)]->normalized noise-floor value."""
    idx = 0 if feature_name == "raw" else 1
    covs, noise = {}, {}
    for (domain, label), feats in feats_dict.items():
        X = feats[idx]
        covs[(domain, label)] = covariance(X)
        noise[(domain, label)] = noise_floor_for_group(X)
    return covs, noise


def report_scope(covs, noise, fold, calib_domains, label):
    i_raw, i_norm = metric1_scope_i(covs, calib_domains)
    ii_raw, ii_norm = metric1_scope_ii(covs, fold, calib_domains)
    nf_i = np.mean([noise[(d, y)] for d in calib_domains for y in (0, 1)])
    nf_ii = np.mean([noise[(fold, y)] for y in (0, 1)] + [noise[(d, y)] for d in calib_domains for y in (0, 1)])
    print(f"  [{label}] fold={fold} scope(i) calib-calib: raw={i_raw:.6g} norm={i_norm:.6g} noise_floor={nf_i:.6g}")
    print(f"  [{label}] fold={fold} scope(ii) held-vs-calib: raw={ii_raw:.6g} norm={ii_norm:.6g} noise_floor={nf_ii:.6g}")
    return dict(scope_i_raw=i_raw, scope_i_norm=i_norm, scope_i_noise=nf_i,
                scope_ii_raw=ii_raw, scope_ii_norm=ii_norm, scope_ii_noise=nf_ii)


def main():
    print("=" * 100)
    print("Feature-mixing/augmentation check: grep found ZERO matches in "
          "train_gated_mask_model.py/gated_mask_model.py/hust_image.py for "
          "mixup/mixing/cutmix/augment/lam/Beta-sampling -- NONE present.")
    print("=" * 100)

    results = []  # list of dicts

    # --- (a) pretrained: model-independent of fold, extract once ---
    print("\n### MODEL: pretrained ###")
    model = load_pretrained_model(None)
    feats = extract_all_domain_class_features(model)
    covs_raw, noise_raw = compute_covs_and_noise(feats, "raw")
    covs_inv, noise_inv = compute_covs_and_noise(feats, "inv")  # identical to raw for pretrained, kept for symmetry
    for fold, calib in GATED_FOLDS_ALL_AS_HELDOUT.items():
        r = report_scope(covs_raw, noise_raw, fold, calib, "pretrained/raw_z")
        results.append(dict(model="pretrained", feature="raw_z", seed=None, fold=fold, **r))
    del model

    # --- (b) GRL-on, 3 seeds ---
    print("\n### MODEL: GRL-on (gate) ###")
    for seed in base.TRAIN_SEEDS:
        for fold, calib in GATED_FOLDS_ALL_AS_HELDOUT.items():
            print(f"  loading GRL-on seed={seed} fold={fold}")
            model = base.load_gated_model(base.ckpt_path(fold, seed))
            feats = extract_all_domain_class_features(model)
            covs_raw, noise_raw = compute_covs_and_noise(feats, "raw")
            covs_inv, noise_inv = compute_covs_and_noise(feats, "inv")
            r_raw = report_scope(covs_raw, noise_raw, fold, calib, f"GRL-on_s{seed}/raw_z")
            r_inv = report_scope(covs_inv, noise_inv, fold, calib, f"GRL-on_s{seed}/z_inv")
            results.append(dict(model="GRL-on", feature="raw_z", seed=seed, fold=fold, **r_raw))
            results.append(dict(model="GRL-on", feature="z_inv", seed=seed, fold=fold, **r_inv))
            del model

    # --- (c) GRL-off seed0 ---
    print("\n### MODEL: GRL-off (seed0) ###")
    for fold, calib in GATED_FOLDS_ALL_AS_HELDOUT.items():
        print(f"  loading GRL-off fold={fold}")
        model = base.load_gated_model(f"checkpoints/coarse5_fold{fold}_gated_nodg_nogrl_s0.pth")
        feats = extract_all_domain_class_features(model)
        covs_raw, noise_raw = compute_covs_and_noise(feats, "raw")
        covs_inv, noise_inv = compute_covs_and_noise(feats, "inv")
        r_raw = report_scope(covs_raw, noise_raw, fold, calib, "GRL-off/raw_z")
        r_inv = report_scope(covs_inv, noise_inv, fold, calib, "GRL-off/z_inv")
        results.append(dict(model="GRL-off", feature="raw_z", seed=0, fold=fold, **r_raw))
        results.append(dict(model="GRL-off", feature="z_inv", seed=0, fold=fold, **r_inv))
        del model

    # --- (d) domain-label-shuffle seed0 ---
    print("\n### MODEL: domain-label-shuffle (seed0) ###")
    for fold, calib in GATED_FOLDS_ALL_AS_HELDOUT.items():
        print(f"  loading shuffle fold={fold}")
        model = base.load_gated_model(f"checkpoints/coarse5_fold{fold}_gated_nodg_shuffled_s0.pth")
        feats = extract_all_domain_class_features(model)
        covs_raw, noise_raw = compute_covs_and_noise(feats, "raw")
        covs_inv, noise_inv = compute_covs_and_noise(feats, "inv")
        r_raw = report_scope(covs_raw, noise_raw, fold, calib, "shuffle/raw_z")
        r_inv = report_scope(covs_inv, noise_inv, fold, calib, "shuffle/z_inv")
        results.append(dict(model="shuffle", feature="raw_z", seed=0, fold=fold, **r_raw))
        results.append(dict(model="shuffle", feature="z_inv", seed=0, fold=fold, **r_inv))
        del model

    print("\n" + "=" * 100)
    print("FULL RESULTS TABLE (one row per model/feature/seed/fold)")
    print("=" * 100)
    header = ("model", "feature", "seed", "fold", "scope_i_norm", "scope_i_noise",
              "scope_ii_norm", "scope_ii_noise")
    print(" | ".join(header))
    for r in results:
        print(" | ".join(str(r.get(h, "")) for h in header))

    print("\nDONE")


if __name__ == "__main__":
    main()
