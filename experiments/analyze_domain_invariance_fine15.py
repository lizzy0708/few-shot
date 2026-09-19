"""
analyze_domain_invariance_fine15.py — Experiment 2 of 4, Stage 1, task 2.

Minimally-adapted copy of experiments/analyze_domain_invariance_tsne.py, retargeted
at the fine15 pipeline (MaskDecompositionModel, gradient-threshold mc/md ->
Gram-Schmidt orthogonalized z_inv) so its domain/class linear-probe accuracy can
be compared side-by-side against the GatedMaskModel (coarse5 gated_nodg) numbers
from Experiment 1.

IMPORTANT SUBSTITUTION (stated explicitly, per task instructions): fine15 has only
ONE checkpoint per fold (no 3-seed training ensemble like coarse5's gated_nodg).
To still report a mean+/-std instead of a single number, we substitute THREE
different stratified train/test splits (different `random_state`) of the SAME
extracted feature set as the "3 repeats". This is NOT seed-ensemble variance
(there is no re-training or re-encoding happening between repeats) -- it only
captures split-sampling noise in the linear probe itself, on a single fixed
feature extractor. Do not treat the resulting std as comparable in kind to the
train-seed std reported for the gated model.

z      = raw backbone feature, pre-decomposition: GAP(out["z"])
z_inv  = post-decomposition feature: GAP(out["z_c_notd"]) (= z * mc_orth, the
         Gram-Schmidt-orthogonalized gradient-threshold mask actually used
         downstream by eval_all_folds.py / eval_dc_folds.py / eval_patch_folds.py)

Domain pool per fold = test_domains U calib_domains from eval_all_folds.FOLDS_FINE,
which is all 15 fine domains for every fold (chance = 1/15 = 0.0667).

NO retraining. Existing checkpoints/fine15_fold{500,600,700,800}.pth only.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from datasets.hust_image import HUSTDataset
from models.mask_decomposition_model import MaskDecompositionModel
from experiments.eval_all_folds import transform, device, FINE_ALL, FOLDS_FINE

FINE_ROOT = "processed_gadf_fine_4096"
MAX_PER_CLASS_PER_DOMAIN = 200
SUBSAMPLE_SEED = 42
SPLIT_SEEDS = [42, 43, 44]  # 3 different splits, substituting for 3 seeds (see module docstring)
TEST_SIZE = 0.2

FINE15_CKPTS = {
    "500": "checkpoints/fine15_fold500.pth",
    "600": "checkpoints/fine15_fold600.pth",
    "700": "checkpoints/fine15_fold700.pth",
    "800": "checkpoints/fine15_fold800.pth",
}


def load_fine15_model(ckpt):
    sd = torch.load(ckpt, map_location=device)
    num_domains = sd["domain_classifier.fc.weight"].shape[0] if "domain_classifier.fc.weight" in sd else len(FINE_ALL)
    model = MaskDecompositionModel(num_classes=2, num_domains=num_domains, encoder_layer="layer4").to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def build_domain_pool_dataset(domains, seed=SUBSAMPLE_SEED, max_per_class=MAX_PER_CLASS_PER_DOMAIN):
    per_domain = []
    for d in domains:
        ds = HUSTDataset(root=FINE_ROOT, domain=d, only_normal=False, transform=transform, all_domains=FINE_ALL)
        normal = [s for s in ds.samples if s[1] == 0]
        anomaly = [s for s in ds.samples if s[1] == 1]
        rng = random.Random(seed)
        rng.shuffle(normal)
        rng.shuffle(anomaly)
        ds.samples = normal[:max_per_class] + anomaly[:max_per_class]
        per_domain.append(ds)
    return ConcatDataset(per_domain)


def extract_features(model, loader):
    """No @torch.no_grad(): MaskDecompositionModel.forward internally uses
    torch.enable_grad() + autograd.grad to compute its gradient-threshold mc/md
    masks (same pattern as eval_all_folds.py's own get_features, which also
    doesn't wrap in no_grad for this reason). Outputs are .detach()'d before
    converting to numpy."""
    z_list, zinv_list, labels, domains = [], [], [], []
    for batch in loader:
        img = batch["image"].to(device)
        out = model(img)
        z = F.adaptive_avg_pool2d(out["z"], 1).flatten(1)
        zinv = F.adaptive_avg_pool2d(out["z_c_notd"], 1).flatten(1)
        z_list.append(z.detach().cpu().numpy())
        zinv_list.append(zinv.detach().cpu().numpy())
        labels.extend(batch["label"].numpy().tolist())
        domains.extend(list(batch["domain_name"]))
    return (np.concatenate(z_list, axis=0), np.concatenate(zinv_list, axis=0),
            np.array(labels), np.array(domains))


def linear_probe_accuracy(features, targets, strat_key, split_seed):
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=TEST_SIZE, random_state=split_seed, stratify=strat_key
    )
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return accuracy_score(y_test, preds)


def run_fold(fold, test_domains, calib_domains, ckpt):
    domain_pool = sorted(set(test_domains) | set(calib_domains))
    n_domains = len(domain_pool)
    chance = 1.0 / n_domains
    print(f"\n=== fold {fold} (fine15) | domain pool = {domain_pool} (n={n_domains}, chance={chance:.4f}) ===")

    model = load_fine15_model(ckpt)
    ds = build_domain_pool_dataset(domain_pool)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    z_feats, zinv_feats, labels, domains = extract_features(model, loader)
    print(f"  extracted {z_feats.shape[0]} samples, z_dim={z_feats.shape[1]}, zinv_dim={zinv_feats.shape[1]}")

    strat_key = np.array([f"{d}_{l}" for d, l in zip(domains, labels)])

    split_results = {"z_domain": [], "zinv_domain": [], "z_class": [], "zinv_class": []}
    for split_seed in SPLIT_SEEDS:
        z_dom = linear_probe_accuracy(z_feats, domains, strat_key, split_seed)
        zinv_dom = linear_probe_accuracy(zinv_feats, domains, strat_key, split_seed)
        z_cls = linear_probe_accuracy(z_feats, labels, strat_key, split_seed)
        zinv_cls = linear_probe_accuracy(zinv_feats, labels, strat_key, split_seed)
        split_results["z_domain"].append(z_dom)
        split_results["zinv_domain"].append(zinv_dom)
        split_results["z_class"].append(z_cls)
        split_results["zinv_class"].append(zinv_cls)
        print(f"    [split_seed={split_seed}] z dom-acc={z_dom:.4f} | zinv dom-acc={zinv_dom:.4f} "
              f"| z cls-acc={z_cls:.4f} | zinv cls-acc={zinv_cls:.4f}")

    summary = {k: (float(np.mean(v)), float(np.std(v))) for k, v in split_results.items()}
    summary["chance"] = chance
    summary["n_domains"] = n_domains
    return summary


def main():
    print(f"Device  : {device}")
    print(f"Root    : {FINE_ROOT}")
    print(f"Split seeds (substituting for 3 seeds; SAME checkpoint, only the train/test split changes): {SPLIT_SEEDS}")

    # Build fold list directly from FOLDS_FINE + our own checkpoint map, ignoring the
    # placeholder ckpt names embedded in FOLDS_FINE (those are eval_all_folds.py's own
    # CLI-override defaults, not real paths).
    fold_configs = []
    for _placeholder_ckpt, test_domains, calib_domains, _all_domains in FOLDS_FINE:
        fold_key = test_domains[0][:3]  # "500" from "500","502","504", etc.
        fold_configs.append((fold_key, test_domains, calib_domains))

    all_summaries = {}
    for fold, test_domains, calib_domains in fold_configs:
        ckpt = FINE15_CKPTS[fold]
        summary = run_fold(fold, test_domains, calib_domains, ckpt)
        all_summaries[fold] = summary

    print("\n" + "=" * 100)
    print("SUMMARY TABLE (fine15, mean+/-std over 3 DIFFERENT TRAIN/TEST SPLITS of the SAME checkpoint's features")
    print("-- NOT seed-ensemble variance; see module docstring)")
    print("=" * 100)
    header = (f"{'Fold':>6} | {'z dom-acc':>16} | {'zinv dom-acc':>16} | "
              f"{'z cls-acc':>16} | {'zinv cls-acc':>16} | {'chance':>8}")
    print(header)
    print("-" * len(header))
    agg = {k: [] for k in ("z_domain", "zinv_domain", "z_class", "zinv_class")}
    for fold, s in all_summaries.items():
        print(f"{fold:>6} | {s['z_domain'][0]:.4f}+/-{s['z_domain'][1]:.4f} "
              f"| {s['zinv_domain'][0]:.4f}+/-{s['zinv_domain'][1]:.4f} "
              f"| {s['z_class'][0]:.4f}+/-{s['z_class'][1]:.4f} "
              f"| {s['zinv_class'][0]:.4f}+/-{s['zinv_class'][1]:.4f} "
              f"| {s['chance']:.4f}")
        for k in agg:
            agg[k].append(s[k][0])
    print("-" * len(header))
    print(f"{'Avg':>6} | {np.mean(agg['z_domain']):>16.4f} | {np.mean(agg['zinv_domain']):>16.4f} "
          f"| {np.mean(agg['z_class']):>16.4f} | {np.mean(agg['zinv_class']):>16.4f} | {1/15:>8.4f}")


if __name__ == "__main__":
    main()
