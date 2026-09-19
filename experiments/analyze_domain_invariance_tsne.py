"""
analyze_domain_invariance_tsne.py — quantitative + visual evidence for domain-invariance
of the coarse5 gated-model (GatedMaskModel, `gated_nodg` checkpoints) — "Experiment 1 of 4"
in a sequence, 2026-09-19.

NO retraining: only extracts features from the existing
checkpoints/coarse5_fold{500,600,700,800}_gated_nodg_s{0,1,2}.pth checkpoints.

Compares:
  z      = raw backbone feature, pre-decomposition (`out["z_pool"]`, i.e. GAP(z) BEFORE the
           class_gate multiply — the same raw pooled feature eval_all_folds.py's baseline
           path and this project's other z_pool usages already mean by "raw").
  z_inv  = post-decomposition feature actually scored in eval_gated_folds.py
           (`z_pool * class_gate`, i.e. GAP(out["z_c_notd"])).

SCOPE DECISION (stated explicitly per the task): "각 fold의 test 도메인에서" is read as
option (a) — per fold, pool ALL domains involved in that fold's eval config (calib domains
+ the held-out test domain). For this repo's coarse5 GATED_FOLDS structure (see
experiments/eval_gated_folds.py), calib_domains ∪ test_domains == {400,500,600,700,800} for
EVERY fold (the held-out domain is simply swapped in for one of the four calib slots), so in
practice every fold's domain pool is the same 5 domains — chance-level domain accuracy is
1/5 = 0.20 for all four folds. This is computed dynamically from each fold's own
(test_domains, calib_domains) tuple rather than hardcoded, in case that assumption is ever
violated by a future fold config.

LINEAR PROBE DESIGN DECISIONS (stated explicitly, not silently assumed):
  - Per-seed-then-average (NOT pooling raw features across the 3 training-seed checkpoints
    into one probe). The 3 seeds are independently-trained encoders whose feature spaces are
    not aligned/comparable dimension-for-dimension; pooling their rows into one classification
    problem would let a probe partly separate by "which seed's coordinate system" rather than
    testing genuine domain/class separability within a single feature space, and would violate
    the i.i.d. assumption a train/test split relies on. Instead: fit one fresh probe per
    (fold, train_seed), report mean±std of test accuracy across the 3 seeds.
  - Fresh `sklearn.linear_model.LogisticRegression` (StandardScaler + LogisticRegression
    pipeline, max_iter=2000) — NOT the model's own GRL-trained `domain_classifier` head. This
    tests true linear separability independent of the adversarial training pressure, unlike
    the project's earlier in-sample `domain_classifier_disc` probe
    (docs/exec-plans/completed/2026-08-domain-gate-weak-grl-sweep.md).
  - Held-out split: stratified 80/20 train_test_split, stratified jointly on (domain, label)
    so both the domain probe and the class probe see a representative split; random_state=42.

t-SNE: sklearn.manifold.TSNE, perplexity=30, learning_rate="auto", init="pca",
random_state=42 — matching this repo's existing convention in
experiments/visualize_tsne_mask.py. One embedding per feature type (z, z_inv) per fold,
reused for both the domain-colored and class-colored plot of that feature.

Sampling for tractability: per domain, up to MAX_PER_CLASS (200) normal + 200 anomaly
samples (deterministic RNG, seed=42) => up to 2000 samples per (fold, seed) extraction.
Same sample set is used for both the t-SNE plot (seed-0 only) and the linear probes
(all 3 seeds) so numbers reported are self-consistent.

Usage:
  conda run -n torch python experiments/analyze_domain_invariance_tsne.py
  conda run -n torch python experiments/analyze_domain_invariance_tsne.py --test_fold 700
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets.hust_image import HUSTDataset
from experiments.eval_all_folds import transform, device
from experiments.eval_gated_folds import GATED_FOLDS, ROOT, ALL_DOMAINS, ckpt_path, load_gated_model

OUT_DIR = "out/analyze/domain_invariance_tsne"
TRAIN_SEEDS = [0, 1, 2]
TSNE_SEED_FOR_PLOTS = 0  # "representative" seed for t-SNE plots (per task instructions)
SUBSAMPLE_SEED = 42
MAX_PER_CLASS_PER_DOMAIN = 200  # up to 200 normal + 200 anomaly per domain
SPLIT_SEED = 42
TEST_SIZE = 0.2


def build_domain_pool_dataset(domains, seed=SUBSAMPLE_SEED, max_per_class=MAX_PER_CLASS_PER_DOMAIN):
    """Balanced (normal+anomaly), deterministic, per-domain-capped ConcatDataset."""
    per_domain = []
    for d in domains:
        ds = HUSTDataset(root=ROOT, domain=d, only_normal=False, transform=transform, all_domains=ALL_DOMAINS)
        normal = [s for s in ds.samples if s[1] == 0]
        anomaly = [s for s in ds.samples if s[1] == 1]
        rng = random.Random(seed)
        rng.shuffle(normal)
        rng.shuffle(anomaly)
        ds.samples = normal[:max_per_class] + anomaly[:max_per_class]
        per_domain.append(ds)
    return ConcatDataset(per_domain)


@torch.no_grad()
def extract_features(model, loader):
    z_list, zinv_list, labels, domains = [], [], [], []
    for batch in loader:
        img = batch["image"].to(device)
        out = model(img)
        z = out["z_pool"]                                          # [B, C], pre-gate (raw)
        zinv = F.adaptive_avg_pool2d(out["z_c_notd"], 1).flatten(1)  # [B, C], post-gate (z_inv)
        z_list.append(z.cpu().numpy())
        zinv_list.append(zinv.cpu().numpy())
        labels.extend(batch["label"].numpy().tolist())
        domains.extend(list(batch["domain_name"]))
    return (np.concatenate(z_list, axis=0), np.concatenate(zinv_list, axis=0),
            np.array(labels), np.array(domains))


def linear_probe_accuracy(features, targets, strat_key):
    """Fresh StandardScaler+LogisticRegression probe, stratified 80/20 split. Returns test acc."""
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=strat_key
    )
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return accuracy_score(y_test, preds)


def plot_tsne(emb, color_values, title, save_path, legend_title):
    plt.figure(figsize=(8, 7))
    unique_values = sorted(set(color_values.tolist()))
    for v in unique_values:
        idx = color_values == v
        plt.scatter(emb[idx, 0], emb[idx, 1], s=14, alpha=0.75, label=str(v))
    plt.title(title)
    plt.legend(title=legend_title, markerscale=1.5)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  [saved] {save_path}")


def run_fold(fold, test_domains, calib_domains, make_tsne_plots):
    domain_pool = sorted(set(test_domains) | set(calib_domains))
    n_domains = len(domain_pool)
    chance = 1.0 / n_domains
    print(f"\n=== fold {fold} | domain pool = {domain_pool} (n={n_domains}, chance={chance:.4f}) ===")

    ds = build_domain_pool_dataset(domain_pool)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    seed_results = {"z_domain": [], "zinv_domain": [], "z_class": [], "zinv_class": []}

    for train_seed in TRAIN_SEEDS:
        model = load_gated_model(ckpt_path(fold, train_seed))
        z_feats, zinv_feats, labels, domains = extract_features(model, loader)
        print(f"  train_seed={train_seed}: extracted {z_feats.shape[0]} samples, "
              f"z_dim={z_feats.shape[1]}, zinv_dim={zinv_feats.shape[1]}")

        strat_key = np.array([f"{d}_{l}" for d, l in zip(domains, labels)])

        z_dom_acc = linear_probe_accuracy(z_feats, domains, strat_key)
        zinv_dom_acc = linear_probe_accuracy(zinv_feats, domains, strat_key)
        z_cls_acc = linear_probe_accuracy(z_feats, labels, strat_key)
        zinv_cls_acc = linear_probe_accuracy(zinv_feats, labels, strat_key)

        seed_results["z_domain"].append(z_dom_acc)
        seed_results["zinv_domain"].append(zinv_dom_acc)
        seed_results["z_class"].append(z_cls_acc)
        seed_results["zinv_class"].append(zinv_cls_acc)

        print(f"    [probe] z    domain-acc={z_dom_acc:.4f} | z_inv domain-acc={zinv_dom_acc:.4f}"
              f" | z    class-acc={z_cls_acc:.4f} | z_inv class-acc={zinv_cls_acc:.4f}")

        if make_tsne_plots and train_seed == TSNE_SEED_FOR_PLOTS:
            label_names = np.array(["normal" if l == 0 else "anomaly" for l in labels])

            print(f"  [t-SNE] fitting on z (train_seed={train_seed}) ...")
            tsne_z = TSNE(n_components=2, perplexity=30, learning_rate="auto",
                          init="pca", random_state=42)
            emb_z = tsne_z.fit_transform(z_feats)
            plot_tsne(emb_z, domains, f"fold {fold}: z (raw) by domain",
                      f"{OUT_DIR}/fold{fold}_z_by_domain.png", "Domain")
            plot_tsne(emb_z, label_names, f"fold {fold}: z (raw) by class",
                      f"{OUT_DIR}/fold{fold}_z_by_class.png", "Class")

            print(f"  [t-SNE] fitting on z_inv (train_seed={train_seed}) ...")
            tsne_zinv = TSNE(n_components=2, perplexity=30, learning_rate="auto",
                             init="pca", random_state=42)
            emb_zinv = tsne_zinv.fit_transform(zinv_feats)
            plot_tsne(emb_zinv, domains, f"fold {fold}: z_inv (decomposed) by domain",
                      f"{OUT_DIR}/fold{fold}_zinv_by_domain.png", "Domain")
            plot_tsne(emb_zinv, label_names, f"fold {fold}: z_inv (decomposed) by class",
                      f"{OUT_DIR}/fold{fold}_zinv_by_class.png", "Class")

    summary = {k: (float(np.mean(v)), float(np.std(v))) for k, v in seed_results.items()}
    summary["chance"] = chance
    summary["n_domains"] = n_domains
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_fold", type=str, default=None, choices=[f[0] for f in GATED_FOLDS])
    ap.add_argument("--no_tsne", action="store_true", help="Skip t-SNE plots, probes only.")
    args = ap.parse_args()

    folds = GATED_FOLDS if args.test_fold is None else [f for f in GATED_FOLDS if f[0] == args.test_fold]

    print(f"Device  : {device}")
    print(f"Out dir : {OUT_DIR}")
    print(f"Train seeds (probe avg): {TRAIN_SEEDS}")
    print(f"t-SNE representative seed: {TSNE_SEED_FOR_PLOTS}")

    all_summaries = {}
    for fold, test_domains, calib_domains in folds:
        summary = run_fold(fold, test_domains, calib_domains, make_tsne_plots=not args.no_tsne)
        all_summaries[fold] = summary

    print("\n" + "=" * 100)
    print("SUMMARY TABLE (mean±std over 3 train-seed probes)")
    print("=" * 100)
    header = (f"{'Fold':>6} | {'z dom-acc':>14} | {'zinv dom-acc':>14} | "
              f"{'z cls-acc':>14} | {'zinv cls-acc':>14} | {'chance':>8}")
    print(header)
    print("-" * len(header))
    agg = {k: [] for k in ("z_domain", "zinv_domain", "z_class", "zinv_class")}
    for fold, s in all_summaries.items():
        print(f"{fold:>6} | {s['z_domain'][0]:.4f}±{s['z_domain'][1]:.4f} "
              f"| {s['zinv_domain'][0]:.4f}±{s['zinv_domain'][1]:.4f} "
              f"| {s['z_class'][0]:.4f}±{s['z_class'][1]:.4f} "
              f"| {s['zinv_class'][0]:.4f}±{s['zinv_class'][1]:.4f} "
              f"| {s['chance']:.4f}")
        for k in agg:
            agg[k].append(s[k][0])
    if len(all_summaries) == 4:
        print("-" * len(header))
        print(f"{'Avg':>6} | {np.mean(agg['z_domain']):>14.4f} | {np.mean(agg['zinv_domain']):>14.4f} "
              f"| {np.mean(agg['z_class']):>14.4f} | {np.mean(agg['zinv_class']):>14.4f} | {0.2:>8.4f}")


if __name__ == "__main__":
    main()
