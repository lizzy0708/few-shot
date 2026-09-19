"""
analyze_domain_invariance_tsne_dw.py — reruns Experiment 1's domain/class linear-probe
methodology against the domain_weight-sweep checkpoints
(`checkpoints/coarse5_fold*_gated_nodg_dw{2,5}_s{0,1,2}.pth`), for direct comparison
against the domain_weight=1.0 gated_nodg baseline numbers.

Does NOT modify experiments/analyze_domain_invariance_tsne.py or
experiments/eval_gated_folds.py. Reuses their pure helper functions and only
reimplements the per-fold driver loop with the dw-tagged checkpoint path + a plain
(use_gate_orth=False) GatedMaskModel loader.

Usage:
  conda run -n torch python experiments/analyze_domain_invariance_tsne_dw.py --dw_tag dw2
  conda run -n torch python experiments/analyze_domain_invariance_tsne_dw.py --dw_tag dw5
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.gated_mask_model import GatedMaskModel
from experiments.eval_gated_folds import GATED_FOLDS
from experiments.eval_all_folds import device
from experiments.analyze_domain_invariance_tsne import (
    build_domain_pool_dataset, extract_features, linear_probe_accuracy,
    TRAIN_SEEDS,
)


def load_gated_model_dw(path):
    sd = torch.load(path, map_location=device)
    num_domains = sd["domain_classifier.weight"].shape[0]
    in_dim = sd["classifier.weight"].shape[1]
    encoder_layer = "layer3" if in_dim == 1024 else "layer4"
    model = GatedMaskModel(
        num_classes=2, num_domains=num_domains,
        encoder_layer=encoder_layer, use_domain_gate=False, use_gate_orth=False,
    ).to(device)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, f"{path}: state_dict mismatch missing={missing} unexpected={unexpected}"
    model.eval()
    return model


def run_fold(fold, test_domains, calib_domains, dw_tag):
    domain_pool = sorted(set(test_domains) | set(calib_domains))
    n_domains = len(domain_pool)
    chance = 1.0 / n_domains
    print(f"\n=== fold {fold} ({dw_tag}) | domain pool = {domain_pool} (n={n_domains}, chance={chance:.4f}) ===")

    ds = build_domain_pool_dataset(domain_pool)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    seed_results = {"z_domain": [], "zinv_domain": [], "z_class": [], "zinv_class": []}

    for train_seed in TRAIN_SEEDS:
        ckpt = f"checkpoints/coarse5_fold{fold}_gated_nodg_{dw_tag}_s{train_seed}.pth"
        model = load_gated_model_dw(ckpt)
        z_feats, zinv_feats, labels, domains = extract_features(model, loader)
        print(f"  train_seed={train_seed}: extracted {z_feats.shape[0]} samples")

        strat_key = np.array([f"{d}_{l}" for d, l in zip(domains, labels)])

        z_dom_acc = linear_probe_accuracy(z_feats, domains, strat_key)
        zinv_dom_acc = linear_probe_accuracy(zinv_feats, domains, strat_key)
        z_cls_acc = linear_probe_accuracy(z_feats, labels, strat_key)
        zinv_cls_acc = linear_probe_accuracy(zinv_feats, labels, strat_key)

        seed_results["z_domain"].append(z_dom_acc)
        seed_results["zinv_domain"].append(zinv_dom_acc)
        seed_results["z_class"].append(z_cls_acc)
        seed_results["zinv_class"].append(zinv_cls_acc)

        print(f"    [probe] z domain-acc={z_dom_acc:.4f} | z_inv domain-acc={zinv_dom_acc:.4f}"
              f" | z class-acc={z_cls_acc:.4f} | z_inv class-acc={zinv_cls_acc:.4f}")

    summary = {k: (float(np.mean(v)), float(np.std(v))) for k, v in seed_results.items()}
    summary["chance"] = chance
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dw_tag", type=str, required=True, choices=["dw2", "dw5"])
    args = ap.parse_args()

    print(f"Device  : {device}")
    print(f"dw_tag  : {args.dw_tag}")
    print(f"Train seeds (probe avg): {TRAIN_SEEDS}")

    all_summaries = {}
    for fold, test_domains, calib_domains in GATED_FOLDS:
        summary = run_fold(fold, test_domains, calib_domains, args.dw_tag)
        all_summaries[fold] = summary

    print("\n" + "=" * 100)
    print(f"SUMMARY TABLE ({args.dw_tag}, mean+/-std over 3 train-seed probes)")
    print("=" * 100)
    header = (f"{'Fold':>6} | {'z dom-acc':>14} | {'zinv dom-acc':>14} | "
              f"{'z cls-acc':>14} | {'zinv cls-acc':>14} | {'chance':>8}")
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
    if len(all_summaries) == 4:
        print("-" * len(header))
        print(f"{'Avg':>6} | {np.mean(agg['z_domain']):>14.4f} | {np.mean(agg['zinv_domain']):>14.4f} "
              f"| {np.mean(agg['z_class']):>14.4f} | {np.mean(agg['zinv_class']):>14.4f} | {0.2:>8.4f}")


if __name__ == "__main__":
    main()
