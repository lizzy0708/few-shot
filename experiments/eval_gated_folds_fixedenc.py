"""
eval_gated_folds_fixedenc.py — 2026-10-08. Does NOT modify eval_gated_folds.py.
New file only. Combines three independent monkey-patches (each already proven
individually in earlier wrapper scripts this project) so the wrap-around-fixed
image root (processed_fixed/) can be evaluated with any feature_key and either
the trained-checkpoint or the fixed-pretrained-encoder model, without touching
eval_gated_folds.py:
  --root <dir>        patches base.ROOT (module constant, not a CLI arg there)
  --ckpt_tag <tag>     patches base.ckpt_path (as in eval_gated_folds_mmd.py)
  --feature_key {z,z_c_notd}  patches base.score_one_model (as in eval_gated_folds_rawz.py)
  --model {trained,pretrained} patches base.load_gated_model + base.TRAIN_SEEDS=[0]
                        for the fixed untrained-ResNet50 baseline (as in
                        eval_pretrained_fewshot.py), when --model pretrained

Usage:
  conda run -n torch python experiments/eval_gated_folds_fixedenc.py --root processed_fixed --ckpt_tag fixedenc --feature_key z_c_notd
  conda run -n torch python experiments/eval_gated_folds_fixedenc.py --root processed_fixed --ckpt_tag fixedenc --feature_key z
  conda run -n torch python experiments/eval_gated_folds_fixedenc.py --root processed_fixed --model pretrained --feature_key z
"""
import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader

from experiments.eval_all_folds import get_features, mahalanobis_score, fit_normal_distribution
from experiments.eval_pretrained_fewshot import load_pretrained_model
import experiments.eval_gated_folds as base


def make_ckpt_path_fn(tag):
    def ckpt_path_fn(fold, seed):
        return f"checkpoints/coarse5_fold{fold}_gated_nodg_{tag}_s{seed}.pth"
    return ckpt_path_fn


def make_score_one_model(feature_key):
    def score_one_model_fk(model, support_ds, calib_normal_ds, query_ds, beta, n_sigma):
        support_loader = DataLoader(support_ds, batch_size=len(support_ds), shuffle=False)
        query_loader = DataLoader(query_ds, batch_size=32, shuffle=False)
        calib_loader = DataLoader(calib_normal_ds, batch_size=32, shuffle=False)
        support_feats, _ = get_features(model, support_loader, feature_key=feature_key)
        query_feats, query_labels = get_features(model, query_loader, feature_key=feature_key)
        calib_feats, _ = get_features(model, calib_loader, feature_key=feature_key)
        support_np, query_np, calib_np = (t.cpu().numpy() for t in (support_feats, query_feats, calib_feats))
        _, prec_np = fit_normal_distribution(calib_np)
        blended = beta * support_np.mean(axis=0) + (1.0 - beta) * calib_np.mean(axis=0)
        return (mahalanobis_score(query_np, blended, prec_np),
                mahalanobis_score(support_np, blended, prec_np),
                mahalanobis_score(calib_np, blended, prec_np),
                query_labels)
    return score_one_model_fk


if __name__ == "__main__":
    pre_ap = argparse.ArgumentParser(add_help=False)
    pre_ap.add_argument("--root", type=str, required=True)
    pre_ap.add_argument("--ckpt_tag", type=str, default=None)
    pre_ap.add_argument("--feature_key", type=str, default="z_c_notd", choices=["z", "z_c_notd"])
    pre_ap.add_argument("--model", type=str, default="trained", choices=["trained", "pretrained"])
    pre_args, remaining = pre_ap.parse_known_args()

    base.ROOT = pre_args.root
    base.score_one_model = make_score_one_model(pre_args.feature_key)

    if pre_args.model == "pretrained":
        base.load_gated_model = lambda path: load_pretrained_model(None)
        base.TRAIN_SEEDS = [0]
    elif pre_args.ckpt_tag:
        base.ckpt_path = make_ckpt_path_fn(pre_args.ckpt_tag)

    sys.argv = [sys.argv[0]] + remaining
    print(f"[eval_gated_folds_fixedenc] root={pre_args.root} ckpt_tag={pre_args.ckpt_tag} "
          f"feature_key={pre_args.feature_key} model={pre_args.model}")
    base.main()
