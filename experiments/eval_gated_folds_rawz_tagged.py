"""
eval_gated_folds_rawz_tagged.py — 2026-10-06. Does NOT modify eval_gated_folds.py or
eval_gated_folds_rawz.py. New file only. Combines eval_gated_folds_rawz.py's raw-z
(feature_key="z") scoring with eval_gated_folds_mmd.py's arbitrary-checkpoint-tag
mechanism (--ckpt_tag), neither of which alone covers this: eval_gated_folds_rawz.py
only targets the default `..._gated_nodg_s{seed}.pth` path (no tag support),
eval_gated_folds_mmd.py only swaps the checkpoint path but always scores z_inv
(default feature_key). Needed to raw-z-evaluate non-standard checkpoint families
such as the GRL-off ablation (checkpoints/coarse5_fold*_gated_nodg_nogrl_s0.pth,
docs/exec-plans/completed/2026-10-gated-nodg-nogrl-ablation.md) without retraining
or touching either existing script.

Usage:
  conda run -n torch python experiments/eval_gated_folds_rawz_tagged.py --ckpt_tag nogrl --train_seeds 0 --beta 0.5 --n_sigma 2.0
"""
import argparse
import sys

import experiments.eval_gated_folds_rawz  # noqa: F401 -- import alone applies its score_one_model monkey-patch
import experiments.eval_gated_folds as base


def make_ckpt_path_fn(tag):
    def ckpt_path_fn(fold, train_seed):
        return f"checkpoints/coarse5_fold{fold}_gated_nodg_{tag}_s{train_seed}.pth"
    return ckpt_path_fn


if __name__ == "__main__":
    pre_ap = argparse.ArgumentParser(add_help=False)
    pre_ap.add_argument("--ckpt_tag", type=str, required=True,
                         help="checkpoint suffix tag, e.g. 'nogrl' for "
                              "checkpoints/coarse5_fold*_gated_nodg_nogrl_s*.pth")
    pre_args, remaining_argv = pre_ap.parse_known_args()
    base.ckpt_path = make_ckpt_path_fn(pre_args.ckpt_tag)
    sys.argv = [sys.argv[0]] + remaining_argv
    print(f"[eval_gated_folds_rawz_tagged] ckpt_tag={pre_args.ckpt_tag} (raw z, feature_key='z')")
    base.main()
