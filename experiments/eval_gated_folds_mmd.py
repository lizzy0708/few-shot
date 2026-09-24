"""
eval_gated_folds_mmd.py — thin wrapper around eval_gated_folds.py, retargeted at the
MMD domain-alignment checkpoints (`checkpoints/coarse5_fold*_gated_nodg_mmd_s{0,1,2}.pth`,
docs/exec-plans/active/2026-09-gated-nodg-mmd.md).

Does NOT modify experiments/eval_gated_folds.py. Monkey-patches only its module-level
`ckpt_path` to point at the mmd-tag checkpoint family -- architecture is identical to
plain nodg (mmd_weight only adds a training-time loss term, no new parameters), so
base.load_gated_model is reused unchanged. Everything else (beta/n_sigma/LedoitWolf/
leakage-fix/run_fold/main) is identical to the already-verified script.

Usage:
  conda run -n torch python experiments/eval_gated_folds_mmd.py --beta 0.5 --n_sigma 2.0
  conda run -n torch python experiments/eval_gated_folds_mmd.py --beta 0.5 --n_sigma 2.0 --train_seeds 0
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import experiments.eval_gated_folds as base


def make_ckpt_path_fn(tag):
    def ckpt_path_mmd(fold, train_seed):
        return f"checkpoints/coarse5_fold{fold}_gated_nodg_{tag}_s{train_seed}.pth"
    return ckpt_path_mmd


if __name__ == "__main__":
    import sys
    pre_ap = argparse.ArgumentParser(add_help=False)
    pre_ap.add_argument("--mmd_tag", type=str, default="mmd",
                         help="checkpoint suffix tag, e.g. 'mmd' (weight=10.0), 'mmd1' (weight=1.0), 'mmd01' (weight=0.1)")
    pre_args, remaining_argv = pre_ap.parse_known_args()
    base.ckpt_path = make_ckpt_path_fn(pre_args.mmd_tag)
    sys.argv = [sys.argv[0]] + remaining_argv
    base.main()
