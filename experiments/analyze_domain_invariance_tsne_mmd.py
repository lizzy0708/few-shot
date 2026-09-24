"""
analyze_domain_invariance_tsne_mmd.py — thin wrapper around
analyze_domain_invariance_tsne.py, retargeted at the MMD domain-alignment checkpoints
(`checkpoints/coarse5_fold*_gated_nodg_mmd_s{0,1,2}.pth`,
docs/exec-plans/active/2026-09-gated-nodg-mmd.md).

Does NOT modify experiments/analyze_domain_invariance_tsne.py. That script does
`from experiments.eval_gated_folds import ckpt_path, ...`, which binds `ckpt_path` as
its OWN module-level name (not a live reference into eval_gated_folds) -- so this
wrapper monkey-patches `probe.ckpt_path` directly (not `eval_gated_folds.ckpt_path`,
which the probe module would not see). Also overrides `probe.TRAIN_SEEDS` to whichever
seeds actually have a trained mmd checkpoint for every fold (auto-detected from disk),
so this works unchanged whether 1 seed or all 3 have been trained so far.

Usage:
  conda run -n torch python experiments/analyze_domain_invariance_tsne_mmd.py --no_tsne
  conda run -n torch python experiments/analyze_domain_invariance_tsne_mmd.py --test_fold 700 --no_tsne
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import experiments.analyze_domain_invariance_tsne as probe
from experiments.eval_gated_folds import GATED_FOLDS


def ckpt_path_mmd(fold, train_seed):
    return f"checkpoints/coarse5_fold{fold}_gated_nodg_mmd_s{train_seed}.pth"


def _detect_available_seeds():
    """Seeds with an mmd checkpoint present for EVERY fold (so run_fold's per-seed loop
    never hits a missing file); checked against candidate seeds 0/1/2."""
    folds = [f[0] for f in GATED_FOLDS]
    available = []
    for seed in (0, 1, 2):
        if all(os.path.exists(ckpt_path_mmd(fold, seed)) for fold in folds):
            available.append(seed)
    return available


probe.ckpt_path = ckpt_path_mmd
probe.TRAIN_SEEDS = _detect_available_seeds()

if __name__ == "__main__":
    print(f"[eval_gated_folds_mmd wrapper] Using mmd checkpoints, TRAIN_SEEDS={probe.TRAIN_SEEDS}")
    if not probe.TRAIN_SEEDS:
        raise SystemExit("No fold has a complete set of mmd checkpoints for any seed yet -- train first.")
    probe.main()
