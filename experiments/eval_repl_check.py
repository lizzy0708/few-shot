"""
eval_repl_check.py — 2026-10-08. Does NOT modify eval_gated_folds.py. New file only.
Reproducibility-gate check for the wrap-around-correction retraining request:
evaluates a single checkpoint tag (default 'repl') with the SAME protocol as
eval_gated_folds.py (reused unchanged via import), additionally computing
Balanced Accuracy (not printed by the base script) from the same predictions.

Usage:
  conda run -n torch python experiments/eval_repl_check.py --ckpt_tag repl --test_fold 500 --train_seeds 1
"""
import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score

from experiments.eval_all_folds import fit_normal_distribution, mahalanobis_score
import experiments.eval_gated_folds as base


def main():
    pre_ap = argparse.ArgumentParser(add_help=False)
    pre_ap.add_argument("--ckpt_tag", type=str, default=None,
                         help="if set, checkpoints/coarse5_fold*_gated_nodg_<tag>_s*.pth; "
                              "if omitted, uses eval_gated_folds.py's default path")
    pre_args, remaining = pre_ap.parse_known_args()
    if pre_args.ckpt_tag:
        tag = pre_args.ckpt_tag

        def ckpt_path_fn(fold, seed):
            return f"checkpoints/coarse5_fold{fold}_gated_nodg_{tag}_s{seed}.pth"
        base.ckpt_path = ckpt_path_fn

    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--n_sigma", type=float, default=2.0)
    ap.add_argument("--train_seeds", type=int, nargs="+", default=base.TRAIN_SEEDS)
    ap.add_argument("--eval_seeds", type=int, nargs="+", default=base.EVAL_SEEDS)
    ap.add_argument("--test_fold", type=str, default=None, choices=[f[0] for f in base.GATED_FOLDS])
    args = ap.parse_args(remaining)

    folds = base.GATED_FOLDS if args.test_fold is None else [f for f in base.GATED_FOLDS if f[0] == args.test_fold]

    for fold, test_domains, calib_domains in folds:
        test_domain = test_domains[0]
        models = {s: base.load_gated_model(base.ckpt_path(fold, s)) for s in args.train_seeds}
        calib_normal_ds = base.build_calib_normal(calib_domains)

        aurocs, accs, f1s, bals = [], [], [], []
        for eval_seed in args.eval_seeds:
            support_ds, query_ds = base.build_support_query(test_domain, eval_seed)
            per_q, per_s, per_c, labels = [], [], [], None
            for s in args.train_seeds:
                q, sc, c, lab = base.score_one_model(models[s], support_ds, calib_normal_ds, query_ds, args.beta, args.n_sigma)
                if labels is None:
                    labels = lab
                per_q.append(q); per_s.append(sc); per_c.append(c)
            ens_q = np.mean(np.stack(per_q), axis=0)
            ens_s = np.mean(np.stack(per_s), axis=0)
            ens_c = np.mean(np.stack(per_c), axis=0)
            threshold = ens_s.mean() + args.n_sigma * ens_c.std()
            preds = (ens_q > threshold).astype(int)
            auroc = roc_auc_score(labels, ens_q)
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, zero_division=0)
            bal = balanced_accuracy_score(labels, preds)
            aurocs.append(auroc); accs.append(acc); f1s.append(f1); bals.append(bal)
            print(f"  fold={fold} eval_seed={eval_seed} AUROC={auroc:.4f} Acc={acc:.4f} F1={f1:.4f} BalAcc={bal:.4f}")

        print(f"SUMMARY fold={fold} tag={pre_args.ckpt_tag} | AUROC={np.mean(aurocs):.4f}±{np.std(aurocs):.4f} "
              f"Acc={np.mean(accs):.4f}±{np.std(accs):.4f} F1={np.mean(f1s):.4f}±{np.std(f1s):.4f} "
              f"BalAcc={np.mean(bals):.4f}±{np.std(bals):.4f}")

    print("DONE")


if __name__ == "__main__":
    main()
