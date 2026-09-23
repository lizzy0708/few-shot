"""
tune_nsigma_rawz.py — fair-comparison follow-up to Experiment 3.

Experiment 3 (eval_gated_folds_rawz.py) reused z_inv's n_sigma=2.0 threshold for raw z's
Mahalanobis scores. If raw z and z_inv live on different scales, that's an unfair
comparison: raw z's Acc/F1 deficit could be a badly-calibrated threshold, not a real
decomposition effect. This script finds each fold's OWN best n_sigma for raw z via
leave-one-calib-domain-out (LOCO) tuning, using only calib-domain labels -- the
held-out test domain's label is never touched during tuning.

Why LOCO-on-calib-domains is leakage-free: for GATED_FOLDS row (fold, test_domain,
calib_domains), calib_domains are 4 OTHER labeled domains in this dataset (they just
aren't the fold's outer test domain). Treating each one as a pseudo-test domain while
scoring it against the *other* calib domains' pooled normal set is textbook nested
cross-validation -- it never reads test_domain's labels. This mirrors the ORIGINAL
z_inv pipeline's own n_sigma selection principle (label-free w.r.t. the outer test
domain): eval_gated_folds.py's n_sigma=2.0 is a fixed hyperparameter never touching
test-domain labels either; here we replace "fixed" with "LOCO-selected" but keep the
same non-negotiable constraint.

Does NOT retrain, does NOT modify eval_gated_folds.py / eval_gated_folds_rawz.py --
imports and reuses run_fold()/build_support_query()/build_calib_normal()/load_gated_model()
unchanged. Importing eval_gated_folds_rawz applies its score_one_model monkey-patch
(z -> raw feature_key) globally in this process, so every score computed here (tuning
AND final re-eval) is on raw z, exactly matching what Experiment 3 measured.

Usage:
  conda run -n torch python experiments/tune_nsigma_rawz.py --beta 0.5
  conda run -n torch python experiments/tune_nsigma_rawz.py --beta 0.5 --tune_eval_seeds 0 1 2
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import experiments.eval_gated_folds as base
import experiments.eval_gated_folds_rawz as rawz  # noqa: F401  (applies score_one_model -> raw z patch)

N_SIGMA_CANDIDATES = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
TUNE_EVAL_SEEDS = [0, 1, 2]  # subset of base.EVAL_SEEDS -- LOCO tuning cost control


def loco_tune_fold(fold, calib_domains, beta, tune_eval_seeds, train_seeds, verbose=True):
    """LOCO n_sigma search for one GATED_FOLDS row. Only ever reads calib_domains'
    own labels (each used once as a pseudo-test domain, scored against the OTHER
    calib domains as its calib set) -- the fold's real test_domain label is never
    read in this function."""
    models = {s: base.load_gated_model(base.ckpt_path(fold, s)) for s in train_seeds}

    # candidate n_sigma -> list of (acc, f1) across (loco_domain, tune_eval_seed)
    per_candidate = {ns: [] for ns in N_SIGMA_CANDIDATES}

    for d in calib_domains:
        other_calib = [c for c in calib_domains if c != d]
        calib_normal_ds = base.build_calib_normal(other_calib)

        for eval_seed in tune_eval_seeds:
            support_ds, query_ds = base.build_support_query(d, eval_seed)

            per_model_query, per_model_support, per_model_calib, ref_labels = [], [], [], None
            for s in train_seeds:
                # n_sigma arg is a required positional but unused inside score_one_model
                # (threshold is computed by the caller, not inside it) -- 0.0 placeholder.
                q, sc, c, ql = base.score_one_model(models[s], support_ds, calib_normal_ds, query_ds, beta, 0.0)
                if ref_labels is None:
                    ref_labels = ql
                per_model_query.append(q)
                per_model_support.append(sc)
                per_model_calib.append(c)

            ens_query = np.mean(np.stack(per_model_query, axis=0), axis=0)
            ens_support = np.mean(np.stack(per_model_support, axis=0), axis=0)
            ens_calib = np.mean(np.stack(per_model_calib, axis=0), axis=0)

            for ns in N_SIGMA_CANDIDATES:
                threshold = ens_support.mean() + ns * ens_calib.std()
                preds = (ens_query > threshold).astype(int)
                acc = accuracy_score(ref_labels, preds)
                f1 = f1_score(ref_labels, preds, zero_division=0)
                per_candidate[ns].append((acc, f1))

            if verbose:
                print(f"    [loco] fold={fold} loco_domain={d} eval_seed={eval_seed} "
                      f"scored ({len(ref_labels)} query samples, {int(ref_labels.sum())} anomaly)")

    summary = {}
    for ns, vals in per_candidate.items():
        accs = [a for a, _ in vals]
        f1s = [f for _, f in vals]
        summary[ns] = {
            "acc_mean": float(np.mean(accs)),
            "f1_mean": float(np.mean(f1s)),
            "combo_mean": float((np.mean(accs) + np.mean(f1s)) / 2.0),
        }
    best_ns = max(N_SIGMA_CANDIDATES, key=lambda ns: summary[ns]["combo_mean"])
    return best_ns, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=0.5, help="proto_beta (same as eval_gated_folds.py)")
    ap.add_argument("--tune_eval_seeds", type=int, nargs="+", default=TUNE_EVAL_SEEDS,
                     help="eval seeds used for LOCO n_sigma tuning (subset of base.EVAL_SEEDS)")
    ap.add_argument("--eval_seeds", type=int, nargs="+", default=base.EVAL_SEEDS,
                     help="eval seeds for the FINAL re-evaluation with tuned n_sigma")
    ap.add_argument("--train_seeds", type=int, nargs="+", default=base.TRAIN_SEEDS)
    ap.add_argument("--out_json", type=str, default="docs/generated/exp3_rawz/tuned_nsigma.json")
    args = ap.parse_args()

    print(f"Device            : {base.device}")
    print(f"beta              : {args.beta}")
    print(f"n_sigma candidates: {N_SIGMA_CANDIDATES}")
    print(f"tune_eval_seeds   : {args.tune_eval_seeds}")
    print(f"final eval_seeds  : {args.eval_seeds}")
    print()

    tuned = {}
    all_summaries = {}
    for fold, test_domains, calib_domains in base.GATED_FOLDS:
        print(f"=== LOCO tuning fold={fold} (test_domain={test_domains[0]} label NOT read here; "
              f"calib_domains={calib_domains}) ===")
        best_ns, summary = loco_tune_fold(fold, calib_domains, args.beta,
                                           args.tune_eval_seeds, args.train_seeds)
        tuned[fold] = best_ns
        all_summaries[fold] = summary
        for ns in N_SIGMA_CANDIDATES:
            s = summary[ns]
            marker = "  <== selected" if ns == best_ns else ""
            print(f"     n_sigma={ns:+.1f} | Acc={s['acc_mean']:.4f} | F1={s['f1_mean']:.4f} "
                  f"| combo={s['combo_mean']:.4f}{marker}")
        print(f"  -> selected n_sigma={best_ns:+.1f} for fold={fold}\n")

    print("Tuned n_sigma per fold:", tuned)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({
            "tuned_n_sigma": tuned,
            "candidates": N_SIGMA_CANDIDATES,
            "tune_eval_seeds": args.tune_eval_seeds,
            "beta": args.beta,
            "loco_summary": all_summaries,
        }, f, indent=2)
    print(f"Wrote {args.out_json}")

    print("\n=== Final raw z evaluation with per-fold LOCO-tuned n_sigma "
          "(test-domain labels used ONLY here, exactly as in the original protocol) ===")
    header = f"{'Fold':>6} | {'n_sigma':>8} | {'AUROC':>16} | {'Acc':>16} | {'F1':>16}"
    print(header)
    print("-" * len(header))
    all_rows = {}
    for fold, test_domains, calib_domains in base.GATED_FOLDS:
        ns = tuned[fold]
        res = base.run_fold(fold, test_domains, calib_domains, args.beta, ns,
                             args.train_seeds, args.eval_seeds)
        all_rows[fold] = res
        print(f"{fold:>6} | {ns:>8.1f} | {res['auroc_mean']:.4f}±{res['auroc_std']:.4f} "
              f"| {res['acc_mean']:.4f}±{res['acc_std']:.4f} "
              f"| {res['f1_mean']:.4f}±{res['f1_std']:.4f}")
    print("-" * len(header))
    avg_auroc = np.mean([r["auroc_mean"] for r in all_rows.values()])
    avg_acc = np.mean([r["acc_mean"] for r in all_rows.values()])
    avg_f1 = np.mean([r["f1_mean"] for r in all_rows.values()])
    print(f"{'Avg':>6} | {'':>8} | {avg_auroc:>16.4f} | {avg_acc:>16.4f} | {avg_f1:>16.4f}")


if __name__ == "__main__":
    main()
