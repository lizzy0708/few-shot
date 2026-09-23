"""Paired (fold, eval_seed) comparison: z_inv(n_sigma=2.0) vs raw z(n_sigma=2.0, Exp.3)
vs raw z(LOCO-tuned n_sigma per fold, Exp.3b -- fair-threshold check)."""
import re, numpy as np
from scipy import stats

pat = re.compile(r"fold=(\d+) eval_seed=(\d) beta.*AUROC=([\d.]+) Acc=([\d.]+) F1=([\d.]+)")
def load(p):
    d = {}
    for line in open(p):
        m = pat.search(line)
        if m:
            d[(m.group(1), int(m.group(2)))] = tuple(map(float, m.group(3, 4, 5)))
    return d

zi = load("docs/generated/exp3_rawz/zinv_baseline_eval.txt")
rw = load("docs/generated/exp3_rawz/rawz_eval.txt")               # raw z, n_sigma=2.0 (reused, Exp.3)
rt = load("docs/generated/exp3_rawz/rawz_tuned_eval.txt")         # raw z, LOCO-tuned n_sigma (Exp.3b)
assert zi.keys() == rw.keys() == rt.keys() and len(zi) == 20
keys = sorted(zi)
names = ["AUROC", "Acc", "F1"]
folds = [k[0] for k in keys]

def paired_report(a, b, label_a, label_b):
    D = np.array([[a[k][i] - b[k][i] for i in range(3)] for k in keys])
    print(f"== Paired diff ({label_a} - {label_b}), n=20 ==")
    for i, n in enumerate(names):
        d = D[:, i]
        t, p = stats.ttest_1samp(d, 0)
        w = stats.wilcoxon(d).pvalue if np.any(d != 0) else float("nan")
        print(f"{n:6s} mean={d.mean():+.4f} sd={d.std(ddof=1):.4f} "
              f"wins({label_a}>{label_b})={int((d>0).sum())}/20 t-p={p:.2g} wilcoxon-p={w:.2g}")
    print(f"\n  per-fold mean diff ({label_a}-{label_b}), wins/5:")
    for f in ["500", "600", "700", "800"]:
        idx = [j for j, x in enumerate(folds) if x == f]
        print("   fold " + f + ": " + " | ".join(
            f"{n} {D[idx, i].mean():+.4f} ({int((D[idx, i] > 0).sum())}/5)" for i, n in enumerate(names)))
    print()
    return D

print("### (1) Original Exp.3: z_inv vs raw z (n_sigma=2.0 reused -- POTENTIALLY UNFAIR) ###\n")
paired_report(zi, rw, "z_inv", "raw(ns=2.0)")

print("### (2) Exp.3b: raw z (n_sigma=2.0 reused) vs raw z (LOCO-tuned n_sigma) -- did tuning help raw z itself? ###\n")
paired_report(rw, rt, "raw(ns=2.0)", "raw(tuned)")

print("### (3) Exp.3b key comparison: z_inv vs raw z (LOCO-tuned n_sigma) -- FAIR comparison ###\n")
paired_report(zi, rt, "z_inv", "raw(tuned)")

print("### Summary averages ###")
for label, d in [("z_inv (ns=2.0)", zi), ("raw z (ns=2.0, Exp.3)", rw), ("raw z (LOCO-tuned, Exp.3b)", rt)]:
    arr = np.array([d[k] for k in keys])
    print(f"{label:28s} AUROC={arr[:,0].mean():.4f}  Acc={arr[:,1].mean():.4f}  F1={arr[:,2].mean():.4f}")
