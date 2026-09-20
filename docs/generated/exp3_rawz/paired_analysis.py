"""Paired (fold, eval_seed) comparison of z_inv vs raw z from the two Experiment-3 logs."""
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

zi, rw = load("docs/generated/exp3_rawz/zinv_baseline_eval.txt"), load("docs/generated/exp3_rawz/rawz_eval.txt")
assert zi.keys() == rw.keys() and len(zi) == 20
keys = sorted(zi)
names = ["AUROC", "Acc", "F1"]
D = np.array([[zi[k][i] - rw[k][i] for i in range(3)] for k in keys])  # z_inv - raw z
folds = [k[0] for k in keys]

print("== Paired diff (z_inv - raw z), n=20 (fold x eval_seed) ==")
for i, n in enumerate(names):
    d = D[:, i]
    t, p = stats.ttest_1samp(d, 0)
    w = stats.wilcoxon(d).pvalue
    print(f"{n:6s} mean={d.mean():+.4f} sd={d.std(ddof=1):.4f} wins(z_inv>raw)={int((d>0).sum())}/20 t-p={p:.2g} wilcoxon-p={w:.2g}")

print("\n== Per-fold paired diff (mean over 5 eval seeds; wins/5) ==")
for f in ["500", "600", "700", "800"]:
    idx = [j for j, x in enumerate(folds) if x == f]
    print(f"fold {f}: " + " | ".join(
        f"{n} {D[idx, i].mean():+.4f} ({int((D[idx, i] > 0).sum())}/5)" for i, n in enumerate(names)))

print("\n== Fold-level (n=4, cluster-robust view: fold means) ==")
fm = np.array([[D[[j for j, x in enumerate(folds) if x == f], i].mean() for i in range(3)] for f in ["500","600","700","800"]])
for i, n in enumerate(names):
    print(f"{n:6s} fold-mean diffs={np.round(fm[:, i], 4).tolist()}  mean={fm[:, i].mean():+.4f}  sd={fm[:, i].std(ddof=1):.4f}")
