"""
계층적 md 채널 분석: md_rpm(물리적 RPM 변동) vs md_batch(측정 배치 변동).

주의: 분류기가 GAP→Linear라 gradient 마스크는 공간 위치에 대해 상수 —
즉 mc/md는 순수 '채널' 마스크다. 따라서 시각화도 채널 수준으로 한다.

Figure 구성 (영문 — 논문용):
  (a) 채널별 md_rpm vs md_batch 강도 산점도 (+상관계수)
  (b) 두 마스크의 채널 강도 분포 (정렬 곡선)
  (c) top-k 채널 집합의 겹침 비율 (k=50/100/200)
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from datasets.hust_image import HUSTDataset
from models.mask_decomposition_model import MaskDecompositionModel
from scripts.eval.eval_all_folds import transform, FINE_ALL

device = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "checkpoints/hier15_md_fold500.pth"

sd = torch.load(CKPT, map_location=device)
model = MaskDecompositionModel(num_classes=2, num_domains=15, encoder_layer="layer4",
                               num_rpm_groups=5, hierarchical_md=True).to(device)
model.load_state_dict(sd, strict=False)
model.eval()

ds = HUSTDataset(root="processed_gadf_fine_4096", domain="500", only_normal=True,
                 transform=transform, all_domains=FINE_ALL)
loader = DataLoader(ds, batch_size=16, shuffle=False)

md_r_ch, md_b_ch = [], []
for bi, batch in enumerate(loader):
    if bi >= 8:
        break
    out = model(batch["image"].to(device))
    md_r_ch.append(out["md_rpm"].mean(dim=(0, 2, 3)).detach().cpu().numpy())
    md_b_ch.append(out["md_batch"].mean(dim=(0, 2, 3)).detach().cpu().numpy())
r = np.mean(md_r_ch, axis=0)   # [2048]
b = np.mean(md_b_ch, axis=0)
corr = np.corrcoef(r, b)[0, 1]

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

# (a) scatter
axes[0].scatter(r, b, s=3, alpha=0.4, color="#1976d2")
axes[0].set_xlabel("md_rpm channel intensity")
axes[0].set_ylabel("md_batch channel intensity")
axes[0].set_title(f"(a) Channel-wise mask intensity (r={corr:.3f})")

# (b) sorted intensity curves
axes[1].plot(np.sort(r)[::-1], label="md_rpm (physical)", color="#1976d2")
axes[1].plot(np.sort(b)[::-1], label="md_batch (measurement)", color="#e65100")
axes[1].set_xlabel("channel rank")
axes[1].set_ylabel("intensity")
axes[1].set_title("(b) Sorted channel intensities")
axes[1].legend(fontsize=9)

# (c) top-k overlap
ks = [50, 100, 200, 400]
overlaps = []
for k in ks:
    top_r = set(np.argsort(r)[-k:])
    top_b = set(np.argsort(b)[-k:])
    overlaps.append(len(top_r & top_b) / k)
axes[2].bar([str(k) for k in ks], overlaps, color="#78909c", edgecolor="black")
axes[2].axhline(np.mean([k / 2048 for k in ks]), ls="--", color="gray",
                label="random expectation")
for i, v in enumerate(overlaps):
    axes[2].text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
axes[2].set_xlabel("top-k channels")
axes[2].set_ylabel("overlap ratio")
axes[2].set_title("(c) Top-k channel set overlap")
axes[2].legend(fontsize=8)
axes[2].set_ylim(0, 1)

plt.suptitle("Hierarchical domain mask decomposition: RPM vs. measurement-batch channels (fold_500)",
             fontsize=11)
plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/md_hierarchy_fold500.png", dpi=200, bbox_inches="tight")
top100 = len(set(np.argsort(r)[-100:]) & set(np.argsort(b)[-100:]))
print(f"저장 완료 | 채널 상관 r={corr:.3f} | top-100 겹침 {top100}/100")
