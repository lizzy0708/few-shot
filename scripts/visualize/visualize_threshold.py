import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from datasets.hust_image import HUSTDataset
from models.mask_decomposition_model import MaskDecompositionModel


def flatten_feature(feat):
    if feat.dim() == 4:
        feat = F.adaptive_avg_pool2d(feat, 1)
        feat = feat.view(feat.size(0), -1)
    return feat


def cosine_score(x, proto):
    x = F.normalize(x, dim=1)
    proto = F.normalize(proto.unsqueeze(0), dim=1)
    return F.cosine_similarity(x, proto, dim=1)


def find_best_threshold(scores, labels):
    thresholds = np.linspace(scores.min(), scores.max(), 500)
    best_acc, best_th = 0, 0
    for th in thresholds:
        preds = (scores > th).astype(int)
        acc = (preds == labels).mean()
        if acc > best_acc:
            best_acc, best_th = acc, th
    return best_th


# ── 설정 ──────────────────────────────────────────
SUPPORT_DOMAIN = "800"
QUERY_DOMAIN   = "800"
CALIB_DOMAINS  = ["400", "500", "600", "700"]
SHOT           = 4
CKPT           = "mask_decomposition.pth"
SEED           = 42

torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

model = MaskDecompositionModel().to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()

# ── prototype 구성 ────────────────────────────────
support_set = HUSTDataset(root="processed", domain=SUPPORT_DOMAIN,
                          only_normal=True, shot=SHOT, transform=transform, seed=SEED)
support_loader = DataLoader(support_set, batch_size=SHOT, shuffle=False)

for batch in support_loader:
    img = batch["image"].to(device)
    out = model(img)
    prototype = flatten_feature(out["z_c_notd"]).mean(dim=0).detach()
    break

# ── query score 수집 ──────────────────────────────
query_set = HUSTDataset(root="processed", domain=QUERY_DOMAIN,
                        only_normal=False, transform=transform)
query_loader = DataLoader(query_set, batch_size=16, shuffle=False)

scores, labels = [], []
for batch in query_loader:
    img = batch["image"].to(device)
    out = model(img)
    z = flatten_feature(out["z_c_notd"])
    sim = cosine_score(z, prototype)
    scores.extend((1.0 - sim).detach().cpu().numpy().tolist())
    labels.extend(batch["label"].numpy().tolist())

scores = np.array(scores)
labels = np.array(labels)

# ── calibration threshold 계산 ────────────────────
calib_scores, calib_labels = [], []
for d in CALIB_DOMAINS:
    ds = HUSTDataset(root="processed", domain=d, only_normal=False, transform=transform)
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    for batch in loader:
        img = batch["image"].to(device)
        out = model(img)
        z = flatten_feature(out["z_c_notd"])
        sim = cosine_score(z, prototype)
        calib_scores.extend((1.0 - sim).detach().cpu().numpy().tolist())
        calib_labels.extend(batch["label"].numpy().tolist())

calib_scores = np.array(calib_scores)
calib_labels = np.array(calib_labels)
calib_th = find_best_threshold(calib_scores, calib_labels)

# support 기반 fixed threshold
support_scores = []
for batch in support_loader:
    img = batch["image"].to(device)
    out = model(img)
    z = flatten_feature(out["z_c_notd"])
    sim = cosine_score(z, prototype)
    support_scores.extend((1.0 - sim).detach().cpu().numpy().tolist())

support_scores = np.array(support_scores)
fixed_th = support_scores.mean() + 2.0 * support_scores.std()

# ── 정확도 계산 ───────────────────────────────────
acc_fixed = ((scores > fixed_th).astype(int) == labels).mean()
acc_calib = ((scores > calib_th).astype(int) == labels).mean()

# ── 시각화 ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("white")

normal_scores  = scores[labels == 0]
anomaly_scores = scores[labels == 1]

bins = np.linspace(0, 1, 60)

for ax, th, title, acc, color in [
    (axes[0], fixed_th,
     f"Fixed Threshold (support 기반)\nThreshold={fixed_th:.3f}  Accuracy={acc_fixed:.4f}",
     acc_fixed, "#e53935"),
    (axes[1], calib_th,
     f"Calibrated Threshold (학습 도메인 기반)\nThreshold={calib_th:.3f}  Accuracy={acc_calib:.4f}",
     acc_calib, "#1e88e5"),
]:
    ax.hist(normal_scores,  bins=bins, alpha=0.6, color="#43a047", label="Normal",  density=True)
    ax.hist(anomaly_scores, bins=bins, alpha=0.6, color="#e53935", label="Anomaly", density=True)
    ax.axvline(th, color=color, linewidth=2.5, linestyle="--", label=f"Threshold={th:.3f}")

    # 놓친 이상 샘플 강조
    missed = anomaly_scores[anomaly_scores <= th]
    if len(missed) > 0:
        ax.hist(missed, bins=bins, alpha=0.4, color="#ff6f00",
                label=f"Missed anomaly ({len(missed)}개)", density=True)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.suptitle(f"Test Domain: {QUERY_DOMAIN}RPM  |  Support: {SHOT}-shot",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("results/threshold_comparison.png", dpi=180, bbox_inches="tight")
plt.show()
print(f"Fixed threshold   → Accuracy: {acc_fixed:.4f}")
print(f"Calibrated threshold → Accuracy: {acc_calib:.4f}")
print("저장 완료: results/threshold_comparison.png")
