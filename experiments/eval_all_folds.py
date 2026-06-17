import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from datasets.hust_image import HUSTDataset
from models.mask_decomposition_model import MaskDecompositionModel

SEEDS = [0, 1, 2, 3, 4]
SHOT  = 4

FOLDS = [
    ("mask_decomposition_fold2.pth", "500", ["400","600","700","800"]),
    ("mask_decomposition_fold3.pth", "600", ["400","500","700","800"]),
    ("mask_decomposition_fold4.pth", "700", ["400","500","600","800"]),
    ("mask_decomposition.pth",       "800", ["400","500","600","700"]),
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_features(model, loader):
    feats, labels = [], []
    for batch in loader:
        img = batch["image"].to(device)
        out = model(img)
        z = out["z_c_notd"]
        z = F.adaptive_avg_pool2d(z, 1).view(z.size(0), -1)
        feats.append(z.detach())
        labels.extend(batch["label"].numpy().tolist())
    return torch.cat(feats, dim=0), np.array(labels)


def cosine_score(x, proto):
    x = F.normalize(x, dim=1)
    proto = F.normalize(proto.unsqueeze(0), dim=1)
    return F.cosine_similarity(x, proto, dim=1)


def get_raw_features(model, loader):
    feats, labels = [], []
    for batch in loader:
        img = batch["image"].to(device)
        out = model(img)
        z = out["z"]
        z = F.adaptive_avg_pool2d(z, 1).view(z.size(0), -1)
        feats.append(z.detach())
        labels.extend(batch["label"].numpy().tolist())
    return torch.cat(feats, dim=0), np.array(labels)


def find_best_threshold(scores, labels):
    thresholds = np.linspace(scores.min(), scores.max(), 500)
    best_acc, best_th = 0, 0
    for th in thresholds:
        acc = ((scores > th).astype(int) == labels).mean()
        if acc > best_acc:
            best_acc, best_th = acc, th
    return best_th


print(f"{'Test':>6} | {'Baseline AUROC':>15} | {'z_inv AUROC':>12} | {'Calibrated Acc':>14} | {'Calibrated F1':>13}")
print("-" * 75)

all_base, all_zinv, all_cacc, all_cf1 = [], [], [], []

for ckpt, test_d, calib_ds in FOLDS:
    model = MaskDecompositionModel().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    base_aurocs, zinv_aurocs, calib_accs, calib_f1s = [], [], [], []

    # calibration 데이터 (전체 사용)
    calib_sets = [HUSTDataset(root="processed", domain=d, only_normal=False, transform=transform) for d in calib_ds]
    calib_loader = DataLoader(ConcatDataset(calib_sets), batch_size=8, shuffle=False)
    calib_feats, calib_labels = get_features(model, calib_loader)

    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)

        support_set = HUSTDataset(root="processed", domain=test_d, only_normal=True,
                                  shot=SHOT, transform=transform, seed=seed)
        support_loader = DataLoader(support_set, batch_size=SHOT, shuffle=False)

        query_set = HUSTDataset(root="processed", domain=test_d, only_normal=False, transform=transform)
        query_loader = DataLoader(query_set, batch_size=8, shuffle=False)

        # prototype
        for batch in support_loader:
            img = batch["image"].to(device)
            out = model(img)
            z = out["z_c_notd"]
            z = F.adaptive_avg_pool2d(z, 1).view(z.size(0), -1)
            prototype = z.mean(dim=0).detach()
            break

        # query scores (z_inv)
        query_feats, query_labels = get_features(model, query_loader)
        zinv_scores = (1.0 - cosine_score(query_feats, prototype)).detach().cpu().numpy()
        zinv_aurocs.append(roc_auc_score(query_labels, zinv_scores))

        # baseline scores (raw z)
        query_raw, _ = get_raw_features(model, query_loader)
        support_raw, _ = get_raw_features(model, support_loader)
        proto_raw = support_raw.mean(dim=0).detach()
        base_scores = (1.0 - cosine_score(query_raw, proto_raw)).detach().cpu().numpy()
        base_aurocs.append(roc_auc_score(query_labels, base_scores))

        # calibrated threshold
        calib_scores = (1.0 - cosine_score(calib_feats, prototype)).detach().cpu().numpy()
        best_th = find_best_threshold(calib_scores, calib_labels)
        preds = (zinv_scores > best_th).astype(int)
        calib_accs.append(accuracy_score(query_labels, preds))
        calib_f1s.append(f1_score(query_labels, preds, zero_division=0))

    bm = np.mean(base_aurocs)
    zm = np.mean(zinv_aurocs)
    am = np.mean(calib_accs)
    fm = np.mean(calib_f1s)

    all_base.append(bm)
    all_zinv.append(zm)
    all_cacc.append(am)
    all_cf1.append(fm)

    print(f"{test_d:>6} | {bm:>15.4f} | {zm:>12.4f} | {am:>14.4f} | {fm:>13.4f}")

print("-" * 75)
print(f"{'Avg':>6} | {np.mean(all_base):>15.4f} | {np.mean(all_zinv):>12.4f} | {np.mean(all_cacc):>14.4f} | {np.mean(all_cf1):>13.4f}")
