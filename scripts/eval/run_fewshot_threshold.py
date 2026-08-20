import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from datasets.hust_image import HUSTDataset
from models.mask_decomposition_model import MaskDecompositionModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def flatten_feature(feat):
    if feat.dim() == 4:
        feat = F.adaptive_avg_pool2d(feat, 1)
        feat = feat.view(feat.size(0), -1)
    return feat


def cosine_score(x, proto):
    x = F.normalize(x, dim=1)
    proto = F.normalize(proto.unsqueeze(0), dim=1)
    return F.cosine_similarity(x, proto, dim=1)


def get_threshold(normal_scores, method="mean_std", k=2.0, percentile=95):
    normal_scores = np.array(normal_scores)

    if method == "mean_std":
        return normal_scores.mean() + k * normal_scores.std()

    elif method == "percentile":
        return np.percentile(normal_scores, percentile)

    else:
        raise ValueError(f"Unknown threshold method: {method}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--support_domain", type=str, required=True)
    parser.add_argument("--query_domain", type=str, required=True)
    parser.add_argument("--shot", type=int, default=4)
    parser.add_argument("--ckpt", type=str, default="mask_decomposition.pth")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--threshold_method", type=str, default="mean_std",
                        choices=["mean_std", "percentile"])
    parser.add_argument("--k", type=float, default=2.0)
    parser.add_argument("--percentile", type=float, default=95)

    args = parser.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("===================================")
    print("Decomposition FSAD + Threshold")
    print("===================================")
    print("Device          :", device)
    print("Support domain  :", args.support_domain)
    print("Query domain    :", args.query_domain)
    print("Shot            :", args.shot)
    print("Seed            :", args.seed)
    print("Checkpoint      :", args.ckpt)
    print("Threshold method:", args.threshold_method)
    print("k               :", args.k)
    print("Percentile      :", args.percentile)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    model = MaskDecompositionModel().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    support_set = HUSTDataset(
        root="processed",
        domain=args.support_domain,
        only_normal=True,
        shot=args.shot,
        transform=transform,
        seed=args.seed,
    )

    support_loader = DataLoader(
        support_set,
        batch_size=args.shot,
        shuffle=False,
    )

    support_features = []

    for batch in support_loader:
        img = batch["image"].to(device)

        out = model(img)
        z_c_notd = flatten_feature(out["z_c_notd"])

        support_features.append(z_c_notd.detach())
        break

    if len(support_features) == 0:
        raise RuntimeError("Failed to build support features.")

    support_features = torch.cat(support_features, dim=0)
    prototype = support_features.mean(dim=0).detach()

    # support normal score로 threshold 계산
    support_sim = cosine_score(support_features, prototype)
    support_normal_scores = 1.0 - support_sim
    support_normal_scores = support_normal_scores.detach().cpu().numpy()

    threshold = get_threshold(
        support_normal_scores,
        method=args.threshold_method,
        k=args.k,
        percentile=args.percentile,
    )

    query_set = HUSTDataset(
        root="processed",
        domain=args.query_domain,
        only_normal=False,
        shot=None,
        transform=transform,
    )

    query_loader = DataLoader(
        query_set,
        batch_size=16,
        shuffle=False,
    )

    scores = []
    labels = []

    for batch in query_loader:
        img = batch["image"].to(device)
        label = batch["label"].cpu().numpy()

        out = model(img)
        z_c_notd = flatten_feature(out["z_c_notd"])

        sim = cosine_score(z_c_notd, prototype)
        anomaly_score = 1.0 - sim

        scores.extend(anomaly_score.detach().cpu().numpy().tolist())
        labels.extend(label.tolist())

    scores = np.array(scores)
    labels = np.array(labels).astype(int)


    print("Score statistics")
    print(f"score min   : {scores.min():.6f}")
    print(f"score max   : {scores.max():.6f}")
    print(f"score mean  : {scores.mean():.6f}")
    print(f"score std   : {scores.std():.6f}")
    print(f"normal mean : {scores[labels == 0].mean():.6f}")
    print(f"normal max  : {scores[labels == 0].max():.6f}")
    print(f"anomaly mean: {scores[labels == 1].mean():.6f}")
    print(f"anomaly min : {scores[labels == 1].min():.6f}")
    preds = (scores > threshold).astype(int)

    auroc = roc_auc_score(labels, scores)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds)

    print("===================================")
    print(f"Threshold : {threshold:.6f}")
    print(f"AUROC     : {auroc:.4f}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print("Confusion Matrix")
    print(cm)
    print("===================================")


if __name__ == "__main__":
    main()