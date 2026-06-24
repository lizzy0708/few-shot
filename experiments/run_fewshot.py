import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score

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


def build_diag_mahalanobis(support_features, reg):
    mean = support_features.mean(dim=0).detach()
    centered = support_features - mean.unsqueeze(0)
    var = centered.var(dim=0, unbiased=False) + reg

    return {
        "mean": mean,
        "inv_var": 1.0 / var,
    }


def diag_mahalanobis_score(x, estimator):
    diff = x - estimator["mean"].unsqueeze(0)
    return torch.sum(diff * diff * estimator["inv_var"].unsqueeze(0), dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="processed")
    parser.add_argument("--support_domain", type=str, required=True)
    parser.add_argument("--query_domain", type=str, required=True)
    parser.add_argument("--shot", type=int, default=4)
    parser.add_argument("--ckpt", type=str, default="mask_decomposition.pth")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_domains", type=int, default=5)
    parser.add_argument("--score_method", type=str, default="mahalanobis",
                        choices=["mahalanobis", "cosine"])
    parser.add_argument("--cov_reg", type=float, default=1e-3)
    args = parser.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("===================================")
    print("Decomposition only FSAD")
    print("===================================")
    print("Device        :", device)
    print("Root          :", args.root)
    print("Support domain:", args.support_domain)
    print("Query domain  :", args.query_domain)
    print("Shot          :", args.shot)
    print("Seed          :", args.seed)
    print("Checkpoint    :", args.ckpt)
    print("Num domains   :", args.num_domains)
    print("Score method  :", args.score_method)
    print("Cov reg       :", args.cov_reg)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    model = MaskDecompositionModel(num_domains=args.num_domains).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    support_set = HUSTDataset(
        root=args.root,
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
    estimator = build_diag_mahalanobis(support_features, args.cov_reg)

    query_set = HUSTDataset(
        root=args.root,
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

        if args.score_method == "cosine":
            sim = cosine_score(z_c_notd, prototype)
            anomaly_score = 1.0 - sim
        elif args.score_method == "mahalanobis":
            anomaly_score = diag_mahalanobis_score(z_c_notd, estimator)
        else:
            raise ValueError(f"Unknown score_method: {args.score_method}")

        scores.extend(anomaly_score.detach().cpu().numpy().tolist())
        labels.extend(label.tolist())

    auroc = roc_auc_score(np.array(labels), np.array(scores))

    print("===================================")
    print(f"AUROC: {auroc:.4f}")
    print("===================================")


if __name__ == "__main__":
    main()
