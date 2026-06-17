import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.metrics import roc_auc_score, average_precision_score

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


def extract_feature(model, img, feature_type):
    out = model(img)

    if feature_type == "original":
        feat = out["z"]

    elif feature_type == "zinv":
        feat = out["z_inv"]

    elif feature_type == "zd":
        feat = out["z_notc_d"]

    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")

    return flatten_feature(feat)


def euclidean_score(query_feat, memory_feat):
    # query_feat: [B, D]
    # memory_feat: [N, D]
    dist = torch.cdist(query_feat, memory_feat, p=2)
    score = dist.min(dim=1)[0]
    return score


def cosine_score(query_feat, memory_feat):
    query_feat = F.normalize(query_feat, dim=1)
    memory_feat = F.normalize(memory_feat, dim=1)

    sim = torch.matmul(query_feat, memory_feat.T)
    max_sim = sim.max(dim=1)[0]

    score = 1.0 - max_sim
    return score


def build_memory_bank(args, model, transform, device):
    datasets = []

    for domain in args.support_domains:
        ds = HUSTDataset(
            root=args.root,
            domain=domain,
            only_normal=True,
            shot=args.shot,
            transform=transform,
            seed=args.seed,
        )
        datasets.append(ds)

    support_set = ConcatDataset(datasets)

    support_loader = DataLoader(
        support_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )

    memory_features = []

    for batch in support_loader:
        img = batch["image"].to(device)

        feat = extract_feature(
            model=model,
            img=img,
            feature_type=args.feature,
        )

        memory_features.append(feat.detach())

    memory_bank = torch.cat(memory_features, dim=0)

    return memory_bank


def evaluate_query(args, model, transform, memory_bank, device):
    query_set = HUSTDataset(
        root=args.root,
        domain=args.query_domain,
        only_normal=False,
        shot=None,
        transform=transform,
    )

    query_loader = DataLoader(
        query_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )

    scores = []
    labels = []

    for batch in query_loader:
        img = batch["image"].to(device)
        label = batch["label"].cpu().numpy()

        feat = extract_feature(
            model=model,
            img=img,
            feature_type=args.feature,
        )

        if args.metric == "euclidean":
            anomaly_score = euclidean_score(feat, memory_bank)

        elif args.metric == "cosine":
            anomaly_score = cosine_score(feat, memory_bank)

        else:
            raise ValueError(f"Unknown metric: {args.metric}")

        scores.extend(anomaly_score.detach().cpu().numpy().tolist())
        labels.extend(label.tolist())

    labels = np.array(labels)
    scores = np.array(scores)

    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)

    return auroc, auprc


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default="processed")

    parser.add_argument(
        "--support_domains",
        type=str,
        nargs="+",
        required=True,
        help="Domains used to build normal memory bank"
    )

    parser.add_argument(
        "--query_domain",
        type=str,
        required=True,
        help="Unseen target domain"
    )

    parser.add_argument(
        "--feature",
        type=str,
        required=True,
        choices=["original", "zinv", "zd"],
        help="Feature type for memory and query"
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to trained decomposition model checkpoint"
    )

    parser.add_argument("--shot", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--metric", type=str, default="euclidean", choices=["euclidean", "cosine"])
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("===================================")
    print("Memory-based FSAD")
    print("===================================")
    print("Device         :", device)
    print("Support domains:", args.support_domains)
    print("Query domain   :", args.query_domain)
    print("Feature        :", args.feature)
    print("Metric         :", args.metric)
    print("Shot/domain    :", args.shot)
    print("Checkpoint     :", args.ckpt)
    print("===================================")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    model = MaskDecompositionModel().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    memory_bank = build_memory_bank(args, model, transform, device)

    print("Memory bank shape:", tuple(memory_bank.shape))

    auroc, auprc = evaluate_query(args, model, transform, memory_bank, device)

    print("===================================")
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    print("===================================")


if __name__ == "__main__":
    main()