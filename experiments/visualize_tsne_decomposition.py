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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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


def collect_features(args, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    datasets = []

    for d in args.domains:
        ds = HUSTDataset(
            root=args.root,
            domain=d,
            only_normal=args.only_normal,
            shot=None,
            transform=transform,
        )

        if args.max_per_domain is not None:
            random.shuffle(ds.samples)
            ds.samples = ds.samples[:args.max_per_domain]

        datasets.append(ds)

    dataset = ConcatDataset(datasets)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )

    model = MaskDecompositionModel(num_classes=args.num_classes, num_domains=args.num_domains).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    features = []
    labels = []
    domains = []

    for batch in loader:
        img = batch["image"].to(device)

        # autograd.grad 사용 때문에 no_grad 사용 X
        out = model(img)

        if args.feature == "original":
            feat = out["z"]

        elif args.feature == "zcd":
            feat = out["z_cd"]

        elif args.feature == "zd":
            feat = out["z_notc_d"]

        elif args.feature == "zinv":
            feat = out["z_inv"]

        else:
            raise ValueError(f"Unknown feature type: {args.feature}")

        feat = flatten_feature(feat)

        features.append(feat.detach().cpu().numpy())
        labels.extend(batch["label"].numpy().tolist())
        domains.extend(list(batch["domain_name"]))

    features = np.concatenate(features, axis=0)
    labels = np.array(labels)
    domains = np.array(domains)

    return features, labels, domains


def plot_tsne(emb, color_values, title, save_path, legend_title):
    plt.figure(figsize=(8, 7))

    unique_values = sorted(list(set(color_values)))

    for v in unique_values:
        idx = color_values == v

        plt.scatter(
            emb[idx, 0],
            emb[idx, 1],
            s=14,
            alpha=0.75,
            label=str(v),
        )

    plt.title(title)
    plt.legend(title=legend_title, markerscale=1.5)
    plt.tight_layout()

    save_dir = os.path.dirname(save_path)
    if save_dir != "":
        os.makedirs(save_dir, exist_ok=True)

    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default="processed")
    parser.add_argument("--domains", type=str, nargs="+", required=True)

    parser.add_argument(
        "--feature",
        type=str,
        required=True,
        choices=["original", "zcd", "zd", "zinv"],
    )

    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_per_domain", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--only_normal",
        action="store_true",
        help="Use only normal samples only",
    )

    parser.add_argument("--save_prefix", type=str, default="results/tsne")
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--num_domains", type=int, default=15)

    args = parser.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("===================================")
    print("t-SNE Visualization")
    print("===================================")
    print("Device      :", device)
    print("Domains     :", args.domains)
    print("Feature     :", args.feature)
    print("Only normal :", args.only_normal)
    print("Checkpoint  :", args.ckpt)
    print("===================================")

    features, labels, domains = collect_features(args, device)

    print("Feature shape:", features.shape)

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=args.seed,
    )

    emb = tsne.fit_transform(features)

    mode = "normal_only" if args.only_normal else "all"

    # Domain 기준
    save_domain = f"{args.save_prefix}_{args.feature}_{mode}_by_domain.png"

    plot_tsne(
        emb,
        domains,
        title=f"t-SNE of {args.feature} feature by domain",
        save_path=save_domain,
        legend_title="Domain",
    )

    # Label 기준 (normal only면 생략)
    if not args.only_normal:
        label_names = np.array([
            "normal" if l == 0 else "anomaly"
            for l in labels
        ])

        save_label = f"{args.save_prefix}_{args.feature}_{mode}_by_label.png"

        plot_tsne(
            emb,
            label_names,
            title=f"t-SNE of {args.feature} feature by label",
            save_path=save_label,
            legend_title="Label",
        )

        print("Saved:", save_label)

    print("Saved:", save_domain)
    print("===================================")


if __name__ == "__main__":
    main()