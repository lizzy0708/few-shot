"""
Disentanglement diagnostic: measures whether z_inv leaks domain information.

Good disentanglement:
  z_inv domain accuracy  → low  (~chance = 1/num_domains)
  z_notc_d domain accuracy → high (>0.7)

Usage:
    python experiments/check_disentangle.py \
        --ckpt mask_decomposition.pth \
        --root processed_cwt \
        --all_domains 400 402 404 500 502 504 600 602 604 700 702 704 800 802 804 \
        --num_classes 5
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from datasets.hust_image import HUSTDataset
from models.mask_decomposition_model import MaskDecompositionModel

device = "cuda" if torch.cuda.is_available() else "cpu"


def check(args):
    all_domains = args.all_domains
    num_domains = len(all_domains)

    datasets = []
    for d in all_domains:
        try:
            ds = HUSTDataset(
                root=args.root,
                domain=d,
                only_normal=False,
                shot=None,
                all_domains=all_domains,
            )
            datasets.append(ds)
        except RuntimeError:
            print(f"  Warning: domain {d} not found in {args.root}, skipping")

    from torch.utils.data import ConcatDataset
    dataset = ConcatDataset(datasets)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

    model = MaskDecompositionModel(
        num_classes=args.num_classes,
        num_domains=num_domains,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    correct_zinv_dom = 0
    correct_zd_dom = 0
    correct_z_dom = 0
    total = 0

    for batch in loader:
        img = batch["image"].to(device)
        domain = batch["domain"].to(device)

        out = model(img)

        # DomainClassifier.forward expects 4D [B,C,H,W]; use fc directly on pooled features
        zinv_pool = F.adaptive_avg_pool2d(out["z_inv"], 1).flatten(1)
        pred_zinv = model.domain_classifier.fc(zinv_pool).argmax(dim=1)
        correct_zinv_dom += (pred_zinv == domain).sum().item()

        zd_pool = F.adaptive_avg_pool2d(out["z_notc_d"], 1).flatten(1)
        pred_zd = model.domain_classifier.fc(zd_pool).argmax(dim=1)
        correct_zd_dom += (pred_zd == domain).sum().item()

        z_pool = F.adaptive_avg_pool2d(out["z"], 1).flatten(1)
        pred_z = model.domain_classifier.fc(z_pool).argmax(dim=1)
        correct_z_dom += (pred_z == domain).sum().item()

        total += domain.size(0)

    chance = 1.0 / num_domains

    print()
    print("=" * 50)
    print("Disentanglement Check")
    print("=" * 50)
    print(f"Checkpoint : {args.ckpt}")
    print(f"Root       : {args.root}")
    print(f"Domains    : {num_domains}  (chance={chance:.3f})")
    print(f"Samples    : {total}")
    print()
    print(f"  raw z    domain acc : {correct_z_dom/total:.4f}  (baseline)")
    print(f"  z_notc_d domain acc : {correct_zd_dom/total:.4f}  ← should be HIGH")
    print(f"  z_inv    domain acc : {correct_zinv_dom/total:.4f}  ← should be LOW (~{chance:.3f})")
    print()
    gap = correct_zd_dom / total - correct_zinv_dom / total
    print(f"  Disentanglement gap (z_notc_d - z_inv): {gap:.4f}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="mask_decomposition.pth")
    parser.add_argument("--root", type=str, default="processed_cwt")
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument(
        "--all_domains", type=str, nargs="+",
        default=["400", "402", "404", "500", "502", "504",
                 "600", "602", "604", "700", "702", "704",
                 "800", "802", "804"],
    )
    args = parser.parse_args()

    print("Device:", device)
    check(args)


if __name__ == "__main__":
    main()
