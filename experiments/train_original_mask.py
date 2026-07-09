"""
train_original_mask.py — 5/1 체크포인트 구조로 재학습.

손실: CE(class) + domain GRL + mask_overlap(mc*md)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from datasets.hust_image import HUSTDataset
from models.original_mask_model import OriginalMaskModel

device = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args):
    set_seed(args.seed)

    num_domains = len(args.all_domains)
    datasets = []
    for domain in args.train_domains:
        datasets.append(HUSTDataset(
            root=args.root, domain=domain, only_normal=False,
            shot=None, all_domains=args.all_domains,
        ))

    dataset = ConcatDataset(datasets)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, drop_last=False)

    model = OriginalMaskModel(
        num_classes=args.num_classes,
        num_domains=num_domains,
        encoder_layer=args.encoder_layer,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    ce = nn.CrossEntropyLoss()

    print("===================================")
    print("Train Original Mask Model (5/1 style)")
    print("===================================")
    print("Device        :", device)
    print("Encoder layer :", args.encoder_layer, f"({model.feature_extractor.out_dim}-dim)")
    print("Train domains :", args.train_domains)
    print("All domains   :", args.all_domains)
    print("Num domains   :", num_domains)
    print("Epochs        :", args.epochs)
    print("Mask weight   :", args.mask_weight)
    print("Domain weight :", args.domain_weight)
    print("Seed          :", args.seed)
    print("Save path     :", args.save_path)
    print("Total samples :", len(dataset))
    print("===================================")

    for epoch in range(args.epochs):
        model.train()
        p = epoch / max(1, args.epochs - 1)
        alpha = float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)

        if epoch < args.warmup_epochs:
            eff_mask_w = 0.0
        else:
            wp = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
            eff_mask_w = args.mask_weight * min(1.0, wp * 2.0)

        total_loss = total_cls = total_dom = total_mask = 0.0
        correct_cls = correct_dom = total = 0

        for batch in loader:
            img = batch["image"].to(device)
            fault_type = batch["fault_type"].long().to(device)
            domain = batch["domain"].long().to(device)
            binary_label = (fault_type > 0).long()

            out = model(img, alpha=alpha, class_label=binary_label)

            cls_loss  = ce(out["class_logits"], binary_label)
            dom_loss  = ce(out["domain_logits"], domain)
            mask_loss = torch.mean(out["mc"] * out["md"])

            loss = cls_loss + args.domain_weight * dom_loss + eff_mask_w * mask_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_cls  += cls_loss.item()
            total_dom  += dom_loss.item()
            total_mask += mask_loss.item()

            correct_cls += (out["class_logits"].argmax(1) == binary_label).sum().item()
            correct_dom += (out["domain_logits"].argmax(1) == domain).sum().item()
            total += fault_type.size(0)

        scheduler.step()
        n = len(loader)
        print(
            f"Epoch [{epoch+1}/{args.epochs}] alpha={alpha:.3f} | mask_w={eff_mask_w:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e} | "
            f"Loss: {total_loss/n:.4f} | Class: {total_cls/n:.4f} | "
            f"Dom: {total_dom/n:.4f} | Mask: {total_mask/n:.4f} | "
            f"Cls Acc: {correct_cls/total:.4f} | Dom Acc: {correct_dom/total:.4f}"
        )

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print("===================================")
    print("Training finished. Saved:", args.save_path)
    print("===================================")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="processed")
    parser.add_argument("--train_domains", type=str, nargs="+", required=True)
    parser.add_argument("--all_domains", type=str, nargs="+",
                        default=["400", "500", "600", "700", "800"])
    parser.add_argument("--encoder_layer", type=str, default="layer3",
                        choices=["layer3", "layer4"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mask_weight", type=float, default=0.1)
    parser.add_argument("--domain_weight", type=float, default=1.0)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="mask_orig.pth")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
