"""
train_simple_mask.py — 6/15 방식 재현.

변경 사항 (fine15 최종 설정 대비):
  - encoder_layer: layer3 (1024-dim) 기본값
  - 손실: CE(class) + domain GRL + mask 직교만 (SupCon/ProtoAlign/Episodic 제거)
  - processed/ 5개 coarse 도메인 (400/500/600/700/800)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from datasets.hust_image import HUSTDataset
from models.mask_decomposition_model import MaskDecompositionModel

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
        ds = HUSTDataset(
            root=args.root,
            domain=domain,
            only_normal=False,
            shot=None,
            all_domains=args.all_domains,
        )
        datasets.append(ds)

    dataset = ConcatDataset(datasets)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, drop_last=False)

    model = MaskDecompositionModel(
        num_classes=args.num_classes,
        num_domains=num_domains,
        encoder_layer=args.encoder_layer,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    ce = nn.CrossEntropyLoss()

    in_dim = model.feature_extractor.out_dim
    print("===================================")
    print("Train Simple Mask Model (6/15 style)")
    print("===================================")
    print("Device        :", device)
    print("Encoder layer :", args.encoder_layer, f"({in_dim}-dim)")
    print("Train domains :", args.train_domains)
    print("All domains   :", args.all_domains)
    print("Num domains   :", num_domains)
    print("Epochs        :", args.epochs)
    print("Batch size    :", args.batch_size)
    print("LR            :", args.lr)
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
            effective_mask_weight = 0.0
        else:
            warmup_progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
            effective_mask_weight = args.mask_weight * min(1.0, warmup_progress * 2.0)

        total_loss = 0.0
        total_class_loss = 0.0
        total_domain_loss = 0.0
        total_mask_loss = 0.0
        correct_cls = 0
        correct_dom = 0
        total = 0

        for batch in loader:
            img = batch["image"].to(device)
            fault_type = batch["fault_type"].long().to(device)
            domain = batch["domain"].long().to(device)

            binary_label = (fault_type > 0).long()

            out = model(img, alpha=alpha, class_label=binary_label)

            z_c_notd = out["z_c_notd"]
            mc = out["mc"]
            md = out["md"]

            class_logits = model.classifier(z_c_notd)
            class_loss   = ce(class_logits, binary_label)
            domain_loss  = ce(out["domain_logits"], domain)
            domain_disc_loss = ce(out["domain_logits_disc"], domain)
            mask_loss    = torch.mean(mc * md)

            loss = (class_loss
                    + args.domain_weight * domain_loss
                    + args.domain_disc_weight * domain_disc_loss
                    + effective_mask_weight * mask_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss       += loss.item()
            total_class_loss += class_loss.item()
            total_domain_loss += domain_loss.item()
            total_mask_loss  += mask_loss.item()

            pred_cls = class_logits.argmax(dim=1)
            pred_dom = out["domain_logits"].argmax(dim=1)
            correct_cls += (pred_cls == binary_label).sum().item()
            correct_dom += (pred_dom == domain).sum().item()
            total += fault_type.size(0)

        n = len(loader)
        cls_acc = correct_cls / total
        dom_acc = correct_dom / total
        scheduler.step()

        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"alpha={alpha:.3f} | mask_w={effective_mask_weight:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e} | "
            f"Loss: {total_loss/n:.4f} | "
            f"Class: {total_class_loss/n:.4f} | "
            f"Dom: {total_domain_loss/n:.4f} | "
            f"Mask(mc*md): {total_mask_loss/n:.4f} | "
            f"Cls Acc: {cls_acc:.4f} | Dom Acc: {dom_acc:.4f}"
        )

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), args.save_path)

    print("===================================")
    print("Training finished")
    print("Saved:", args.save_path)
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
    parser.add_argument("--domain_disc_weight", type=float, default=1.0)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="mask_simple.pth")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
