import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from datasets.hust_image import HUSTDataset
from models.gated_mask_model import GatedMaskModel

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
        ds = HUSTDataset(root=args.root, domain=domain, only_normal=False, shot=None,
                          all_domains=args.all_domains)
        datasets.append(ds)
    dataset = ConcatDataset(datasets)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=2, drop_last=False)

    model = GatedMaskModel(num_classes=args.num_classes, num_domains=num_domains,
                            encoder_layer=args.encoder_layer,
                            use_domain_gate=(not args.no_domain_gate),
                            domain_gate_grl=args.domain_gate_grl).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    ce = nn.CrossEntropyLoss()

    print("===================================")
    print("Train GatedMaskModel (learnable sigmoid gates)")
    print("===================================")
    print("Device        :", device)
    print("Train domains :", args.train_domains)
    print("All domains   :", args.all_domains)
    print("Num domains   :", num_domains)
    print("Epochs        :", args.epochs)
    print("Batch size    :", args.batch_size)
    print("LR            :", args.lr)
    print("Use domain gate:", not args.no_domain_gate)
    print("Domain gate GRL:", args.domain_gate_grl)
    print("Mask weight   :", args.mask_weight)
    print("Domain weight :", args.domain_weight)
    print("DomDisc weight:", args.domain_disc_weight)
    print("Seed          :", args.seed)
    print("Save path     :", args.save_path)
    print("Total samples :", len(dataset))
    print("===================================")

    for epoch in range(args.epochs):
        model.train()

        p = epoch / max(1, args.epochs - 1)
        alpha = float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)

        total_loss = total_class = total_dom = total_domdisc = total_mask = 0.0
        correct_cls = correct_dom = total = 0

        for batch in loader:
            img = batch["image"].to(device)
            fault_type = batch["fault_type"].long().to(device)
            domain = batch["domain"].long().to(device)
            binary_label = (fault_type > 0).long()

            out = model(img, alpha=alpha)

            class_loss = ce(out["class_logits"], binary_label)
            domain_loss = ce(out["domain_logits"], domain)
            if out["domain_logits_disc"] is not None:
                domain_disc_loss = ce(out["domain_logits_disc"], domain)
            else:
                domain_disc_loss = torch.tensor(0.0, device=device)
            mask_loss = out["mask_loss"]

            loss = (class_loss
                    + args.domain_weight * domain_loss
                    + args.domain_disc_weight * domain_disc_loss
                    + args.mask_weight * mask_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_class += class_loss.item()
            total_dom += domain_loss.item()
            total_domdisc += domain_disc_loss.item()
            total_mask += mask_loss.item()

            correct_cls += (out["class_logits"].argmax(dim=1) == binary_label).sum().item()
            if out["domain_logits_disc"] is not None:
                correct_dom += (out["domain_logits_disc"].argmax(dim=1) == domain).sum().item()
            total += fault_type.size(0)

        n = len(loader)
        scheduler.step()
        print(f"Epoch [{epoch+1}/{args.epochs}] alpha={alpha:.3f} | "
              f"lr={scheduler.get_last_lr()[0]:.2e} | "
              f"Loss: {total_loss/n:.4f} | Class: {total_class/n:.4f} | "
              f"Dom(GRL): {total_dom/n:.4f} | DomDisc: {total_domdisc/n:.4f} | "
              f"Mask(gate_c*gate_d): {total_mask/n:.4f} | "
              f"Cls Acc: {correct_cls/total:.4f} | DomDisc Acc: {correct_dom/total:.4f}")

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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mask_weight", type=float, default=0.1)
    parser.add_argument("--domain_weight", type=float, default=1.0)
    parser.add_argument("--domain_disc_weight", type=float, default=1.0)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--encoder_layer", type=str, default="layer3", choices=["layer3", "layer4"])
    parser.add_argument("--no_domain_gate", action="store_true",
                        help="z_inv = z*class_gate만 사용(도메인 게이트 제거). mask loss도 자동 제외됨. "
                             "domain_classifier/GRL을 통한 encoder invariance 학습은 계속 유지.")
    parser.add_argument("--domain_gate_grl", action="store_true",
                        help="domain_gate_net 학습을 detach+직접분류 대신 GRL(adversarial)로 전환. "
                             "encoder는 여전히 detach로 보호(z_pool.detach()*domain_gate에 GRL).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
