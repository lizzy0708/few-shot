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
from models.vit_mask_decomposition_model import ViTMaskDecompositionModel


def normal_contrastive_loss(z_pool, labels, temperature=0.07):
    z = F.normalize(z_pool, dim=1)
    sim = z @ z.T / temperature
    is_normal = (labels == 0)
    B = len(labels)
    eye = torch.eye(B, device=labels.device)
    mask_pos = (is_normal.unsqueeze(1) & is_normal.unsqueeze(0)).float() * (1 - eye)
    mask_denom = (1 - eye)
    logits_max, _ = sim.max(dim=1, keepdim=True)
    sim_exp = torch.exp(sim - logits_max.detach())
    denom = (sim_exp * mask_denom).sum(dim=1, keepdim=True)
    log_prob = sim - logits_max.detach() - torch.log(denom + 1e-8)
    n_pos = mask_pos.sum(dim=1)
    valid = is_normal & (n_pos > 0)
    loss = -(mask_pos * log_prob).sum(dim=1) / (n_pos + 1e-8)
    return loss[valid].mean() if valid.any() else torch.tensor(0.0, device=z.device)


def prototype_alignment_loss(z_pool, labels, domains):
    z = F.normalize(z_pool, dim=1)
    normals = z[labels == 0]
    dom_ids = domains[labels == 0]
    unique_doms = dom_ids.unique()
    if len(unique_doms) < 2:
        return torch.tensor(0.0, device=z_pool.device)
    protos = []
    for d in unique_doms:
        mask = (dom_ids == d)
        if mask.sum() > 0:
            protos.append(normals[mask].mean(dim=0))
    protos = torch.stack(protos)
    global_proto = protos.mean(dim=0)
    return ((protos - global_proto) ** 2).sum(dim=1).mean()


def episodic_proto_loss(z_pool, labels, n_support=4, n_episodes=4):
    device = z_pool.device
    normal_idx = (labels == 0).nonzero(as_tuple=True)[0]
    if len(normal_idx) < n_support + 1 or not (labels == 1).any():
        return torch.tensor(0.0, device=device)
    losses = []
    for _ in range(n_episodes):
        perm = torch.randperm(len(normal_idx), device=device)[:n_support]
        support_idx = normal_idx[perm]
        prototype = z_pool[support_idx].mean(dim=0).detach()
        dists = ((z_pool - prototype) ** 2).sum(dim=1).sqrt()
        thresh = dists[support_idx].detach().mean()
        logits = dists - thresh
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        losses.append(loss)
    return torch.stack(losses).mean()


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
    datasets = []
    for domain in args.train_domains:
        ds = HUSTDataset(
            root=args.root, domain=domain,
            only_normal=False, shot=None,
            all_domains=args.all_domains,
        )
        datasets.append(ds)

    dataset = ConcatDataset(datasets)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=2, drop_last=False)

    num_domains = len(args.all_domains)
    model = ViTMaskDecompositionModel(
        num_classes=args.num_classes,
        num_domains=num_domains
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    print("===================================")
    print("Train ViT Mask Decomposition Model")
    print("===================================")
    print(f"Device       : {device}")
    print(f"Train domains: {args.train_domains}")
    print(f"Num domains  : {num_domains}")
    print(f"Epochs       : {args.epochs}")
    print(f"Batch size   : {args.batch_size}")
    print(f"LR           : {args.lr}")
    print(f"Domain weight: {args.domain_weight}")
    print(f"SupCon weight: {args.supcon_weight}")
    print(f"Proto weight : {args.proto_weight}")
    print(f"Episodic wt  : {args.episodic_weight}")
    print(f"Seed         : {args.seed}")
    print(f"Save path    : {args.save_path}")
    print(f"Total samples: {len(dataset)}")
    print("===================================")

    for epoch in range(args.epochs):
        model.train()
        p = epoch / max(1, args.epochs - 1)
        alpha = float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)

        total_loss = total_class = total_domain = 0.0
        total_supcon = total_proto = total_episodic = 0.0
        correct_cls = correct_dom = total = 0

        for batch in loader:
            img = batch["image"].to(device)
            fault_type = batch["fault_type"].long().to(device)
            domain = batch["domain"].long().to(device)

            if domain.min() < 0 or domain.max() >= num_domains:
                raise ValueError(f"Domain label out of range [0, {num_domains-1}]")

            binary_label = (fault_type > 0).long()

            out = model(img, alpha=alpha, class_label=binary_label)
            z_c_notd = out["z_c_notd"]

            class_logits = model.classifier(z_c_notd)
            class_loss   = ce(class_logits, binary_label)
            domain_loss  = ce(out["domain_logits"], domain)

            z_inv_pool = F.adaptive_avg_pool2d(z_c_notd, 1).flatten(1)

            supcon_loss   = normal_contrastive_loss(z_inv_pool, binary_label, args.temperature)
            proto_loss    = prototype_alignment_loss(z_inv_pool, binary_label, domain)
            episodic_loss = episodic_proto_loss(z_inv_pool, binary_label, n_support=4, n_episodes=4)

            loss = (class_loss
                    + args.domain_weight   * domain_loss
                    + args.supcon_weight   * supcon_loss
                    + args.proto_weight    * proto_loss
                    + args.episodic_weight * episodic_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss     += loss.item()
            total_class    += class_loss.item()
            total_domain   += domain_loss.item()
            total_supcon   += supcon_loss.item()
            total_proto    += proto_loss.item()
            total_episodic += episodic_loss.item()

            pred_cls = class_logits.argmax(dim=1)
            pred_dom = out["domain_logits"].argmax(dim=1)
            correct_cls += (pred_cls == binary_label).sum().item()
            correct_dom += (pred_dom == domain).sum().item()
            total += fault_type.size(0)

        n = len(loader)
        print(
            f"Epoch [{epoch+1}/{args.epochs}] alpha={alpha:.3f} | "
            f"Loss: {total_loss/n:.4f} | Class: {total_class/n:.4f} | "
            f"Dom: {total_domain/n:.4f} | SupCon: {total_supcon/n:.4f} | "
            f"Proto: {total_proto/n:.4f} | Episodic: {total_episodic/n:.4f} | "
            f"Cls Acc: {correct_cls/total:.4f} | Dom Acc: {correct_dom/total:.4f}"
        )

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True) if os.path.dirname(args.save_path) else None
    torch.save(model.state_dict(), args.save_path)
    print(f"\nSaved: {args.save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="processed_gadf_fine_4096")
    parser.add_argument("--train_domains", type=str, nargs="+", required=True)
    parser.add_argument("--all_domains", type=str, nargs="+",
                        default=["400","402","404","500","502","504",
                                 "600","602","604","700","702","704","800","802","804"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--domain_weight", type=float, default=1.0)
    parser.add_argument("--supcon_weight", type=float, default=0.5)
    parser.add_argument("--proto_weight", type=float, default=0.1)
    parser.add_argument("--episodic_weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="vit_mask_decomposition.pth")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
