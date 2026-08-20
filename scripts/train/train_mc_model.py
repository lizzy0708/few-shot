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
from models.mc_model import MCModel


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
    z_norm = F.normalize(z_pool, dim=1)
    losses = []
    for _ in range(n_episodes):
        perm = torch.randperm(len(normal_idx), device=device)[:n_support]
        support_idx = normal_idx[perm]
        prototype = F.normalize(z_norm[support_idx].mean(dim=0).detach(), dim=0)
        dists = ((z_norm - prototype) ** 2).sum(dim=1).sqrt()
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
        datasets.append(HUSTDataset(
            root=args.root, domain=domain,
            only_normal=False, shot=None,
            all_domains=args.all_domains,
        ))
    dataset = ConcatDataset(datasets)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=2, drop_last=False)

    num_domains = len(args.all_domains)
    model = MCModel(num_classes=args.num_classes, num_domains=num_domains).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    print("=" * 50)
    print("MCModel Training (mc-only, GRL on z_c, positive on z_d)")
    print("=" * 50)
    print(f"Device         : {device}")
    print(f"Train domains  : {args.train_domains}")
    print(f"Total samples  : {len(dataset)}")
    print(f"Epochs         : {args.epochs}")
    print(f"Batch size     : {args.batch_size}")
    print(f"LR             : {args.lr}")
    print(f"domain_weight  : {args.domain_weight}  (adversarial on z_c)")
    print(f"domain_d_weight: {args.domain_d_weight} (positive on z_d)")
    print(f"supcon_weight  : {args.supcon_weight}")
    print(f"proto_weight   : {args.proto_weight}")
    print(f"episodic_weight: {args.episodic_weight}")
    print(f"warmup_epochs  : {args.warmup_epochs}")
    print(f"Save path      : {args.save_path}")
    print("=" * 50)

    for epoch in range(args.epochs):
        model.train()
        p = epoch / max(1, args.epochs - 1)
        alpha = float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)

        total_loss = total_cls = total_dom = total_dom_d = 0.0
        total_sc = total_pa = total_ep = 0.0
        correct_cls = correct_dom = correct_dom_d = total = 0

        for batch in loader:
            img = batch["image"].to(device)
            fault_type = batch["fault_type"].long().to(device)
            domain = batch["domain"].long().to(device)
            binary_label = (fault_type > 0).long()

            out = model(img, alpha=alpha, class_label=binary_label)
            z_c = out["z_c"]
            z_d = out["z_d"]

            z_c_pool = F.adaptive_avg_pool2d(z_c, 1).flatten(1)
            z_d_pool = F.adaptive_avg_pool2d(z_d, 1).flatten(1)

            class_logits = model.classifier(z_c)
            class_loss   = ce(class_logits, binary_label)
            domain_loss  = ce(out["domain_logits"], domain)       # adversarial (low)
            domain_d_loss = ce(out["domain_logits_d"], domain)    # positive (high)

            supcon_loss   = normal_contrastive_loss(z_c_pool, binary_label, args.temperature)
            proto_loss    = prototype_alignment_loss(z_c_pool, binary_label, domain)
            episodic_loss = episodic_proto_loss(z_c_pool, binary_label)

            loss = (class_loss
                    + args.domain_weight   * domain_loss
                    + args.domain_d_weight * domain_d_loss
                    + args.supcon_weight   * supcon_loss
                    + args.proto_weight    * proto_loss
                    + args.episodic_weight * episodic_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss  += loss.item()
            total_cls   += class_loss.item()
            total_dom   += domain_loss.item()
            total_dom_d += domain_d_loss.item()
            total_sc    += supcon_loss.item()
            total_pa    += proto_loss.item()
            total_ep    += episodic_loss.item()

            correct_cls   += (class_logits.argmax(1) == binary_label).sum().item()
            correct_dom   += (out["domain_logits"].argmax(1) == domain).sum().item()
            correct_dom_d += (out["domain_logits_d"].argmax(1) == domain).sum().item()
            total += img.size(0)

        n = len(loader)
        print(
            f"Epoch [{epoch+1}/{args.epochs}] alpha={alpha:.3f} | "
            f"Loss={total_loss/n:.4f} | Cls={total_cls/n:.4f} | "
            f"Dom(adv)={total_dom/n:.4f} | Dom(pos)={total_dom_d/n:.4f} | "
            f"SC={total_sc/n:.4f} | EP={total_ep/n:.4f} | "
            f"ClsAcc={correct_cls/total:.4f} | "
            f"DomAcc(adv)={correct_dom/total:.4f} | "
            f"DomAcc(pos)={correct_dom_d/total:.4f}"
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
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--domain_weight", type=float, default=1.0)
    parser.add_argument("--domain_d_weight", type=float, default=1.0)
    parser.add_argument("--supcon_weight", type=float, default=0.5)
    parser.add_argument("--proto_weight", type=float, default=0.1)
    parser.add_argument("--episodic_weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="mc_model.pth")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
