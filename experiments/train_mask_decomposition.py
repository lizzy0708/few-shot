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
from models.mask_decomposition_model import MaskDecompositionModel
from models.original_mask_model import OriginalMaskModel


def normal_contrastive_loss(z_pool, labels, temperature=0.07):
    """
    One-class contrastive loss on z_inv:
      - Normal-Normal pairs (positive): attract
      - Normal-Anomaly pairs (negative): repel
      - Anomaly-Anomaly pairs          : ignored
    Only normal samples are used as anchors.
    """
    z = F.normalize(z_pool, dim=1)
    sim = z @ z.T / temperature  # [B, B]

    is_normal = (labels == 0)  # [B]
    B = len(labels)
    eye = torch.eye(B, device=labels.device)

    mask_pos = (is_normal.unsqueeze(1) & is_normal.unsqueeze(0)).float()
    mask_pos = mask_pos * (1 - eye)

    mask_denom = (1 - eye)

    logits_max, _ = sim.max(dim=1, keepdim=True)
    sim_exp = torch.exp(sim - logits_max.detach())

    denom = (sim_exp * mask_denom).sum(dim=1, keepdim=True)
    log_prob = sim - logits_max.detach() - torch.log(denom + 1e-8)

    n_pos = mask_pos.sum(dim=1)
    valid = is_normal & (n_pos > 0)
    loss = -(mask_pos * log_prob).sum(dim=1) / (n_pos + 1e-8)
    return loss[valid].mean() if valid.any() else torch.tensor(0.0, device=z.device)


def episodic_proto_loss(z_pool, labels, domains=None, n_support=4, n_episodes=4):
    """4-shot 테스트 시나리오 시뮬레이션.
    domains 제공 시: 단일 도메인에서 support 선택 (test-time과 동일).
    domains=None 시: 기존 방식 (배치에서 랜덤 선택).
    """
    device = z_pool.device

    if not (labels == 1).any():
        return torch.tensor(0.0, device=device)

    z_norm = F.normalize(z_pool, dim=1)
    losses = []

    for _ in range(n_episodes):
        if domains is not None:
            # test-time 시뮬레이션: 단일 도메인에서 support 4개 선택
            unique_doms = domains.unique()
            # 정상 샘플이 충분한 도메인만 후보
            valid_doms = [d for d in unique_doms
                          if ((domains == d) & (labels == 0)).sum() >= n_support]
            if not valid_doms:
                continue
            dom = valid_doms[torch.randint(len(valid_doms), (1,)).item()]
            dom_normal_idx = ((domains == dom) & (labels == 0)).nonzero(as_tuple=True)[0]
            perm = torch.randperm(len(dom_normal_idx), device=device)[:n_support]
            support_idx = dom_normal_idx[perm]
        else:
            normal_idx = (labels == 0).nonzero(as_tuple=True)[0]
            if len(normal_idx) < n_support + 1:
                continue
            perm = torch.randperm(len(normal_idx), device=device)[:n_support]
            support_idx = normal_idx[perm]

        prototype = F.normalize(z_norm[support_idx].mean(dim=0).detach(), dim=0)
        dists = ((z_norm - prototype) ** 2).sum(dim=1).sqrt()
        thresh = dists[support_idx].detach().mean()
        logits = dists - thresh
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        losses.append(loss)

    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)


def prototype_alignment_loss(z_pool, labels, domains):
    """정상 샘플의 domain별 prototype이 z_inv 공간에서 가까워야 함."""
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


device = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_coarse_domain_map(all_domains):
    """500/502/504 → 같은 index (RPM 그룹 기준)."""
    rpm_groups = sorted(set(int(d) // 100 * 100 for d in all_domains))
    rpm_to_idx = {rpm: idx for idx, rpm in enumerate(rpm_groups)}
    return {str(d): rpm_to_idx[int(d) // 100 * 100] for d in all_domains}


def train(args):
    set_seed(args.seed)

    if args.coarse:
        domain_map  = make_coarse_domain_map(args.all_domains)
        num_domains = len(set(domain_map.values()))
    else:
        domain_map  = None
        num_domains = len(args.all_domains)

    datasets = []
    for domain in args.train_domains:
        ds = HUSTDataset(
            root=args.root,
            domain=domain,
            only_normal=False,
            shot=None,
            all_domains=args.all_domains,
            domain_map=domain_map,
        )
        datasets.append(ds)

    dataset = ConcatDataset(datasets)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=False,
    )

    if args.original:
        model = OriginalMaskModel(
            num_classes=args.num_classes,
            num_domains=num_domains,
            encoder_layer='layer3',
        ).to(device)
    else:
        model = MaskDecompositionModel(
            num_classes=args.num_classes,
            num_domains=num_domains
        ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    ce = nn.CrossEntropyLoss()

    print("===================================")
    print("Train Mask Decomposition Model (fine15)")
    print("===================================")
    print("Device        :", device)
    print("Train domains :", args.train_domains)
    print("All domains   :", args.all_domains)
    print("Domain mode   :", f"coarse ({num_domains} groups)" if args.coarse else f"fine ({num_domains} domains)")
    if args.coarse:
        print("Domain map    :", domain_map)
    print("Num domains   :", num_domains)
    print("Epochs        :", args.epochs)
    print("Batch size    :", args.batch_size)
    print("LR            :", args.lr)
    print("Mask weight   :", args.mask_weight)
    print("Domain weight :", args.domain_weight)
    print("DomDisc weight:", args.domain_disc_weight)
    print("DomInv weight :", args.domain_inv_weight)
    print("SupCon weight :", args.supcon_weight)
    print("Proto weight  :", args.proto_weight)
    print("Temperature   :", args.temperature)
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
        total_domain_disc_loss = 0.0
        total_domain_inv_loss = 0.0
        total_mask_loss = 0.0
        total_supcon_loss = 0.0
        total_proto_loss = 0.0
        total_episodic_loss = 0.0

        correct_cls = 0
        correct_dom = 0
        total = 0

        for batch in loader:
            img = batch["image"].to(device)
            fault_type = batch["fault_type"].long().to(device)
            domain = batch["domain"].long().to(device)

            if domain.min() < 0 or domain.max() >= num_domains:
                raise ValueError(
                    f"Domain label must be in [0, {num_domains - 1}], "
                    f"but got min={domain.min().item()}, max={domain.max().item()}."
                )

            binary_label = (fault_type > 0).long()

            out = model(img, alpha=alpha, class_label=binary_label)

            mc = out["mc"]
            md = out["md"]

            if args.original:
                # Notion 6/15 원본: class + domain(GRL) + mask_orth 만 사용
                class_logits = out["class_logits"]
                class_loss   = ce(class_logits, binary_label)
                domain_loss  = ce(out["domain_logits"], domain)
                mask_loss    = torch.mean(mc * md)
                loss = (class_loss
                        + args.domain_weight * domain_loss
                        + effective_mask_weight * mask_loss)
                domain_disc_loss = torch.tensor(0.0, device=device)
                domain_inv_loss  = torch.tensor(0.0, device=device)
                supcon_loss      = torch.tensor(0.0, device=device)
                proto_loss       = torch.tensor(0.0, device=device)
                episodic_loss    = torch.tensor(0.0, device=device)
            else:
                z_c_notd = out["z_c_notd"]
                class_logits    = model.classifier(z_c_notd)
                class_loss      = ce(class_logits, binary_label)
                domain_loss     = ce(out["domain_logits"], domain)
                domain_disc_loss = ce(out["domain_logits_disc"], domain)
                domain_inv_loss = ce(out["domain_logits_inv"], domain)
                mask_loss       = torch.mean(mc * md)

                z_inv_pool = F.adaptive_avg_pool2d(z_c_notd, 1).flatten(1)

                supcon_loss = normal_contrastive_loss(
                    z_inv_pool, binary_label, temperature=args.temperature
                )
                proto_loss = prototype_alignment_loss(z_inv_pool, binary_label, domain)
                episodic_loss = episodic_proto_loss(
                    z_inv_pool, binary_label,
                    domains=domain if args.domain_episodes else None,
                    n_support=4, n_episodes=4
                )

                loss = (class_loss
                        + args.domain_weight * domain_loss
                        + args.domain_disc_weight * domain_disc_loss
                        + args.domain_inv_weight * domain_inv_loss
                        + effective_mask_weight * mask_loss
                        + args.supcon_weight * supcon_loss
                        + args.proto_weight * proto_loss
                        + args.episodic_weight * episodic_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss            += loss.item()
            total_class_loss      += class_loss.item()
            total_domain_loss     += domain_loss.item()
            total_domain_disc_loss += domain_disc_loss.item()
            total_domain_inv_loss += domain_inv_loss.item()
            total_mask_loss       += mask_loss.item()
            total_supcon_loss     += supcon_loss.item()
            total_proto_loss      += proto_loss.item()
            total_episodic_loss   += episodic_loss.item()

            pred_cls = class_logits.argmax(dim=1)
            pred_dom = out["domain_logits"].argmax(dim=1)

            correct_cls += (pred_cls == binary_label).sum().item()
            correct_dom += (pred_dom == domain).sum().item()
            total += fault_type.size(0)

        avg_loss             = total_loss / len(loader)
        avg_class_loss       = total_class_loss / len(loader)
        avg_domain_loss      = total_domain_loss / len(loader)
        avg_domain_disc_loss = total_domain_disc_loss / len(loader)
        avg_domain_inv_loss  = total_domain_inv_loss / len(loader)
        avg_mask_loss        = total_mask_loss / len(loader)
        avg_supcon_loss      = total_supcon_loss / len(loader)
        avg_proto_loss       = total_proto_loss / len(loader)
        avg_episodic_loss    = total_episodic_loss / len(loader)

        cls_acc = correct_cls / total
        dom_acc = correct_dom / total

        scheduler.step()

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"alpha={alpha:.3f} | mask_w={effective_mask_weight:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e} | "
            f"Loss: {avg_loss:.4f} | "
            f"Class: {avg_class_loss:.4f} | "
            f"Dom: {avg_domain_loss:.4f} | "
            f"DomDisc: {avg_domain_disc_loss:.4f} | "
            f"DomInv: {avg_domain_inv_loss:.4f} | "
            f"Mask(mc*md): {avg_mask_loss:.4f} | "
            f"SupCon: {avg_supcon_loss:.4f} | "
            f"Proto: {avg_proto_loss:.4f} | "
            f"Episodic: {avg_episodic_loss:.4f} | "
            f"Cls Acc: {cls_acc:.4f} | "
            f"Dom Acc: {dom_acc:.4f}"
        )

    save_dir = os.path.dirname(args.save_path)
    if save_dir != "":
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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mask_weight", type=float, default=0.1)
    parser.add_argument("--domain_weight", type=float, default=1.0)
    parser.add_argument("--supcon_weight", type=float, default=0.5)
    parser.add_argument("--proto_weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--episodic_weight", type=float, default=1.0)
    parser.add_argument("--domain_disc_weight", type=float, default=1.0)
    parser.add_argument("--domain_inv_weight", type=float, default=0.0)
    parser.add_argument("--domain_episodes", action="store_true",
                        help="Use domain-specific support in episodic loss (matches test-time)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="mask_decomposition.pth")
    parser.add_argument("--coarse", action="store_true",
                        help="500/502/504 → 같은 도메인 index (RPM 그룹 기준)")
    parser.add_argument("--original", action="store_true",
                        help="Notion 6/15 원본: OriginalMaskModel (layer3, z*mc*(1-md), 3-loss만 사용)")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
