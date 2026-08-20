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
from models.channel_mask_model import ChannelMaskModel


def normal_contrastive_loss(z, labels, temperature=0.07):
    z = F.normalize(z, dim=1)
    sim = z @ z.T / temperature
    B = len(labels)
    eye = torch.eye(B, device=labels.device)
    is_normal = (labels == 0)
    mask_pos = (is_normal.unsqueeze(1) & is_normal.unsqueeze(0)).float() * (1 - eye)
    mask_denom = 1 - eye
    logits_max, _ = sim.max(dim=1, keepdim=True)
    sim_exp = torch.exp(sim - logits_max.detach())
    denom = (sim_exp * mask_denom).sum(dim=1, keepdim=True)
    log_prob = sim - logits_max.detach() - torch.log(denom + 1e-8)
    n_pos = mask_pos.sum(dim=1)
    valid = is_normal & (n_pos > 0)
    loss = -(mask_pos * log_prob).sum(dim=1) / (n_pos + 1e-8)
    return loss[valid].mean() if valid.any() else torch.tensor(0.0, device=z.device)


def prototype_alignment_loss(z, labels, domains):
    z = F.normalize(z, dim=1)
    normals = z[labels == 0]
    dom_ids = domains[labels == 0]
    unique_doms = dom_ids.unique()
    if len(unique_doms) < 2:
        return torch.tensor(0.0, device=z.device)
    protos = [normals[dom_ids == d].mean(0) for d in unique_doms if (dom_ids == d).sum() > 0]
    protos = torch.stack(protos)
    return ((protos - protos.mean(0)) ** 2).sum(dim=1).mean()


def episodic_proto_loss(z, labels, domains=None, n_support=4, n_episodes=4):
    device = z.device
    if not (labels == 1).any():
        return torch.tensor(0.0, device=device)
    z_norm = F.normalize(z, dim=1)
    losses = []
    for _ in range(n_episodes):
        if domains is not None:
            valid_doms = [d for d in domains.unique()
                          if ((domains == d) & (labels == 0)).sum() >= n_support]
            if not valid_doms:
                continue
            dom = valid_doms[torch.randint(len(valid_doms), (1,)).item()]
            idx = ((domains == dom) & (labels == 0)).nonzero(as_tuple=True)[0]
        else:
            idx = (labels == 0).nonzero(as_tuple=True)[0]
            if len(idx) < n_support + 1:
                continue
        perm = torch.randperm(len(idx), device=device)[:n_support]
        support_idx = idx[perm]
        proto = F.normalize(z_norm[support_idx].mean(0).detach(), dim=0)
        dists = ((z_norm - proto) ** 2).sum(dim=1).sqrt()
        thresh = dists[support_idx].detach().mean()
        loss = F.binary_cross_entropy_with_logits(dists - thresh, labels.float())
        losses.append(loss)
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)


device = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_epoch_md(model, loader):
    """
    전체 학습 데이터를 no_grad로 한 번 순회하여 md를 계산.
    배치당 샘플 부족으로 인한 노이즈를 제거.
    """
    model.eval()
    all_zp, all_labels, all_domains = [], [], []
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            ft  = batch["fault_type"].long().to(device)
            dom = batch["domain"].long().to(device)
            z   = model.feature_extractor(img)
            zp  = F.adaptive_avg_pool2d(z, 1).flatten(1)
            all_zp.append(zp)
            all_labels.append((ft > 0).long())
            all_domains.append(dom)
    model.train()
    return ChannelMaskModel.compute_md(
        torch.cat(all_zp),
        torch.cat(all_labels),
        torch.cat(all_domains),
    )


def train(args):
    set_seed(args.seed)

    datasets = []
    for domain in args.train_domains:
        datasets.append(HUSTDataset(
            root=args.root, domain=domain, only_normal=False,
            shot=None, all_domains=args.all_domains,
        ))
    dataset = ConcatDataset(datasets)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, drop_last=True)

    num_domains = len(args.all_domains)
    model = ChannelMaskModel(num_classes=args.num_classes, num_domains=num_domains).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    ce = nn.CrossEntropyLoss()

    print("=" * 50)
    print("ChannelMaskModel (channel-wise domain variance mask)")
    print("=" * 50)
    print(f"Device        : {device}")
    print(f"Train domains : {args.train_domains}")
    print(f"Epochs        : {args.epochs}  |  Batch : {args.batch_size}  |  LR : {args.lr}")
    print(f"Warmup        : {args.warmup_epochs} epochs (md disabled)")
    print(f"Num domains   : {num_domains}")
    print(f"Weights  class={args.class_weight} domain={args.domain_weight} "
          f"supcon={args.supcon_weight} proto={args.proto_weight} episodic={args.episodic_weight}")
    print(f"Seed : {args.seed}  |  Save : {args.save_path}")
    print(f"Total samples : {len(dataset)}")
    print("=" * 50)

    md = None  # epoch 시작 전 초기화

    for epoch in range(args.epochs):
        model.train()

        p     = epoch / max(1, args.epochs - 1)
        alpha = float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)
        use_md = (epoch >= args.warmup_epochs)

        # ── epoch 단위 md 계산 (전체 데이터, no_grad) ────────────────────
        if use_md:
            md = compute_epoch_md(model, loader)
            md_nonzero_ratio = (md > 0.1).float().mean().item()
            print(f"  [md] nonzero(>0.1)={md_nonzero_ratio:.3f}  "
                  f"mean={md.mean():.4f}  max={md.max():.4f}")

        totals = dict(loss=0, cls=0, dom=0, supcon=0, proto=0, episodic=0,
                      mc_mean=0, mc_md_overlap=0)
        correct_cls, correct_dom, total = 0, 0, 0

        for batch in loader:
            img          = batch["image"].to(device)
            fault_type   = batch["fault_type"].long().to(device)
            domain       = batch["domain"].long().to(device)
            binary_label = (fault_type > 0).long()

            # ── forward ────────────────────────────────────────────────────
            out = model(img, class_label=binary_label, md=md, alpha=alpha)
            z_inv  = out["z_inv"]
            mc     = out["mc"]

            class_logits  = model.classifier(z_inv)
            class_loss    = ce(class_logits, binary_label)
            domain_loss   = ce(out["domain_logits"], domain)

            supcon_loss   = normal_contrastive_loss(z_inv, binary_label, args.temperature)
            proto_loss    = prototype_alignment_loss(z_inv, binary_label, domain)
            episodic_loss = episodic_proto_loss(
                z_inv, binary_label,
                domains=domain if args.domain_episodes else None,
            )

            loss = (args.class_weight   * class_loss
                    + args.domain_weight  * domain_loss
                    + args.supcon_weight  * supcon_loss
                    + args.proto_weight   * proto_loss
                    + args.episodic_weight * episodic_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            totals["loss"]     += loss.item()
            totals["cls"]      += class_loss.item()
            totals["dom"]      += domain_loss.item()
            totals["supcon"]   += supcon_loss.item()
            totals["proto"]    += proto_loss.item()
            totals["episodic"] += episodic_loss.item()
            totals["mc_mean"]  += mc.mean().item()
            if md is not None:
                totals["mc_md_overlap"] += (mc * md.unsqueeze(0)).mean().item()

            correct_cls += (class_logits.argmax(1) == binary_label).sum().item()
            correct_dom += (out["domain_logits"].argmax(1) == domain).sum().item()
            total       += binary_label.size(0)

        n = len(loader)
        md_str = f"mc*md={totals['mc_md_overlap']/n:.4f} | " if use_md else "md=OFF | "

        print(
            f"Epoch [{epoch+1:2d}/{args.epochs}] alpha={alpha:.2f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e} | "
            f"Loss={totals['loss']/n:.4f} | "
            f"Cls={totals['cls']/n:.4f} | "
            f"Dom={totals['dom']/n:.4f} | "
            f"SupCon={totals['supcon']/n:.4f} | "
            f"Ep={totals['episodic']/n:.4f} | "
            f"{md_str}"
            f"ClsAcc={correct_cls/total:.4f} | DomAcc={correct_dom/total:.4f}"
        )
        scheduler.step()

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"\nSaved → {args.save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",         type=str,   default="processed_gadf_fine_4096")
    parser.add_argument("--train_domains",type=str,   nargs="+", required=True)
    parser.add_argument("--all_domains",  type=str,   nargs="+",
                        default=["500","502","504",
                                 "600","602","604","700","702","704","800","802","804"])
    parser.add_argument("--epochs",       type=int,   default=20)
    parser.add_argument("--batch_size",   type=int,   default=16)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--num_classes",  type=int,   default=2)
    parser.add_argument("--warmup_epochs",type=int,   default=3,
                        help="md 비활성화 초기 epoch 수 (encoder 안정화 후 md 적용)")
    parser.add_argument("--class_weight", type=float, default=1.0)
    parser.add_argument("--domain_weight",type=float, default=1.0)
    parser.add_argument("--supcon_weight",type=float, default=0.5)
    parser.add_argument("--proto_weight", type=float, default=0.1)
    parser.add_argument("--episodic_weight", type=float, default=1.0)
    parser.add_argument("--temperature",  type=float, default=0.07)
    parser.add_argument("--domain_episodes", action="store_true")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--save_path",    type=str,   default="channel_mask_fold_500.pth")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
