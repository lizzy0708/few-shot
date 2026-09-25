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
from models.gated_mask_model import GatedMaskModel
from utils.mmd import multi_domain_mmd_loss
from utils.domain_balanced_sampler import DomainBalancedBatchSampler

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

    if args.shuffle_domain_labels:
        # 2026-09-25 control experiment: does the Acc/F1 gain from z_inv come from
        # genuine domain-adversarial learning (GRL removing real domain info), or
        # from the class_gate/mask structure itself acting as a regularizer
        # independent of whether the domain labels it's trained against are real?
        # Fixed (not per-epoch) label permutation across all training samples --
        # each sample keeps a consistent but WRONG domain label for the whole run,
        # so GRL/domain_classifier learn to fit noise instead of real domain
        # structure, while the gate/mask architecture and class_loss path are
        # completely unchanged. Only `datasets[i].samples` (in-memory, this
        # training run only) is mutated -- does not touch files on disk or any
        # other script's view of HUSTDataset.
        rng = np.random.RandomState(args.seed)
        all_true_domains = [s[2] for ds in datasets for s in ds.samples]
        shuffled_domains = rng.permutation(all_true_domains)
        ptr = 0
        for ds in datasets:
            new_samples = []
            for s in ds.samples:
                path, label, _domain_idx, domain_name, fault_type, rpm_label, batch_label = s
                new_samples.append((path, label, int(shuffled_domains[ptr]), domain_name,
                                     fault_type, rpm_label, batch_label))
                ptr += 1
            ds.samples = new_samples

    dataset = ConcatDataset(datasets)

    if args.balanced_batch:
        # 2026-09-24 (utils/domain_balanced_sampler.py): isolate whether the
        # mmd_weight sweep's failure was caused by noisy per-batch MMD estimates
        # (small/unbalanced per-domain batch counts), independent of mmd_weight itself.
        sampler = DomainBalancedBatchSampler(
            dataset, batch_size=args.batch_size,
            num_domains=len(args.train_domains), seed=args.seed,
        )
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=2)
    else:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=2, drop_last=False)

    model = GatedMaskModel(num_classes=args.num_classes, num_domains=num_domains,
                            encoder_layer=args.encoder_layer,
                            use_domain_gate=(not args.no_domain_gate),
                            domain_gate_grl=args.domain_gate_grl,
                            use_gate_orth=args.use_gate_orth).to(device)

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
    print("Use gate orth :", args.use_gate_orth)
    print("Mask weight   :", args.mask_weight)
    print("Domain weight :", args.domain_weight)
    print("DomDisc weight:", args.domain_disc_weight)
    print("MMD weight    :", args.mmd_weight)
    print("Balanced batch:", args.balanced_batch)
    print("Shuffled dom labels:", args.shuffle_domain_labels)
    print("Seed          :", args.seed)
    print("Save path     :", args.save_path)
    print("Total samples :", len(dataset))
    print("===================================")

    for epoch in range(args.epochs):
        model.train()

        p = epoch / max(1, args.epochs - 1)
        alpha = float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)

        total_loss = total_class = total_dom = total_domdisc = total_mask = total_mmd = 0.0
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

            # 2026-09-24 MMD attempt (see utils/mmd.py docstring): discriminator-free
            # alignment of z_inv's per-domain distributions, added alongside the
            # existing GRL domain_loss (not a replacement for it). Computed on the same
            # GAP-pooled z_inv space that eval-time Mahalanobis scoring and the
            # independent domain-invariance probe both use.
            z_inv_pooled = F.adaptive_avg_pool2d(out["z_inv"], 1).flatten(1)
            mmd_loss = multi_domain_mmd_loss(z_inv_pooled, domain)

            loss = (class_loss
                    + args.domain_weight * domain_loss
                    + args.domain_disc_weight * domain_disc_loss
                    + args.mask_weight * mask_loss
                    + args.mmd_weight * mmd_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_class += class_loss.item()
            total_dom += domain_loss.item()
            total_domdisc += domain_disc_loss.item()
            total_mask += mask_loss.item()
            total_mmd += mmd_loss.item()

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
              f"Mask(gate_c*gate_d): {total_mask/n:.4f} | MMD: {total_mmd/n:.4f} | "
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
    parser.add_argument("--mmd_weight", type=float, default=0.0,
                        help="2026-09-24 MMD domain-alignment attempt (utils/mmd.py): weight on "
                             "multi-bandwidth RBF-kernel MMD^2 between calib-domain z_inv "
                             "distributions in each batch, added alongside (not replacing) the "
                             "existing GRL domain_loss. Default 0.0 = fully off, reproduces prior "
                             "gated_nodg behavior exactly (mmd_loss computed but not backpropped "
                             "through the weighted sum when weight=0). Discriminator-free "
                             "alternative to the two failed GRL-based attempts (Gram-Schmidt "
                             "orthogonalization, raising domain_weight) -- see "
                             "docs/exec-plans/completed/2026-09-gated-nodg-mmd.md for the measured "
                             "starting value (10.0) and rationale.")
    parser.add_argument("--balanced_batch", action="store_true",
                        help="2026-09-24 (utils/domain_balanced_sampler.py): use a "
                             "DomainBalancedBatchSampler so every batch has exactly "
                             "batch_size // len(train_domains) samples from each train "
                             "domain, instead of plain shuffle=True (which gives each "
                             "domain only that many *on average*, with high variance). "
                             "Isolates whether the mmd_weight sweep's failure was caused "
                             "by noisy per-batch MMD estimates. Requires batch_size "
                             "divisible by len(train_domains).")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--encoder_layer", type=str, default="layer3", choices=["layer3", "layer4"])
    parser.add_argument("--no_domain_gate", action="store_true",
                        help="z_inv = z*class_gate만 사용(도메인 게이트 제거). mask loss도 자동 제외됨. "
                             "domain_classifier/GRL을 통한 encoder invariance 학습은 계속 유지.")
    parser.add_argument("--domain_gate_grl", action="store_true",
                        help="domain_gate_net 학습을 detach+직접분류 대신 GRL(adversarial)로 전환. "
                             "encoder는 여전히 detach로 보호(z_pool.detach()*domain_gate에 GRL).")
    parser.add_argument("--use_gate_orth", action="store_true",
                        help="Stage 2 (2026-09-19): class_gate를 domain_classifier.weight의 행 "
                             "부분공간에 대해 Gram-Schmidt 직교화한 뒤 z_inv를 구성 (fine15의 "
                             "mc_orth와 유사하나 동일하지 않음 — models/gated_mask_model.py의 "
                             "_orthogonalize_gate_against_domain 참고). class_logits 경로는 "
                             "영향받지 않음 — 기존 nodg와 학습 dynamics 동일, z_inv만 변경.")
    parser.add_argument("--shuffle_domain_labels", action="store_true",
                        help="2026-09-25 control experiment: train GRL/domain_classifier against "
                             "a fixed random permutation of the true domain labels (class_loss "
                             "and the gate/mask architecture are unaffected) -- isolates whether "
                             "z_inv's Acc/F1 gain over raw z comes from genuine domain-adversarial "
                             "learning or from the class_gate structure itself acting as a "
                             "regularizer independent of label truthfulness. Default off, fully "
                             "backward-compatible.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
