import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.metrics import roc_auc_score

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


def max_cosine_anomaly_score(query_feat, memory_bank):
    query_feat = F.normalize(query_feat, dim=1)
    memory_bank = F.normalize(memory_bank, dim=1)

    sim = torch.matmul(query_feat, memory_bank.t())
    max_sim, _ = torch.max(sim, dim=1)

    anomaly_score = 1.0 - max_sim
    return anomaly_score


def extract_components(model, loader, device):
    zc_list = []
    zd_list = []
    labels = []

    for batch in loader:
        img = batch["image"].to(device)
        label = batch["label"]

        out = model(img)

        z_c_notd = flatten_feature(out["z_c_notd"])
        z_notc_d = flatten_feature(out["z_notc_d"])

        zc_list.append(z_c_notd.detach())
        zd_list.append(z_notc_d.detach())
        labels.extend(label.cpu().numpy().tolist())

    zc = torch.cat(zc_list, dim=0)
    zd = torch.cat(zd_list, dim=0)
    labels = np.array(labels)

    return zc, zd, labels


def build_mixed_memory(
    zc_support,
    zd_pool,
    mix_num=5,
    alpha=0.5,
    mix_mode="centered",
):
    memory_list = [zc_support]

    n_support = zc_support.size(0)
    n_pool = zd_pool.size(0)

    zd_mean = zd_pool.mean(dim=0, keepdim=True)

    for _ in range(mix_num):
        idx = torch.randint(0, n_pool, (n_support,), device=zc_support.device)
        sampled_zd = zd_pool[idx]

        if mix_mode == "add":
            z_mix = zc_support + alpha * sampled_zd
            memory_list.append(z_mix)

        elif mix_mode == "centered":
            # domain-specific 성분의 평균을 제거해서 방향 변화만 주는 방식
            domain_shift = sampled_zd - zd_mean
            z_mix = zc_support + alpha * domain_shift
            memory_list.append(z_mix)

        elif mix_mode == "plus_minus":
            # 같은 domain shift를 + / - 양방향으로 추가
            domain_shift = sampled_zd - zd_mean
            z_mix_plus = zc_support + alpha * domain_shift
            z_mix_minus = zc_support - alpha * domain_shift

            memory_list.append(z_mix_plus)
            memory_list.append(z_mix_minus)

        else:
            raise ValueError(f"Unknown mix_mode: {mix_mode}")

    memory_bank = torch.cat(memory_list, dim=0)
    return memory_bank.detach()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--support_domain", type=str, required=True)
    parser.add_argument("--query_domain", type=str, required=True)
    parser.add_argument("--mix_domains", type=str, nargs="+", required=True)

    parser.add_argument("--shot", type=int, default=4)
    parser.add_argument("--mix_num", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--mix_mode", type=str, default="centered",
                        choices=["add", "centered", "plus_minus"])

    parser.add_argument("--ckpt", type=str, default="mask_decomposition.pth")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("===================================")
    print("Simplified XDomainMix FSAD")
    print("===================================")
    print("Device        :", device)
    print("Support domain:", args.support_domain)
    print("Query domain  :", args.query_domain)
    print("Mix domains   :", args.mix_domains)
    print("Shot          :", args.shot)
    print("Mix num       :", args.mix_num)
    print("Alpha         :", args.alpha)
    print("Mix mode      :", args.mix_mode)
    print("Seed          :", args.seed)
    print("Checkpoint    :", args.ckpt)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    model = MaskDecompositionModel().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    support_set = HUSTDataset(
        root="processed",
        domain=args.support_domain,
        only_normal=True,
        shot=args.shot,
        transform=transform,
        seed=args.seed,
    )

    support_loader = DataLoader(
        support_set,
        batch_size=args.shot,
        shuffle=False,
    )

    mix_sets = []

    for d in args.mix_domains:
        mix_sets.append(
            HUSTDataset(
                root="processed",
                domain=d,
                only_normal=True,
                shot=None,
                transform=transform,
            )
        )

    mix_dataset = ConcatDataset(mix_sets)

    mix_loader = DataLoader(
        mix_dataset,
        batch_size=16,
        shuffle=False,
    )

    zc_support, _, _ = extract_components(model, support_loader, device)
    _, zd_pool, _ = extract_components(model, mix_loader, device)

    zc_support = zc_support.to(device)
    zd_pool = zd_pool.to(device)

    memory_bank = build_mixed_memory(
        zc_support=zc_support,
        zd_pool=zd_pool,
        mix_num=args.mix_num,
        alpha=args.alpha,
        mix_mode=args.mix_mode,
    )

    print("Memory bank shape:", memory_bank.shape)

    query_set = HUSTDataset(
        root="processed",
        domain=args.query_domain,
        only_normal=False,
        shot=None,
        transform=transform,
    )

    query_loader = DataLoader(
        query_set,
        batch_size=16,
        shuffle=False,
    )

    scores = []
    labels = []

    for batch in query_loader:
        img = batch["image"].to(device)
        label = batch["label"].cpu().numpy()

        out = model(img)
        z_query = flatten_feature(out["z_c_notd"])

        anomaly_score = max_cosine_anomaly_score(z_query, memory_bank)

        scores.extend(anomaly_score.detach().cpu().numpy().tolist())
        labels.extend(label.tolist())

    auroc = roc_auc_score(np.array(labels), np.array(scores))

    print("===================================")
    print(f"AUROC: {auroc:.4f}")
    print("===================================")


if __name__ == "__main__":
    main()