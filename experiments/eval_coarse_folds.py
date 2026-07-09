"""
eval_coarse_folds.py  —  코스 도메인 전용 평가 스크립트.

Fine eval과의 차이:
  - 4-fold LOO: 500 / 600 / 700 / 800 RPM 그룹 단위
  - Support: sub-batch(x00, x02, x04)에서 균등 샘플링
    → 예) 4-shot = 500에서 2개 + 502에서 1개 + 504에서 1개
    → prototype이 세 측정 배치를 모두 커버
  - Query: 코스 도메인 전체 (500+502+504 통합)
  - Calib: 학습 RPM 그룹 전체 정상 샘플
    → fine 대비 3× 더 많은 normal → 더 안정적인 LedoitWolf covariance

사용법:
  python experiments/eval_coarse_folds.py \\
    --ckpt_fold_500 resnet50_coarse_fold_500.pth \\
    [--ckpt_fold_600 ...] [--ckpt_fold_700 ...] [--ckpt_fold_800 ...] \\
    [--n_sigma 0.0] [--pca_dim 128]
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

from datasets.hust_image import HUSTDataset
from models.mask_decomposition_model import MaskDecompositionModel

# ── 상수 ──────────────────────────────────────────────────────────────────────
FINE_ROOT   = "processed_gadf_fine_4096"   # sub-batch 분리 샘플링용
COARSE_ROOT = "processed_gadf_coarse_4096" # query / calib용

# (coarse test RPM, [fine sub-batches], [calib coarse domains])
COARSE_FOLDS = [
    ("500", ["500", "502", "504"], ["600", "700", "800"]),
    ("600", ["600", "602", "604"], ["500", "700", "800"]),
    ("700", ["700", "702", "704"], ["500", "600", "800"]),
    ("800", ["800", "802", "804"], ["500", "600", "700"]),
]
ALL_COARSE = ["500", "600", "700", "800"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
device = "cuda" if torch.cuda.is_available() else "cpu"


# ── 유틸 ──────────────────────────────────────────────────────────────────────
def get_features(model, loader, feature_key="z_c_notd"):
    feats, labels = [], []
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            out = model(img)
            z = out[feature_key]
            z = F.adaptive_avg_pool2d(z, 1).view(z.size(0), -1)
            feats.append(z.detach())
            labels.extend(batch["label"].numpy().tolist())
    return torch.cat(feats), np.array(labels)


def mahalanobis_score(x, mean, prec):
    diff = x - mean
    return np.sqrt(np.maximum((diff @ prec) * diff, 0).sum(axis=1))


def make_fine_to_coarse_map():
    """500/502/504→0, 600/602/604→1, 700/702/704→2, 800/802/804→3"""
    fine_all = [f"{rpm+offset}" for rpm in [500,600,700,800] for offset in [0,2,4]]
    coarse_idx = {str(rpm): i for i, rpm in enumerate([500,600,700,800])}
    return {d: coarse_idx[str(int(d)//100*100)] for d in fine_all}


FINE_TO_COARSE = make_fine_to_coarse_map()


def stratified_support(fine_sub_batches, shot, seed):
    """
    fine sub-batches(["500","502","504"])에서 균등하게 shot개 정상 샘플 추출.
    shot=4 → 각 배치에서 최대한 균등하게 (e.g. 2+1+1)
    """
    n = len(fine_sub_batches)
    base, rem = divmod(shot, n)
    per_batch = [base + (1 if i < rem else 0) for i in range(n)]

    all_samples = []
    for sub, k in zip(fine_sub_batches, per_batch):
        try:
            ds = HUSTDataset(
                root=FINE_ROOT, domain=sub, only_normal=True,
                shot=k, transform=transform,
                seed=seed,
                domain_map=FINE_TO_COARSE,
            )
            all_samples.append(ds)
        except RuntimeError:
            pass

    if not all_samples:
        return None
    return ConcatDataset(all_samples)


# ── 메인 평가 ─────────────────────────────────────────────────────────────────
def run(ckpt_map, seeds, shot, n_sigma, pca_dim, num_classes):
    header = (f"{'Test':>6} | {'Base AUROC':>10} | {'Base Acc':>8} | {'Base F1':>7} |"
              f" {'z_inv AUROC':>11} | {'z_inv Acc':>9} | {'z_inv F1':>8}")
    print(header)
    print("-" * len(header))

    all_base, all_zinv = [], []
    all_base_acc, all_base_f1, all_cacc, all_cf1 = [], [], [], []

    for coarse_test, fine_subs, calib_coarse in COARSE_FOLDS:
        ckpt = ckpt_map.get(coarse_test)
        if ckpt is None or not os.path.exists(ckpt):
            print(f"  [{coarse_test}] checkpoint 없음 — 생략")
            continue

        # ── 모델 로드 (num_domains는 checkpoint에서 자동 감지) ──────────────
        sd = torch.load(ckpt, map_location=device)
        num_domains = sd["domain_classifier_inv.fc.weight"].shape[0]
        model = MaskDecompositionModel(
            num_classes=num_classes, num_domains=num_domains
        ).to(device)
        model.load_state_dict(sd, strict=False)
        model.eval()

        # ── Calib: 코스 calib 도메인의 정상 샘플 전체 ─────────────────────
        domain_map = {d: i for i, d in enumerate(ALL_COARSE)}
        calib_sets = []
        for d in calib_coarse:
            try:
                calib_sets.append(HUSTDataset(
                    root=COARSE_ROOT, domain=d, only_normal=True,
                    transform=transform,
                    all_domains=ALL_COARSE, domain_map=domain_map,
                ))
            except RuntimeError:
                pass

        calib_loader = DataLoader(ConcatDataset(calib_sets), batch_size=32, shuffle=False)
        calib_feats, _ = get_features(model, calib_loader)
        calib_np = calib_feats.cpu().numpy()

        pca_model = None
        if pca_dim > 0 and pca_dim < calib_np.shape[1]:
            pca_model = PCA(n_components=pca_dim, random_state=42)
            calib_np = pca_model.fit_transform(calib_np)

        lw = LedoitWolf().fit(calib_np)
        prec_np = lw.precision_

        # ── Query: 코스 test 도메인 전체 ──────────────────────────────────
        try:
            query_ds = HUSTDataset(
                root=COARSE_ROOT, domain=coarse_test, only_normal=False,
                transform=transform,
                all_domains=ALL_COARSE, domain_map=domain_map,
            )
        except RuntimeError:
            print(f"  [{coarse_test}] query 데이터 없음 — 생략")
            continue

        query_loader = DataLoader(query_ds, batch_size=32, shuffle=False)
        query_feats, query_labels = get_features(model, query_loader)
        if len(np.unique(query_labels)) < 2:
            continue
        query_np = query_feats.cpu().numpy()

        base_aurocs, zinv_aurocs = [], []
        base_accs, base_f1s, accs, f1s = [], [], [], []

        for seed in seeds:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

            # ── Support: sub-batch 균등 샘플링 ───────────────────────────
            support_ds = stratified_support(fine_subs, shot, seed)
            if support_ds is None:
                continue

            support_loader = DataLoader(support_ds, batch_size=shot, shuffle=False)
            support_feats, _ = get_features(model, support_loader)
            support_np = support_feats.cpu().numpy()

            if pca_model is not None:
                q_np = pca_model.transform(query_np)
                s_np = pca_model.transform(support_np)
            else:
                q_np, s_np = query_np.copy(), support_np.copy()

            prototype_np = s_np.mean(axis=0)
            zinv_scores = mahalanobis_score(q_np, prototype_np, prec_np)
            sup_scores  = mahalanobis_score(s_np, prototype_np, prec_np)
            calib_scores = mahalanobis_score(calib_np, prototype_np, prec_np)
            threshold = sup_scores.mean() + n_sigma * calib_scores.std()

            zinv_aurocs.append(roc_auc_score(query_labels, zinv_scores))
            preds = (zinv_scores > threshold).astype(int)
            accs.append(accuracy_score(query_labels, preds))
            f1s.append(f1_score(query_labels, preds, zero_division=0))

            # ── Baseline: cosine similarity ────────────────────────────
            from torchvision.models import resnet50
            class _B(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    bb = resnet50(pretrained=True)
                    self.enc = torch.nn.Sequential(
                        bb.conv1, bb.bn1, bb.relu, bb.maxpool,
                        bb.layer1, bb.layer2, bb.layer3,
                    )
                def forward(self, x):
                    return F.adaptive_avg_pool2d(self.enc(x), 1).flatten(1)

            if not hasattr(run, "_baseline"):
                run._baseline = _B().to(device).eval()

            with torch.no_grad():
                qb = torch.cat([run._baseline(b["image"].to(device))
                                for b in query_loader]).detach()
                sb = torch.cat([run._baseline(b["image"].to(device))
                                for b in support_loader]).detach()

            proto_b = F.normalize(sb.mean(0, keepdim=True), dim=1)
            base_scores = (1 - F.cosine_similarity(
                F.normalize(qb, dim=1), proto_b)).cpu().numpy()
            base_aurocs.append(roc_auc_score(query_labels, base_scores))
            preds_b = (base_scores > base_scores.mean()).astype(int)
            base_accs.append(accuracy_score(query_labels, preds_b))
            base_f1s.append(f1_score(query_labels, preds_b, zero_division=0))

        if not zinv_aurocs:
            continue

        ba = np.mean(base_aurocs); bac = np.mean(base_accs); bf = np.mean(base_f1s)
        za = np.mean(zinv_aurocs); zac = np.mean(accs);      zf = np.mean(f1s)

        print(f"  {coarse_test:>5}   | {ba:>10.4f} | {bac:>8.4f} | {bf:>7.4f} |"
              f" {za:>11.4f} | {zac:>9.4f} | {zf:>8.4f}")

        all_base.append(ba); all_zinv.append(za)
        all_base_acc.append(bac); all_base_f1.append(bf)
        all_cacc.append(zac); all_cf1.append(zf)

    if all_zinv:
        print("-" * len(header))
        print(f"  {'Avg':>5}   | {np.mean(all_base):>10.4f} | {np.mean(all_base_acc):>8.4f} |"
              f" {np.mean(all_base_f1):>7.4f} | {np.mean(all_zinv):>11.4f} |"
              f" {np.mean(all_cacc):>9.4f} | {np.mean(all_cf1):>8.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_fold_500", type=str, default=None)
    parser.add_argument("--ckpt_fold_600", type=str, default=None)
    parser.add_argument("--ckpt_fold_700", type=str, default=None)
    parser.add_argument("--ckpt_fold_800", type=str, default=None)
    parser.add_argument("--seeds",    type=int, nargs="+", default=[0,1,2,3,4])
    parser.add_argument("--shot",     type=int, default=4)
    parser.add_argument("--n_sigma",  type=float, default=0.0)
    parser.add_argument("--pca_dim",  type=int,   default=128)
    parser.add_argument("--num_classes", type=int, default=2)
    args = parser.parse_args()

    ckpt_map = {
        "500": args.ckpt_fold_500,
        "600": args.ckpt_fold_600,
        "700": args.ckpt_fold_700,
        "800": args.ckpt_fold_800,
    }

    print(f"Device : {device}")
    print(f"Shot   : {args.shot} (sub-batch 균등 샘플링)")
    print(f"Seeds  : {args.seeds}")
    print()
    run(ckpt_map, args.seeds, args.shot, args.n_sigma, args.pca_dim, args.num_classes)


if __name__ == "__main__":
    main()
