"""
eval_patch_folds.py — z_inv patch memory 기반 few-shot 이상탐지 (파일럿).

현행 GAP 방식과의 차이:
  - z_inv [B,2048,7,7]에서 GAP 대신 49개 patch 벡터를 유지
  - support 4장 → patch memory (4×49=196개)
  - query patch별 최근접 memory 거리 → top-k 평균 = image score
  - threshold: support LOO (각 장을 나머지 3장 memory로 스코어) 평균 — 정상 4개만 사용

파일럿 범위: --test_fold로 단일 fold, {mahal, cosine} × {top5, max} + hybrid 비교.
사전 체크: z_inv patch norm 분포 (mask sparsity 확인).

사용:
  python experiments/eval_patch_folds.py \
    --ckpt checkpoints/fine15_fold500.pth --test_fold 500 --seeds 0 1 2 3 4
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
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

from datasets.hust_image import HUSTDataset
from models.mask_decomposition_model import MaskDecompositionModel
from experiments.eval_all_folds import transform, FINE_ALL

device = "cuda" if torch.cuda.is_available() else "cpu"

FINE_FOLDS = {
    "500": (["500", "502", "504"],
            ["400", "402", "404", "600", "602", "604", "700", "702", "704", "800", "802", "804"]),
    "600": (["600", "602", "604"],
            ["400", "402", "404", "500", "502", "504", "700", "702", "704", "800", "802", "804"]),
    "700": (["700", "702", "704"],
            ["400", "402", "404", "500", "502", "504", "600", "602", "604", "800", "802", "804"]),
    "800": (["800", "802", "804"],
            ["400", "402", "404", "500", "502", "504", "600", "602", "604", "700", "702", "704"]),
}


@torch.no_grad()
def get_patch_features(model, loader):
    """z_inv를 GAP 없이 patch 단위로 반환: [N, 49, 2048] + 이미지 단위 GAP [N, 2048]."""
    patches, gaps, labels = [], [], []
    for batch in loader:
        img = batch["image"].to(device)
        out = model(img)
        z = out["z_c_notd"]                      # [B, C, 7, 7]
        B, C, H, W = z.shape
        p = z.view(B, C, H * W).permute(0, 2, 1)  # [B, 49, C]
        patches.append(p.detach().cpu())
        gaps.append(z.mean(dim=(2, 3)).detach().cpu())
        labels.extend(batch["label"].numpy().tolist())
    return torch.cat(patches), torch.cat(gaps), np.array(labels)


def patch_norm_report(patches: torch.Tensor, name: str):
    """mask sparsity 사전 체크: patch L2 norm 분포."""
    norms = patches.norm(dim=-1).flatten().numpy()
    q = np.percentile(norms, [0, 10, 25, 50, 75, 90, 100])
    near_zero = (norms < 1e-4).mean()
    print(f"[사전체크:{name}] patch norm 분위수 0/10/25/50/75/90/100 = "
          + "/".join(f"{v:.3g}" for v in q) + f" | ~0 비율 {near_zero:.1%}")
    return near_zero


def score_patches(query_p, memory_p, mode, pca=None, prec=None, topk=5):
    """query_p [Nq,49,C], memory_p [M,C] → 이미지 점수 [Nq]."""
    Nq, P, C = query_p.shape
    q = query_p.reshape(-1, C).numpy()
    m = memory_p.numpy()
    if mode == "mahal":
        q = pca.transform(q) if pca is not None else q
        m = pca.transform(m) if pca is not None else m
        # pairwise Mahalanobis^2: (x-y) P (x-y)^T
        qP = q @ prec
        mP = m @ prec
        d2 = (qP * q).sum(1)[:, None] + (mP * m).sum(1)[None, :] - 2 * (qP @ m.T)
        d = np.sqrt(np.clip(d2, 0, None))
        patch_scores = d.min(axis=1)
    else:  # cosine
        qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
        mn = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-8)
        sim = qn @ mn.T
        patch_scores = 1.0 - sim.max(axis=1)
    patch_scores = patch_scores.reshape(Nq, P)
    top = np.sort(patch_scores, axis=1)[:, -topk:]
    return top.mean(axis=1)


def support_loo_threshold(support_p, mode, pca=None, prec=None, topk=5):
    """support 4장 leave-one-out 스코어 평균 = threshold (n_sigma=0 철학)."""
    n = support_p.shape[0]
    scores = []
    for i in range(n):
        mem = torch.cat([support_p[j] for j in range(n) if j != i])  # [(n-1)*49, C]
        s = score_patches(support_p[i:i+1], mem, mode, pca, prec, topk)
        scores.append(s[0])
    return float(np.mean(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--test_fold", type=str, required=True, choices=list(FINE_FOLDS))
    ap.add_argument("--root", type=str, default="processed_gadf_fine_4096")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--shot", type=int, default=4)
    ap.add_argument("--pca_dim", type=int, default=128)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    test_domains, calib_domains = FINE_FOLDS[args.test_fold]

    sd = torch.load(args.ckpt, map_location=device)
    num_domains = sd["domain_classifier.fc.weight"].shape[0] if "domain_classifier.fc.weight" in sd else 15
    model = MaskDecompositionModel(num_classes=2, num_domains=num_domains, encoder_layer="layer4").to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    # ---- calib: 정상 patch로 공분산/PCA 적합 ----
    calib_sets = [HUSTDataset(root=args.root, domain=d, only_normal=True,
                              transform=transform, all_domains=FINE_ALL)
                  for d in calib_domains]
    calib_loader = DataLoader(ConcatDataset(calib_sets), batch_size=32, shuffle=False)
    calib_p, calib_gap, _ = get_patch_features(model, calib_loader)

    near_zero = patch_norm_report(calib_p, "calib")
    if near_zero > 0.5:
        print("경고: patch 절반 이상이 0 — mask sparsity로 설계 무효 가능성. 중단 검토.")

    calib_flat = calib_p.reshape(-1, calib_p.shape[-1]).numpy()
    # patch 수가 많으므로 서브샘플로 적합 (메모리/속도)
    idx = np.random.RandomState(42).choice(len(calib_flat), min(50000, len(calib_flat)), replace=False)
    pca = PCA(n_components=args.pca_dim, random_state=42).fit(calib_flat[idx])
    lw = LedoitWolf().fit(pca.transform(calib_flat[idx]))
    prec = lw.precision_

    # ---- GAP 경로 준비 (hybrid용): 기존 프로토콜과 동일 (PCA 128 + LedoitWolf) ----
    calib_gap_np = calib_gap.numpy()
    gap_pca = PCA(n_components=args.pca_dim, random_state=42).fit(calib_gap_np)
    gap_lw = LedoitWolf().fit(gap_pca.transform(calib_gap_np))
    gap_prec = gap_lw.precision_

    def gap_scores(feats_np, proto_np):
        d = gap_pca.transform(feats_np) - gap_pca.transform(proto_np[None])[0]
        return np.sqrt(np.clip((d @ gap_prec * d).sum(1), 0, None))

    # ---- 결과 집계 ----
    conds = [("mahal", args.topk), ("mahal", 1), ("cosine", args.topk), ("cosine", 1),
             ("hybrid", args.topk)]
    agg = {c: {"auroc": [], "acc": [], "f1": []} for c in conds}

    for d in test_domains:
        query_ds = HUSTDataset(root=args.root, domain=d, only_normal=False,
                               transform=transform, all_domains=FINE_ALL)
        support_pool = HUSTDataset(root=args.root, domain=d, only_normal=True,
                                   transform=transform, all_domains=FINE_ALL)
        query_loader = DataLoader(query_ds, batch_size=32, shuffle=False)
        query_p, query_gap, query_labels = get_patch_features(model, query_loader)

        for seed in args.seeds:
            rng = random.Random(seed)
            sup_idx = rng.sample(range(len(support_pool)), args.shot)
            sup_imgs = torch.stack([support_pool[i]["image"] for i in sup_idx]).to(device)
            with torch.no_grad():
                z = model(sup_imgs)["z_c_notd"]
            sup_p = z.view(z.size(0), z.size(1), -1).permute(0, 2, 1).detach().cpu()  # [4,49,C]

            # query에서 support 중복 제거 불필요 (query는 정상+이상 전체, 관례 유지)
            sup_gap = z.mean(dim=(2, 3)).detach().cpu().numpy()
            proto_gap = sup_gap.mean(axis=0)

            for mode, topk in conds:
                mem = sup_p.reshape(-1, sup_p.shape[-1])
                if mode == "hybrid":
                    # 성분 1: GAP Mahalanobis (기존 프로토콜)
                    q_gap = gap_scores(query_gap.numpy(), proto_gap)
                    c_gap = gap_scores(calib_gap_np, proto_gap)
                    s_gap = gap_scores(sup_gap, proto_gap)
                    # 성분 2: patch cosine-top5
                    q_pat = score_patches(query_p, mem, "cosine", topk=topk)
                    c_pat = score_patches(calib_p, mem, "cosine", topk=topk)
                    s_pat_thr = support_loo_threshold(sup_p, "cosine", topk=topk)
                    # calib 정상 분포로 z-정규화 후 평균
                    zq = ((q_gap - c_gap.mean()) / (c_gap.std() + 1e-8)
                          + (q_pat - c_pat.mean()) / (c_pat.std() + 1e-8)) / 2
                    thr = ((s_gap.mean() - c_gap.mean()) / (c_gap.std() + 1e-8)
                           + (s_pat_thr - c_pat.mean()) / (c_pat.std() + 1e-8)) / 2
                    scores = zq
                else:
                    kw = dict(pca=pca, prec=prec) if mode == "mahal" else dict()
                    scores = score_patches(query_p, mem, mode, topk=topk, **kw)
                    thr = support_loo_threshold(sup_p, mode, topk=topk, **kw)
                preds = (scores > thr).astype(int)
                agg[(mode, topk)]["auroc"].append(roc_auc_score(query_labels, scores))
                agg[(mode, topk)]["acc"].append(accuracy_score(query_labels, preds))
                agg[(mode, topk)]["f1"].append(f1_score(query_labels, preds))

    print(f"\n=== fold_{args.test_fold} patch scoring (seeds={args.seeds}, "
          f"{len(test_domains)} sub-domains) ===")
    for (mode, topk), m in agg.items():
        label = f"{mode}-top{topk}"
        print(f"[{label:12s}] AUROC {np.mean(m['auroc']):.4f} | "
              f"Acc {np.mean(m['acc']):.4f} | F1 {np.mean(m['f1']):.4f}")


if __name__ == "__main__":
    main()
