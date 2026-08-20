"""
eval_dc_folds.py — Distribution Calibration (Free Lunch, ICLR'21 변형) 평가 파일럿.

4-shot support의 빈약한 프로토타입 통계를 calib 도메인 통계로 보정:
  1) Tukey 변환 (x^λ, λ=0.5) — 특징 가우시안화 (z_inv는 비음수라 적용 가능)
  2) 프로토타입 보정 — support 평균을 가장 가까운 k개 calib 도메인 정상 평균과 결합
     proto_cal = (1-β)·support_mean + β·mean(nearest_k_calib_means)

재학습 없음 (fine15 체크포인트 사용). threshold 프로토콜 유지 (support 점수 평균).

사용:
  python scripts/eval/eval_dc_folds.py --ckpt checkpoints/fine15_fold500.pth --test_fold 500
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

from datasets.hust_image import HUSTDataset
from models.mask_decomposition_model import MaskDecompositionModel
from scripts.eval.eval_all_folds import transform, FINE_ALL, get_features
from scripts.eval.eval_patch_folds import FINE_FOLDS

device = "cuda" if torch.cuda.is_available() else "cpu"


def tukey(x: np.ndarray, lam: float = 0.5) -> np.ndarray:
    return np.power(np.clip(x, 0, None) + 1e-6, lam)


def mahal(feats, proto, prec):
    d = feats - proto
    return np.sqrt(np.clip((d @ prec * d).sum(1), 0, None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--test_fold", type=str, required=True, choices=list(FINE_FOLDS))
    ap.add_argument("--root", type=str, default="processed_gadf_fine_4096")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--shot", type=int, default=4)
    ap.add_argument("--pca_dim", type=int, default=128)
    ap.add_argument("--k_near", type=int, default=2)
    args = ap.parse_args()

    test_domains, calib_domains = FINE_FOLDS[args.test_fold]

    sd = torch.load(args.ckpt, map_location=device)
    model = MaskDecompositionModel(num_classes=2, num_domains=15, encoder_layer="layer4").to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    # calib 정상: 도메인별로 분리 로드 (도메인별 평균 필요)
    calib_feats_by_dom = {}
    for d in calib_domains:
        ds = HUSTDataset(root=args.root, domain=d, only_normal=True,
                         transform=transform, all_domains=FINE_ALL)
        feats, _ = get_features(model, DataLoader(ds, batch_size=32, shuffle=False))
        calib_feats_by_dom[d] = feats.cpu().numpy()

    calib_all = np.concatenate(list(calib_feats_by_dom.values()))

    # (조건 라벨, tukey 사용, beta)
    CONDS = [("baseline", False, 0.0), ("tukey", True, 0.0),
             ("dc_b0.2", False, 0.2), ("dc_b0.5", False, 0.5),
             ("tukey+dc_b0.2", True, 0.2), ("tukey+dc_b0.5", True, 0.5)]

    # 변환별 PCA/공분산/도메인 평균 사전 적합
    spaces = {}
    for use_tukey in (False, True):
        c = tukey(calib_all) if use_tukey else calib_all
        pca = PCA(n_components=args.pca_dim, random_state=42).fit(c)
        cp = pca.transform(c)
        prec = LedoitWolf().fit(cp).precision_
        dom_means = {d: pca.transform(tukey(f) if use_tukey else f).mean(axis=0)
                     for d, f in calib_feats_by_dom.items()}
        spaces[use_tukey] = (pca, prec, dom_means)

    agg = {c[0]: {"auroc": [], "acc": [], "f1": []} for c in CONDS}

    for d in test_domains:
        query_ds = HUSTDataset(root=args.root, domain=d, only_normal=False,
                               transform=transform, all_domains=FINE_ALL)
        support_pool = HUSTDataset(root=args.root, domain=d, only_normal=True,
                                   transform=transform, all_domains=FINE_ALL)
        q_feats, q_labels = get_features(model, DataLoader(query_ds, batch_size=32, shuffle=False))
        q_np = q_feats.cpu().numpy()

        for seed in args.seeds:
            rng = random.Random(seed)
            sup_idx = rng.sample(range(len(support_pool)), args.shot)
            sup_imgs = torch.stack([support_pool[i]["image"] for i in sup_idx]).to(device)
            out = model(sup_imgs)
            z = out["z_c_notd"]
            sup_np = torch.nn.functional.adaptive_avg_pool2d(z, 1).flatten(1).detach().cpu().numpy()

            for cname, use_tukey, beta in CONDS:
                pca, prec, dom_means = spaces[use_tukey]
                s = pca.transform(tukey(sup_np) if use_tukey else sup_np)
                q = pca.transform(tukey(q_np) if use_tukey else q_np)
                proto = s.mean(axis=0)
                if beta > 0:
                    means = np.stack(list(dom_means.values()))
                    near = means[np.argsort(np.linalg.norm(means - proto, axis=1))[:args.k_near]]
                    proto = (1 - beta) * proto + beta * near.mean(axis=0)
                scores = mahal(q, proto, prec)
                thr = mahal(s, proto, prec).mean()   # n_sigma=0 프로토콜
                preds = (scores > thr).astype(int)
                agg[cname]["auroc"].append(roc_auc_score(q_labels, scores))
                agg[cname]["acc"].append(accuracy_score(q_labels, preds))
                agg[cname]["f1"].append(f1_score(q_labels, preds))

    print(f"\n=== fold_{args.test_fold} DC 평가 (seeds={args.seeds}) ===")
    for cname, m in agg.items():
        print(f"[{cname:14s}] AUROC {np.mean(m['auroc']):.4f} | "
              f"Acc {np.mean(m['acc']):.4f} | F1 {np.mean(m['f1']):.4f}")


if __name__ == "__main__":
    main()
