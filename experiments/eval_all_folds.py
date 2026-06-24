import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.covariance import LedoitWolf

from torchvision.models import resnet50
from datasets.hust_image import HUSTDataset
from models.mask_decomposition_model import MaskDecompositionModel
from models.vit_mask_decomposition_model import ViTMaskDecompositionModel


class PretrainedExtractor(torch.nn.Module):
    """학습 없이 pretrained ResNet50 feature만 추출 (baseline용)"""
    def __init__(self):
        super().__init__()
        backbone = resnet50(pretrained=True)
        self.encoder = torch.nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3,
        )

    def forward(self, x):
        z = self.encoder(x)
        return F.adaptive_avg_pool2d(z, 1).flatten(1)


# Coarse-grained folds (5 domains, one held-out domain at a time)
FOLDS_COARSE = [
    ("mask_decomposition_fold2.pth", ["500"],             ["400","600","700","800"], ["400","500","600","700","800"]),
    ("mask_decomposition_fold3.pth", ["600"],             ["400","500","700","800"], ["400","500","600","700","800"]),
    ("mask_decomposition_fold4.pth", ["700"],             ["400","500","600","800"], ["400","500","600","700","800"]),
    ("mask_decomposition.pth",       ["800"],             ["400","500","600","700"], ["400","500","600","700","800"]),
]

# Fine-grained folds (15 domains, one speed group held out at a time)
FINE_ALL = ["400","402","404","500","502","504","600","602","604","700","702","704","800","802","804"]
FOLDS_FINE = [
    ("mask_decomposition_fold_500.pth", ["500","502","504"],
     ["400","402","404","600","602","604","700","702","704","800","802","804"], FINE_ALL),
    ("mask_decomposition_fold_600.pth", ["600","602","604"],
     ["400","402","404","500","502","504","700","702","704","800","802","804"], FINE_ALL),
    ("mask_decomposition_fold_700.pth", ["700","702","704"],
     ["400","402","404","500","502","504","600","602","604","800","802","804"], FINE_ALL),
    ("mask_decomposition.pth",          ["800","802","804"],
     ["400","402","404","500","502","504","600","602","604","700","702","704"], FINE_ALL),
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_features(model, loader, feature_key="z_c_notd"):
    feats, labels = [], []
    for batch in loader:
        img = batch["image"].to(device)
        out = model(img)
        z = out[feature_key]
        z = F.adaptive_avg_pool2d(z, 1).view(z.size(0), -1)
        feats.append(z.detach())
        labels.extend(batch["label"].numpy().tolist())
    return torch.cat(feats, dim=0), np.array(labels)


def mahalanobis_score(x: np.ndarray, mean: np.ndarray, prec: np.ndarray) -> np.ndarray:
    """Mahalanobis distance from mean using precision matrix (inverse covariance)."""
    diff = x - mean
    return np.sqrt(np.maximum((diff @ prec) * diff, 0).sum(axis=1))


def fit_normal_distribution(normal_feats: np.ndarray):
    """LedoitWolf shrinkage covariance on normal features → (mean, precision)."""
    lw = LedoitWolf().fit(normal_feats)
    return lw.location_, lw.precision_


def normal_threshold(scores: np.ndarray, n_sigma: float = 2.0) -> float:
    """Threshold = mean + n_sigma * std of normal-sample scores (no label leakage)."""
    return scores.mean() + n_sigma * scores.std()


def run_folds(folds, root, seeds, shot, num_classes, calib_pct=95.0, vit=False):
    all_base, all_zinv = [], []
    all_base_acc, all_base_f1, all_cacc, all_cf1 = [], [], [], []

    header = (f"{'Test':>12} | {'Base AUROC':>10} | {'Base Acc':>8} | {'Base F1':>7} |"
              f" {'z_inv AUROC':>11} | {'z_inv Acc':>9} | {'z_inv F1':>8}")
    print(header)
    print("-" * len(header))

    for ckpt, test_domains, calib_domains, all_domains in folds:
        if not os.path.exists(ckpt):
            print(f"  Checkpoint not found: {ckpt} — skipping fold")
            continue

        num_domains = len(all_domains)

        if vit:
            model = ViTMaskDecompositionModel(
                num_classes=num_classes,
                num_domains=num_domains,
            ).to(device)
        else:
            model = MaskDecompositionModel(
                num_classes=num_classes,
                num_domains=num_domains,
            ).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
        model.eval()

        # Baseline용 pretrained extractor (학습 안 함)
        baseline_extractor = PretrainedExtractor().to(device)
        baseline_extractor.eval()

        base_aurocs, zinv_aurocs = [], []
        base_accs, base_f1s, accs, f1s = [], [], [], []

        # 학습 도메인 정상 샘플로 정상 분포 추정 (공분산)
        calib_normal_sets = []
        for d in calib_domains:
            try:
                calib_normal_sets.append(HUSTDataset(
                    root=root, domain=d, only_normal=True,
                    transform=transform, all_domains=all_domains,
                ))
            except RuntimeError:
                pass
        calib_normal_loader = DataLoader(ConcatDataset(calib_normal_sets), batch_size=32, shuffle=False)
        calib_normal_feats, _ = get_features(model, calib_normal_loader)
        calib_normal_np = calib_normal_feats.cpu().numpy()

        def extract_pretrained(loader):
            feats, lbls = [], []
            with torch.no_grad():
                for batch in loader:
                    img = batch["image"].to(device)
                    feats.append(baseline_extractor(img).detach())
                    lbls.extend(batch["label"].numpy().tolist())
            return torch.cat(feats, dim=0), np.array(lbls)

        _, prec_np = fit_normal_distribution(calib_normal_np)

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            for d in test_domains:
                try:
                    support_ds = HUSTDataset(
                        root=root, domain=d, only_normal=True,
                        shot=shot, transform=transform,
                        seed=seed, all_domains=all_domains,
                    )
                    query_ds = HUSTDataset(
                        root=root, domain=d, only_normal=False,
                        transform=transform, all_domains=all_domains,
                    )
                except RuntimeError:
                    continue

                support_loader = DataLoader(support_ds, batch_size=shot, shuffle=False)
                query_loader   = DataLoader(query_ds,   batch_size=32,   shuffle=False)

                query_feats, query_labels = get_features(model, query_loader)
                if len(np.unique(query_labels)) < 2:
                    continue

                support_feats, _ = get_features(model, support_loader)
                prototype_np = support_feats.cpu().numpy().mean(axis=0)

                # z_inv: Mahalanobis score
                query_np = query_feats.cpu().numpy()
                zinv_scores = mahalanobis_score(query_np, prototype_np, prec_np)
                zinv_aurocs.append(roc_auc_score(query_labels, zinv_scores))

                # Support 기반 threshold: mean + 2*std (4-shot)
                support_zinv_scores = mahalanobis_score(support_feats.cpu().numpy(), prototype_np, prec_np)
                threshold = support_zinv_scores.mean() + 2.0 * support_zinv_scores.std()
                preds = (zinv_scores > threshold).astype(int)
                accs.append(accuracy_score(query_labels, preds))
                f1s.append(f1_score(query_labels, preds, zero_division=0))

                # Baseline: pretrained ResNet50 + cosine
                query_base, _ = extract_pretrained(query_loader)
                support_base, _ = extract_pretrained(support_loader)
                proto_base = F.normalize(support_base.mean(dim=0, keepdim=True), dim=1)
                base_scores = (1.0 - F.cosine_similarity(F.normalize(query_base, dim=1), proto_base)).cpu().numpy()
                base_aurocs.append(roc_auc_score(query_labels, base_scores))

                support_base_scores = (1.0 - F.cosine_similarity(F.normalize(support_base, dim=1), proto_base)).cpu().numpy()
                base_threshold = support_base_scores.mean() + 2.0 * support_base_scores.std()
                base_preds = (base_scores > base_threshold).astype(int)
                base_accs.append(accuracy_score(query_labels, base_preds))
                base_f1s.append(f1_score(query_labels, base_preds, zero_division=0))

        bm = np.mean(base_aurocs)
        bam = np.mean(base_accs)
        bfm = np.mean(base_f1s)
        zm = np.mean(zinv_aurocs)
        am = np.mean(accs)
        fm = np.mean(f1s)

        all_base.append(bm)
        all_base_acc.append(bam)
        all_base_f1.append(bfm)
        all_zinv.append(zm)
        all_cacc.append(am)
        all_cf1.append(fm)

        test_label = "+".join(test_domains)
        print(f"{test_label:>12} | {bm:>10.4f} | {bam:>8.4f} | {bfm:>7.4f} |"
              f" {zm:>11.4f} | {am:>9.4f} | {fm:>8.4f}")

    print("-" * len(header))
    print(f"{'Avg':>12} | {np.mean(all_base):>10.4f} | {np.mean(all_base_acc):>8.4f} | {np.mean(all_base_f1):>7.4f} |"
          f" {np.mean(all_zinv):>11.4f} | {np.mean(all_cacc):>9.4f} | {np.mean(all_cf1):>8.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="processed",
                        help="Processed data root (processed / processed_cwt / processed_fine)")
    parser.add_argument("--num_classes", type=int, default=5,
                        help="ClassClassifier output size (must match checkpoint)")
    parser.add_argument("--mode", type=str, default="coarse",
                        choices=["coarse", "fine"],
                        help="coarse=5-domain LOO, fine=15-domain group LOO")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--shot", type=int, default=4)
    # Optional per-fold checkpoint overrides (coarse mode)
    parser.add_argument("--ckpt_fold_500", type=str, default=None)
    parser.add_argument("--ckpt_fold_600", type=str, default=None)
    parser.add_argument("--ckpt_fold_700", type=str, default=None)
    parser.add_argument("--ckpt_fold_800", type=str, default="mask_decomposition.pth")
    parser.add_argument("--test_fold", type=str, default=None,
                        help="Only evaluate one fold, e.g. '500', '600', '700', '800'")
    parser.add_argument("--calib_pct", type=float, default=95.0,
                        help="Calibration normal score percentile for threshold (default 95)")
    parser.add_argument("--vit", action="store_true",
                        help="Use ViTMaskDecompositionModel instead of ResNet-based model")
    args = parser.parse_args()

    print(f"Device   : {device}")
    print(f"Root     : {args.root}")
    print(f"Mode     : {args.mode}")
    print(f"Classes  : {args.num_classes}")
    print(f"Seeds    : {args.seeds}")
    print(f"Shot     : {args.shot}")
    print()

    if args.mode == "coarse":
        folds = list(FOLDS_COARSE)
        if args.ckpt_fold_500: folds[0] = (args.ckpt_fold_500,) + folds[0][1:]
        if args.ckpt_fold_600: folds[1] = (args.ckpt_fold_600,) + folds[1][1:]
        if args.ckpt_fold_700: folds[2] = (args.ckpt_fold_700,) + folds[2][1:]
        if args.ckpt_fold_800: folds[3] = (args.ckpt_fold_800,) + folds[3][1:]
    else:
        folds = list(FOLDS_FINE)
        if args.ckpt_fold_500: folds[0] = (args.ckpt_fold_500,) + folds[0][1:]
        if args.ckpt_fold_600: folds[1] = (args.ckpt_fold_600,) + folds[1][1:]
        if args.ckpt_fold_700: folds[2] = (args.ckpt_fold_700,) + folds[2][1:]
        if args.ckpt_fold_800: folds[3] = (args.ckpt_fold_800,) + folds[3][1:]

    if args.test_fold:
        folds = [f for f in folds if args.test_fold in f[1]]
        if not folds:
            print(f"No fold found for test_fold={args.test_fold}")
            return

    run_folds(folds, args.root, args.seeds, args.shot, args.num_classes, args.calib_pct, args.vit)


if __name__ == "__main__":
    main()
