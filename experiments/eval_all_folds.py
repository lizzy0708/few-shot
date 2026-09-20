import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torchvision.models import resnet50
from datasets.hust_image import HUSTDataset
from models.mask_decomposition_model import MaskDecompositionModel
from models.inv_encoder_model import InvEncoderModel
from models.mc_model import MCModel
from models.channel_mask_model import ChannelMaskModel
from models.original_mask_model import OriginalMaskModel, FeatureExtractor


class RawPretrainedModel(torch.nn.Module):
    """Ablation stage 1: untrained pretrained ResNet50 (encoder_layer), no mc/md decomposition at all."""
    def __init__(self, encoder_layer='layer3'):
        super().__init__()
        self.feature_extractor = FeatureExtractor(encoder_layer=encoder_layer)

    def forward(self, x, **kwargs):
        z = self.feature_extractor(x)
        return {"z_c_notd": z, "z": z}


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

# cuDNN algorithm selection is nondeterministic across runs/batch sizes; this feeds
# tiny feature-level noise into Youden's J threshold selection (argmax over observed
# scores), which can jump between candidate thresholds and swing Acc/F1 by 10-20pp
# even though AUROC barely moves. Force determinism so eval is bit-reproducible.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except TypeError:
    torch.use_deterministic_algorithms(True)


def get_features(model, loader, feature_key="z_c_notd", md=None):
    feats, labels = [], []
    for batch in loader:
        img = batch["image"].to(device)
        out = model(img) if md is None else model(img, md=md)
        z = out[feature_key]
        z = F.adaptive_avg_pool2d(z, 1).view(z.size(0), -1)
        feats.append(z.detach())
        labels.extend(batch["label"].numpy().tolist())
    return torch.cat(feats, dim=0), np.array(labels)


def compute_md_from_calib(model, calib_normal_loader):
    """calib 도메인 정상 샘플로 md_calib 계산 (ChannelMaskModel 전용)."""
    zp_list, dom_list = [], []
    with torch.no_grad():
        for batch in calib_normal_loader:
            img = batch["image"].to(device)
            z = model.feature_extractor(img)
            zp = F.adaptive_avg_pool2d(z, 1).flatten(1)
            zp_list.append(zp)
            dom_list.append(batch["domain"].long().to(device))
    zp_all = torch.cat(zp_list)
    dom_all = torch.cat(dom_list)
    labels_all = torch.zeros(len(zp_all), dtype=torch.long, device=device)
    return ChannelMaskModel.compute_md(zp_all, labels_all, dom_all)


def plot_confusion_matrix(cm: np.ndarray, title: str, save_path: str,
                          auroc: float = None, acc: float = None,
                          f1: float = None, prec: float = None, rec: float = None):
    """혼동행렬을 heatmap 이미지로 저장."""
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)

    classes = ['Normal', 'Anomaly']
    ax.set_xticks([0, 1]); ax.set_xticklabels(classes, fontsize=12)
    ax.set_yticks([0, 1]); ax.set_yticklabels(classes, fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)

    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = val / total * 100
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, f'{val:,}\n({pct:.1f}%)', ha='center', va='center',
                    fontsize=11, color=color, fontweight='bold')

    metrics_lines = []
    if auroc is not None: metrics_lines.append(f'AUROC={auroc:.4f}')
    if acc   is not None: metrics_lines.append(f'Acc={acc:.4f}')
    if f1    is not None: metrics_lines.append(f'F1={f1:.4f}')
    if prec  is not None: metrics_lines.append(f'Prec={prec:.4f}')
    if rec   is not None: metrics_lines.append(f'Rec={rec:.4f}')
    subtitle = '  |  '.join(metrics_lines)

    ax.set_title(f'{title}\n{subtitle}', fontsize=11, pad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [CM 이미지 저장] {save_path}")


def mahalanobis_score(x: np.ndarray, mean: np.ndarray, prec: np.ndarray) -> np.ndarray:
    """Mahalanobis distance from mean using precision matrix (inverse covariance)."""
    diff = x - mean
    return np.sqrt(np.maximum((diff @ prec) * diff, 0).sum(axis=1))


def fit_normal_distribution(normal_feats: np.ndarray):
    """LedoitWolf shrinkage covariance on normal features → (mean, precision)."""
    lw = LedoitWolf().fit(normal_feats)
    return lw.location_, lw.precision_


def fit_diag_distribution(normal_feats: np.ndarray, reg: float = 1e-3) -> np.ndarray:
    """Diagonal covariance: precision = diag(1 / (var + reg))."""
    var = normal_feats.var(axis=0) + reg
    return np.diag(1.0 / var)


def normal_threshold(scores: np.ndarray, n_sigma: float = 2.0) -> float:
    """Threshold = mean + n_sigma * std of normal-sample scores (no label leakage)."""
    return scores.mean() + n_sigma * scores.std()


def youden_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Threshold at maximum Youden's J = TPR - FPR (from calib normal+anomaly)."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return float(thresholds[np.argmax(tpr - fpr)])


def f1_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Threshold at maximum F1 score (from calib normal+anomaly)."""
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1s = 2 * precision * recall / (precision + recall + 1e-9)
    return float(thresholds[np.argmax(f1s[:-1])])


def acc_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Threshold at maximum Accuracy (from calib normal+anomaly)."""
    thresholds = np.unique(scores)
    best_acc, best_th = 0.0, thresholds[0]
    for th in thresholds:
        preds = (scores >= th).astype(int)
        acc = (preds == labels).mean()
        if acc > best_acc:
            best_acc, best_th = acc, th
    return float(best_th)


def make_coarse_domain_map(all_domains):
    """500/502/504 → 같은 index (RPM 그룹 기준)."""
    rpm_groups = sorted(set(int(d) // 100 * 100 for d in all_domains))
    rpm_to_idx = {rpm: idx for idx, rpm in enumerate(rpm_groups)}
    return {str(d): rpm_to_idx[int(d) // 100 * 100] for d in all_domains}


def run_folds(folds, root, seeds, shot, num_classes, calib_pct=95.0, use_l2=False, n_sigma=2.0, use_classifier=False, proto_beta=1.0, model_type="mask", pca_dim=0, coarse=False, use_youden=False, use_cosine=False, use_f1_thresh=False, use_acc_thresh=False, use_zmc=False, use_nomd=False, eval_batch_size=32, cov_type="auto"):
    all_base, all_zinv = [], []
    all_base_acc, all_base_f1, all_cacc, all_cf1 = [], [], [], []
    all_zinv_std, all_cacc_std, all_cf1_std = [], [], []
    all_prec, all_rec = [], []
    global_preds, global_labels = [], []

    header = (f"{'Test':>12} | {'Base AUROC':>10} | {'Base Acc':>8} | {'Base F1':>7} |"
              f" {'z_inv AUROC':>11} | {'z_inv Acc':>9} | {'z_inv F1':>8} | {'Prec':>6} | {'Rec':>6}")
    print(header)
    print("-" * len(header))

    if use_nomd:
        feature_key = "z_mc_nomd"
    elif use_zmc:
        feature_key = "z_mc"
    else:
        feature_key = "z_c_notd"

    for ckpt, test_domains, calib_domains, all_domains in folds:
        if not os.path.exists(ckpt):
            print(f"  Checkpoint not found: {ckpt} — skipping fold")
            continue

        # coarse 모드: 500/502/504 → 같은 도메인 index
        if coarse:
            domain_map  = make_coarse_domain_map(all_domains)
            num_domains = len(set(domain_map.values()))
        else:
            domain_map  = None
            num_domains = len(all_domains)

        # 체크포인트에서 실제 num_domains 및 encoder_layer 감지
        _sd = torch.load(ckpt, map_location=device)
        for key in ("domain_classifier.weight", "domain_classifier_inv.fc.weight"):
            if key in _sd:
                num_domains = _sd[key].shape[0]
                break
        # classifier 가중치 열 수로 encoder_layer 감지 (fc wrapper 유무 모두 처리)
        ckpt_encoder_layer = 'layer4'
        for cls_key in ("classifier.fc.weight", "classifier.weight"):
            if cls_key in _sd:
                if _sd[cls_key].shape[1] == 1024:
                    ckpt_encoder_layer = 'layer3'
                break

        if model_type == "inv":
            model = InvEncoderModel(
                num_classes=num_classes,
                num_domains=num_domains,
            ).to(device)
        elif model_type == "mc":
            model = MCModel(
                num_classes=num_classes,
                num_domains=num_domains,
            ).to(device)
        elif model_type == "channel":
            model = ChannelMaskModel(num_classes=num_classes, num_domains=num_domains).to(device)
        elif model_type == "original":
            model = OriginalMaskModel(
                num_classes=num_classes,
                num_domains=num_domains,
                encoder_layer=ckpt_encoder_layer,
            ).to(device)
        elif model_type == "raw_pretrained":
            model = RawPretrainedModel(encoder_layer=ckpt_encoder_layer).to(device)
        else:
            # 계층적 md 체크포인트 감지 (disc_rpm 헤드 존재 여부)
            hier_md = "domain_classifier_disc_rpm.fc.weight" in _sd
            hier_rpm_groups = (_sd["domain_classifier_disc_rpm.fc.weight"].shape[0]
                               if hier_md else None)
            model = MaskDecompositionModel(
                num_classes=num_classes,
                num_domains=num_domains,
                encoder_layer=ckpt_encoder_layer,
                num_rpm_groups=hier_rpm_groups,
                hierarchical_md=hier_md,
            ).to(device)
        if model_type != "raw_pretrained":
            # raw_pretrained: pretrained ImageNet weights only, ckpt path only used
            # for the file-exists gate + encoder_layer detection above — no state to load.
            model.load_state_dict(_sd, strict=False)
        model.eval()

        # Baseline용 pretrained extractor (학습 안 함)
        baseline_extractor = PretrainedExtractor().to(device)
        baseline_extractor.eval()

        base_aurocs, zinv_aurocs = [], []
        base_accs, base_f1s, accs, f1s, precs, recs = [], [], [], [], [], []
        fold_all_preds, fold_all_labels = [], []
        fold_base_preds, fold_base_labels = [], []
        # seed 단위로 분리 저장 (도메인 분산과 섞이지 않도록 seed별 평균 후 std 계산)
        seed_zinv_auroc = {s: [] for s in seeds}
        seed_acc = {s: [] for s in seeds}
        seed_f1 = {s: [] for s in seeds}
        seed_base_auroc = {s: [] for s in seeds}
        seed_base_acc = {s: [] for s in seeds}
        seed_base_f1 = {s: [] for s in seeds}
        # 서브도메인별로도 분리 저장 (500/502/504를 묶지 않고 각각 5-seed mean±std)
        domain_zinv_auroc = {d: [] for d in test_domains}
        domain_acc = {d: [] for d in test_domains}
        domain_f1 = {d: [] for d in test_domains}

        # Calib 로딩: youden/cosine 모드는 정상+이상 모두, 아니면 정상만
        calib_sets = []
        for d in calib_domains:
            try:
                calib_sets.append(HUSTDataset(
                    root=root, domain=d, only_normal=(not use_youden and not use_cosine),
                    transform=transform, all_domains=all_domains,
                    domain_map=domain_map,
                ))
            except RuntimeError:
                pass
        calib_loader = DataLoader(ConcatDataset(calib_sets), batch_size=eval_batch_size, shuffle=False)

        md_calib = None
        if model_type == "channel":
            # ChannelMaskModel은 정상 loader로 md 계산 (youden여부 무관)
            if use_youden:
                calib_norm_sets = []
                for d in calib_domains:
                    try:
                        calib_norm_sets.append(HUSTDataset(
                            root=root, domain=d, only_normal=True,
                            transform=transform, all_domains=all_domains,
                            domain_map=domain_map,
                        ))
                    except RuntimeError:
                        pass
                md_calib = compute_md_from_calib(
                    model, DataLoader(ConcatDataset(calib_norm_sets), batch_size=eval_batch_size, shuffle=False)
                )
            else:
                md_calib = compute_md_from_calib(model, calib_loader)

        calib_feats, calib_labels_all = get_features(model, calib_loader, feature_key=feature_key, md=md_calib)
        calib_np_all = calib_feats.cpu().numpy()

        # 정상 샘플만 분리 (covariance fitting용)
        if use_youden:
            calib_normal_np = calib_np_all[calib_labels_all == 0]
        else:
            calib_normal_np = calib_np_all

        # PCA dimensionality reduction (fit on calib normals)
        pca_model = None
        if pca_dim > 0 and pca_dim < calib_normal_np.shape[1]:
            pca_model = PCA(n_components=pca_dim, random_state=42)
            calib_normal_np = pca_model.fit_transform(calib_normal_np)
            if use_youden:
                calib_np_all = pca_model.transform(calib_np_all)

        def extract_pretrained(loader):
            feats, lbls = [], []
            with torch.no_grad():
                for batch in loader:
                    img = batch["image"].to(device)
                    feats.append(baseline_extractor(img).detach())
                    lbls.extend(batch["label"].numpy().tolist())
            return torch.cat(feats, dim=0), np.array(lbls)

        effective_cov = cov_type if cov_type != "auto" else ("diag" if use_youden else "ledoit")
        if effective_cov == "diag":
            prec_np = fit_diag_distribution(calib_normal_np)
        else:
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
                        domain_map=domain_map,
                    )
                    # 🔴 2026-09-17 leakage fix: query_ds previously included every sample in
                    # the domain (no shot filtering applied to it), so the exact 4 normal
                    # windows drawn into support_ds were also scored as query "normal" samples.
                    # Exclude them explicitly by path.
                    support_paths = set(s[0] for s in support_ds.samples)
                    query_ds = HUSTDataset(
                        root=root, domain=d, only_normal=False,
                        transform=transform, all_domains=all_domains,
                        domain_map=domain_map,
                        exclude_paths=support_paths,
                    )
                    query_paths = set(s[0] for s in query_ds.samples)
                    assert support_paths.isdisjoint(query_paths), (
                        f"support/query leakage in domain={d} seed={seed}: "
                        f"{support_paths & query_paths}"
                    )
                except RuntimeError:
                    continue

                support_loader = DataLoader(support_ds, batch_size=shot, shuffle=False)
                query_loader   = DataLoader(query_ds,   batch_size=eval_batch_size,   shuffle=False)

                query_feats, query_labels = get_features(model, query_loader, feature_key=feature_key, md=md_calib)
                if len(np.unique(query_labels)) < 2:
                    continue

                support_feats, _ = get_features(model, support_loader, feature_key=feature_key, md=md_calib)
                prototype_np = support_feats.cpu().numpy().mean(axis=0)

                # z_inv scoring
                query_np = query_feats.cpu().numpy()
                support_np = support_feats.cpu().numpy()

                # apply PCA if fitted
                if pca_model is not None:
                    query_np   = pca_model.transform(query_np)
                    support_np = pca_model.transform(support_np)

                prototype_np = support_np.mean(axis=0)

                if use_cosine:
                    def cosine_dist(x, proto):
                        x_n = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
                        p_n = proto / (np.linalg.norm(proto) + 1e-8)
                        return 1.0 - (x_n @ p_n)
                    zinv_scores = cosine_dist(query_np, prototype_np)
                    calib_scores_thresh = cosine_dist(calib_np_all, prototype_np)
                    threshold = youden_threshold(calib_scores_thresh, calib_labels_all)
                elif use_l2:
                    zinv_scores = np.sqrt(((query_np - prototype_np) ** 2).sum(axis=1))
                    support_zinv_scores = np.sqrt(((support_np - prototype_np) ** 2).sum(axis=1))
                    threshold = support_zinv_scores.mean()
                elif use_classifier:
                    fc_w = model.classifier.fc.weight.detach().cpu().numpy()
                    fc_b = model.classifier.fc.bias.detach().cpu().numpy()
                    query_logits = query_np @ fc_w.T + fc_b
                    support_logits = support_np @ fc_w.T + fc_b
                    zinv_scores = query_logits[:, 1] - query_logits[:, 0]
                    support_anom_scores = support_logits[:, 1] - support_logits[:, 0]
                    threshold = support_anom_scores.mean()
                elif use_f1_thresh:
                    zinv_scores = mahalanobis_score(query_np, prototype_np, prec_np)
                    calib_scores_thresh = mahalanobis_score(calib_np_all, prototype_np, prec_np)
                    threshold = f1_threshold(calib_scores_thresh, calib_labels_all)
                elif use_acc_thresh:
                    zinv_scores = mahalanobis_score(query_np, prototype_np, prec_np)
                    calib_scores_thresh = mahalanobis_score(calib_np_all, prototype_np, prec_np)
                    threshold = acc_threshold(calib_scores_thresh, calib_labels_all)
                elif use_youden:
                    zinv_scores = mahalanobis_score(query_np, prototype_np, prec_np)
                    calib_scores_thresh = mahalanobis_score(calib_np_all, prototype_np, prec_np)
                    threshold = youden_threshold(calib_scores_thresh, calib_labels_all)
                else:
                    # Mahalanobis + support mean + n_sigma*calib_std
                    zinv_scores = mahalanobis_score(query_np, prototype_np, prec_np)
                    support_zinv_scores = mahalanobis_score(support_np, prototype_np, prec_np)
                    calib_zinv_scores = mahalanobis_score(calib_normal_np, prototype_np, prec_np)
                    threshold = support_zinv_scores.mean() + n_sigma * calib_zinv_scores.std()

                from sklearn.metrics import precision_score, recall_score
                zinv_aurocs.append(roc_auc_score(query_labels, zinv_scores))
                preds = (zinv_scores > threshold).astype(int)
                accs.append(accuracy_score(query_labels, preds))
                f1s.append(f1_score(query_labels, preds, zero_division=0))
                precs.append(precision_score(query_labels, preds, zero_division=0))
                recs.append(recall_score(query_labels, preds, zero_division=0))
                seed_zinv_auroc[seed].append(zinv_aurocs[-1])
                seed_acc[seed].append(accs[-1])
                seed_f1[seed].append(f1s[-1])
                domain_zinv_auroc[d].append(zinv_aurocs[-1])
                domain_acc[d].append(accs[-1])
                domain_f1[d].append(f1s[-1])
                fold_all_preds.extend(preds.tolist())
                fold_all_labels.extend(query_labels.tolist())

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
                fold_base_preds.extend(base_preds.tolist())
                fold_base_labels.extend(query_labels.tolist())
                seed_base_auroc[seed].append(base_aurocs[-1])
                seed_base_acc[seed].append(base_accs[-1])
                seed_base_f1[seed].append(base_f1s[-1])

        bm = np.mean(base_aurocs)
        bam = np.mean(base_accs)
        bfm = np.mean(base_f1s)
        zm = np.mean(zinv_aurocs)
        am = np.mean(accs)
        fm = np.mean(f1s)

        # seed별 평균(도메인 3개 묶음) → 5-seed 간 std (fine mode에서만 의미 있음: coarse는 fold당 도메인 1개라 seed 평균=단일 원소)
        seed_auroc_means = [np.mean(v) for v in seed_zinv_auroc.values() if v]
        seed_acc_means   = [np.mean(v) for v in seed_acc.values() if v]
        seed_f1_means    = [np.mean(v) for v in seed_f1.values() if v]
        z_std = np.std(seed_auroc_means) if len(seed_auroc_means) > 1 else 0.0
        a_std = np.std(seed_acc_means) if len(seed_acc_means) > 1 else 0.0
        f_std = np.std(seed_f1_means) if len(seed_f1_means) > 1 else 0.0

        base_seed_auroc_means = [np.mean(v) for v in seed_base_auroc.values() if v]
        base_seed_acc_means   = [np.mean(v) for v in seed_base_acc.values() if v]
        base_seed_f1_means    = [np.mean(v) for v in seed_base_f1.values() if v]
        bz_std = np.std(base_seed_auroc_means) if len(base_seed_auroc_means) > 1 else 0.0
        ba_std = np.std(base_seed_acc_means) if len(base_seed_acc_means) > 1 else 0.0
        bf_std = np.std(base_seed_f1_means) if len(base_seed_f1_means) > 1 else 0.0

        all_base.append(bm)
        all_base_acc.append(bam)
        all_base_f1.append(bfm)
        all_zinv.append(zm)
        all_cacc.append(am)
        all_cf1.append(fm)
        all_zinv_std.append(z_std)
        all_cacc_std.append(a_std)
        all_cf1_std.append(f_std)

        pm = np.mean(precs) if precs else 0.0
        rm = np.mean(recs)  if recs  else 0.0
        all_prec.append(pm)
        all_rec.append(rm)

        test_label = "+".join(test_domains)
        print(f"{test_label:>12} | {bm:>10.4f} | {bam:>8.4f} | {bfm:>7.4f} |"
              f" {zm:>11.4f} | {am:>9.4f} | {fm:>8.4f} | {pm:>6.4f} | {rm:>6.4f}")
        print(f"  [5-seed mean±std] AUROC {zm:.4f}±{z_std:.4f} | Acc {am:.4f}±{a_std:.4f} | F1 {fm:.4f}±{f_std:.4f}"
              f"  (n_seeds={len(seed_auroc_means)})")
        print(f"  [Base 5-seed mean±std] AUROC {bm:.4f}±{bz_std:.4f} | Acc {bam:.4f}±{ba_std:.4f} | F1 {bfm:.4f}±{bf_std:.4f}"
              f"  (n_seeds={len(base_seed_auroc_means)})")

        # 서브도메인별 breakdown (500/502/504를 묶지 않고 개별 표시, 각각 5-seed mean±std)
        if len(test_domains) > 1:
            for d in test_domains:
                dz, da, df = domain_zinv_auroc[d], domain_acc[d], domain_f1[d]
                if not dz:
                    continue
                print(f"    - {d:>8} | AUROC {np.mean(dz):.4f}±{np.std(dz):.4f}"
                      f" | Acc {np.mean(da):.4f}±{np.std(da):.4f}"
                      f" | F1 {np.mean(df):.4f}±{np.std(df):.4f}  (n_seeds={len(dz)})")

        # Confusion matrix for this fold
        if fold_all_labels:
            cm = confusion_matrix(fold_all_labels, fold_all_preds, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            print(f"  [z_inv CM] TN={tn:5d} FP={fp:5d} FN={fn:5d} TP={tp:5d}"
                  f"  | Sens={tp/(tp+fn+1e-9):.4f} Spec={tn/(tn+fp+1e-9):.4f}")
            fold_acc  = (tn+tp)/(tn+fp+fn+tp)
            fold_prec = tp/(tp+fp+1e-9)
            fold_rec  = tp/(tp+fn+1e-9)
            fold_f1   = 2*tp/(2*tp+fp+fn+1e-9)
            plot_confusion_matrix(
                cm, title=f"Fold {test_label} Confusion Matrix",
                save_path=f"results/cm_fold_{test_label}.png",
                auroc=zm, acc=fold_acc, f1=fold_f1, prec=fold_prec, rec=fold_rec,
            )
            global_preds.extend(fold_all_preds)
            global_labels.extend(fold_all_labels)

    print("-" * len(header))
    avg_prec = np.mean(all_prec) if all_prec else 0.0
    avg_rec  = np.mean(all_rec)  if all_rec  else 0.0
    print(f"{'Avg':>12} | {np.mean(all_base):>10.4f} | {np.mean(all_base_acc):>8.4f} | {np.mean(all_base_f1):>7.4f} |"
          f" {np.mean(all_zinv):>11.4f} | {np.mean(all_cacc):>9.4f} | {np.mean(all_cf1):>8.4f} | {avg_prec:>6.4f} | {avg_rec:>6.4f}")
    if all_zinv_std:
        print(f"  [fold별 5-seed std의 평균 — 참고용, fold간 pooled std 아님]"
              f" AUROC ±{np.mean(all_zinv_std):.4f} | Acc ±{np.mean(all_cacc_std):.4f} | F1 ±{np.mean(all_cf1_std):.4f}")

    if global_labels:
        gcm = confusion_matrix(global_labels, global_preds, labels=[0, 1])
        gtn, gfp, gfn, gtp = gcm.ravel()
        print()
        gtotal = gtn + gfp + gfn + gtp
        print("=== 전체 혼동행렬 (z_inv, 모든 fold 합산) ===")
        print(f"              Pred Normal  Pred Anomaly")
        print(f"  True Normal    {gtn:7d}     {gfp:7d}")
        print(f"  True Anomaly   {gfn:7d}     {gtp:7d}")
        gacc  = (gtn+gtp)/gtotal
        grec  = gtp/(gtp+gfn+1e-9)
        gspec = gtn/(gtn+gfp+1e-9)
        gprec = gtp/(gtp+gfp+1e-9)
        gf1   = 2*gtp/(2*gtp+gfp+gfn+1e-9)
        print(f"  Accuracy      = {gacc:.4f}")
        print(f"  Recall(Sens)  = {grec:.4f}")
        print(f"  Specificity   = {gspec:.4f}")
        print(f"  Precision     = {gprec:.4f}")
        print(f"  F1            = {gf1:.4f}")
        plot_confusion_matrix(
            gcm, title="Overall Confusion Matrix (All Folds)",
            save_path="results/cm_overall.png",
            acc=gacc, f1=gf1, prec=gprec, rec=grec,
        )


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
    parser.add_argument("--use_l2", action="store_true",
                        help="Use L2 distance + support mean threshold (matches episodic training)")
    parser.add_argument("--n_sigma", type=float, default=0.0,
                        help="Threshold = support_mean + n_sigma * calib_std (default 0.0)")
    parser.add_argument("--use_classifier", action="store_true",
                        help="Threshold-free: use trained classifier with support bias correction")
    parser.add_argument("--proto_beta", type=float, default=1.0,
                        help="Prototype blend: β*test_proto + (1-β)*calib_centroid (default 1.0 = test only)")
    parser.add_argument("--model_type", type=str, default="mask",
                        choices=["mask", "inv", "mc", "vit", "channel", "original", "raw_pretrained"],
                        help="Model type: mask=MaskDecompositionModel, inv=InvEncoderModel, mc=MCModel, channel=ChannelMaskModel, original=OriginalMaskModel, raw_pretrained=untrained pretrained ResNet50 (ablation stage 1, no decomposition)")
    parser.add_argument("--pca_dim", type=int, default=0,
                        help="PCA dim before Mahalanobis (0=no PCA, e.g. 64, 128, 256)")
    parser.add_argument("--coarse", action="store_true",
                        help="500/502/504를 같은 domain index로 묶어서 평가")
    parser.add_argument("--youden", action="store_true",
                        help="Youden's J threshold + diag covariance (6/15 방식)")
    parser.add_argument("--cosine", action="store_true",
                        help="코사인 유사도 기반 scoring + Youden threshold (원본 방식)")
    parser.add_argument("--f1_thresh", action="store_true",
                        help="Calib F1 최대화 threshold")
    parser.add_argument("--acc_thresh", action="store_true",
                        help="Calib Accuracy 최대화 threshold (정확도 우선)")
    parser.add_argument("--zmc", action="store_true",
                        help="z*mc feature 사용 (Gram-Schmidt 없이, 5/1 체크포인트 방식)")
    parser.add_argument("--nomd", action="store_true",
                        help="z*mc*(1-md) feature 사용 (Notion 6/15 원본 공식)")
    parser.add_argument("--eval_batch_size", type=int, default=32,
                        help="calib/query DataLoader batch size (GPU 메모리 부족 시 낮출 것)")
    parser.add_argument("--cov_type", type=str, default="auto", choices=["auto", "diag", "ledoit"],
                        help="auto: youden=diag/else=ledoit (기존 동작). diag/ledoit로 강제 지정 가능")
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

    run_folds(folds, args.root, args.seeds, args.shot, args.num_classes, args.calib_pct, args.use_l2, args.n_sigma, args.use_classifier, args.proto_beta, args.model_type, args.pca_dim, coarse=args.coarse, use_youden=args.youden, use_cosine=args.cosine, use_f1_thresh=args.f1_thresh, use_acc_thresh=args.acc_thresh, use_zmc=args.zmc, use_nomd=args.nomd, eval_batch_size=args.eval_batch_size, cov_type=args.cov_type)


if __name__ == "__main__":
    main()
