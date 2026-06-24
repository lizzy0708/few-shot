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

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

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


def extract_feature(model, img, feature_type):
    out = model(img)

    if feature_type == "zinv":
        feat = out["z_inv"]
    elif feature_type == "original":
        feat = out["z"]
    elif feature_type == "zd":
        feat = out["z_notc_d"]
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")

    return flatten_feature(feat)


def cosine_anomaly_score(x, proto):
    x = F.normalize(x, dim=1)
    proto = F.normalize(proto.unsqueeze(0), dim=1)
    return 1.0 - F.cosine_similarity(x, proto, dim=1)


def build_mahalanobis_estimator(mean_features, cov_features, cov_mode, reg, shrinkage):
    mean = mean_features.mean(dim=0).detach()
    centered = cov_features - cov_features.mean(dim=0, keepdim=True)

    if cov_mode == "diag":
        var = centered.var(dim=0, unbiased=False)
        var = var + reg
        return {
            "mean": mean,
            "cov_mode": cov_mode,
            "inv_var": 1.0 / var,
        }

    if cov_mode != "full":
        raise ValueError(f"Unknown covariance mode: {cov_mode}")

    n = cov_features.size(0)
    dim = cov_features.size(1)

    if n <= 1:
        cov = torch.eye(dim, device=cov_features.device, dtype=cov_features.dtype)
    else:
        cov = centered.t().matmul(centered) / float(n - 1)

    diag_cov = torch.diag(torch.diag(cov))
    cov = (1.0 - shrinkage) * cov + shrinkage * diag_cov
    cov = cov + reg * torch.eye(dim, device=cov.device, dtype=cov.dtype)
    precision = torch.linalg.pinv(cov)

    return {
        "mean": mean,
        "cov_mode": cov_mode,
        "precision": precision,
    }


def mahalanobis_score(x, estimator):
    diff = x - estimator["mean"].unsqueeze(0)

    if estimator["cov_mode"] == "diag":
        return torch.sum(diff * diff * estimator["inv_var"].unsqueeze(0), dim=1)

    projected = diff.matmul(estimator["precision"])
    return torch.sum(projected * diff, dim=1)


def build_multi_domain_dataset(root, domains, transform, seed):
    domain_list = domains.split(",")
    datasets = []

    for d in domain_list:
        d = d.strip()

        dataset = HUSTDataset(
            root=root,
            domain=d,
            only_normal=False,
            shot=None,
            transform=transform,
            seed=seed,
        )

        datasets.append(dataset)
        print(f"Loaded calibration domain {d}: {len(dataset)} samples")

    return ConcatDataset(datasets)


def extract_features_and_labels(model, loader, device, feature_type):
    features = []
    labels = []

    model.eval()

    for batch in loader:
        img = batch["image"].to(device)
        label = batch["label"].cpu().numpy()

        img.requires_grad_(True)
        feat = extract_feature(model, img, feature_type)

        features.append(feat.detach())
        labels.extend(label.tolist())

    features = torch.cat(features, dim=0)
    labels = np.array(labels).astype(int)

    return features, labels


def compute_scores(features, labels, score_method, prototype=None, estimator=None):
    scores = []

    if score_method == "cosine":
        anomaly_score = cosine_anomaly_score(features, prototype)
    elif score_method == "mahalanobis":
        anomaly_score = mahalanobis_score(features, estimator)
    else:
        raise ValueError(f"Unknown score_method: {score_method}")

    scores.extend(anomaly_score.detach().cpu().numpy().tolist())

    scores = np.array(scores)

    return scores, labels


def select_covariance_features(source, support_features, calib_features, calib_labels):
    if source == "support":
        return support_features

    calib_normal = calib_features[torch.from_numpy(calib_labels == 0).to(calib_features.device)]

    if calib_normal.size(0) == 0:
        raise RuntimeError("No normal samples found in calibration data for covariance estimation.")

    if source == "calib_normal":
        return calib_normal

    if source == "support_calib_normal":
        return torch.cat([support_features, calib_normal], dim=0)

    raise ValueError(f"Unknown covariance source: {source}")


def find_best_threshold(scores, labels, metric="f1"):
    thresholds = np.linspace(scores.min(), scores.max(), 500)

    best_threshold = None
    best_value = -1.0
    best_result = None

    for th in thresholds:
        preds = (scores > th).astype(int)

        acc = accuracy_score(labels, preds)
        balanced_acc = balanced_accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, zero_division=0)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)

        if metric == "f1":
            value = f1
        elif metric == "accuracy":
            value = acc
        elif metric == "balanced_accuracy":
            value = balanced_acc
        elif metric == "youden":
            tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
            tpr = tp / (tp + fn + 1e-12)
            fpr = fp / (fp + tn + 1e-12)
            value = tpr - fpr
        elif metric == "recall":
            value = recall
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if value > best_value:
            best_value = value
            best_threshold = th
            best_result = {
                "accuracy": acc,
                "balanced_accuracy": balanced_acc,
                "f1": f1,
                "precision": precision,
                "recall": recall,
            }

    return best_threshold, best_result


def evaluate_with_threshold(scores, labels, threshold):
    preds = (scores > threshold).astype(int)

    auroc = roc_auc_score(labels, scores)
    acc = accuracy_score(labels, preds)
    balanced_acc = balanced_accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds)

    return auroc, acc, balanced_acc, f1, precision, recall, cm


def print_score_statistics(scores, labels, title):
    print("===================================")
    print(title)
    print(f"score min    : {scores.min():.6f}")
    print(f"score max    : {scores.max():.6f}")
    print(f"score mean   : {scores.mean():.6f}")
    print(f"score std    : {scores.std():.6f}")

    if np.sum(labels == 0) > 0:
        print(f"normal mean  : {scores[labels == 0].mean():.6f}")
        print(f"normal max   : {scores[labels == 0].max():.6f}")

    if np.sum(labels == 1) > 0:
        print(f"anomaly mean : {scores[labels == 1].mean():.6f}")
        print(f"anomaly min  : {scores[labels == 1].min():.6f}")

    print("===================================")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="processed")
    parser.add_argument("--support_domain", type=str, required=True)
    parser.add_argument("--calib_domain", type=str, required=True)
    parser.add_argument("--query_domain", type=str, required=True)
    parser.add_argument("--shot", type=int, default=4)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_domains", type=int, default=5)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--select_metric", type=str, default="f1",
                        choices=["f1", "accuracy", "balanced_accuracy", "youden", "recall"])
    parser.add_argument("--feature", type=str, default="zinv",
                        choices=["zinv", "original", "zd"])
    parser.add_argument("--score_method", type=str, default="mahalanobis",
                        choices=["mahalanobis", "cosine"])
    parser.add_argument("--cov_mode", type=str, default="diag",
                        choices=["diag", "full"])
    parser.add_argument("--cov_source", type=str, default="support_calib_normal",
                        choices=["support", "calib_normal", "support_calib_normal"])
    parser.add_argument("--cov_reg", type=float, default=1e-3)
    parser.add_argument("--cov_shrinkage", type=float, default=0.1)

    args = parser.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("===================================")
    print("FSAD + Calibrated Threshold")
    print("===================================")
    print("Device        :", device)
    print("Root          :", args.root)
    print("Support domain:", args.support_domain)
    print("Calib domain  :", args.calib_domain)
    print("Query domain  :", args.query_domain)
    print("Shot          :", args.shot)
    print("Checkpoint    :", args.ckpt)
    print("Num domains   :", args.num_domains)
    print("Select metric :", args.select_metric)
    print("Feature       :", args.feature)
    print("Score method  :", args.score_method)
    print("Cov mode      :", args.cov_mode)
    print("Cov source    :", args.cov_source)
    print("Cov reg       :", args.cov_reg)
    print("Cov shrinkage :", args.cov_shrinkage)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    model = MaskDecompositionModel(num_domains=args.num_domains).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    # 1. Few-shot support normal feature 추출
    support_set = HUSTDataset(
        root=args.root,
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

    support_features, support_labels = extract_features_and_labels(
        model=model,
        loader=support_loader,
        device=device,
        feature_type=args.feature,
    )
    prototype = support_features.mean(dim=0).detach()

    print("Support normal features extracted.")
    print("Support feature shape:", support_features.shape)

    # 2. Calibration domain에서 score 계산
    calib_set = build_multi_domain_dataset(
        root=args.root,
        domains=args.calib_domain,
        transform=transform,
        seed=args.seed,
    )

    calib_loader = DataLoader(
        calib_set,
        batch_size=args.batch_size,
        shuffle=False,
    )

    print("Extracting calibration features...")
    calib_features, calib_labels = extract_features_and_labels(
        model=model,
        loader=calib_loader,
        device=device,
        feature_type=args.feature,
    )

    estimator = None
    if args.score_method == "mahalanobis":
        cov_features = select_covariance_features(
            source=args.cov_source,
            support_features=support_features,
            calib_features=calib_features,
            calib_labels=calib_labels,
        )

        estimator = build_mahalanobis_estimator(
            mean_features=support_features,
            cov_features=cov_features,
            cov_mode=args.cov_mode,
            reg=args.cov_reg,
            shrinkage=args.cov_shrinkage,
        )

        print("Mahalanobis estimator built.")
        print("Mean source      : support normal")
        print("Cov feature shape:", cov_features.shape)

    print("Computing calibration scores...")
    calib_scores, calib_labels = compute_scores(
        features=calib_features,
        labels=calib_labels,
        score_method=args.score_method,
        prototype=prototype,
        estimator=estimator,
    )

    print_score_statistics(
        calib_scores,
        calib_labels,
        title="Calibration Score Statistics"
    )

    # 3. Calibration domain에서 threshold 선택
    best_threshold, calib_result = find_best_threshold(
        scores=calib_scores,
        labels=calib_labels,
        metric=args.select_metric,
    )

    print("===================================")
    print("Selected Threshold from Calibration")
    print(f"Threshold : {best_threshold:.6f}")
    print(f"Calib Acc : {calib_result['accuracy']:.4f}")
    print(f"Calib BAcc: {calib_result['balanced_accuracy']:.4f}")
    print(f"Calib F1  : {calib_result['f1']:.4f}")
    print(f"Calib Pre : {calib_result['precision']:.4f}")
    print(f"Calib Rec : {calib_result['recall']:.4f}")
    print("===================================")

    # 4. Query domain 평가
    query_set = HUSTDataset(
        root=args.root,
        domain=args.query_domain,
        only_normal=False,
        shot=None,
        transform=transform,
    )

    print(f"Loaded query domain {args.query_domain}: {len(query_set)} samples")

    query_loader = DataLoader(
        query_set,
        batch_size=args.batch_size,
        shuffle=False,
    )

    print("Extracting query features...")
    query_features, query_labels = extract_features_and_labels(
        model=model,
        loader=query_loader,
        device=device,
        feature_type=args.feature,
    )

    print("Computing query scores...")
    query_scores, query_labels = compute_scores(
        features=query_features,
        labels=query_labels,
        score_method=args.score_method,
        prototype=prototype,
        estimator=estimator,
    )

    print_score_statistics(
        query_scores,
        query_labels,
        title="Query Score Statistics"
    )

    auroc, acc, balanced_acc, f1, precision, recall, cm = evaluate_with_threshold(
        scores=query_scores,
        labels=query_labels,
        threshold=best_threshold,
    )

    print("===================================")
    print("Final Query Result")
    print(f"Threshold : {best_threshold:.6f}")
    print(f"AUROC     : {auroc:.4f}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Balanced Acc: {balanced_acc:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print("Confusion Matrix")
    print(cm)
    print("===================================")


if __name__ == "__main__":
    main()
