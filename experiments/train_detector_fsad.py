import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from datasets.hust_image import HUSTDataset
from models.mask_decomposition_model import MaskDecompositionModel
from models.anomaly_detector import AnomalyDetector


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


def extract_features(decomp_model, loader, device):
    features = []
    labels = []

    decomp_model.eval()

    for batch in loader:
        img = batch["image"].to(device)
        label = batch["label"].float()

        
        img.requires_grad_(True)

        out = decomp_model(img)
        z = flatten_feature(out["z_c_notd"])

        features.append(z.detach().cpu())
        labels.append(label)

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)

    return features, labels


def train_detector(detector, train_features, train_labels, device, epochs=50, lr=1e-3):
    detector.train()

    train_features = train_features.to(device)
    train_labels = train_labels.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(detector.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        logits = detector(train_features)
        loss = criterion(logits, train_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}] Loss: {loss.item():.4f}")


@torch.no_grad()
def evaluate_detector(detector, test_features, test_labels, device, threshold=0.5):
    detector.eval()

    test_features = test_features.to(device)

    logits = detector(test_features)
    probs = torch.sigmoid(logits).cpu().numpy()

    y_true = test_labels.numpy().astype(int)
    y_score = probs
    y_pred = (y_score >= threshold).astype(int)

    auroc = roc_auc_score(y_true, y_score)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    return auroc, acc, f1, precision, recall


def build_multi_domain_dataset(domains, transform, seed):
    domain_list = domains.split(",")
    datasets = []

    for d in domain_list:
        d = d.strip()

        dataset = HUSTDataset(
            root="processed",
            domain=d,
            only_normal=False,
            shot=None,
            transform=transform,
            seed=seed,
        )

        datasets.append(dataset)
        print(f"Loaded train domain {d}: {len(dataset)} samples")

    return ConcatDataset(datasets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_domain", type=str, required=True)
    parser.add_argument("--test_domain", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("===================================")
    print("Train Anomaly Detector FSAD")
    print("===================================")
    print("Device       :", device)
    print("Train domain :", args.train_domain)
    print("Test domain  :", args.test_domain)
    print("Checkpoint   :", args.ckpt)
    print("Epochs       :", args.epochs)
    print("LR           :", args.lr)
    print("Threshold    :", args.threshold)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    decomp_model = MaskDecompositionModel().to(device)
    decomp_model.load_state_dict(torch.load(args.ckpt, map_location=device))
    decomp_model.eval()

    train_set = build_multi_domain_dataset(
        domains=args.train_domain,
        transform=transform,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
    )

    test_set = HUSTDataset(
        root="processed",
        domain=args.test_domain,
        only_normal=False,
        shot=None,
        transform=transform,
    )

    print(f"Loaded test domain {args.test_domain}: {len(test_set)} samples")

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
    )

    print("Extracting train features...")
    train_features, train_labels = extract_features(decomp_model, train_loader, device)

    print("Extracting test features...")
    test_features, test_labels = extract_features(decomp_model, test_loader, device)

    input_dim = train_features.size(1)

    print("Feature dim:", input_dim)
    print("Train features:", train_features.shape)
    print("Test features :", test_features.shape)

    detector = AnomalyDetector(input_dim=input_dim).to(device)

    print("Training detector...")
    train_detector(
        detector=detector,
        train_features=train_features,
        train_labels=train_labels,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
    )

    print("Evaluating detector...")
    auroc, acc, f1, precision, recall = evaluate_detector(
        detector=detector,
        test_features=test_features,
        test_labels=test_labels,
        device=device,
        threshold=args.threshold,
    )

    print("===================================")
    print(f"AUROC     : {auroc:.4f}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print("===================================")


if __name__ == "__main__":
    main()