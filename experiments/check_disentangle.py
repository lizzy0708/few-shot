import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from datasets.hust_image import HUSTDataset
from models.decomposition_model import DecompositionModel

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_partial(model, path):
    checkpoint = torch.load(path)
    model_dict = model.state_dict()

    filtered_dict = {}
    for k, v in checkpoint.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            filtered_dict[k] = v

    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)


def check():

    dataset = HUSTDataset(root="./processed")
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = DecompositionModel().to(device)

    load_partial(model, "decomposition.pth")

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for img, label, domain in loader:

            img = img.to(device)
            domain = domain.to(device)

            zc, zd, Mc, Md, class_logits, domain_logits = model(img)

            # Fc로 domain 예측
            pred_logits = model.domain_head(zc)
            pred = torch.argmax(pred_logits, dim=1)

            correct += (pred == domain).sum().item()
            total += domain.size(0)

    acc = correct / total
    print("\n==============================")
    print(f"Domain accuracy using Fc: {acc:.4f}")
    print("==============================\n")


if __name__ == "__main__":
    print("Device:", device)
    check()