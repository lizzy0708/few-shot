import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from datasets.hust_image import HUSTDataset
from models.mask_decomposition_model import MaskDecompositionModel


device = "cuda" if torch.cuda.is_available() else "cpu"


def train(args):
    datasets = []

    for domain in args.train_domains:
        ds = HUSTDataset(
            root=args.root,
            domain=domain,
            only_normal=False,
            shot=None,
        )
        datasets.append(ds)

    dataset = ConcatDataset(datasets)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=False,
    )

    num_domains = len(args.all_domains)

    model = MaskDecompositionModel(
        num_classes=2,
        num_domains=num_domains
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    print("===================================")
    print("Train Mask Decomposition Model")
    print("===================================")
    print("Device       :", device)
    print("Train domains:", args.train_domains)
    print("All domains  :", args.all_domains)
    print("Num domains  :", num_domains)
    print("Epochs       :", args.epochs)
    print("Batch size   :", args.batch_size)
    print("LR           :", args.lr)
    print("Mask weight  :", args.mask_weight)
    print("Domain weight:", args.domain_weight)
    print("Save path    :", args.save_path)
    print("Total samples:", len(dataset))
    print("===================================")

    for epoch in range(args.epochs):
        model.train()

        total_loss = 0.0
        total_class_loss = 0.0
        total_domain_loss = 0.0
        total_mask_loss = 0.0

        correct_cls = 0
        correct_dom = 0
        total = 0

        for batch in loader:
            img = batch["image"].to(device)
            label = batch["label"].long().to(device)
            domain = batch["domain"].long().to(device)

            # domain label sanity check
            if domain.min() < 0 or domain.max() >= num_domains:
                raise ValueError(
                    f"Domain label must be in [0, {num_domains - 1}], "
                    f"but got min={domain.min().item()}, max={domain.max().item()}. "
                    f"Check datasets/hust_image.py domain mapping."
                )

            out = model(img)

            z_c_notd = out["z_c_notd"]
            z_notc_d = out["z_notc_d"]
            mc = out["mc"]
            md = out["md"]

            # class-relevant, domain-invariant feature should predict class
            class_logits = model.classifier(z_c_notd)

            # domain-relevant feature should predict domain
            domain_logits = model.domain_classifier(z_notc_d)

            class_loss = ce(class_logits, label)
            domain_loss = ce(domain_logits, domain)

            # reduce overlap between class mask and domain mask
            mask_loss = torch.mean(mc * md)

            loss = class_loss + args.domain_weight * domain_loss + args.mask_weight * mask_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_class_loss += class_loss.item()
            total_domain_loss += domain_loss.item()
            total_mask_loss += mask_loss.item()

            pred_cls = class_logits.argmax(dim=1)
            pred_dom = domain_logits.argmax(dim=1)

            correct_cls += (pred_cls == label).sum().item()
            correct_dom += (pred_dom == domain).sum().item()
            total += label.size(0)

        avg_loss = total_loss / len(loader)
        avg_class_loss = total_class_loss / len(loader)
        avg_domain_loss = total_domain_loss / len(loader)
        avg_mask_loss = total_mask_loss / len(loader)

        cls_acc = correct_cls / total
        dom_acc = correct_dom / total

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"Loss: {avg_loss:.4f} | "
            f"Class: {avg_class_loss:.4f} | "
            f"Domain: {avg_domain_loss:.4f} | "
            f"Mask: {avg_mask_loss:.4f} | "
            f"Cls Acc: {cls_acc:.4f} | "
            f"Dom Acc: {dom_acc:.4f}"
        )

    save_dir = os.path.dirname(args.save_path)
    if save_dir != "":
        os.makedirs(save_dir, exist_ok=True)

    torch.save(model.state_dict(), args.save_path)

    print("===================================")
    print("Training finished")
    print("Saved:", args.save_path)
    print("===================================")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default="processed")

    parser.add_argument(
        "--train_domains",
        type=str,
        nargs="+",
        required=True,
        help="Domains used for training, e.g., 400 500 600 700"
    )

    parser.add_argument(
        "--all_domains",
        type=str,
        nargs="+",
        default=["400", "500", "600", "700", "800"],
        help="All domain list for domain classifier mapping"
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mask_weight", type=float, default=0.1)
    parser.add_argument("--domain_weight", type=float, default=1.0)
    parser.add_argument("--save_path", type=str, default="decomposition.pth")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()