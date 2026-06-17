import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class HUSTDataset(Dataset):
    def __init__(
        self,
        root="./processed",
        domain=None,
        only_normal=False,
        shot=None,
        transform=None,
        seed=42,
    ):
        self.samples = []

        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        # Fixed domain mapping
        self.domain_map = {
            "400": 0,
            "500": 1,
            "600": 2,
            "700": 3,
            "800": 4,
        }

        if domain is not None:
            domain_list = [str(domain)]
        else:
            domain_list = sorted([
                d for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d))
            ])

        for domain_name in domain_list:
            domain_name = str(domain_name)

            if domain_name not in self.domain_map:
                continue

            domain_idx = self.domain_map[domain_name]
            domain_path = os.path.join(root, domain_name)

            if not os.path.isdir(domain_path):
                continue

            for label_name in ["normal", "anomaly"]:
                if only_normal and label_name != "normal":
                    continue

                label_path = os.path.join(domain_path, label_name)

                if not os.path.exists(label_path):
                    continue

                label = 0 if label_name == "normal" else 1

                for fname in sorted(os.listdir(label_path)):
                    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                        path = os.path.join(label_path, fname)
                        self.samples.append(
                            (path, label, domain_idx, domain_name)
                        )

        if shot is not None:
            normal_samples = [s for s in self.samples if s[1] == 0]

            random.seed(seed)
            random.shuffle(normal_samples)

            self.samples = normal_samples[:shot]

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found. root={root}, domain={domain}, "
                f"only_normal={only_normal}, shot={shot}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, domain_idx, domain_name = self.samples[idx]

        img = Image.open(path).convert("RGB")
        img = self.transform(img)

        return {
            "image": img,
            "label": label,
            "domain": domain_idx,
            "domain_name": domain_name,
            "path": path,
        }