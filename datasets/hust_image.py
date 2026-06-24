import os
import re
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Maps filename prefix → fault type index
# N=normal, I=inner race, O=outer race, B=ball, compound(IB/IO/OB)=4
_FAULT_MAP = {'N': 0, 'I': 1, 'O': 2, 'B': 3, 'IB': 4, 'IO': 4, 'OB': 4}


def _parse_fault_type(path: str) -> int:
    m = re.match(r'^([A-Z]+)\d+_', os.path.basename(path))
    if m:
        return _FAULT_MAP.get(m.group(1), -1)
    return -1


class HUSTDataset(Dataset):
    def __init__(
        self,
        root="./processed",
        domain=None,
        only_normal=False,
        shot=None,
        transform=None,
        seed=42,
        all_domains=None,
        domain_map=None,
    ):
        self.samples = []

        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        if domain_map is not None:
            self.domain_map = {str(k): int(v) for k, v in domain_map.items()}
        else:
            if all_domains is None:
                all_domains = sorted([
                    d for d in os.listdir(root)
                    if os.path.isdir(os.path.join(root, d))
                ])

            self.domain_map = {
                str(domain_name): idx
                for idx, domain_name in enumerate([str(d) for d in all_domains])
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
                raise ValueError(
                    f"Domain {domain_name} is not in domain_map. "
                    f"Available domains: {sorted(self.domain_map.keys())}"
                )

            domain_idx = self.domain_map[domain_name]
            domain_path = os.path.join(root, domain_name)

            if not os.path.isdir(domain_path):
                continue

            # RPM label (0~4): 400→0, 500→1, 600→2, 700→3, 800→4
            # Batch label (0~2): x00→0, x02→1, x04→2
            domain_num = int(domain_name)
            rpm_label   = domain_num // 100 - 4
            batch_label = (domain_num % 100) // 2

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
                        fault_type = _parse_fault_type(fname)
                        self.samples.append(
                            (path, label, domain_idx, domain_name, fault_type,
                             rpm_label, batch_label)
                        )

        if shot is not None:
            normal_samples = [s for s in self.samples if s[1] == 0]  # s[1] = binary label

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
        path, label, domain_idx, domain_name, fault_type, rpm_label, batch_label = self.samples[idx]

        img = Image.open(path).convert("RGB")
        img = self.transform(img)

        return {
            "image": img,
            "label": label,             # binary: 0=normal, 1=anomaly
            "domain": domain_idx,       # 15-way (fine) or 5-way (coarse)
            "domain_name": domain_name,
            "path": path,
            "fault_type": fault_type,   # N=0,I=1,O=2,B=3,compound=4
            "rpm_label": rpm_label,     # 5-way: 400→0, 500→1, ..., 800→4
            "batch_label": batch_label, # 3-way: x00→0, x02→1, x04→2
        }
