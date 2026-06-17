import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


# Gradient Reversal Layer
class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha=1.0):
    return GRL.apply(x, alpha)


# Encoder: feature z 생성
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = resnet50(pretrained=True)

        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
        )

    def forward(self, x):
        return self.encoder(x)  # [B, 1024, H, W]


# Class Classifier: normal / anomaly
class ClassClassifier(nn.Module):
    def __init__(self, in_dim=1024, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, z):
        z_pool = F.adaptive_avg_pool2d(z, 1).flatten(1)
        return self.fc(z_pool)


# Domain Classifier: 400 / 500 / 600 / 700 / 800
class DomainClassifier(nn.Module):
    def __init__(self, in_dim=1024, num_domains=5):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_domains)

    def forward(self, z):
        z_pool = F.adaptive_avg_pool2d(z, 1).flatten(1)
        return self.fc(z_pool)


class MaskDecompositionModel(nn.Module):
    def __init__(self, num_classes=2, num_domains=5):
        super().__init__()

        self.feature_extractor = FeatureExtractor()
        self.classifier = ClassClassifier(in_dim=1024, num_classes=num_classes)
        self.domain_classifier = DomainClassifier(in_dim=1024, num_domains=num_domains)

    def forward(self, x, alpha=1.0):
        # 1. Feature extraction
        z = self.feature_extractor(x)  # [B, C, H, W]
        z.requires_grad_(True)

        # 2. Class prediction
        class_logits = self.classifier(z)

        # 3. Domain prediction with GRL
        z_grl = grad_reverse(z, alpha)
        domain_logits = self.domain_classifier(z_grl)

        # 4. Importance score 계산
        # class/domain prediction에 크게 기여한 feature 위치를 gradient로 계산
        class_score = class_logits.max(dim=1)[0].sum()
        domain_score = domain_logits.max(dim=1)[0].sum()

        grad_c = torch.autograd.grad(
            class_score,
            z,
            create_graph=True,
            retain_graph=True
        )[0]

        grad_d = torch.autograd.grad(
            domain_score,
            z,
            create_graph=True,
            retain_graph=True
        )[0]

        # 5. Soft mask 생성
        # mc: class-relevant mask
        # md: domain-relevant mask
        mc = torch.relu(grad_c)
        md = torch.relu(grad_d)

        mc = mc / (mc.amax(dim=(1, 2, 3), keepdim=True) + 1e-6)
        md = md / (md.amax(dim=(1, 2, 3), keepdim=True) + 1e-6)

        # 6. Feature decomposition
        # class + domain feature
        z_cd = z * mc * md

        # class-relevant & domain-invariant feature
        # 논문에서 memory bank에 저장할 핵심 feature
        z_c_notd = z * mc * (1 - md)

        # domain-relevant feature
        z_notc_d = z * (1 - mc) * md

        # irrelevant feature
        z_notc_notd = z * (1 - mc) * (1 - md)

        # 7. Proposed feature
        # 본 연구의 domain-invariant normal feature
        z_inv = z_c_notd

        return {
            "z": z,
            "class_logits": class_logits,
            "domain_logits": domain_logits,
            "mc": mc,
            "md": md,
            "z_cd": z_cd,
            "z_c_notd": z_c_notd,
            "z_notc_d": z_notc_d,
            "z_notc_notd": z_notc_notd,
            "z_inv": z_inv,
        }