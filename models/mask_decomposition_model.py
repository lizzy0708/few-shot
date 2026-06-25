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
            backbone.layer4,
        )

    def forward(self, x):
        return self.encoder(x)  # [B, 2048, H, W]


# Class Classifier: normal / anomaly
class ClassClassifier(nn.Module):
    def __init__(self, in_dim=1024, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, z):
        z_pool = F.adaptive_avg_pool2d(z, 1).flatten(1)
        return self.fc(z_pool)


# Domain Classifier: 15-way fine-grained domain
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
        self.classifier = ClassClassifier(in_dim=2048, num_classes=num_classes)
        self.domain_classifier = DomainClassifier(in_dim=2048, num_domains=num_domains)
        # domain adversarial directly on z_inv (forces z_inv itself to be domain-blind)
        self.domain_classifier_inv = DomainClassifier(in_dim=2048, num_domains=num_domains)

    def forward(self, x, alpha=1.0, class_label=None):
        with torch.enable_grad():
            # 1. Feature extraction
            z = self.feature_extractor(x)  # [B, C, H, W]
            z.requires_grad_(True)

            B = z.size(0)

            # 2. Class logits
            class_logits = self.classifier(z)

            # 3. class_score for mc: use GT label when available
            if class_label is not None:
                class_score = class_logits[torch.arange(B, device=z.device), class_label].sum()
            else:
                class_score = class_logits.max(dim=1)[0].sum()

            # 4. domain_score for md: GRL-path (adversarial domain classifier)
            z_grl = grad_reverse(z, alpha)
            domain_logits = self.domain_classifier(z_grl)
            domain_score = domain_logits.max(dim=1)[0].sum()

            grad_c = torch.autograd.grad(
                class_score, z, create_graph=True, retain_graph=True
            )[0]

            grad_d = torch.autograd.grad(
                domain_score, z, create_graph=True, retain_graph=True
            )[0]

            # 5. Soft masks
            mc = torch.relu(grad_c)
            md = torch.relu(grad_d)

            mc = mc / (mc.amax(dim=(1, 2, 3), keepdim=True) + 1e-6)
            md = md / (md.amax(dim=(1, 2, 3), keepdim=True) + 1e-6)

            # 6. Feature decomposition
            z_cd        = z * mc * md
            z_c_notd    = z * mc * (1 - md)   # class-relevant, domain-invariant
            z_notc_d    = z * (1 - mc) * md   # domain-relevant
            z_notc_notd = z * (1 - mc) * (1 - md)
            z_inv       = z_c_notd

            # direct domain adversarial on z_inv: GRL(z_inv) → domain_classifier_inv
            z_inv_grl = grad_reverse(z_c_notd, alpha)
            domain_logits_inv = self.domain_classifier_inv(z_inv_grl)

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
            "domain_logits_inv": domain_logits_inv,
        }
