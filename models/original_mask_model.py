"""
original_mask_model.py — 5/1 checkpoint 구조 재현.

구조:
  - FeatureExtractor: layer3 (1024-dim)
  - ClassClassifier: 1024 → 2 (binary)
  - DomainClassifier: 1024 → num_domains (GRL)
  - z_inv = z * mc * (1 - md)  (단순 element-wise, Gram-Schmidt 없음)
  - md: domain_classifier forward (no GRL) 로 gradient 계산
  - GRL: encoder를 domain-blind으로 만드는 adversarial loss용
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


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


class FeatureExtractor(nn.Module):
    def __init__(self, encoder_layer='layer3'):
        super().__init__()
        backbone = resnet50(pretrained=True)
        layers = [
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3,
        ]
        if encoder_layer == 'layer4':
            layers.append(backbone.layer4)
            self.out_dim = 2048
        else:
            self.out_dim = 1024
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class OriginalMaskModel(nn.Module):
    def __init__(self, num_classes=2, num_domains=5, encoder_layer='layer3'):
        super().__init__()
        self.feature_extractor = FeatureExtractor(encoder_layer=encoder_layer)
        in_dim = self.feature_extractor.out_dim
        self.classifier = nn.Linear(in_dim, num_classes)
        self.domain_classifier = nn.Linear(in_dim, num_domains)

    def forward(self, x, alpha=1.0, class_label=None):
        with torch.enable_grad():
            z = self.feature_extractor(x)  # [B, C, H, W]
            z.requires_grad_(True)
            B = z.size(0)

            z_pool = F.adaptive_avg_pool2d(z, 1).flatten(1)

            # --- mc: class gradient mask ---
            class_logits = self.classifier(z_pool)
            if class_label is not None:
                class_score = class_logits[torch.arange(B, device=z.device), class_label].sum()
            else:
                class_score = class_logits.max(dim=1)[0].sum()

            grad_c = torch.autograd.grad(
                class_score, z, create_graph=True, retain_graph=True
            )[0]
            mc = torch.relu(grad_c)
            mc = mc / (mc.amax(dim=(1, 2, 3), keepdim=True) + 1e-6)

            # --- md: domain gradient mask (forward without GRL) ---
            z_for_md = z.detach().requires_grad_(True)
            z_pool_md = F.adaptive_avg_pool2d(z_for_md, 1).flatten(1)
            domain_score = self.domain_classifier(z_pool_md).max(dim=1)[0].sum()
            grad_d = torch.autograd.grad(domain_score, z_for_md)[0].detach()
            md = torch.relu(grad_d)
            md = md / (md.amax(dim=(1, 2, 3), keepdim=True) + 1e-6)

            # --- z_inv = z * mc * (1 - md) ---
            z_inv = z * mc * (1.0 - md)

            # --- GRL domain loss ---
            z_grl = grad_reverse(z, alpha)
            z_pool_grl = F.adaptive_avg_pool2d(z_grl, 1).flatten(1)
            domain_logits = self.domain_classifier(z_pool_grl)

        return {
            "z": z,
            "z_c_notd": z_inv,
            "z_inv": z_inv,
            "class_logits": class_logits,
            "domain_logits": domain_logits,
            "mc": mc,
            "md": md,
        }
