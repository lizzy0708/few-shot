import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


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


class ViTFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)

    def forward(self, x):
        # patch tokens: [B, 196, 768] (CLS 제외)
        features = self.vit.forward_features(x)  # [B, 197, 768]
        patch_tokens = features[:, 1:, :]         # [B, 196, 768]
        # [B, 768, 14, 14] 로 reshape → avg_pool2d 등 기존 코드와 호환
        B = patch_tokens.size(0)
        return patch_tokens.permute(0, 2, 1).reshape(B, 768, 14, 14)


class ClassClassifier(nn.Module):
    def __init__(self, in_dim=768, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, z):
        z_pool = F.adaptive_avg_pool2d(z, 1).flatten(1)
        return self.fc(z_pool)


class DomainClassifier(nn.Module):
    def __init__(self, in_dim=768, num_domains=15):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_domains)

    def forward(self, z):
        z_pool = F.adaptive_avg_pool2d(z, 1).flatten(1)
        return self.fc(z_pool)


class ViTMaskDecompositionModel(nn.Module):
    def __init__(self, num_classes=2, num_domains=15):
        super().__init__()
        self.feature_extractor = ViTFeatureExtractor()
        self.classifier = ClassClassifier(in_dim=768, num_classes=num_classes)
        self.domain_classifier = DomainClassifier(in_dim=768, num_domains=num_domains)

    def forward(self, x, alpha=1.0, class_label=None):
        with torch.enable_grad():
            z = self.feature_extractor(x)  # [B, 768, 14, 14]
            z.requires_grad_(True)

            B = z.size(0)

            class_logits = self.classifier(z)

            if class_label is not None:
                class_score = class_logits[torch.arange(B, device=z.device), class_label].sum()
            else:
                class_score = class_logits.max(dim=1)[0].sum()

            z_grl = grad_reverse(z, alpha)
            domain_logits = self.domain_classifier(z_grl)
            domain_score = domain_logits.max(dim=1)[0].sum()

            # create_graph=False: ViT에서 2차 미분은 너무 느림
            # z_inv = z * mc * (1-md) → loss backprop은 z를 통해 정상 흐름
            grad_c = torch.autograd.grad(
                class_score, z, create_graph=False, retain_graph=True
            )[0]
            grad_d = torch.autograd.grad(
                domain_score, z, create_graph=False, retain_graph=True
            )[0]

            mc = torch.relu(grad_c)
            md = torch.relu(grad_d)

            mc = mc / (mc.amax(dim=(1, 2, 3), keepdim=True) + 1e-6)
            md = md / (md.amax(dim=(1, 2, 3), keepdim=True) + 1e-6)

            z_cd        = z * mc * md
            z_c_notd    = z * mc * (1 - md)
            z_notc_d    = z * (1 - mc) * md
            z_notc_notd = z * (1 - mc) * (1 - md)
            z_inv       = z_c_notd

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
