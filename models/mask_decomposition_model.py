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
    def __init__(self, encoder_layer='layer4'):
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
        return self.encoder(x)  # [B, C, H, W]


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
    def __init__(self, num_classes=2, num_domains=5, encoder_layer='layer4'):
        super().__init__()

        self.feature_extractor = FeatureExtractor(encoder_layer=encoder_layer)
        in_dim = self.feature_extractor.out_dim
        self.classifier = ClassClassifier(in_dim=in_dim, num_classes=num_classes)
        self.domain_classifier = DomainClassifier(in_dim=in_dim, num_domains=num_domains)
        self.domain_classifier_inv = DomainClassifier(in_dim=in_dim, num_domains=num_domains)
        # Discriminative domain classifier: trained to identify domains (no GRL).
        # Used solely for computing md — gradient w.r.t. z gives true domain-relevant regions.
        # Updated only via domain_logits_disc (z.detach()) in training, so encoder is unaffected.
        self.domain_classifier_disc = DomainClassifier(in_dim=in_dim, num_domains=num_domains)

    def forward(self, x, alpha=1.0, class_label=None):
        with torch.enable_grad():
            z = self.feature_extractor(x)  # [B, C, H, W]
            z.requires_grad_(True)
            B = z.size(0)

            # --- mc: class gradient mask ---
            class_logits = self.classifier(z)
            if class_label is not None:
                class_score = class_logits[torch.arange(B, device=z.device), class_label].sum()
            else:
                class_score = class_logits.max(dim=1)[0].sum()

            grad_c = torch.autograd.grad(
                class_score, z, create_graph=True, retain_graph=True
            )[0]
            mc = torch.relu(grad_c)
            mc = mc / (mc.amax(dim=(1, 2, 3), keepdim=True) + 1e-6)

            # --- md: domain gradient mask from discriminative classifier ---
            # z_for_md is detached so this gradient does not affect the encoder;
            # grad_d is also detached so md is a fixed mask during the main backward pass.
            z_for_md = z.detach().requires_grad_(True)
            domain_score_disc = self.domain_classifier_disc(z_for_md).max(dim=1)[0].sum()
            grad_d = torch.autograd.grad(domain_score_disc, z_for_md)[0].detach()
            md = torch.relu(grad_d)
            md = md / (md.amax(dim=(1, 2, 3), keepdim=True) + 1e-6)

            # --- GRL adversarial path (forces encoder to be domain-blind) ---
            z_grl = grad_reverse(z, alpha)
            domain_logits = self.domain_classifier(z_grl)

            # --- Gradient orthogonalization: remove domain direction from mc ---
            # Gram-Schmidt: mc_orth = mc - proj(mc onto md)
            mc_flat = mc.view(B, -1)
            md_flat = md.view(B, -1)
            md_unit = F.normalize(md_flat, dim=1, eps=1e-6)
            mc_orth_flat = mc_flat - (mc_flat * md_unit).sum(dim=1, keepdim=True) * md_unit
            mc_orth = mc_orth_flat.view_as(mc).relu()
            mc_orth = mc_orth / (mc_orth.amax(dim=(1, 2, 3), keepdim=True) + 1e-6)

            z_inv = z * mc_orth  # class-relevant, domain-orthogonal

            # --- optional: GRL on z_inv ---
            z_inv_grl = grad_reverse(z_inv, alpha)
            domain_logits_inv = self.domain_classifier_inv(z_inv_grl)

            # --- disc classifier training signal (z.detach() → only disc weights update) ---
            domain_logits_disc = self.domain_classifier_disc(z.detach())

        return {
            "z": z,
            "class_logits": class_logits,
            "domain_logits": domain_logits,
            "domain_logits_disc": domain_logits_disc,
            "mc": mc,
            "md": md,
            "mc_orth": mc_orth,
            "z_mc": z * mc,           # class-relevant only (no domain removal)
            "z_mc_nomd": z * mc * (1 - md),  # Notion 6/15 원본: z * mc * (1-md)
            "z_c_notd": z_inv,   # backward compat with eval script
            "z_inv": z_inv,
            "domain_logits_inv": domain_logits_inv,
        }
