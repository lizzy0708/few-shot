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
    def __init__(self, num_classes=2, num_domains=5, encoder_layer='layer4',
                 num_rpm_groups=None, hierarchical_md=False, num_batches=3):
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
        # 계층적 도메인 (선택): RPM 그룹(물리적 변동) 전용 adversarial 헤드.
        # 15-way flat 헤드는 RPM 변동과 측정 배치 변동을 동일 취급하므로,
        # RPM 축(테스트 hold-out 축)을 명시적으로 지우는 5-way 헤드를 병행한다.
        self.domain_classifier_rpm = (
            DomainClassifier(in_dim=in_dim, num_domains=num_rpm_groups)
            if num_rpm_groups else None
        )
        # 계층적 md (2단계): md를 RPM 변동(물리)과 배치 변동(측정)으로 분리 계산.
        # disc 헤드들은 z.detach() 위에서만 학습 → encoder 무영향 (1단계 GRL 방식과 다름)
        self.hierarchical_md = hierarchical_md
        if hierarchical_md:
            n_rpm = num_rpm_groups or max(num_domains // 3, 1)
            self.domain_classifier_disc_rpm = DomainClassifier(in_dim=in_dim, num_domains=n_rpm)
            self.domain_classifier_disc_batch = DomainClassifier(in_dim=in_dim, num_domains=num_batches)

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
            md_rpm, md_batch = None, None
            if self.hierarchical_md:
                # 계층적 md: RPM(물리)·배치(측정) disc에서 각각 마스크 → 합집합(max)
                z_md_r = z.detach().requires_grad_(True)
                score_r = self.domain_classifier_disc_rpm(z_md_r).max(dim=1)[0].sum()
                g_r = torch.autograd.grad(score_r, z_md_r)[0].detach()
                md_rpm = torch.relu(g_r)
                md_rpm = md_rpm / (md_rpm.amax(dim=(1, 2, 3), keepdim=True) + 1e-6)

                z_md_b = z.detach().requires_grad_(True)
                score_b = self.domain_classifier_disc_batch(z_md_b).max(dim=1)[0].sum()
                g_b = torch.autograd.grad(score_b, z_md_b)[0].detach()
                md_batch = torch.relu(g_b)
                md_batch = md_batch / (md_batch.amax(dim=(1, 2, 3), keepdim=True) + 1e-6)

                md = torch.max(md_rpm, md_batch)
            else:
                z_for_md = z.detach().requires_grad_(True)
                domain_score_disc = self.domain_classifier_disc(z_for_md).max(dim=1)[0].sum()
                grad_d = torch.autograd.grad(domain_score_disc, z_for_md)[0].detach()
                md = torch.relu(grad_d)
                md = md / (md.amax(dim=(1, 2, 3), keepdim=True) + 1e-6)

            # --- GRL adversarial path (forces encoder to be domain-blind) ---
            z_grl = grad_reverse(z, alpha)
            domain_logits = self.domain_classifier(z_grl)
            domain_logits_rpm = (
                self.domain_classifier_rpm(z_grl)
                if self.domain_classifier_rpm is not None else None
            )

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
            if self.hierarchical_md:
                domain_logits_disc_rpm = self.domain_classifier_disc_rpm(z.detach())
                domain_logits_disc_batch = self.domain_classifier_disc_batch(z.detach())
            else:
                domain_logits_disc_rpm = domain_logits_disc_batch = None

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
            "domain_logits_rpm": domain_logits_rpm,
            "domain_logits_disc_rpm": domain_logits_disc_rpm,
            "domain_logits_disc_batch": domain_logits_disc_batch,
            "md_rpm": md_rpm,
            "md_batch": md_batch,
        }
