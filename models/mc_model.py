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


class ClassClassifier(nn.Module):
    def __init__(self, in_dim=2048, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, z):
        return self.fc(F.adaptive_avg_pool2d(z, 1).flatten(1))


class DomainClassifier(nn.Module):
    def __init__(self, in_dim=2048, num_domains=15):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_domains)

    def forward(self, z):
        return self.fc(F.adaptive_avg_pool2d(z, 1).flatten(1))


class MCModel(nn.Module):
    """
    md 제거, GRL을 z_c에 직접 적용.

    핵심 변경점 (vs MaskDecompositionModel):
    - md 계산 없음 → circular GRL 문제 해소
    - GRL을 z 전체가 아닌 z_c = z ⊙ mc에 적용
    - z_d = z ⊙ (1-mc)에 긍정적 domain classifier → domain acc 높게 유지
    - 결과: z_c는 class-relevant + domain-invariant
             z_d는 domain-relevant (high domain acc)
    """

    def __init__(self, num_classes=2, num_domains=15):
        super().__init__()

        backbone = resnet50(pretrained=True)
        self.encoder = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )

        self.classifier = ClassClassifier(in_dim=2048, num_classes=num_classes)

        # z_c 에 adversarial (GRL): z_c가 domain 구분 못하게 강제
        self.domain_classifier = DomainClassifier(in_dim=2048, num_domains=num_domains)

        # z_d 에 positive: domain 정보를 명시적으로 보존 → domain acc 높게
        self.domain_classifier_d = DomainClassifier(in_dim=2048, num_domains=num_domains)

    def forward(self, x, alpha=1.0, class_label=None):
        with torch.enable_grad():
            z = self.encoder(x)         # [B, 2048, 7, 7]
            z.requires_grad_(True)

            B = z.size(0)

            # class score for mc
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

            # z_c: class-relevant (no md → GRL on z_c handles domain invariance)
            z_c = z * mc
            # z_d: class-irrelevant (domain-relevant)
            z_d = z * (1 - mc)

            # adversarial on z_c → make z_c domain-invariant
            z_c_grl = grad_reverse(z_c, alpha)
            domain_logits = self.domain_classifier(z_c_grl)

            # positive on z_d → preserve domain info in z_d
            domain_logits_d = self.domain_classifier_d(z_d)

        return {
            "z": z,
            "z_c": z_c,
            "z_d": z_d,
            "z_c_notd": z_c,           # eval_all_folds.py 호환
            "mc": mc,
            "class_logits": class_logits,
            "domain_logits": domain_logits,        # adversarial (z_c, low domain acc)
            "domain_logits_d": domain_logits_d,    # positive  (z_d, high domain acc)
        }
