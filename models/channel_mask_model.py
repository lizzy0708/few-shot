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
    def __init__(self):
        super().__init__()
        backbone = resnet50(pretrained=True)
        self.encoder = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )

    def forward(self, x):
        return self.encoder(x)  # [B, 2048, 7, 7]


class ChannelMaskModel(nn.Module):
    """
    채널 단위 마스크 + GRL 도메인 불변 모델.

    mc  : 공간 gradient를 채널 방향 mean-pool → 채널별 class 중요도 [B, 2048]
    md  : epoch 단위로 전체 정상 샘플의 도메인 간 채널 분산 → 도메인 민감 채널 [2048]
    GRL : encoder-level domain 불변성 (fine15와 동일)

    z_inv = z_pool * mc * (1 − md)

    md 출처가 domain classifier gradient가 아니라 실제 feature 분산이므로
    GRL 충돌 없음. 추론 시에도 calib 도메인 통계로 md 계산 가능.
    """

    def __init__(self, num_classes=2, num_domains=12):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.classifier = nn.Linear(2048, num_classes)
        self.domain_classifier = nn.Linear(2048, num_domains)

    def forward(self, x, class_label=None, md=None, alpha=1.0):
        with torch.enable_grad():
            z = self.feature_extractor(x)   # [B, 2048, 7, 7]
            z.requires_grad_(True)
            B = z.size(0)

            z_pool = F.adaptive_avg_pool2d(z, 1).flatten(1)  # [B, 2048]
            class_logits = self.classifier(z_pool)

            if class_label is not None:
                class_score = class_logits[torch.arange(B, device=z.device), class_label].sum()
            else:
                class_score = class_logits.max(dim=1)[0].sum()

            # mc: 공간 gradient → 채널 방향 집계 (GradCAM weight)
            grad_c = torch.autograd.grad(
                class_score, z, create_graph=True, retain_graph=True
            )[0]                                        # [B, 2048, 7, 7]
            mc = torch.relu(grad_c).mean(dim=(2, 3))   # [B, 2048]
            mc = mc / (mc.amax(dim=1, keepdim=True) + 1e-6)

            # GRL: encoder-level 도메인 불변성
            domain_logits = self.domain_classifier(grad_reverse(z_pool, alpha))

            # z_inv: class 중요 채널 유지, domain 민감 채널 억제
            if md is not None:
                md_bc = md.unsqueeze(0).expand(B, -1).clamp(0.0, 1.0)
                z_inv = z_pool * mc * (1.0 - md_bc)
            else:
                z_inv = z_pool * mc

        return {
            "z_pool": z_pool,
            "class_logits": class_logits,
            "domain_logits": domain_logits,
            "mc": mc,
            "z_inv": z_inv,
            "z_c_notd": z_inv.unsqueeze(-1).unsqueeze(-1),  # eval 호환 [B,2048,1,1]
        }

    @staticmethod
    def compute_md(z_pool, labels, domains):
        """
        정상 샘플을 도메인별로 묶어 채널 평균을 구한 뒤
        도메인 간 채널 분산을 반환 → md [C].

        z_pool  : [N, C]  (detach 불필요, 내부에서 처리)
        labels  : [N]     0=normal, 1=anomaly
        domains : [N]     domain index
        """
        device = z_pool.device
        z_det = z_pool.detach()
        normal_mask = (labels == 0)

        domain_means = []
        for d in domains[normal_mask].unique():
            samples = z_det[normal_mask & (domains == d)]
            if len(samples) >= 1:
                domain_means.append(samples.mean(0))

        if len(domain_means) < 2:
            return torch.zeros(z_pool.size(1), device=device)

        stacked = torch.stack(domain_means)   # [n_dom, C]
        var = stacked.var(dim=0)              # [C]
        return var / (var.max() + 1e-6)
