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


class InvEncoderModel(nn.Module):
    """
    Mask decomposition 없이 projection head로 domain-invariant feature 학습.
    GRL을 z_inv에 직접 적용 → 순환 모순 해소.

    forward output의 'z_c_notd' 키는 eval_all_folds.py 호환을 위해
    z_inv를 4D로 unsqueeze한 것 (GAP 후 inv_dim 차원 feature로 평가).
    """

    def __init__(self, num_classes=2, num_domains=15, inv_dim=512):
        super().__init__()

        backbone = resnet50(pretrained=True)
        self.encoder = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )

        # 2048 → inv_dim: domain-invariant feature space
        self.inv_proj = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, inv_dim),
            nn.BatchNorm1d(inv_dim),
        )

        self.classifier = nn.Linear(inv_dim, num_classes)

        # GRL applied directly on z_inv → true domain adversarial on the feature we use
        self.domain_classifier = nn.Sequential(
            nn.Linear(inv_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_domains),
        )

    def forward(self, x, alpha=1.0, class_label=None):
        z = self.encoder(x)                                 # [B, 2048, 7, 7]
        z_pool = F.adaptive_avg_pool2d(z, 1).flatten(1)    # [B, 2048]

        z_inv = self.inv_proj(z_pool)                       # [B, inv_dim]

        class_logits = self.classifier(z_inv)

        z_inv_grl = grad_reverse(z_inv, alpha)
        domain_logits = self.domain_classifier(z_inv_grl)

        # z_c_notd: 4D fake tensor for eval_all_folds.py compatibility
        z_c_notd = z_inv.unsqueeze(-1).unsqueeze(-1)       # [B, inv_dim, 1, 1]

        return {
            "z_inv": z_inv,
            "z_c_notd": z_c_notd,
            "class_logits": class_logits,
            "domain_logits": domain_logits,
        }
