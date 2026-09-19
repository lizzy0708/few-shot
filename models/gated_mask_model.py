"""
gated_mask_model.py — gradient-threshold mc/md를 학습 가능한 sigmoid gate로 교체.

구조:
  - FeatureExtractor: layer3 (1024-dim, coarse5 기본값과 동일)
  - class_gate_net / domain_gate_net: pooled z(1024-d) -> MLP -> sigmoid -> [0,1] 연속 마스크
  - class_logits = classifier(z_pool * class_gate)      -> class_loss가 encoder+class_gate_net 학습
  - domain_logits(GRL) = domain_classifier(GRL(z_pool))  -> domain_loss(adversarial)가 encoder 학습 (기존과 동일, gate 무관)
  - domain_logits_disc = domain_classifier_disc(z_pool.detach() * domain_gate)
        -> domain_disc_loss(non-adversarial)가 domain_gate_net 학습
        (domain_gate를 adversarial 경로에 직접 넣으면 gate가 0으로 collapse해버리는
         퇴화해를 학습해버리는 문제가 있어, 대신 "이 채널들로 도메인을 맞혀봐"라는
         일반 분류 신호로 domain_gate를 학습시킴 — 어떤 채널이 도메인 정보를 담고
         있는지 학습하게 하는 게 목적이므로 이쪽이 md의 원래 의도에 더 부합)
  - z_inv = z * class_gate * (1 - domain_gate)  (기존 공식 유지, gate는 채널별로 broadcast)
  - mask_loss = mean(class_gate * domain_gate)  (orthogonality, 완전히 미분 가능 — 기존처럼
        autograd.grad 2-pass 트릭 불필요)

  use_domain_gate=False 옵션: z_inv = z * class_gate만 사용(도메인 게이트 항 제거).
  domain_classifier_disc/domain_gate_net 자체를 생성하지 않으므로 mask orthogonality
  loss도 자동으로 없음(mask_loss=0 고정). domain_classifier(GRL)를 통한 encoder
  domain-invariance 학습은 옵션과 무관하게 항상 유지됨.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.original_mask_model import FeatureExtractor, grad_reverse


class GateNet(nn.Module):
    def __init__(self, in_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_dim),
            nn.Sigmoid(),
        )

    def forward(self, z_pool):
        return self.net(z_pool)


class GatedMaskModel(nn.Module):
    def __init__(self, num_classes=2, num_domains=5, encoder_layer='layer3', gate_hidden=256,
                 use_domain_gate=True, domain_gate_grl=False):
        super().__init__()
        self.feature_extractor = FeatureExtractor(encoder_layer=encoder_layer)
        in_dim = self.feature_extractor.out_dim
        self.classifier = nn.Linear(in_dim, num_classes)
        self.domain_classifier = nn.Linear(in_dim, num_domains)
        self.class_gate_net = GateNet(in_dim, gate_hidden)

        self.use_domain_gate = use_domain_gate
        # domain_gate_grl=False (기존): domain_gate_net은 detach된 z_pool로 도메인을 "맞히는"
        #   일반 분류 신호로 학습 -> 도메인 식별에 쓰이는 채널을 찾아내는 것이 목적, z_inv에서만 억제.
        # domain_gate_grl=True (신규): encoder는 여전히 detach로 보호하되, domain_gate_net 자신에게는
        #   GRL을 걸어 "이 게이트를 곱하면 도메인을 못 맞히게" adversarial하게 직접 학습시킴 —
        #   gate가 곧바로 domain-invariance를 목적함수로 갖게 됨 (기존의 2단계 방식과 다름).
        self.domain_gate_grl = domain_gate_grl
        if use_domain_gate:
            self.domain_classifier_disc = nn.Linear(in_dim, num_domains)
            self.domain_gate_net = GateNet(in_dim, gate_hidden)

    def forward(self, x, alpha=1.0):
        z = self.feature_extractor(x)                       # [B, C, H, W]
        z_pool = F.adaptive_avg_pool2d(z, 1).flatten(1)      # [B, C]

        class_gate = self.class_gate_net(z_pool)             # [B, C] in (0,1)
        class_logits = self.classifier(z_pool * class_gate)

        # domain_classifier/GRL is always trained (encoder domain-invariance pressure,
        # identical to the use_domain_gate=True setup) — independent of whether domain_gate
        # is applied to z_inv.
        domain_logits = self.domain_classifier(grad_reverse(z_pool, alpha))

        mc = class_gate.unsqueeze(-1).unsqueeze(-1)

        if self.use_domain_gate:
            domain_gate = self.domain_gate_net(z_pool)                     # [B, C] in (0,1)
            masked = z_pool.detach() * domain_gate      # encoder always protected (detach)
            if self.domain_gate_grl:
                domain_logits_disc = self.domain_classifier_disc(grad_reverse(masked, alpha))
            else:
                domain_logits_disc = self.domain_classifier_disc(masked)
            md = domain_gate.unsqueeze(-1).unsqueeze(-1)
            z_inv = z * mc * (1.0 - md)
            mask_loss = (class_gate * domain_gate).mean()
            md_out = domain_gate
        else:
            # z_inv = z * class_gate only — no domain_gate term, so orthogonality
            # loss (class_gate vs domain_gate) is meaningless and dropped automatically.
            domain_logits_disc = None
            z_inv = z * mc
            mask_loss = torch.tensor(0.0, device=z.device)
            md_out = None

        return {
            "z": z,
            "z_pool": z_pool,
            "z_c_notd": z_inv,
            "z_inv": z_inv,
            "class_logits": class_logits,
            "domain_logits": domain_logits,
            "domain_logits_disc": domain_logits_disc,
            "mc": class_gate,
            "md": md_out,
            "mask_loss": mask_loss,
        }
