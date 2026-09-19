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

  use_gate_orth=True 옵션 (2026-09-19, Stage 2 도메인-누수 대응):
  Experiment 1/2 (도메인 불변성 정량화)에서 nodg z_inv가 독립 프로브 기준
  domain-acc 57~65%(chance 20%)로 거의 줄지 않음을 확인 — nodg에는 class_gate가
  domain 방향을 명시적으로 제거하는 항이 전혀 없었기 때문(z_inv = z*class_gate뿐,
  GRL이 encoder를 얼마나 깨끗하게 만들었는지에 전적으로 의존). fine15의 Gram-Schmidt
  (mc_orth = mc - proj(mc onto md), models/mask_decomposition_model.py 147-156행)를
  거울삼아 class_gate에도 명시적 anti-domain 직교화 항을 추가한다.
  fine15와의 차이(자세한 설계 이유는 _orthogonalize_gate 참고):
    1) fine15는 샘플별 공간(gradient) 마스크 vs 샘플별 gradient 기반 md 1개 벡터를
       직교화; 여기서는 채널 전용 class_gate([B,C], 공간 해상도 없음) vs
       domain_classifier의 학습된 가중치 행(row) 부분공간(고정, 배치 불변)을 직교화.
    2) fine15는 md 벡터 1개만 투영 제거; 여기서는 domain_classifier의 행 최대
       num_domains(5)개가 이루는 부분공간 전체를 QR로 한 번에 제거.
    3) 샘플별 autograd.grad(2-pass) 재도입은 GatedMaskModel이 애초에 gradient-threshold의
       4-shot 취약성(RELIABILITY.md §9)을 피하려고 만들어진 것과 상충되므로 의도적으로
       배제 — 대신 domain_classifier 자신의(이미 GRL로 adversarial 학습 중인) 학습된
       가중치 방향을 사용한다.
    4) 새 학습 파라미터를 추가하지 않음(domain_classifier.weight를 그대로 재사용) —
       따라서 기존 nodg 체크포인트의 state_dict key/shape와 100% 동일, use_gate_orth는
       순수 forward-pass 동작 플래그일 뿐 구조를 바꾸지 않음.
  직교화는 "class_gate를 먼저 직교화한 뒤 z에 곱한다"는 순서로 적용한다(fine15와 동일한
  "마스크를 먼저 직교화 → z에 곱한다" 순서, z_inv를 만든 뒤 사후 직교화하지 않음).
  class_logits(class_loss 경로)는 원래의 (직교화 안 된) z_pool*class_gate를 그대로
  사용 — 학습 손실/dynamics는 기존 nodg와 완전히 동일하게 유지하고, z_inv(다운스트림
  Mahalanobis 특징)만 바뀐다.
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
                 use_domain_gate=True, domain_gate_grl=False, use_gate_orth=False):
        super().__init__()
        self.feature_extractor = FeatureExtractor(encoder_layer=encoder_layer)
        in_dim = self.feature_extractor.out_dim
        self.classifier = nn.Linear(in_dim, num_classes)
        self.domain_classifier = nn.Linear(in_dim, num_domains)
        self.class_gate_net = GateNet(in_dim, gate_hidden)

        # Stage-2 domain-leakage fix (2026-09-19): see module docstring "use_gate_orth=True"
        # section for full rationale + explicit delta from fine15's Gram-Schmidt mechanism.
        # No new parameters are introduced by this flag — it only changes forward-pass
        # behavior, so old (non-orth) checkpoints remain structurally loadable.
        self.use_gate_orth = use_gate_orth

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

    def _orthogonalize_gate_against_domain(self, gate):
        """
        Stage-2 anti-domain orthogonalization (2026-09-19).

        Mirrors fine15's Gram-Schmidt step (models/mask_decomposition_model.py, lines
        147-156: `mc_orth = relu(mc - proj(mc onto md_unit)); mc_orth /= max(mc_orth)`)
        but is NOT identical -- differences, stated explicitly per the design requirement:

          1. fine15 projects out a single per-sample GRADIENT-based domain-relevance
             vector `md` (torch.autograd.grad of the domain score w.r.t. the full spatial
             z tensor, recomputed every forward call). This method instead projects out
             the FIXED (batch-invariant, no autograd.grad) subspace spanned by
             `domain_classifier.weight`'s rows -- the directions the domain classifier
             itself currently uses to discriminate domains, already being pushed toward
             domain-invariance via the GRL loss on z_pool.
          2. fine15 has exactly one md vector per sample to remove. Here there can be up
             to num_domains (5) directions; they are orthogonalized against as a whole
             subspace via QR decomposition (torch.linalg.qr), not a single projection.
          3. This operates on the channel-only class_gate ([B, C], no spatial dimension --
             GatedMaskModel's gate has no spatial resolution to begin with, unlike fine15's
             full [B,C,H,W] gradient masks), not on a spatial mask.
          4. Deliberately NOT a per-sample torch.autograd.grad mechanism: GatedMaskModel
             was introduced specifically to move away from that fragility on 4-shot samples
             (RELIABILITY.md §9); reintroducing it here for "domain md" would regress that.

        Applied to the RAW gate (before the gate multiplies into z), matching fine15's
        "orthogonalize the mask, then multiply" order -- not "multiply then orthogonalize
        the result". See module docstring for the reasoning (Experiment 2 found gate values
        are already ~80% near-extreme 0/1; orthogonalizing post-multiply would mostly act on
        already-near-zero channels and barely touch the selection itself).
        """
        W = self.domain_classifier.weight  # [num_domains, C]
        # Orthonormal basis (columns of Q) spanning the row space of W, via QR on W^T.
        Q, _ = torch.linalg.qr(W.t().float(), mode="reduced")  # Q: [C, k], k = min(C, num_domains)
        Q = Q.to(gate.dtype)
        proj = (gate @ Q) @ Q.t()          # [B, C] component of gate lying in the domain subspace
        gate_orth = gate - proj
        gate_orth = F.relu(gate_orth)
        gate_orth = gate_orth / (gate_orth.amax(dim=1, keepdim=True) + 1e-6)
        return gate_orth

    def forward(self, x, alpha=1.0):
        z = self.feature_extractor(x)                       # [B, C, H, W]
        z_pool = F.adaptive_avg_pool2d(z, 1).flatten(1)      # [B, C]

        class_gate = self.class_gate_net(z_pool)             # [B, C] in (0,1)
        # class_logits ALWAYS uses the raw (non-orthogonalized) gated feature -- training
        # signal for class_loss/domain_loss/domain_disc_loss is identical to the existing
        # nodg recipe regardless of use_gate_orth. Only z_inv (below) differs.
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
            if self.use_gate_orth:
                class_gate_orth = self._orthogonalize_gate_against_domain(class_gate)
                mc_orth = class_gate_orth.unsqueeze(-1).unsqueeze(-1)
                z_inv = z * mc_orth
            else:
                class_gate_orth = None
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
            "mc_orth": class_gate_orth if self.use_gate_orth else None,
            "md": md_out,
            "mask_loss": mask_loss,
        }
