# ARCHITECTURE.md

모델 구조 단일 소스. 3개 트랙이 공존한다 — 트랙을 섞어 비교하지 않는다(`AGENTS.md` 참고).

## 공통 백본

```
입력 x [B, 3, 224, 224] (GADF/CWT 이미지)
  → ResNet50 (ImageNet pretrained)
  → conv1 → bn1 → relu → maxpool → layer1 → layer2 → layer3 (→ layer4, 옵션)
  → z [B, 1024, 14, 14]  (layer3)  또는  [B, 2048, 7, 7] (layer4)
```

domain-invariant feature `z_inv`를 뽑아 4-shot Mahalanobis few-shot 이상탐지에 사용한다는
목표는 세 트랙 공통. 마스크(`z_inv` 만드는 방법)와 학습 손실만 다르다.

---

## 트랙 1: coarse5 legacy — gradient-threshold mask (`models/original_mask_model.py`)

```
mc = ReLU(∂class_score/∂z)  / max     # class-relevant mask (gradient 기반, non-differentiable end-to-end)
md = ReLU(∂domain_score/∂z) / max     # domain-relevant mask
z_inv = z ⊙ mc ⊙ (1 - md)
```
- `create_graph=True`로 2-pass autograd 필요 (mc/md 자체가 gradient의 함수).
- 상세: `METHOD_615.md`, 공식 재현 수치: coarse5 avg AUROC 0.9450 / Acc 0.8333 / F1 0.8856 (§6.2).

## 트랙 2: fine15 확정 — 동일 gradient mask + 보조 loss (`models/mask_decomposition_model.py`)

트랙 1과 마스크 수식은 동일, ResNet50 layer4(2048-dim) + SupCon/ProtoAlign/EpisodicProto 보조 손실
추가. **논문 본편.** 상세: `CLAUDE.md`, `TRAINING.md`, `INFERENCE.md`.
확정 수치: AUROC 0.9239 / Acc 0.8683 / F1 0.9250 (fine15, 4-fold 평균, n_sigma=0.0).

## 트랙 3: coarse5 gated — learnable sigmoid gate (`models/gated_mask_model.py`, 2026-08~ 활성 개발)

gradient-threshold mc/md를 학습 가능한 sigmoid gate로 교체. 배경(왜 바꿨는지)은
`docs/design-docs/gated-mask-architecture.md`.

```
z_pool = GAP(z)                                    # [B, 1024]

class_gate = class_gate_net(z_pool)                # GateNet: MLP→Sigmoid, [B,1024] in (0,1)
class_logits = classifier(z_pool * class_gate)     # class_loss가 encoder+class_gate_net 학습

domain_logits = domain_classifier(GRL(z_pool, alpha))   # 항상 존재, domain_weight로 on/off
                                                          # (현재 최종 구조: domain_weight=0, 즉 OFF)

# domain_gate 학습 신호 — 두 변형 (docs/design-docs/domain-gate-training-signal.md):
#   (A) detach+직접분류(default, domain_gate_grl=False):
domain_gate = domain_gate_net(z_pool)
domain_logits_disc = domain_classifier_disc(z_pool.detach() * domain_gate)   # non-adversarial

#   (B) weak-GRL(domain_gate_grl=True, 2026-08-16 도입, 스윕 중):
domain_logits_disc = domain_classifier_disc(GRL(z_pool.detach() * domain_gate, alpha))
# 인코더는 항상 detach로 보호 — domain_gate_net/domain_classifier_disc만 adversarial 신호를 받음

z_inv = z * class_gate * (1 - domain_gate)          # use_domain_gate=False면 z_inv = z*class_gate만
mask_loss = mean(class_gate * domain_gate)          # orthogonality, 완전히 미분 가능
```

학습 손실:
```
L = class_loss
  + domain_weight      * domain_loss        (GRL, encoder 전체에 adversarial)
  + domain_disc_weight * domain_disc_loss    (domain_gate 학습 신호, A 또는 B)
  + mask_weight        * mask_loss
```

추론(4-shot Mahalanobis, `n_sigma=2.0` leakage-free):
```
prototype = mean(z_inv(4-shot support))
[optional] blended_prototype = proto_beta * prototype + (1-proto_beta) * calib_centroid
precision = LedoitWolf().fit(calib_domains의 z_inv).precision_
score(q) = sqrt((z_inv(q) - blended_prototype)^T precision (z_inv(q) - blended_prototype))
threshold = mean(support scores) + n_sigma * std(calib scores)
```

**현재 확정 최고 (2026-08-17, 세션 최고 기록)**: `use_domain_gate=False`(class_gate만) +
`domain_weight=1.0`(GRL on) + 학습 3-seed 앙상블 + `proto_beta=0.5`(no domain-centering),
4-fold 평균 **AUROC 0.9540 / Acc 0.9053 / F1 0.9429**. 이 설정은 raw-vs-z_inv 대조실험에서
z_inv가 실제로 raw를 이기는(도메인 불변성이 진짜 기여하는) 유일한 최고기록이자, gradient-threshold
(레거시 mc/md)와의 직접 대조에서도 AUROC·안정성 둘 다 앞선 설정이다 —
`docs/design-docs/gate-vs-gradient-threshold.md`. 전체 수치·비교는 `docs/generated/results.md`.

**(B) weak-GRL은 폐기됨** (2026-08-17): `domain_disc_weight ∈ {0.05,0.1,0.2,0.3}` 전부에서
`domain_gate`가 정확히 0.0으로 collapse — sigmoid gate가 saturation에 갇혀 gradient가 사라지는
구조적 문제. 기본값은 계속 (A) detach+직접분류. 상세: `RELIABILITY.md` §7,
`docs/exec-plans/completed/2026-08-domain-gate-weak-grl-sweep.md`.

### GateNet
```python
GateNet: Linear(in_dim, 256) → ReLU → Linear(256, in_dim) → Sigmoid
```

---

## 알려진 구조적 함정 (다시 밟지 말 것)

- **z_inv에 md를 무조건 곱하는 구조는 domain_classifier가 미학습이면 붕괴한다.**
  `domain_weight=0`(GRL off)일 때 트랙 1식 gradient-threshold md는 random-init 상태로
  z_inv를 오염시켜 fold별 극단적 bimodal 불안정을 유발했음(500/700 양호, 600/800 붕괴,
  mask_weight와 무관 — 구조적 원인, RELIABILITY.md 참고). 트랙 3의 독립적
  domain_gate_net 학습 신호(A/B)가 이 문제의 해결책.
- **mc/md는 GAP→Linear 분류기 구조상 순수 채널 마스크**(공간적 선택성 없음, gradient가
  공간 상수) — patch-level 확장이 이득 없었던 구조적 이유(`ROADMAP_EXTENSION.md` §A).
