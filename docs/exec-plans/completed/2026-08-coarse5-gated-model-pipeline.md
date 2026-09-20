# coarse5 gated-model 파이프라인 확립 (2026-08-13 ~ 08-16)

## 목표
gradient-threshold mc/md의 구조적 결함(`docs/design-docs/gated-mask-architecture.md`)을
해결하고, coarse5 트랙에서 fine15 수준(또는 그 이상)의 leakage-free 성능을 확립한다.

## 진행 순서와 결과

1. **raw ResNet50+GADF baseline 정량화** — 분해+loss가 실제로 기여하는지 확인.
2. **3-stage ablation**(raw → decomposition-only → decomposition+orthogonal loss) —
   decomposition-only 단계는 stage1/stage3 결과가 거의 동일해 중간 실행 스킵(효과 없으면
   전량 실행하지 않는다는 원칙 적용).
3. **domain-centered Mahalanobis** (학습 없음, raw feature) — 유의미한 개선 확인, 특히 fold 800.
4. **GatedMaskModel 도입**(학습 가능 sigmoid gate) — `docs/design-docs/gated-mask-architecture.md`.
5. **domain_gate 억제 강도 스윕**(`z_inv = z*class_gate*(1-α·domain_gate)`, α∈{0.3,0.5,0.7,1.0})
   → **α=0.3 채택**, 이후 모든 실험의 고정값.
6. **학습 seed 분산 분해**(train-seed 3개 × eval-seed 5개) — fold 800이 학습-재현성 자체의
   문제임을 확인(다른 fold는 eval-seed 분산이 지배적).
7. **3-seed 학습 앙상블** — 특히 fold 800에서 크게 개선(AUROC 0.7748±0.0431 →
   0.8478±0.0159, domain-centering 적용).
8. **fault-type 진단**(800, 600 도메인) — Outer race(O) 결함이 병목의 실체임을 규명
   (AUROC 0.61~0.70 vs I/B/Compound 0.84~1.00). Envelope analysis로 O-fault 신호 자체는
   개선 가능하나(kurtosis 3~9배) 현재 모델로 즉시 전이되지는 않음 — 재학습이 다음 단계.
9. **proto_beta 구현**(`docs/design-docs/proto-beta-blending.md`) — β=0.5 no-centering이
   domain-centering을 앞섬.
10. **500/600/700 fold에 학습 seed 1,2 추가** → 4-fold 모두 3-seed 앙상블 가능.

## 최종 확정 수치 (2026-08-16)

3-seed 학습 앙상블 + `proto_beta=0.5`(no centering), n_sigma=2.0, LedoitWolf:

| Fold | AUROC | Acc | F1 |
|---|---|---|---|
| 500 | 0.9920±0.0002 | 0.9634±0.0040 | 0.9790±0.0022 |
| 600 | 0.9034±0.0021 | 0.8921±0.0085 | 0.9363±0.0060 |
| 700 | 0.9852±0.0044 | 0.9351±0.0197 | 0.9629±0.0108 |
| 800 | 0.8898±0.0027 | 0.8631±0.0115 | 0.9174±0.0088 |
| **Avg** | **0.9426** | **0.9134** | **0.9489** |

domain-centering 단일-seed 대비: AUROC +0.0312 / Acc +0.0327 / F1 +0.0224.
전체 표: `docs/generated/results.md`.

## 재현 명령

```bash
# 체크포인트 학습 (fold별 x seed 0/1/2)
conda run -n torch python experiments/train_gated_mask_model.py \
  --root processed --train_domains <calib 4개> --all_domains 400 500 600 700 800 \
  --epochs 10 --batch_size 16 --lr 1e-4 --num_classes 2 \
  --mask_weight 0.1 --domain_weight 0.0 --domain_disc_weight 1.0 \
  --seed {0,1,2} --save_path checkpoints/coarse5_fold{d}_gated_noGRL_s{seed}.pth

# 평가: scratchpad/ensemble_all_folds_beta05.py (3-seed 앙상블 + proto_beta=0.5 일반화 버전)
```

## 미시도 후보 (다음 단계)
`docs/research-specs/index.md` 참고. 현재 진행 중: domain_gate weak-GRL —
`docs/exec-plans/active/2026-08-domain-gate-weak-grl-sweep.md`.
