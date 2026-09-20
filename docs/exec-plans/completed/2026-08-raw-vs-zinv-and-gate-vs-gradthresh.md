# 도메인 불변성 실증 + gate vs gradient-threshold 대조 (2026-08-17, 완료)

## 계기

당시 세션 최고(noGRL 3-seed 앙상블, AUROC 0.9426)가 "도메인 불변성 기반 few-shot
이상탐지"라는 연구 목표에 실제로 부합하는지 질문받음 — 그 기록을 만든 두 요소(3-seed
앙상블, proto_beta)가 domain-invariance와 무관한 일반적 개선이었기 때문.

## 실행 순서와 결과

1. **raw z_pool vs z_inv, noGRL 계열 대조** (같은 체크포인트, proto_beta+3-seed 앙상블
   동일 적용, z_inv on/off만 다르게): **raw가 이김**(AUROC 0.9522 vs 0.9426, 4-fold 중
   3개에서 raw 우위) — 이 기록은 도메인 불변성이 기여했다는 근거가 안 됨. 원인:
   noGRL 체크포인트는 `domain_weight=0`이라 encoder 자체가 domain-adversarial 학습을
   받은 적이 없음.
2. **같은 대조를 GRL-on 계열(`gated_nodg`, class_gate만)에 단일seed로 재확인**: **z_inv가
   이김**(0.9486 vs 0.9338) — GRL을 켜야 불변성이 실제로 작동.
3. **GRL-on 계열도 3-seed 앙상블까지 완성**하기 위해 `gated_nodg` seed 1,2 학습(8런).
   결과: AUROC **0.9540**/Acc 0.9053/F1 0.9429 — 이전 최고(0.9426) 갱신, 그리고 이번엔
   불변성 근거가 정당한 최고기록.
4. **"gate가 굳이 필요했나"** 질문 제기 → gate는 애초에 `domain_weight=0`일 때만 생기는
   구조적 버그를 고치려고 도입한 것이라, GRL-on에서는 레거시 gradient-threshold(mc/md,
   `OriginalMaskModel`)도 문제없을 수 있다는 가설. 기존 학습 스크립트
   (`train_original_mask.py`)로 동일 레시피 3-seed×4fold(12런) 재학습.
5. **동일 프로토콜로 gate vs gradient-threshold 직접 대조**: gate가 AUROC(0.9540 vs
   0.9444)와 안정성(fold 800에서 gradient-threshold만 5개 중 2개 eval-seed 완전 붕괴) 둘 다
   앞섬 → **gate가 실제로 더 나은 설계였다**는 결론.
6. 평가 스크립트 버그 한 번 잡음: gradient-threshold는 calib/support(라벨 이미 확정)
   추출 시에도 label-free(max-logit)로 mc를 계산해 불필요한 불안정을 만들었음 — GT label
   명시로 수정(`RELIABILITY.md` §9).

## 최종 결론 (2026-08-17 확정)

- **새 세션 최고**: gate(class_gate) + GRL-on + 3-seed 앙상블 + proto_beta=0.5,
  AUROC 0.9540 / Acc 0.9053 / F1 0.9429.
- 이 기록은 (a) 도메인 불변성이 실제로 기여한다는 근거가 있고, (b) gate가 레거시
  gradient-threshold보다 성능·안정성 둘 다 앞선다는 근거도 있어 논문 서술에 안전하게
  쓸 수 있음.
- `docs/generated/results.md`에 전체 비교표, `docs/design-docs/gate-vs-gradient-threshold.md`에
  설계 근거 정리.

## 남은 것

- 이전 noGRL 최고기록(0.9426)은 "가장 높은 Acc/F1"으로는 여전히 유효하지만, 도메인
  불변성 주장의 근거로는 쓰지 않기로 함 — `docs/generated/results.md`에 그 캐비엇 명시.
- fold 800의 gradient-threshold 붕괴처럼, "4-shot 극소 표본에서의 강건성"이 이제 이
  프로젝트의 또 다른 정량화 가능한 기여 포인트가 될 수 있음 — 별도 정리 후보.
