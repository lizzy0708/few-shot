# domain_gate weak-GRL 도입 + weight 스윕 — ❌ 종료 (2026-08-16~17, gate collapse로 폐기)

> **결론 먼저**: 4개 weight(0.05/0.1/0.2/0.3) 전부에서 `domain_gate`가 정확히 0.0으로
> collapse(mean=std=min=max=0.000000, 직접 forward pass로 확인). "weak"하게 준 weight가
> 실제로는 전혀 약하지 않았다 — sigmoid gate가 saturation 구간에 갇히면 gradient가 사라져
> 한번 collapse되면 못 돌아오는 구조적 불안정. weight 크기와 무관하게 100% 재현.
> **held-out 최종 평가(4-fold 3-seed) 실행 취소** — 어차피 이미 확보한 class_gate-only
> 결과와 동일할 것이 확실해 비용 대비 정보가 없음. `docs/design-docs/domain-gate-training-signal.md`
> 상세, `RELIABILITY.md` §7에 재발 방지 기록.

## 목적
`docs/design-docs/domain-gate-training-signal.md`의 (B) weak-GRL 변형이 (A) detach+직접분류보다
더 domain-invariant한 z_inv를 만드는지 확인. GRL 강도는 `domain_disc_weight ∈
{0.05, 0.1, 0.2, 0.3}`로 스윕(encoder는 항상 detach로 보호되므로 "weak"는 loss weight로
조절, alpha 스케줄은 기존과 동일 `2/(1+exp(-10p))-1`).

## 절차 (held-out을 보기 전에 weight를 확정 — `docs/design-docs/threshold-protocol.md` 원칙)

1. **스윕 학습**: 4 weight × 2 train seed = 8 런. calib 도메인 구성은 fold 800 기준
   (400/500/600/700, held-out 800은 이 단계에서 전혀 사용하지 않음) — 코드:
   `scratchpad/domain_gate_grl_sweep.py`.
2. **선택 기준 (1차, 확정 전 유일 근거)**: 각 weight의 domain probe F1
   (calib 도메인 자체에 대한 `domain_logits_disc` macro-F1, chance=1/5=0.2에 가까울수록
   좋음 — domain 정보가 잘 지워졌다는 뜻) + 2개 학습 seed 간 std(안정성). closest-to-chance
   우선, std는 tie-break.
3. weight 확정 후, 4-fold 5-seed(fold 800은 3-seed 앙상블) held-out 평가를 **그 weight로
   딱 한 번** 수행. proto_beta=0.5, no centering, LedoitWolf, n_sigma=2.0
   (`docs/exec-plans/completed/2026-08-coarse5-gated-model-pipeline.md`와 동일 프로토콜).
4. 기존 최고(β=0.5, no-centering, 단일 seed AUROC 0.9151 / 3-seed 앙상블 AUROC 0.9426)와 비교.

## 스코프 축소(의도적)

- weight 선택은 fold 800의 calib 구성 하나만 대표로 사용(4-fold 전체 스윕은 8×4×2=64 런으로
  비용 과다). 최종 채택 weight는 4-fold 전체에 동일 적용.
- domain probe F1은 별도 held-out split 없이 calib 데이터 자체에 대한 in-sample 값 —
  weight 간 **상대 비교**용으로만 사용(절대값의 낙관 편향은 모든 weight에 동일하게 적용되어
  랭킹에는 영향 없음).
- 안정성 std는 2 train seed 기준(3 seed보다 약한 추정이나, 비용 대비 판단).

## 실제 결과 (2026-08-17)

| weight | mean F1 (calib probe) | std (2 seed) | 실제 domain_gate |
|---|---|---|---|
| 0.05 | 0.1081 | 0.0000 | 0.0 (완전 collapse) |
| 0.10 | 0.1081 | 0.0000 | 0.0 (완전 collapse) |
| 0.20 | 0.1081 | 0.0000 | 0.0 (완전 collapse) |
| 0.30 | 0.1081 | 0.0000 | 0.0 (완전 collapse) |

F1이 chance(0.2)보다도 낮은 이유: domain_gate≡0이면 `domain_classifier_disc`가 항상 0벡터를
입력받아 아무것도 구분 못 하고 학습 도메인 중 최다수 도메인만 계속 찍음(로그의 DomDisc Acc가
4개 weight·2개 seed 전부에서 **동일하게 0.2757**로 수렴한 것이 그 증거 — 학습 신호가 전혀
반영되지 않고 있다는 뜻). "domain 정보가 잘 지워졌다"는 낙관적 해석이 아니라 게이트 자체가
죽은 것.

## 진단: 왜 weight를 낮춰도 collapse가 똑같이 일어났나

로그 상 mask_loss(`class_gate*domain_gate`)가 **모든 조합에서 epoch 2부터 정확히 0.0000**으로
떨어짐 — epoch 1(alpha=0, 아직 adversarial 압력 없음)에는 정상, epoch 2(alpha=0.505로 급격히
상승)에서 즉시 collapse. Sigmoid는 입력이 커지면 gradient가 0에 가까워지는(saturating) 함수라,
일단 0 근처로 밀리기 시작하면 그 지점에서 gradient가 사라져 더 이상 못 빠져나오는 흡수 상태가
된다. loss weight(0.05~0.3)를 낮추는 건 "얼마나 세게 미는가"만 조절할 뿐 "얼마나 쉽게
흡수되는가"라는 정성적 불안정성 자체는 못 건드림 — 그래서 weight를 6배 차이로 줘도 결과가
안 바뀜.

## 이 결과가 뒷받침하는 것

`docs/design-docs/domain-gate-training-signal.md`의 (A) detach+직접분류를 처음 채택한 이유
(collapse-to-zero 위험 회피)가 사후적으로 다시 한번 확인됨 — (B) weak-GRL은 encoder를
detach로 보호해도 domain_gate_net 자체의 collapse까지는 못 막는다는 것이 실증됨.

## 재시도한다면 (미시도, 다음 후보)

- gate 출력에 하한(floor) 강제: `domain_gate = 0.05 + 0.9*sigmoid(...)` 식으로 saturation
  자체를 봉쇄
- adversarial loss 대신 gate 값에 대한 명시적 entropy/L2 정규화로 collapse 억제
- GRL 강도(loss weight)가 아니라 alpha 스케줄 자체를 훨씬 완만하게(현재 10 epoch 내 0→1 급상승)

## 상태
- [x] 모델/학습 스크립트에 `domain_gate_grl` 플래그 추가, forward/backward 스모크 테스트 통과
- [x] 8-run 스윕 학습 완료 (`scratchpad/domain_gate_grl_sweep.log`)
- [x] weight 선택 — **불가**: 전 weight 동일 collapse, 선택할 대상 자체가 없음
- [x] held-out 평가 — **실행 취소** (collapse 상태로는 무의미)
- [x] `RELIABILITY.md` §7, `docs/research-specs/index.md` H5 갱신, `completed/`로 이동
