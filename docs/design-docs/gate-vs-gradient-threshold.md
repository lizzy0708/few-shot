# gate가 gradient-threshold(레거시 mc/md)보다 나은가 — 직접 대조 결론

## 배경

`gated-mask-architecture.md`에서 gate로 바꾼 원래 이유는 `domain_weight=0`(GRL off) 상황의
구조적 붕괴를 고치기 위해서였다. 그런데 그 붕괴는 GRL이 꺼져 있을 때만 생기는 문제라, "GRL을
켜두면 원래 gradient-threshold(레거시) 방식도 멀쩡했을 텐데, gate가 진짜 필요했나?"라는
질문이 자연스럽게 남았다(2026-08-17 세션 중 제기됨).

## 확인 순서

1. **raw z_pool vs z_inv, GRL-off 계열**: raw가 z_inv를 이김(AUROC 0.9522 vs 0.9426) —
   이 계열에서는 애초에 불변성 메커니즘이 기여하지 않고 있었다.
2. **raw z_pool vs z_inv, GRL-on 계열(gate, class_gate만)**: z_inv가 raw를 이김
   (0.9486 vs 0.9338, 단일seed) — GRL을 켜야 z_inv가 실제로 의미를 갖는다는 것 확인.
3. **gate vs gradient-threshold, 둘 다 GRL-on, 완전 동일 프로토콜(proto_beta=0.5, no
   centering, LedoitWolf, n_sigma=2.0, 3-seed 앙상블)로 직접 대조**.

## 결과

| | AUROC(4-fold avg) | 비고 |
|---|---|---|
| gate (class_gate) | **0.9540** | fold 800 포함 5개 eval-seed 전부 안정 |
| gradient-threshold (원본 mc/md) | 0.9444 | fold 800에서 5개 중 2개 eval-seed가 threshold 완전 붕괴 |

fold별 AUROC은 gate가 4개 중 3개(600/700/800)에서 이기고 500만 gradient-threshold가 근소
우위. 하지만 진짜 차이는 **안정성**이다: gradient-threshold는 fold 800처럼 어려운 도메인에서
특정 4-shot 조합을 뽑으면 Acc/F1이 0에 가깝게 완전히 무너지는 반면(`RELIABILITY.md` §9),
gate는 같은 fold에서 그런 붕괴가 전혀 없다.

## 왜 gradient-threshold만 이렇게 불안정한가

gradient-threshold의 mc/md는 샘플마다 `torch.autograd.grad`로 개별 계산되는 값이라, 이상치
gradient 하나가 4-shot support set(표본 크기 4로 극도로 작음)에 섞이면 prototype·threshold
전체를 쉽게 왜곡시킨다. gate(class_gate_net)는 학습된 고정 함수(MLP→Sigmoid)라 입력이
달라져도 출력이 매끄럽게 변하고, 특정 샘플 하나 때문에 극단적으로 튀지 않는다 — 학습
가능한 gate가 원래 목표였던 "GRL-off 붕괴 방지"뿐 아니라 **"작은 support set에 대한
강건성"**이라는, 처음엔 의도하지 않았던 이점도 갖고 있었던 셈이다.

## 결론

**gate가 필요했다.** 성능(AUROC)도 근소하게 낫고, 결정적으로 4-shot이라는 극소 표본
시나리오에서 gradient-threshold보다 훨씬 강건하다 — 논문의 "왜 gate로 설계했는가" 절에
쓸 수 있는 근거.

## 평가 스크립트 메모 (재현 시 주의)

gradient-threshold 모델(`OriginalMaskModel`) 평가 시, calib/support는 `only_normal=True`로
이미 라벨이 확정(0)되어 있으므로 `class_label=0`을 명시해서 mc를 계산해야 한다
(`CLAUDE.md` "GT label로 mc 계산" 설계 결정과 동일 이유). query만 실제로 라벨을 모르니
`class_label=None`(max-logit) 사용. 처음엔 calib/support까지 `class_label=None`으로 잘못
평가해서 fold 800/600에 훨씬 심한 threshold 붕괴가 나왔었다(수정 전: fold800 5/5 eval-seed
전부 붕괴, 수정 후: 5개 중 2개만 — 나머지는 진짜 모델의 취약성).
