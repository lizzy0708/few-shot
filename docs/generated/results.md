# 확정 결과 (자동 갱신 대상 — 새 공식 수치 나올 때마다 갱신)

수기 관리 중 (별도 자동 생성 스크립트 없음 — TODO: eval 로그에서 파싱하는 스크립트 추가 고려).

## fine15 (논문 본편, `checkpoints/fine15_fold*.pth`, 수정 금지)

| Fold | Base AUROC | Base Acc | Base F1 | z_inv AUROC | z_inv Acc | z_inv F1 |
|------|-----------|---------|--------|------------|----------|---------|
| 500  | 0.8848 | 0.8357 | 0.8987 | 0.9507 | 0.8847 | 0.9357 |
| 600  | 0.9013 | 0.8497 | 0.9092 | 0.9260 | 0.8666 | 0.9252 |
| 700  | 0.8918 | 0.8338 | 0.8998 | 0.9086 | 0.8607 | 0.9197 |
| 800  | 0.9050 | 0.8645 | 0.9216 | 0.9104 | 0.8611 | 0.9192 |
| **Avg** | 0.8957 | 0.8459 | 0.9073 | **0.9239** | **0.8683** | **0.9250** |

## coarse5 legacy (Youden threshold, label-leaking — 참고용, 논문 인용 금지)

| Fold | AUROC | Acc | F1 |
|------|-------|-----|-----|
| 500  | 0.9898 | 0.9452 | 0.9672 |
| 600  | 0.9321 | 0.8247 | 0.8862 |
| 700  | 0.9992 | 0.9013 | 0.9389 |
| 800  | 0.8588 | 0.6620 | 0.7502 |
| **Avg** | 0.9450 | 0.8333 | 0.8856 |

## coarse5 gated (n_sigma=2.0, leakage-free) — 현재 활성 개발 트랙

### 세션 최고 (2026-08-17): gate(class_gate) + GRL-on 3-seed 앙상블 + proto_beta=0.5, no-centering

`checkpoints/coarse5_fold*_gated_nodg_s{0,1,2}.pth` (`use_domain_gate=False`, `domain_weight=1.0`).

| Fold | AUROC | Acc | F1 |
|------|-------|-----|-----|
| 500  | 0.9829±0.0020 | 0.9533±0.0039 | 0.9731±0.0023 |
| 600  | 0.9500±0.0018 | 0.9034±0.0174 | 0.9429±0.0123 |
| 700  | 0.9820±0.0039 | 0.9323±0.0124 | 0.9611±0.0069 |
| 800  | 0.9008±0.0015 | 0.8325±0.0266 | 0.8948±0.0201 |
| **Avg** | **0.9540** | **0.9053** | **0.9429** |

AUROC 기준 세션 최고. `docs/design-docs/gate-vs-gradient-threshold.md`에 이 결과에 이르게 된
raw-vs-z_inv 대조실험 경위와, gate가 gradient-threshold(레거시)보다 나은 이유(안정성) 기록.

### 이전 최고 (2026-08-16, GRL-off): 3-seed 앙상블 + proto_beta=0.5, no-centering

`checkpoints/coarse5_fold*_gated_noGRL_s{0,1,2}.pth` (domain_weight=0.0). **주의**: 이 계열은
raw-vs-z_inv 대조실험에서 raw feature가 z_inv를 이겨(AUROC 0.9522 vs 0.9426) 도메인 불변성
기여가 실증되지 않음 — Acc/F1은 GRL-on보다 근소 우위지만 "도메인 불변성 기반" 주장의 근거로는
부적합. 상세: `docs/exec-plans/completed/2026-08-raw-vs-zinv-and-gate-vs-gradthresh.md`.

| Fold | AUROC | Acc | F1 |
|------|-------|-----|-----|
| 500  | 0.9920±0.0002 | 0.9634±0.0040 | 0.9790±0.0022 |
| 600  | 0.9034±0.0021 | 0.8921±0.0085 | 0.9363±0.0060 |
| 700  | 0.9852±0.0044 | 0.9351±0.0197 | 0.9629±0.0108 |
| 800  | 0.8898±0.0027 | 0.8631±0.0115 | 0.9174±0.0088 |
| **Avg** | 0.9426 | **0.9134** | **0.9489** |

### gradient-threshold(레거시 mc/md) + GRL-on, 3-seed 앙상블 (비교용, gate가 이김)

`checkpoints/coarse5_fold*_origmask_s{0,1,2}.pth` (`OriginalMaskModel`, 동일 프로토콜).

| Fold | AUROC | Acc | F1 |
|------|-------|-----|-----|
| 500  | 0.9876±0.0010 | 0.9649±0.0033 | 0.9797±0.0020 |
| 600  | 0.9440±0.0018 | 0.9069±0.0072 | 0.9456±0.0050 |
| 700  | 0.9672±0.0081 | 0.9167±0.0144 | 0.9518±0.0079 |
| 800  | 0.8786±0.0027 | 0.5898±0.3549\* | 0.5588±0.4561\* |
| **Avg** | 0.9444 | 0.8446\* | 0.8590\* |

\* fold 800에서 5개 eval-seed 중 2개가 threshold 완전 붕괴(Acc 0.155/F1 0.000) —
gradient-threshold 고유의 취약성, `RELIABILITY.md` §9.

### raw z_pool (분해 없음) vs z_inv(분해) — GRL on/off 대조

| 학습 설정 | raw z_pool AUROC | z_inv AUROC | 승자 |
|---|---|---|---|
| GRL off (domain_weight=0), 3-seed | **0.9522** | 0.9426 | raw |
| GRL on (domain_weight=1.0), 단일seed | 0.9338 | 0.9486 | z_inv |
| **GRL on (domain_weight=1.0), 3-seed 완전 매칭** | 0.9450 | **0.9540** | **z_inv** |

3-seed 앙상블까지 양쪽 다 완전히 맞춘 최종 비교(2026-08-17): Acc(0.8684→0.9053, +0.0369),
F1(0.9174→0.9429, +0.0255)도 z_inv가 전 지표 우위. fold별로는 500/600/700에서 z_inv 승,
800만 raw가 근소 우위. 도메인 불변성(z_inv)이 실제로 기여하려면 encoder가 GRL로
domain-adversarial 학습을 받아야 한다는 근거. 상세: `docs/design-docs/gate-vs-gradient-threshold.md`.

### 절대 baseline (단일 seed, 참고용)

| 설정 | AUROC | Acc | F1 |
|---|---|---|---|
| raw ResNet50+GADF (학습 전혀 안 함, 분해 없음) | ~0.83* | ~0.75* | ~0.83* |
| domain-centering (raw feature, 학습 없음) | 개선 확인(정확 수치는 Notion) | | |
| gated class_gate+domain_gate(α=0.3)+centering, GRL-off | 0.9114 | 0.8807 | 0.9265 |

\* raw baseline 수치는 3-stage ablation 실험 기준 근사치 — 정확한 수치는 Notion
"⚠️ Raw ResNet50+GADF vs 현재 방법 정량 비교" 참고.

### 종료된 방향
domain_gate weak-GRL 스윕 — collapse로 폐기, `RELIABILITY.md` §7,
`docs/exec-plans/completed/2026-08-domain-gate-weak-grl-sweep.md`.
