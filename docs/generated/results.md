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

#### 실험 3 (2026-09-20): leakage-fix 스크립트로 재현 + paired 검증

`experiments/eval_gated_folds_rawz.py`(z_inv→raw z만 교체, 나머지는 `eval_gated_folds.py` 재사용)와
`eval_gated_folds.py`를 같은 환경에서 순차 실행(3 train-seed × 4 fold × 5 eval-seed, β=0.5, n_σ=2.0).
위 08-17 표를 **재현**한 것이지 새 수치가 아님 (raw 0.9449/0.8684/0.9174 ≈ 0.9450/0.8684/0.9174).

| | AUROC | Acc | F1 |
|---|---|---|---|
| z_inv (재실행) | 0.9539 | 0.9054 | 0.9430 |
| raw z | 0.9449 | 0.8684 | 0.9174 |
| Δ (z_inv − raw), paired n=20 | +0.0090 (p=0.07, 15/20승) | +0.0369 (p<1e-6, 19/20승) | +0.0256 (p<1e-6, 19/20승) |

- **Acc/F1**: 4개 fold 전부에서 z_inv 우위(fold별 +0.016~+0.048), 노이즈 범위 밖.
- **AUROC**: +0.009는 노이즈 범위 안. fold 700(+0.040)이 끌고 가고 fold 800은 반대(−0.0145, 0/5승).
  fold 단위(n=4) 기준 평균 +0.009, sd 0.023 → 유의하다고 말할 수 없음.
- 한계: eval-seed(support 재추출) 변동만 반영. 3 train-seed는 앙상블로 합쳐져 train-seed 분산은 미반영.
  raw z는 체크포인트가 z_pool*class_gate 경로로 학습된 모델의 gate 이전 feature라 "분해 없이 학습한 모델"이 아님.
- Acc/F1 차이(+0.037)가 AUROC 차이(+0.009)보다 훨씬 큰 것은 임계값 보정 차이가 섞였을 가능성을 시사
  (raw z 공간에서 support mean + 2σ가 덜 맞음) — **가설, 미검증**.
- 해석 주의: Exp.1(도메인 acc 60.8%→60.2%)상 gate가 도메인 정보를 거의 못 지우므로, 이 이득을
  "도메인 불변성" 덕으로 귀속할 근거는 없음. class_gate의 채널 재가중 효과일 수 있음(대조군 없음).

원본 로그·paired 분석: `docs/generated/exp3_rawz/`.

#### 실험 3b (2026-09-23): threshold 공정성 검증 — raw z 전용 n_sigma를 LOCO로 재선정

배경: 실험 3은 z_inv용 n_sigma=2.0을 raw z에도 그대로 재사용했음 — feature space가 다르면
불공정한 비교일 수 있음. `experiments/tune_nsigma_rawz.py`(신규, 기존 스크립트 무수정)로
raw z 전용 n_sigma를 **target(test) 도메인 라벨 없이** leave-one-calib-domain-out(LOCO)
방식으로 fold별 재선정: 각 fold의 calib 도메인 4개 중 하나씩을 돌아가며 "가짜 test 도메인"으로
삼고(라벨 사용 가능 — nested CV의 inner validation, outer test 도메인과 무관), 나머지 3개
calib 도메인의 정상 데이터로 scoring해 후보 {-1.0 ~ 3.0, 0.5 간격} 중 Acc/F1 평균이 최고인
n_sigma를 선택(3 tune-eval-seed × 3 train-seed 앙상블). 선정된 n_sigma로 원래 프로토콜대로
(target 도메인 라벨 사용) 3×4×5 재평가.

**선정된 n_sigma**: fold500=1.5, fold600=2.5, fold700=2.0(=원래 값과 동일), fold800=2.5

| | n_sigma | AUROC | Acc | F1 |
|---|---|---|---|---|
| z_inv (기존) | 2.0 | 0.9539 | 0.9054 | 0.9430 |
| raw z (실험3, n_sigma 재사용) | 2.0 | 0.9449 | 0.8684 | 0.9174 |
| **raw z (실험3b, LOCO 자체 튜닝)** | fold별 | 0.9449 | **0.8592** | **0.9102** |

**결과: threshold를 raw z에 맞게 다시 골라도 z_inv와의 격차가 줄지 않고 오히려 커짐**
(Acc 격차 +0.0369→+0.0462, F1 격차 +0.0256→+0.0327, paired n=20 기준 z_inv가 20/20승,
p<1e-5). raw z 자체 기준으로도 튜닝 전(n_sigma=2.0 재사용)이 튜닝 후보다 Acc/F1이 더 높음
(11/20승, p<0.01) — LOCO로 고른 threshold가 실제 held-out 도메인에는 과적합되어 오히려
전이가 덜 됨(특히 fold800: Acc 0.7847→0.7581). AUROC는 threshold와 무관한 지표라 당연히
불변(0.9449, 완전히 동일).

**해석**: 실험 3의 Acc/F1 격차(+3.7%p/+2.6%p)는 threshold 불공정 비교의 산물이 아님 —
가장 공정하게 골라도(그리고 그 결과가 오히려 원래보다 나쁨) 격차는 유지·확대됨. **decomposition의
Acc/F1 기여는 실재하는 것으로 결론.** AUROC 기여(+0.009)가 노이즈 범위 안이라는 실험 3 결론은
그대로 유지(threshold와 무관하므로 이 실험이 바꿀 수 없는 부분).

원본 로그·3-way paired 분석: `docs/generated/exp3_rawz/` (`tune_nsigma_run_full.txt`,
`tuned_nsigma.json`, `rawz_tuned_eval.txt`, `paired_analysis_tuned.py/.txt`).

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

## 실험 A (2026-09-23): n_sigma FPR-Recall trade-off (Table 2 / z_inv 파이프라인)

`experiments/sweep_nsigma_zinv.py`(신규, `eval_gated_folds.py` 무수정 — import 재사용).
모델 forward pass는 기존 Table 2 평가와 동일하게 (fold, eval_seed)당 1번만 수행하고,
n_sigma 9개 후보 {-1.0 ~ 3.0, 0.5 간격}는 이미 계산된 ensemble score 배열에 threshold
연산만 다르게 적용해 스윕(추가 forward pass 없음). target 도메인 라벨은 최종 평가
단계에서만 사용(n_sigma 선택 자체에는 미사용 — 기존 프로토콜과 동일).

**평균(4 fold) Specificity/Recall/BalAcc vs n_sigma**:

| n_sigma | Specificity | Recall | Balanced Acc |
|---|---|---|---|
| -1.0 | 0.228 | 0.990 | 0.609 |
| 0.0 | 0.435 | 0.977 | 0.706 |
| 1.0 | 0.597 | 0.959 | 0.778 |
| **2.0 (현재)** | **0.720** | **0.937** | **0.828** |
| 2.5 | 0.771 | 0.923 | 0.847 |
| 3.0 | 0.811 | 0.908 | 0.859 |

전체 표(9개 후보 전부, fold별): `docs/generated/expA_fpr_recall/nsigma_sweep.json`.
ROC operating point 시각화: `roc_nsigma_operating_points.png`, BalAcc 곡선: `balanced_acc_vs_nsigma.png`.

**핵심 발견**: 테스트한 범위 {-1.0~3.0} 안에서 Balanced Accuracy가 4개 fold 전부 **단조
증가** — 내부 극값(peak)이 없음. 즉 이 범위 안에서는 "BalAcc 최댓값" 기준의 명확한 최적
n_sigma를 못 찾음(3.0이 매번 최고). 다만 한계 트레이드오프(marginal trade-off)로 보면
n_sigma=2.0→2.5 구간은 fold 500/600/700에서 Recall 손실이 매우 작으면서(−0.3~1.4%p)
Specificity가 뚜렷이 개선(+3.9~7.2%p)되는 "무릎점" — **n_sigma=2.5가 합리적인 대안**.
fold 800만 예외로 같은 구간에서 Recall 손실이 더 큼(−3.6%p) — AUROC가 가장 낮은(0.9008)
도메일수록 Spec/Recall 트레이드오프가 더 가파름(실험 C의 동기와 일치).

**현재 n_sigma=2.0의 위치**: 4 fold 평균 Recall 93.7%/Specificity 72.0%로, Recall을
우선하는 쪽에 있음 — 이상탐지에서 미탐지(false negative)가 오탐(false positive)보다
비용이 큰 경우 합리적 선택. Recall을 조금 양보하고 오탐을 줄이려면 n_sigma=2.5를 고려할
가치 있음(BalAcc +0.019, 전체 Recall −1.4%p 수준).

**실험 3b LOCO 값(raw z, 참고용)과의 일관성**: 실험 3b는 raw z 공간에서 fold별
n_sigma를 500=1.5/600=2.5/700=2.0/800=2.5로 선정했음 — **주의: 다른 feature space(raw
z)에서 나온 값이라 이 실험(z_inv)과 직접 비교 불가**, 방향성만 참고. fold 700(=2.0,
현재값과 동일)과 fold 600/800(=2.5, 이번 실험의 "무릎점" 추천과 일치)은 방향이 맞고,
fold 500만 반대 방향(3b는 현재보다 낮은 1.5, 이번 실험은 z_inv 기준 fold 500이 가장
여유 있어 2.5~3.0도 Recall 손실이 작음)— feature space가 다른 만큼 완전한 교차검증은
아님.

재현: `conda run -n torch python experiments/sweep_nsigma_zinv.py --beta 0.5`

## 실험 B (2026-09-23~24): K-shot sensitivity (Table 2 / z_inv 파이프라인)

`experiments/sweep_kshot_zinv.py`(신규, `eval_gated_folds.py` 무수정 — 모듈레벨 `SHOT`
상수를 호출 시점에 몽키패치). K∈{1,2,8,16}을 n_sigma=2.0(현재값) 고정하고 재평가, K=4는
기존 Table 2 결과를 재사용(재실행 안 함).

| K | AUROC | Acc | F1 |
|---|---|---|---|
| 1 | 0.9499 | 0.8806 | 0.9255 |
| 2 | 0.9525 | 0.9029 | 0.9414 |
| **4 (현재)** | **0.9539** | **0.9054** | **0.9430** |
| 8 | 0.9547 | 0.9081 | 0.9440 |
| 16 | 0.9545 | 0.9135 | 0.9479 |

**한계 이득(K 1→2→4→8→16)**: AUROC +0.0026 / +0.0014 / +0.0008 / -0.0002, Acc +0.0223 /
+0.0025 / +0.0027 / +0.0054, F1 +0.0158 / +0.0016 / +0.0010 / +0.0039 — **K=1→2 구간에서
대부분의 이득이 발생하고 K=2 이후로는 거의 포화**. 현재 K=4는 이미 포화 구간 안에 있어
합리적 선택.

**fold 800(가장 저조)은 K를 늘려도 개선되지 않음** — AUROC가 K 전체(1~16)에서 0.898~0.902
범위 안에 머무르며 뚜렷한 추세가 없고(K=2가 오히려 최고), Acc/F1도 K=2에서 최고(Acc
0.8468/F1 0.9060)였다가 K=4/8에서 하락 후 K=16에서 재상승 — **단조 개선 패턴 없음**. 반면
fold 600은 K=1→4에서 뚜렷한 개선(AUROC 0.9316→0.9500, Acc 0.8639→0.9034)을 보인 뒤 포화.

**해석**: fold 800의 낮은 성능은 few-shot support 통계 추정의 불안정성(K가 작아서 생기는
문제) 때문이 아님 — K를 4배(16)까지 늘려도 개선되지 않으므로, **원인은 support 개수가
아니라 도메인 800 자체의 신호 분리도(separability)에 있을 가능성이 높음**(실험 C의 배경
잡음 가설과 정합). 반대로 fold 600은 shot 수에 민감해 현재 K=4가 그 fold에는 다소 이른
포화 지점일 수 있음.

원본 로그·전체 표: `docs/generated/expB_kshot/` (`kshot_results.json`, `run_full.txt`,
`kshot_auroc_acc.png`).

재현: `conda run -n torch python experiments/sweep_kshot_zinv.py --beta 0.5 --n_sigma 2.0 --k_values 1 2 8 16`

## 🔴 MMD 도메인 정렬 시도 (2026-09-24) — domain leakage 개선 방향 최종 실패, 완전 종료

GRL(직교화)/domain_weight 스윕에 이은 세 번째이자 마지막 시도: discriminator 없이 도메인 간
z_inv 분포 거리(multi-bandwidth RBF-kernel MMD)를 직접 최소화(`utils/mmd.py`,
`--mmd_weight 10.0`, 기존 GRL loss는 유지한 채 보조 항으로 추가). 1-seed×4-fold로 빠르게
판정(3-seed 확장 없이 종료 — 아래 판정 참고).

| | domain-acc | class-acc | AUROC | Acc | F1 |
|---|---|---|---|---|---|
| 기존(GRL만) | 0.6017 | 0.9790 | 0.9540 | 0.9053 | 0.9429 |
| +직교화(실패) | 0.5958 | 0.9800 | 0.9548 | 0.8960 | 0.9359 |
| +domain_weight=2.0(실패) | 0.5579 | 0.9621 | 0.9231 | 0.8536 | 0.9087 |
| **+MMD(실패)** | **0.4531** | **0.8925** | **0.9224** | **0.7285** | **0.7976** |

**판정**: domain-acc는 −14.9pp로 사전 등록한 성공 기준(5pp 이상 감소)을 충분히 넘김 — 세
시도 중 유일하게 이 조건을 만족. 그러나 AUROC/Acc/F1(각각 −1pp 이내 요구)이 −3.2/−17.7/−14.5pp로
대폭 초과 위반(fold 600은 Acc가 0.90→0.50로 반토막). 두 조건은 AND이므로 **종합 실패**.

**결론**: 3가지 서로 다른 메커니즘(GRL 직교화, GRL 강화, discriminator-free MMD) 모두
"domain-acc 유의 감소 + Acc/F1 유지"를 동시에 만족 못함. **domain leakage 개선 방향 완전
종료** — `checkpoints/coarse5_fold*_gated_nodg_s{0,1,2}.pth`(dw=1.0, 비직교화)가 유일한
유효 선택지로 확정. 논문에는 z_inv가 도메인 정보를 완전히 제거하지 못한다는 한계를 정직하게
기술할 것.

**추가 확인(같은 날, mmd_weight={1.0, 0.1} 재시도)**: weight를 낮춰도 판정 기준을 만족하는
지점이 없음을 확인 — monotonic trade-off만 관찰됨(weight↓ ⟹ Acc/F1 손상↓이지만 domain-acc
감소폭도 노이즈 범위 안으로 함께 줄어듦):

| mmd_weight | domain-acc Δ | Acc Δ | F1 Δ | 조건1(5pp↓) | 조건2(±1pp) |
|---|---|---|---|---|---|
| 0.1 | −1.04pp | −2.00pp | −1.50pp | ❌ | ❌ |
| 1.0 | −2.61pp | −1.29pp | −0.92pp | ❌ | 근접(F1만) |
| 10.0 | −14.86pp | −17.68pp | −14.53pp | ✅ | ❌❌ |

세 값 모두 두 조건 동시 충족 실패 — **3-seed 확장이나 추가 weight 탐색 없이 완전 종료**.

상세: `docs/exec-plans/completed/2026-09-gated-nodg-mmd.md`. 원본 로그: `docs/generated/mmd_attempt/`.

**추가 확인(같은 날, mmd_weight=1.0 + domain-balanced batch, 4번째이자 정말 마지막 시도)**:
"batch가 불균형해서(4개 calib 도메인에 batch_size=16 → 도메인당 평균 4개) MMD 추정 자체가
노이즈였다"는 가설을 직접 검증 — `utils/domain_balanced_sampler.py`로 매 batch가 도메인당
정확히 16개(batch_size=64)씩 뽑히도록 고정, mmd_weight=1.0(스윕에서 가장 균형 잡혔던 값)
고정, batch 구성만 격리해서 재학습:

| | domain-acc | Δ | AUROC Δ | Acc Δ | F1 Δ | 조건1 | 조건2 |
|---|---|---|---|---|---|---|---|
| MMD w=1.0 (불균형, 기존) | 0.5756 | −2.61pp | −1.31pp | −1.29pp | −0.92pp | ❌ | ❌ |
| **MMD w=1.0 + balanced batch** | **0.5887** | **−1.30pp** | **−1.88pp** | −0.96pp | −0.69pp | ❌ | ❌ |

**가설 반증**: 노이즈 가설이 맞다면 batch를 균형화했을 때 domain-acc 감소폭이 더 커져야
하는데, 실제로는 오히려 **더 작아졌다**(−2.61pp → −1.30pp, weight=0.1 수준으로 약화)
— 동시에 AUROC 손상은 더 커짐(−1.88pp). 즉 batch 불균형/노이즈는 실패의 원인이 아니었음이
확인됨. 근본 원인은 다른 곳(RBF 커널 MMD와 z_inv 분포 구조의 부정합, GRL·MMD 두 신호의
상쇄 등)에 있을 가능성이 있으나 추가 검증하지 않음 — 사전 약속대로 여기서 종료.

**domain leakage 개선 방향은 이제 4개 메커니즘(직교화, domain_weight, MMD unbalanced ×3
weight, MMD balanced batch), 총 6개 설정 전부 실패로 완전히 최종 종료.**
`checkpoints/coarse5_fold*_gated_nodg_s{0,1,2}.pth`(dw=1.0, GRL만)가 논문 Table 2의
유일한 유효 체크포인트 패밀리로 확정. 상세: `docs/exec-plans/completed/2026-09-mmd-balanced-batch.md`,
원본 로그: `docs/generated/mmd_balanced_attempt/`.

## 🟢 대조군 실험: 도메인 라벨 shuffle (2026-09-25) — Acc/F1 개선의 메커니즘 규명

domain leakage 개선 방향은 종료됐지만("도메인 불변성이 개선 안 됨"), z_inv의 Acc/F1은
여전히 raw z보다 유의하게 높음(실험 3/3b, p<0.001). 이 개선이 (a) 진짜 도메인 적대적
학습(GRL) 때문인지, (b) class_gate/mask 구조 자체의 재가중 효과(도메인 라벨 진위와
무관)인지 직접 검증. `--shuffle_domain_labels`(신규): 학습 샘플의 진짜 도메인 라벨을
seed 고정 permutation으로 무작위 재배정한 채 GRL/domain_classifier를 학습(class_loss/
gate 구조는 완전히 동일하게 유지). 1-seed×4-fold.

| | AUROC | Acc | F1 |
|---|---|---|---|
| raw z (decomposition 없음) | 0.9449 | 0.8684 | 0.9174 |
| z_inv (진짜 도메인 라벨) | 0.9540 | 0.9053 | 0.9429 |
| **z_inv (shuffle 도메인 라벨)** | **0.9465** | **0.9041** | **0.9428** |

**gain 분해(raw z 대비)**: shuffle이 얻는 gain(Acc +3.57pp, F1 +2.54pp)이 진짜 라벨
버전 전체 gain(Acc +3.69pp, F1 +2.55pp)의 **96.7%/99.6%**를 차지 — 진짜 라벨의 추가
기여는 사실상 0(Acc +0.12pp, F1 +0.01pp, 노이즈 범위). 반대로 **AUROC**는 gain의
82.4%(+0.75pp/+0.91pp)가 shuffle에는 없고 진짜 라벨에서만 나타남(절대 크기는 작음 —
raw z AUROC가 이미 0.9449로 높아 개선 여지 자체가 작았음).

**결론(성공/실패 프레임 아님, 그대로의 발견)**: Acc/F1 개선은 거의 전적으로
class_gate/mask 구조 자체의 재가중 효과이며 도메인 라벨의 진위와 무관함 — 논문에 명시할
가치 있는 독립적 발견. AUROC만 예외적으로 진짜 도메인 적대적 학습이 작지만 일관된 추가
기여(+0.75pp)를 함 — 지표별로 구분해서 서술할 것. 독립 도메인 프로브(z_inv domain-acc
0.5962)는 진짜 라벨 버전(0.6017)과 사실상 동일 — encoder가 진짜 라벨을 학습한 적 없으므로
도메인 불변성 지표로 해석 불가, 기존 결론(GRL이 도메인 정보를 유의하게 제거 못함)과
일관될 뿐 새로운 정보 없음.

상세: `docs/exec-plans/completed/2026-09-shuffled-domain-label-control.md`. 원본 로그:
`docs/generated/shuffled_domain_attempt/`.
