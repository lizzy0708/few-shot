# coarse5 gated_nodg — MMD 도메인 정렬 시도 (2026-09-24, domain-leakage 방향 마지막 시도)

## 배경

Stage 2(Gram-Schmidt 직교화)와 domain_weight 스윕(2.0/5.0) 둘 다 실패로 종료됨
(`docs/exec-plans/completed/2026-09-gated-nodg-orth.md`,
`docs/exec-plans/completed/2026-09-gated-nodg-domain-weight-sweep.md`). 두 시도 모두
GRL + 단일 domain_classifier discriminator 구조에 의존 — "그 discriminator 하나만
속이면 되는 국지적 해"에 빠지는 DANN의 잘 알려진 한계에 취약. 이번 시도는 discriminator
없이 도메인 간 z_inv 분포 거리(RBF 커널 MMD)를 직접 최소화하는 완전히 다른 메커니즘.

**이번이 도메인 불변성 개선 방향의 마지막 시도** — 실패 시 이 방향 완전 종료, 현재 결과
(도메인 정보 제거 안 됨, Acc/F1은 개선됨)를 논문에 정직하게 반영.

## 구현

- `utils/mmd.py`(신규): multi-bandwidth RBF 커널 MMD² (median heuristic × {1,2,4,8,16}).
  GAP-pooled z_inv([B,1024])에 대해, 배치 안에 있는 calib 도메인들 사이 pairwise MMD²를
  평균. 학습 파라미터 없음(순수 함수).
- `experiments/train_gated_mask_model.py`: `--mmd_weight` 인자 추가(기본값 0.0 = 완전
  비활성, 기존 동작과 100% 동일 — 하위 호환). 기존 GRL(domain_loss)은 제거하지 않고
  유지, MMD는 보조 항으로 추가. `total_mmd` 로깅 추가.
- **mmd_weight=10.0 채택 근거**: 학습 전 미학습 모델의 실제 배치로 측정 —
  초기 MMD²≈0.034, 초기 domain_loss(GRL CE)≈1.62(=ln5, 미학습 5-way 균등 예측의 이론값).
  사용자 지정 후보 {0.1, 1.0, 10.0} 중 가장 큰 10.0을 선택해야 MMD 항(≈0.34)이
  domain_loss 항(1.0×1.62=1.62) 대비 무시 못 할 규모(약 20%)가 됨 — 1.0이나 0.1은
  domain_loss 대비 각각 2%, 0.2%로 사실상 무의미. 스윕 없이 이 값 하나로 진행.
- 스모크 테스트(1 epoch, fold 500 train_domains): 정상 종료, MMD 0.034→0.0054로 감소
  확인(실제로 최적화되고 있음), Cls Acc 85.2%, NaN/Inf 없음.

## 재현 명령 (1-seed 우선, 성공 조짐 있으면 seed 1/2 추가)

```bash
declare -A CALIB
CALIB[500]="400 600 700 800"
CALIB[600]="400 500 700 800"
CALIB[700]="400 500 600 800"
CALIB[800]="400 500 600 700"

for fold in 500 600 700 800; do
  for seed in 0; do   # 1-seed 우선 실행 — 성공 조짐 있으면 1 2 추가
    conda run -n torch python experiments/train_gated_mask_model.py \
      --root processed --train_domains ${CALIB[$fold]} --all_domains 400 500 600 700 800 \
      --epochs 10 --batch_size 16 --lr 1e-4 --num_classes 2 \
      --mask_weight 0.1 --domain_weight 1.0 --domain_disc_weight 1.0 --mmd_weight 10.0 \
      --no_domain_gate \
      --seed ${seed} \
      --save_path checkpoints/coarse5_fold${fold}_gated_nodg_mmd_s${seed}.pth
  done
done
```

기존 nodg 런(mmd_weight=0.0 상당)과 `--mmd_weight 10.0` 하나만 다름 — 나머지 인자는
domain_weight 스윕 문서와 완전히 동일. 기존 체크포인트(`..._gated_nodg_s{0,1,2}.pth`,
`..._gated_nodg_orth_s{0,1,2}.pth`, `..._gated_nodg_dw{2,5}_s{0,1,2}.pth`)는 덮어쓰지 않음.

## 검증 스크립트

- `experiments/eval_gated_folds.py`를 무수정 재사용(체크포인트 경로 패턴만 다름 —
  `ckpt_path`를 `coarse5_fold{fold}_gated_nodg_mmd_s{seed}.pth`로 바꾸는 thin wrapper,
  기존 orth/dw wrapper와 동일 패턴).
- `experiments/analyze_domain_invariance_tsne.py`도 동일 패턴.

## 판정 기준 (사전 고정, 결과 보고 후 임의 변경 금지)

- **성공**: domain-acc가 노이즈 범위(±1~4%p)를 넘어 5%p 이상 유의하게 감소, AND
  AUROC/Acc/F1이 각각 -1%p 이내로 유지.
- **실패**: 위 조건 미달 시 실패로 결론, 도메인 불변성 개선 방향 완전 종료.

## 결과 (2026-09-24 확정, 1-seed × 4-fold — 판정 기준 미달로 seed 1/2 추가 없이 즉시 종료)

**MMD는 domain-acc를 유의미하게 줄이는 데는 성공했으나(성공 조건 1 충족), Acc/F1을 판정 기준을
훨씬 초과해 파괴함(성공 조건 2 대실패)** — 종합 실패. 3연속 시도(직교화/domain_weight/MMD)
전부 실패로 이 방향 완전 종료.

| | domain-acc (z_inv) | class-acc (z_inv) | AUROC | Acc | F1 |
|---|---|---|---|---|---|
| 기존(GRL만) | 0.6017 | 0.9790 | 0.9540 | 0.9053 | 0.9429 |
| +직교화(실패) | 0.5958 | 0.9800 | 0.9548 | 0.8960 | 0.9359 |
| +domain_weight=2.0(실패) | 0.5579 | 0.9621 | 0.9231 | 0.8536 | 0.9087 |
| **+MMD(mmd_weight=10.0, 실패)** | **0.4531** | **0.8925** | **0.9224** | **0.7285** | **0.7976** |
| Δ(MMD − 기존) | **−0.1486** (14.9pp↓) | −0.0865 | −0.0316 | **−0.1768** | **−0.1453** |

fold별 AUROC/Acc/F1 (`experiments/eval_gated_folds_mmd.py --train_seeds 0`):

| Fold | AUROC | Acc | F1 | (기존 대비 Δ) |
|---|---|---|---|---|
| 500 | 0.9790±0.0040 | 0.8864±0.0063 | 0.9294±0.0043 | Acc −6.7pp |
| 600 | 0.8613±0.0106 | **0.5023±0.0390** | **0.5919±0.0452** | Acc −40.1pp(!) |
| 700 | 0.9409±0.0166 | 0.8446±0.0073 | 0.9014±0.0050 | Acc −8.8pp |
| 800 | 0.9084±0.0027 | 0.6807±0.0118 | 0.7677±0.0108 | Acc −15.2pp |
| **Avg** | **0.9224** | **0.7285** | **0.7976** | |

fold별 domain-acc/class-acc (`experiments/analyze_domain_invariance_tsne_mmd.py --no_tsne`):

| Fold | z domain-acc | z_inv domain-acc | z class-acc | z_inv class-acc |
|---|---|---|---|---|
| 500 | 0.4050 | 0.4175 | 0.8800 | 0.8825 |
| 600 | 0.4575 | 0.4300 | 0.8900 | 0.8975 |
| 700 | 0.4700 | 0.4375 | 0.8875 | 0.8875 |
| 800 | 0.4975 | 0.5275 | 0.8975 | 0.9025 |

**판정 (사전 고정 기준 적용)**:
- 조건 1(domain-acc 5pp 이상 감소): **충족** — 14.9pp 감소, 직교화(−0.6pp)·domain_weight(−4.4pp) 둘 다 5pp를 못 넘겼던 것과 달리 MMD는 확실히 넘김. discriminator-free 메커니즘이 실제로 "국지적 해"를 피하고 진짜 분포 정렬을 만들어냈다는 뜻으로 해석됨.
- 조건 2(AUROC/Acc/F1 각각 −1pp 이내): **대실패** — Acc −17.7pp, F1 −14.5pp, AUROC −3.2pp 전부 기준을 몇 배~십수 배 초과. fold 600은 Acc가 절반 수준(0.90→0.50)으로 붕괴.
- **최종 판정: 실패.** 두 조건은 AND이므로 조건 1 충족만으로는 성공이 아님.

**원인 추정(미검증)**: mmd_weight=10.0이 과도하게 강해 class-discriminative 구조까지 함께
무너뜨렸을 가능성 — batch_size=16을 4개 calib 도메인이 나눠 쓰면 도메인당 평균 4개뿐이라
MMD 추정 자체가 노이즈가 크고, 이 노이즈 섞인 gradient가 class_gate/classifier 학습을
방해했을 수 있음. 더 작은 mmd_weight나 더 큰 batch_size로 재시도하면 다른 결과가 나올 수
있으나, **사용자 지시("이번이 마지막 시도")에 따라 추가 스윕 없이 여기서 종료**.

**최종 결론(3연속 방향 전체)**: 직교화·domain_weight·MMD 세 가지 서로 다른 메커니즘 모두
"domain-acc를 유의하게 줄이면서 Acc/F1을 유지"라는 조건을 동시에 만족시키지 못함(MMD만
domain-acc 조건은 넘었으나 Acc/F1을 완전히 희생함) — **domain leakage 개선 방향은 이것으로
완전히 종료**. `checkpoints/coarse5_fold*_gated_nodg_s{0,1,2}.pth`(dw=1.0, 비직교화, GRL만)가
유일한 유효 선택지로 확정. 논문에는 "z_inv가 도메인 정보를 완전히 제거하지 못한다"는 한계를
정직하게 기록하고, Acc/F1 개선(decomposition의 실재 효과, 실험 3/3b로 별도 확정됨)과는
구분해서 서술할 것.

---

## 추가: mmd_weight 스윕 {1.0, 0.1} — 2026-09-24, 최종 종료 확정

mmd_weight=10.0의 Acc/F1 붕괴가 weight 과다 때문인지 확인하기 위해 1.0/0.1로 낮춰 재시도
(1-seed×4-fold, 그 외 설정 동일). `experiments/eval_gated_folds_mmd.py`/
`experiments/analyze_domain_invariance_tsne_mmd.py`에 `--mmd_tag` 옵션을 추가해 체크포인트
패밀리(`_mmd1_`/`_mmd01_`)를 구분.

| mmd_weight | domain-acc | class-acc | AUROC | Acc | F1 | domain-acc Δ | Acc Δ | F1 Δ |
|---|---|---|---|---|---|---|---|---|
| 0(기존) | 0.6017 | 0.9790 | 0.9540 | 0.9053 | 0.9429 | — | — | — |
| 0.1 | 0.5913 | 0.9756 | 0.9453 | 0.8853 | 0.9279 | −1.04pp | −2.00pp | −1.50pp |
| 1.0 | 0.5756 | 0.9800 | 0.9409 | 0.8924 | 0.9337 | −2.61pp | −1.29pp | −0.92pp |
| 10.0 | 0.4531 | 0.8925 | 0.9224 | 0.7285 | 0.7976 | −14.86pp | −17.68pp | −14.53pp |

**명확한 monotonic trade-off 확인**: weight를 낮출수록 Acc/F1 손상은 줄지만(0.1/1.0은 F1
−1pp 근방까지 회복) domain-acc 감소폭도 함께 줄어들어 노이즈 범위(±1~4pp, 기존 직교화
−0.6pp/domain_weight −4.4pp와 같은 수준) 안으로 들어옴 — **0.1, 1.0 둘 다 판정 기준의
조건 1(domain-acc 5pp 이상 감소)을 충족 못함**. 조건 2(Acc/F1 −1pp 이내)는 1.0에서 F1만
근접 충족(−0.92pp)하지만 조건 1이 이미 실패라 무의미.

**최종 판정**: 테스트한 mmd_weight {0.1, 1.0, 10.0} 중 domain-acc 조건과 성능 유지 조건을
동시에 만족하는 값은 없음. 사전 약속대로 **3-seed 확장 없이, 추가 mmd_weight 탐색 없이
여기서 완전 종료**. domain leakage 개선 방향(직교화/domain_weight/MMD 3가지, 총 5개 설정)
전체가 실패로 마무리됨 — `checkpoints/coarse5_fold*_gated_nodg_s{0,1,2}.pth`(dw=1.0, GRL만)
확정 유지.

재현: `docs/generated/mmd_attempt/train_sweep.txt`, `eval_mmd{1,01}.txt`, `probe_mmd{1,01}.txt`.
