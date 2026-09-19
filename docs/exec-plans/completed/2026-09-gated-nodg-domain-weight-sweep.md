# coarse5 gated_nodg — domain_weight sweep (2026-09-19, last iteration on domain-leakage direction)

## 배경

Stage 2(anti-domain 직교화, `docs/exec-plans/completed/2026-09-gated-nodg-orth.md`)는
domain-acc를 사실상 못 줄였고(0.6017→0.5958, noise-level) Acc/F1만 유의미하게
악화시켰다(0.9053/0.9429→0.8960/0.9359). Experiment 2 Stage 1 진단에서
`domain_weight=1.0`은 스윕된 적 없는 argparse 기본값일 뿐임을 확인 — 이번이 "도메인 누수
축소" 방향의 마지막 시도: GRL 강도(`domain_weight`) 자체를 올려서(2.0, 5.0) encoder
수준에서 domain 정보를 더 강하게 지우면 독립 프로브 기준 domain-acc가 실제로 줄어드는지
확인한다. 코드 변경 없음 — `train_gated_mask_model.py`는 이미 `--domain_weight`를
받으므로 CLI 인자만 바뀐다(`--use_gate_orth`는 사용하지 않음 — 평범한 nodg 구조 그대로).

## 재현 명령 (정확히 사용된 형태, 24런: 3 seed × 4 fold × 2 domain_weight)

```bash
declare -A CALIB
CALIB[500]="400 600 700 800"
CALIB[600]="400 500 700 800"
CALIB[700]="400 500 600 800"
CALIB[800]="400 500 600 700"

for dw in 2.0 5.0; do
  dw_tag=$(echo $dw | cut -d. -f1)   # 2.0 -> "2", 5.0 -> "5"
  for fold in 500 600 700 800; do
    for seed in 0 1 2; do
      conda run -n torch python experiments/train_gated_mask_model.py \
        --root processed --train_domains ${CALIB[$fold]} --all_domains 400 500 600 700 800 \
        --epochs 10 --batch_size 16 --lr 1e-4 --num_classes 2 \
        --mask_weight 0.1 --domain_weight ${dw} --domain_disc_weight 1.0 \
        --no_domain_gate \
        --seed ${seed} \
        --save_path checkpoints/coarse5_fold${fold}_gated_nodg_dw${dw_tag}_s${seed}.pth
    done
  done
done
```

원본 `gated_nodg`(domain_weight=1.0) 런과 **`--domain_weight`만 다르다** — 나머지
모든 인자(`--train_domains`/`--all_domains`/`--epochs`/`--batch_size`/`--lr`/
`--mask_weight`/`--domain_disc_weight`/`--no_domain_gate`)는
`docs/exec-plans/completed/2026-09-gated-nodg-orth.md`에 기록된 것과 완전히 동일
(그 문서와 마찬가지로, 원본 `gated_nodg` 정확한 커맨드가 끝내 발견되지 않아
"argparse 기본값 + 문서에 확인된 플래그" 재현 기준을 그대로 계승). `--use_gate_orth`는
사용하지 않음(Stage 2 직교화와 무관한 평범한 nodg 구조).

체크포인트 출력: `checkpoints/coarse5_fold{500,600,700,800}_gated_nodg_dw{2,5}_s{0,1,2}.pth`
(24개). 기존 `..._gated_nodg_s{0,1,2}.pth`(dw=1.0) 및
`..._gated_nodg_orth_s{0,1,2}.pth`는 덮어쓰지 않음.

## 검증 스크립트

- `experiments/eval_gated_folds.py`를 건드리지 않고 체크포인트 경로만 monkey-patch하는
  thin wrapper 재사용 패턴(Stage 2의 `eval_gated_folds_orth.py`와 동일 방식) —
  dw=2.0/5.0 각각에 대해 별도 wrapper 또는 파라미터화된 wrapper.
- `experiments/analyze_domain_invariance_tsne.py`도 동일 패턴으로 재사용.

## 결과 (2026-09-20 확정, dw=5.0 사용자 지시로 중단)

**dw=2.0 (12/12 런 완료, 완전 평가됨) — 판정 기준 미달, 실패:**

| domain_weight | domain-acc (z_inv) | class-acc (z_inv) | AUROC | Acc | F1 |
|---|---|---|---|---|---|
| 1.0 (기존) | 0.6017 | 0.9790 | 0.9540 | 0.9053 | 0.9429 |
| 2.0 | 0.5579 | 0.9621 | 0.9231 | 0.8536 | 0.9087 |

Δ(2.0−1.0): domain-acc −0.0438(기준 ≥5pp 미달), class-acc −0.0169, AUROC −0.0309,
Acc −0.0517, F1 −0.0342 (기준 ~1pp 이내 크게 초과) — fold 700이 가장 크게 악화
(AUROC 0.9820→0.9048, −7.7pp).

**dw=5.0 (사용자 명시적 지시로 중단, 미완성 — 사용 불가):** 3/12 런만 완료
(`checkpoints/coarse5_fold500_gated_nodg_dw5_s{0,1,2}.pth`만 존재, fold 600/700/800은
전혀 학습되지 않음 — fold600 seed0은 학습 도중 kill되어 체크포인트 미생성). dw=2.0에서
이미 AUROC/Acc/F1이 기준을 크게 벗어난 결과가 나온 뒤, dw=5.0(더 강한 GRL)이 그 추세를
반전시킬 근거가 없다고 판단해 사용자가 명시적으로 학습을 중단시킴 — 이 3개 체크포인트는
**부분/미완성 스윕의 잔여물이며, 4-fold 비교나 어떤 결론의 근거로도 사용하지 않는다.**
필요시 정리(삭제) 여부는 별도 판단.

**최종 결론: "도메인 누수 축소를 위한 domain_weight 스윕" 방향은 여기서 종료한다.**
두 시도(Stage 2 직교화, Stage 3 domain_weight 스윕) 모두 판정 기준을 충족하지 못했고,
dw=2.0은 Stage 2보다 훨씬 나쁜 비용/효과 트레이드오프를 보였다. 이 방향의 추가 변형은
사용자의 명시적 요청 없이 제안하지 않는다. `checkpoints/coarse5_fold*_gated_nodg_s{0,1,2}.pth`
(domain_weight=1.0, 비직교화 원본)가 이 조사 전체를 통틀어 유일하게 유효한 선택지로 남는다.
