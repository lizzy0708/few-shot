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

## 결과

전사는 대화 기록 참고(진행 중 — dw=2.0 완료 시 중간 보고, dw=5.0까지 완료 시 최종 보고).
"도메인 누수 축소" 방향의 마지막 반복 — 두 값 모두 판정 기준(≥5pp domain-acc 감소 AND
AUROC/Acc/F1 ~1pp 이내 유지) 미달 시 이 방향은 여기서 종료.
