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

## 결과

(학습·평가 완료 후 기입 — 진행 중)
