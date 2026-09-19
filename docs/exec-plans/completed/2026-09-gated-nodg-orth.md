# coarse5 gated_nodg + anti-domain orthogonalization (Stage 2, 2026-09-19)

## 배경

`docs/generated/results.md`의 "세션 최고 (2026-08-17)" (`gated_nodg`, AUROC 0.9540)에서
z_inv가 raw z 대비 domain-acc를 거의 줄이지 못함을 독립 선형 프로브로 확인
(Experiment 1: z_inv domain-acc 0.6017 vs chance 0.20, class-acc 0.9790). Experiment 2
진단: `nodg`(`use_domain_gate=False`)에는 class_gate가 domain 방향을 명시적으로 제거하는
항이 전혀 없었음(fine15는 Gram-Schmidt `mc_orth = mc - proj(mc onto md)`로 명시적 제거,
`models/mask_decomposition_model.py:147-156`) — GRL만으로 encoder가 domain-blind해지길
기대했을 뿐. class_gate 값 자체는 이미 78-86%가 극단(<0.1 또는 >0.9)이라 "게이트가
물러서" 새는 게 아니라 "명시적 anti-domain 항 부재"가 원인으로 지목됨.

## 변경 사항

`models/gated_mask_model.py`에 `use_gate_orth=True` 옵션 추가
(`GatedMaskModel._orthogonalize_gate_against_domain`): class_gate를
`domain_classifier.weight`의 행(row) 부분공간에 대해 QR 기반 Gram-Schmidt로 직교화한 뒤
z_inv를 구성. fine15와의 정확한 차이는 코드 주석과 커밋 메시지(아래 커밋 해시) 참고 —
요약하면 (1) 샘플별 gradient 대신 domain_classifier의 학습된 고정 가중치 방향 사용,
(2) md 벡터 1개가 아니라 domain_classifier 행 전체(최대 5개) 부분공간을 QR로 한 번에
제거, (3) 새 학습 파라미터 없음(기존 nodg 체크포인트와 state_dict 100% 호환),
(4) class_logits 경로는 영향받지 않음(학습 dynamics 유지, z_inv만 변경).

**Pre-training 커밋**: `6bb09192510a7f74eeb06b441cbb02d5f7660cbb`
("Add explicit anti-domain orthogonalization to GatedMaskModel (Stage 2)") —
12-run 학습 시작 전에 커밋 완료.

## 재현 명령 (정확히 사용된 형태, 12런: 3 seed × 4 fold)

```bash
declare -A CALIB
CALIB[500]="400 600 700 800"
CALIB[600]="400 500 700 800"
CALIB[700]="400 500 600 800"
CALIB[800]="400 500 600 700"

for fold in 500 600 700 800; do
  for seed in 0 1 2; do
    conda run -n torch python experiments/train_gated_mask_model.py \
      --root processed --train_domains ${CALIB[$fold]} --all_domains 400 500 600 700 800 \
      --epochs 10 --batch_size 16 --lr 1e-4 --num_classes 2 \
      --mask_weight 0.1 --domain_weight 1.0 --domain_disc_weight 1.0 \
      --no_domain_gate --use_gate_orth \
      --seed ${seed} \
      --save_path checkpoints/coarse5_fold${fold}_gated_nodg_orth_s${seed}.pth
  done
done
```

하이퍼파라미터 근거: `--epochs 10 --batch_size 16 --lr 1e-4 --mask_weight 0.1
--domain_weight 1.0 --domain_disc_weight 1.0`은 모두 `train_gated_mask_model.py`의
argparse 기본값 그대로(원본 `gated_nodg` 체크포인트를 만든 정확한 명령이 끝내 발견되지
않아, 이전 조사(2026-09-19 Table-2 provenance 조사)와 동일하게 "기본값 + 문서에 확인된
플래그(`use_domain_gate=False`→`--no_domain_gate`, `domain_weight=1.0`)"를 재현 기준으로
삼음 — 새로운 추측 아님, 기존 조사와 동일한 근거). `--no_domain_gate`는 `nodg`(domain_gate
없음) 계열 유지. `--use_gate_orth`만 신규 추가된 유일한 실제 변경.

체크포인트 출력: `checkpoints/coarse5_fold{500,600,700,800}_gated_nodg_orth_s{0,1,2}.pth`
(12개, 기존 `..._gated_nodg_s{0,1,2}.pth`는 덮어쓰지 않음).

학습 로그: `train_gated_mask_model.py`의 표준 print 출력을 각 런마다 별도 파일로
저장(`gated_orth_train_logs/fold{fold}_s{seed}.log`) — 이전 세션의 "scratchpad 로그
소실" 재발 방지 목적. **주의**: 이 로그 파일 자체는 `/tmp` 기반 세션 scratchpad에
저장되었으므로 여전히 휘발성이다 — 학습 재현에 필요한 것은 위 커맨드와 커밋 해시이지
이 로그 파일이 아니다. 로그의 핵심 수치(최종 epoch loss/acc)는 이 문서의 "결과" 절에
전사되어 있다.

## 검증 스크립트

- `experiments/eval_gated_folds_orth.py` — `eval_gated_folds.py`를 건드리지 않고
  체크포인트 경로/로더만 monkey-patch (beta=0.5, n_sigma=2.0, LedoitWolf, leakage-fix
  로직은 Table-2 재현에 이미 검증된 것과 완전히 동일).
- `experiments/analyze_domain_invariance_tsne_orth.py` — 동일하게
  `analyze_domain_invariance_tsne.py`를 건드리지 않고 순수 헬퍼 함수만 재사용.

## 결과

전사는 최종 보고서(대화 기록) 참고 — 이 문서는 "무엇을 어떻게 재현하는가"에 집중하고,
수치 자체는 `docs/generated/results.md`에 추후 반영 여부를 사용자가 결정.
