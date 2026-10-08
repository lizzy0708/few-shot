# Step 0: 재현 확인 게이트 (wrap-around 보정 재학습 전 필수 선행 단계)

## 목적
GADF→uint8 wrap-around 보정 이미지로 재학습하기 전, 현재 코드/환경으로 Table 2를
재현 가능한지 먼저 확인(fold 500, seed 1, 1회). 통과해야 본 작업(보정 이미지 생성·
재학습) 진행.

## 코드 변경
없음 — `experiments/train_gated_mask_model.py`/`experiments/eval_gated_folds_mmd.py`
전부 기존 CLI 인자로 충분. 새 파일 `experiments/eval_repl_check.py`만 추가(BalAcc까지
계산하는 비교 전용, `eval_gated_folds.py` 미수정·재사용).

## 학습 커맨드 (원본 processed/, fold 500, seed 1만, 1회)
```bash
conda run -n torch python experiments/train_gated_mask_model.py \
  --root processed --train_domains 400 600 700 800 --all_domains 400 500 600 700 800 \
  --epochs 10 --batch_size 16 --lr 1e-4 --num_classes 2 \
  --mask_weight 0.1 --domain_weight 1.0 --domain_disc_weight 1.0 \
  --no_domain_gate --seed 1 \
  --save_path checkpoints/coarse5_fold500_gated_nodg_repl_s1.pth
```
(기존 `coarse5_fold500_gated_nodg_s1.pth`와 완전히 동일한 하이퍼파라미터 — `_repl_`
태그만 다름, 덮어쓰지 않음.)

## 평가 커맨드
```bash
conda run -n torch python experiments/eval_repl_check.py --test_fold 500 --train_seeds 1 --beta 0.5 --n_sigma 2.0
```

## 통과 기준 (사전 고정)
AUROC 차이(|repl − 기존|) ≤ 0.005. 통과 못하면 중단, 차이와 가능한 원인 보고.

## 결과 (2026-10-08) — 통과 ✅

| | AUROC | Acc | F1 | BalAcc |
|---|---|---|---|---|
| 기존 `coarse5_fold500_gated_nodg_s1.pth` | 0.9761±0.0023 | 0.9496±0.0047 | 0.9709±0.0027 | 0.8785±0.0168 |
| 신규 `coarse5_fold500_gated_nodg_repl_s1.pth` | 0.9761±0.0023 | 0.9496±0.0047 | 0.9709±0.0027 | 0.8785±0.0168 |
| **AUROC 차이** | **0.0000** | | | |

**통과**(기준 0.005 이내, 실제 0.0000). 체크포인트 파일 자체는 MD5가 다름(`2973c319...` vs `dea5efa6...`, 크기도 다름 — 즉 버그로 같은 파일을 두 번 평가한 게 아니라 진짜 독립적으로 재학습된 별개 모델)인데도 평가 지표가 소수점 4자리까지 완전히 일치 — `torch.backends.cudnn.deterministic=True` 등 기존 결정론적 설정이 실제로 완벽히 작동함을 확인. 현재 코드/환경으로 Table 2 재현 가능 — **wrap-around 보정 재학습 본 작업 진행 가능**.

원본 로그: `docs/generated/repl_check/{train,eval_repl,eval_orig}.log`.
