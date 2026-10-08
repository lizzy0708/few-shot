# GADF wrap-around 보정 이미지 재생성·재학습 (2026-10-08)

0단계(재현 확인) 통과 후 본 작업. 기존 파일(`processed/`, `processed/make_gadf.py`,
`checkpoints/coarse5_fold*_gated_nodg_s{0,1,2}.pth` 등) 전혀 수정·덮어쓰기 안 함.

## 1. 이미지 재생성
새 파일 `processed/make_gadf_fixed.py`(기존 `make_gadf.py` 복제, uint8 변환 줄만
`np.clip(np.round((gaf_img+1)/2*255), 0, 255).astype(np.uint8)`로 교체) →
새 폴더 `processed_fixed/`.
```bash
conda run -n torch python processed/make_gadf_fixed.py
```

## 2. 학습
`train_gated_mask_model.py`는 이미 `--root` 인자를 지원하므로 **새 래퍼 불필요** —
`--root processed_fixed`만 추가해서 기존 스크립트 그대로 사용.
```bash
conda run -n torch python experiments/train_gated_mask_model.py \
  --root processed_fixed --train_domains <calib 4개> --all_domains 400 500 600 700 800 \
  --epochs 10 --batch_size 16 --lr 1e-4 --num_classes 2 \
  --mask_weight 0.1 --domain_weight 1.0 --domain_disc_weight 1.0 \
  --no_domain_gate --seed {0,1,2} \
  --save_path checkpoints/coarse5_fold<fold>_gated_nodg_fixedenc_s<seed>.pth
```
fold {500,600,700,800} × seed {0,1,2} = 12 run.

## 3. 평가
`eval_gated_folds.py`의 `ROOT`는 CLI 인자가 아닌 모듈 상수라 새 래퍼 필요 — 새 파일
`experiments/eval_gated_folds_fixedenc.py`(root + ckpt-tag + feature_key + pretrained
전환을 전부 monkey-patch로 처리, 기존 `eval_gated_folds.py` 미수정).
```bash
# z_inv / raw z (보정 이미지, 학습된 체크포인트)
conda run -n torch python experiments/eval_gated_folds_fixedenc.py --root processed_fixed --ckpt_tag fixedenc --feature_key z_c_notd --beta 0.5 --n_sigma 2.0
conda run -n torch python experiments/eval_gated_folds_fixedenc.py --root processed_fixed --ckpt_tag fixedenc --feature_key z --beta 0.5 --n_sigma 2.0
# pretrained baseline (보정 이미지, 학습 없음, 추론만)
conda run -n torch python experiments/eval_gated_folds_fixedenc.py --root processed_fixed --model pretrained --feature_key z --beta 0.5 --n_sigma 2.0
```

## 사전 고정 규칙
원본 Table 2(`processed/`)가 대표 결과, 보정 결과(`processed_fixed/`)는 "저장 방식
민감도" 참고용으로만 보고 — 결과를 보고 이 지위를 바꾸지 않음.

## 결과

### 1. 이미지 재생성 — 검증 통과
- 총 47,663장(원본과 정확히 동일), 도메인별 정상 1,497장(5개 도메인 전부) — 일치
- (a) 샘플 20장: float gaf_img로부터 `round((v+1)/2*255)` 직접 재계산한 값과 저장된 PNG가 **20/20 완전 일치**(max_abs_diff=0)
- (c) 히스토그램(예시 `N502_419840.png`): 원본 OLD(0~5구간 668 / 250~255구간 527, 비대칭=wrap-around 흔적) vs NEW(474/474, 완전 대칭) — 대각선(GAF=0)이 중간 회색대(123~133)에 OLD 654개→NEW 1054개로 늘어 올바르게 매핑됨을 확인. **wrap-around 제거 확인됨.**

(학습·평가는 진행 중 — 완료 후 이어서 추가)
