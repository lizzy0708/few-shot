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
(생성·학습·평가 완료 후 추가)
