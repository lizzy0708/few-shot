# 추론 (Inference)

## 스크립트
`scripts/eval/eval_all_folds.py`

## 방식
4-shot few-shot anomaly detection (정상 샘플 4개만 사용)

## 흐름

### Step 1. Calibration (학습 도메인 기반 precision 추정)
```
calib 도메인(학습 도메인) 정상 샘플 전부
  → 모델 forward → z_inv 추출 → GAP → [N, 2048]
  → L2 normalize → [N, 2048] (unit sphere)
  → LedoitWolf 공분산 추정 → precision matrix prec_n [2048, 2048]
```

LedoitWolf: shrinkage 기반 공분산 추정 → 고차원(2048)에서 안정적

### Step 2. Prototype 생성 (test 도메인 support samples)
```
test 도메인 정상 샘플 4개 (support set)
  → 모델 forward → z_inv → GAP → [4, 2048]
  → L2 normalize → mean → re-normalize
  → prototype [2048,]
```

### Step 3. Scoring (query 샘플)
```
test 도메인 전체 샘플 (query set)
  → 모델 forward → z_inv → GAP → [N, 2048]
  → L2 normalize → [N, 2048]

Mahalanobis distance:
  score(q) = sqrt((q_n - proto_n)^T × prec_n × (q_n - proto_n))
```

### Step 4. Threshold & Classification
```
support 4개의 Mahalanobis score 계산
threshold = mean(support scores)     ← n_sigma=0.0 (σ 항 없음)

pred = (score > threshold) ? 1(anomaly) : 0(normal)
```

## 핵심 설계 결정

| 항목 | 결정 | 이유 |
|------|------|------|
| L2 normalize 후 거리 계산 | scale 불변 | 도메인마다 feature magnitude가 달라 threshold 불안정 |
| threshold = support_mean (n_sigma=0) | 학습-추론 일치 | EpisodicProto도 support_mean을 threshold로 사용 |
| Mahalanobis (LedoitWolf) | 방향성 거리 | cosine/L2보다 AUROC 높음 (0.9239 vs 0.8993) |
| calib precision, test prototype | 구조 분리 | precision은 학습 도메인 분포, prototype은 test 도메인 적응 |

## 평가
```
seeds [0, 1, 2, 3, 4] × 3개 test 도메인 = 15회 평균
지표: AUROC (ranking), Acc (정확도), F1 (precision/recall 조화평균)
주 지표: Acc, F1
```

## 실행 명령

### 단일 fold (빠른 검증)
```bash
conda run -n torch python scripts/eval/eval_all_folds.py \
  --root processed_gadf_fine_4096 --mode fine --num_classes 2 \
  --ckpt_fold_500 checkpoints/fine15_fold500.pth \
  --test_fold 500
```

### 전체 4 fold
```bash
conda run -n torch python scripts/eval/eval_all_folds.py \
  --root processed_gadf_fine_4096 --mode fine --num_classes 2 \
  --ckpt_fold_500 checkpoints/fine15_fold500.pth \
  --ckpt_fold_600 checkpoints/fine15_fold600.pth \
  --ckpt_fold_700 checkpoints/fine15_fold700.pth \
  --ckpt_fold_800 checkpoints/fine15_fold800.pth
```

## 현재 최종 결과 (n_sigma=0.0)

| Fold | Base Acc | Base F1 | z_inv Acc | z_inv F1 | z_inv AUROC |
|------|---------|--------|----------|---------|------------|
| 500+502+504 | 0.8357 | 0.8987 | 0.8847 | 0.9357 | 0.9507 |
| 600+602+604 | 0.8497 | 0.9092 | 0.8666 | 0.9252 | 0.9260 |
| 700+702+704 | 0.8338 | 0.8998 | 0.8607 | 0.9197 | 0.9086 |
| 800+802+804 | 0.8645 | 0.9216 | 0.8611 | 0.9192 | 0.9104 |
| **Avg** | **0.8459** | **0.9073** | **0.8683** | **0.9250** | **0.9239** |

## 실패한 추론 방법들

| 방법 | 결과 |
|------|------|
| n_sigma=2.0 (support_std 기반) | fold_700/800 threshold 과도하게 높음 |
| calib_std 기반 n_sigma | n_sigma 증가할수록 Acc 하락 |
| L2 distance (use_l2) | AUROC 0.8993 < Mahalanobis 0.9239 |
| 학습된 classifier 직접 사용 | hold-out domain에서 역전 (AUROC ~0.09) |
