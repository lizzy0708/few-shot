# 6/15 Method 문서

## 개요

`processed/` (coarse, 5-domain) 데이터셋에서  
**4-shot 정상 샘플만으로 bearing fault 이상탐지**

Notion 6/15 기록 목표치: **AUROC 0.9913 / Acc 0.9484 / F1 0.9683**

---

## 1. 모델 구조

### FeatureExtractor
```
ResNet50 (ImageNet pretrained)
  → conv1 → bn1 → relu → maxpool
  → layer1 → layer2 → layer3
  출력: z [B, 1024, 14, 14]
```

### Classifier / DomainClassifier
```
ClassClassifier:   GAP(z) → Linear(1024, 2)   # normal / anomaly 
DomainClassifier:  GAP(z) → Linear(1024, 5)   # 400/500/600/700/800 (GRL)
```

### Feature Decomposition
```
mc = ReLU(∂class_score/∂z)  / max    # class-relevant mask
md = ReLU(∂domain_score/∂z) / max    # domain-relevant mask

z_inv = z ⊙ mc ⊙ (1 - md)           # 클래스 관련, 도메인 불변
```

### 학습 손실
```
L = CE(class_logits, label)           # 정상/이상 분류
  + domain_weight × CE(domain_logits, domain)  # 도메인 적대 (GRL)
  + mask_weight   × mean(mc ⊙ md)             # 마스크 직교성
```

---

## 2. 학습 설정

| 항목 | 값 |
|------|----|
| 데이터 | `processed/` (coarse, GADF 224×224) |
| 도메인 | 5개: 400/500/600/700/800 = 베어링 타입 6204/6205/6206/6207/6208 (RPM 아님 — [원 논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC10327369/) 확인, 2026-08-13 정정) |
| Epochs | 10 |
| Batch size | 16 |
| LR | 1e-4 (Adam) |
| mask_weight | 0.1 |
| domain_weight | 1.0 |
| 추가 손실 | 없음 (SupCon / Proto / Episodic 없음) |
| Seed | 고정 없음 (랜덤) |

### LOO Fold 구조

| Fold | Train 도메인 | Test 도메인 | 체크포인트 |
|------|------------|-----------|---------|
| fold_500 | 400, 600, 700, 800 | 500 | `checkpoints/decomposition_400_600_700_800.pth` |
| fold_600 | 400, 500, 700, 800 | 600 | `checkpoints/decomposition_400_500_700_800.pth` |
| fold_700 | 400, 500, 600, 800 | 700 | `checkpoints/decomposition_400_500_600_800.pth` |
| fold_800 | 400, 500, 600, 700 | 800 | `checkpoints/decomposition_400_500_600_700.pth` |

---

## 3. 평가 방법 (원본 — 코사인 유사도)

초기 eval 스크립트 (`git cf6404c`)의 방법:

```
1. calib 도메인(4개) 전체 샘플(정상+이상) → z_inv 추출
2. support 4개 정상 샘플 → z_inv prototype (mean)
3. query 샘플 z_inv → 1 - cosine_similarity(z_inv, prototype) = anomaly score
4. threshold: calib scores로 grid search → best accuracy
5. AUROC / Acc / F1 계산
```

Seeds: [0, 1, 2, 3, 4] → 5-seed 평균

### 현재 eval 방법 (Mahalanobis + Youden)

이후 변경된 방법 (현재 `eval_all_folds.py`):

```
1. calib 도메인 정상 샘플 → z_inv → 대각 공분산 정밀행렬 추정 (diag precision)
2. support 4개 정상 → z_inv prototype (mean)
3. query z_inv → Mahalanobis distance from prototype = anomaly score
4. threshold: Youden's J (calib 정상+이상 ROC에서 TPR-FPR 최대)
5. AUROC / Acc / F1 + confusion matrix
```

---

## 4. 학습 명령어 (재현용)

```bash
# fold_500
conda run -n torch python experiments/train_mask_decomposition.py \
  --root processed \
  --train_domains 400 600 700 800 \
  --all_domains 400 500 600 700 800 \
  --epochs 10 --mask_weight 0.1 --domain_weight 1.0 \
  --save_path checkpoints/decomposition_400_600_700_800.pth

# fold_600
conda run -n torch python experiments/train_mask_decomposition.py \
  --root processed \
  --train_domains 400 500 700 800 \
  --all_domains 400 500 600 700 800 \
  --epochs 10 --mask_weight 0.1 --domain_weight 1.0 \
  --save_path checkpoints/decomposition_400_500_700_800.pth

# fold_700
conda run -n torch python experiments/train_mask_decomposition.py \
  --root processed \
  --train_domains 400 500 600 800 \
  --all_domains 400 500 600 700 800 \
  --epochs 10 --mask_weight 0.1 --domain_weight 1.0 \
  --save_path checkpoints/decomposition_400_500_600_800.pth

# fold_800
conda run -n torch python experiments/train_mask_decomposition.py \
  --root processed \
  --train_domains 400 500 600 700 \
  --all_domains 400 500 600 700 800 \
  --epochs 10 --mask_weight 0.1 --domain_weight 1.0 \
  --save_path checkpoints/decomposition_400_500_600_700.pth
```

## 5. 평가 명령어

```bash
# 현재 eval (Mahalanobis + Youden, no PCA, seeds 0~4)
conda run -n torch python experiments/eval_all_folds.py \
  --root processed --mode coarse --num_classes 2 \
  --ckpt_fold_500 checkpoints/decomposition_400_600_700_800.pth \
  --ckpt_fold_600 checkpoints/decomposition_400_500_700_800.pth \
  --ckpt_fold_700 checkpoints/decomposition_400_500_600_800.pth \
  --ckpt_fold_800 checkpoints/decomposition_400_500_600_700.pth \
  --pca_dim 0 --youden --seeds 0 1 2 3 4
```

---

## 6. 공식 기록 (coarse5 성능)

### 6.1 역대 최고 — 재현 불가 (기록용)

> **AUROC 0.9913 / Acc 0.9484 / F1 0.9683 (2026-06-15 Notion 기록)**
> 당시 fold_600/800 체크포인트가 삭제되어 **재현 불가**. seed 미고정 학습이라
> 재학습으로도 동일 수치 보장 안 됨 (멀티시드 재학습 시도로도 미달 → 후보 전량 폐기).
> 논문에는 재현 가능한 6.2 수치를 사용할 것.

| Fold | 6/15 AUROC (재현 불가) |
|------|------------------|
| 500  | 0.9948 |
| 600  | 0.9923 |
| 700  | 1.0000 |
| 800  | 0.9779 |
| **Avg** | **0.9913** |

### 6.2 현재 best — 재현 가능 (공식 수치, Notion 6/29 확정)

fold 500/700은 5/1 학습분, fold 600/800은 멀티시드 재학습 중 best seed.

| Fold | 체크포인트 | AUROC | Acc | F1 |
|------|-----------|-------|-----|-----|
| 500  | `coarse5_fold500_best.pth` (구 decomposition_400_600_700_800, 5/1) | 0.9898 | 0.9452 | 0.9672 |
| 600  | `coarse5_fold600_best.pth` (구 decomp_fold_600_s3, best seed) | 0.9321 | 0.8247 | 0.8862 |
| 700  | `coarse5_fold700_best.pth` (구 decomposition_400_500_600_800, 5/1) | 0.9992 | 0.9013 | 0.9389 |
| 800  | `coarse5_fold800_best.pth` (구 decomp_fold_800_s2, best seed) | 0.8588 | 0.6620 | 0.7502 |
| **Avg** | | **0.9450** | **0.8333** | **0.8856** |

- fold_800이 병목 (base AUROC 0.8349 — 도메인 자체가 어려움)
- fine15와 직접 비교 주의: threshold 프로토콜이 다름
  (coarse5 = Youden, calib 이상 라벨 사용 / fine15 = support mean, 정상 4개만)

### 현재 보유 체크포인트 (2026-07-09 정리: fold별 best만 보관, 탈락 후보 55개 삭제)

| 파일 | 용도 |
|------|------|
| `checkpoints/coarse5_fold{500,600,700,800}_best.pth` | coarse5 fold별 best (6.2 표) |
| `checkpoints/fine15_fold{500-800}.pth` | **fine15 최종 체크포인트 (수정 금지)** |
