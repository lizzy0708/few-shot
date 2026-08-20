# CLAUDE.md

## Goal

**도메인 불변성 기반 정상 특징 메모리 퓨샷 이상탐지 (Few-shot Anomaly Detection)**

미학습 RPM 도메인(예: 700/702/704)에서 정상 샘플 4개만으로 이상 탐지.

- 원 논문: AUROC + cosine similarity + 5개 coarse domain
- 저널 확장: **15개 fine-grained domain + Acc/F1** + z_inv 도메인 불변성 강화

---

## Setup

```bash
conda activate torch
pip install -r requirements.txt
```

Python: `/home/smai9/anaconda3/envs/torch/bin/python`  
Data root: `processed_gadf_fine_4096/` (GADF 변환 이미지, 224×224)

---

## Architecture

### MaskDecompositionModel (`models/mask_decomposition_model.py`)

```
입력 x [B,3,224,224]
  → ResNet50 (layer1~layer4) → z [B, 2048, 7, 7]
  → ClassClassifier(GAP→Linear) → class_logits [B,2]   (binary: normal/anomaly)
  → DomainClassifier(GAP→Linear) → domain_logits [B,15] (via GRL on z)

Gradient masks:
  mc = ReLU(∂class_score/∂z) / max     ← class-relevant
  md = ReLU(∂domain_score/∂z) / max    ← domain-relevant

Feature decomposition:
  z_inv = z_c_notd = z ⊙ mc ⊙ (1 - md)  ← class-relevant, domain-invariant
```

### Inference (Few-shot, 4-shot)

1. support 4개 정상 샘플 → z_inv 추출 → prototype (mean)
2. LedoitWolf shrinkage로 calib domain normal 공분산 추정
3. Mahalanobis distance → threshold = support_mean + 2σ
4. query domain Acc/F1/AUROC 평가

### Training Loss

```
L = CE(classifier(z_inv), binary_label)        ← class loss
  + λ_d  · CE(domain_cls(GRL(z)), domain)       ← domain adversarial (z 전체)
  + λ_m  · mean(mc ⊙ md)                        ← mask orthogonality
  + λ_sc · SupCon(z_inv, binary_label)          ← normal 클러스터링
  + λ_pa · ProtoAlign(z_inv, binary_label, domain) ← 도메인간 prototype 정렬
  + λ_ep · EpisodicProto(z_inv, binary_label)   ← 4-shot 테스트 시나리오 시뮬레이션
```

EpisodicProto: 각 배치에서 정상 4개→prototype→L2거리→BCE (학습-추론 정렬)

---

## Key Design Decisions

| 결정 | 이유 |
|------|------|
| GADF 224×224 | 시간-주파수 2D 표현으로 ResNet 활용 |
| ResNet50 layer4 (2048-dim) | layer3보다 AUROC/Acc 일관 향상 |
| Binary class label | 5-class mc는 anomaly 정보를 보존 못 함 |
| Soft mask (XdomainMix 수정) | binary mask 대신 soft → 안정적 학습 |
| GT label로 mc 계산 | max logit 사용 시 오분류 샘플 불안정 |
| Mahalanobis + LedoitWolf | 고차원에서 cosine보다 분포 기반 거리로 정확도 향상 |
| EpisodicProto loss | 학습-추론 disconnect 해소 (L2 threshold 시뮬레이션) |

---

## Current Results (fine15 최종 설정: layer4, 20 epochs, n_sigma=0.0, pca_dim=128)

| Fold | Base AUROC | Base Acc | Base F1 | z_inv AUROC | z_inv Acc | z_inv F1 |
|------|-----------|---------|--------|------------|----------|---------|
| 500  | 0.8848 | 0.8357 | 0.8987 | **0.9507** | **0.8847** | **0.9357** |
| 600  | 0.9013 | 0.8497 | 0.9092 | 0.9260 | 0.8666 | 0.9252 |
| 700  | 0.8918 | 0.8338 | 0.8998 | 0.9086 | **0.8607** | **0.9197** |
| 800  | 0.9050 | 0.8645 | 0.9216 | 0.9104 | 0.8611 | 0.9192 |
| **Avg** | **0.8957** | **0.8459** | **0.9073** | **0.9239** | **0.8683** | **0.9250** |

---

## Checkpoints (현재 best — fold별 best만 보관, 2026-07-09 정리)

| 파일 | Fold | 설정 | 비고 |
|------|------|------|------|
| `checkpoints/fine15_fold{500,600,700,800}.pth` | 각 fold hold-out | fine15 (layer4) | 메인 결과 (위 표), 수정 금지 |
| `checkpoints/coarse5_fold{500,600,700,800}_best.pth` | 각 fold hold-out | coarse5 (layer3) | avg AUROC 0.9450 / Acc 0.8333 / F1 0.8856 (METHOD_615.md §6.2) |

- coarse5 역대 최고(AUROC 0.9913, 6/15)는 체크포인트 소실로 **재현 불가** — 논문 인용 금지, 기록은 METHOD_615.md §6.1
- 두 설정은 threshold 프로토콜이 달라 직접 비교 주의 (coarse5=Youden/이상 라벨 사용, fine15=support mean/정상만)

---

## Commands

### 학습 (fold_500, fine15 설정)

```bash
conda run -n torch python scripts/train/train_mask_decomposition.py \
  --root processed_gadf_fine_4096 \
  --train_domains 400 402 404 600 602 604 700 702 704 800 802 804 \
  --all_domains 400 402 404 500 502 504 600 602 604 700 702 704 800 802 804 \
  --epochs 20 --num_classes 2 --warmup_epochs 3 \
  --mask_weight 0.1 --domain_weight 1.0 \
  --supcon_weight 0.5 --proto_weight 0.1 --episodic_weight 1.0 \
  --seed 42 --save_path checkpoints/fine15_fold500.pth
```

### 평가 (전체 4 fold)

```bash
conda run -n torch python scripts/eval/eval_all_folds.py \
  --root processed_gadf_fine_4096 --mode fine --num_classes 2 \
  --ckpt_fold_500 checkpoints/fine15_fold500.pth \
  --ckpt_fold_600 checkpoints/fine15_fold600.pth \
  --ckpt_fold_700 checkpoints/fine15_fold700.pth \
  --ckpt_fold_800 checkpoints/fine15_fold800.pth
```

### 평가 (단일 fold)

```bash
conda run -n torch python scripts/eval/eval_all_folds.py \
  --root processed_gadf_fine_4096 --mode fine --num_classes 2 \
  --ckpt_fold_500 checkpoints/fine15_fold500.pth --test_fold 500
```

---

## HUST 데이터셋 구조

| 파일 그룹 | 의미 |
|----------|------|
| N400/500/.../800 | 측정 배치 1 (fs=24.93 kHz) |
| N402/502/.../802 | 측정 배치 2 (fs=24.22 kHz) |
| N404/504/.../804 | 측정 배치 3 (fs≈23.0 kHz) |

5개 RPM 조건 × 3회 반복 = 15개 도메인. B400, IB400 없음 (400 RPM ball/compound 미수집).

---

## 시도했으나 효과 없었던 것들

| 방법 | 결과 |
|------|------|
| 40 epoch + cosine LR (v4) | fold_500 Acc 0.8786 < fine15 0.8968 (과적합) |
| L2 inference (use_l2) | AUROC 0.8993 vs 0.9239 (Mahalanobis가 우월) |
| 95th percentile threshold | Acc/F1 하락 |
| cross-val std threshold (calib_std) | n_sigma 증가할수록 Acc 하락 |
| Mahalanobis episodic training | 60~80분/fold (학습 불가) |
| ViT backbone | 도메인 loss 폭발 (21.7), OOM |
| episodic_weight 튜닝 (0.5, 2.0) | 유의미한 차이 없음 |
| z_inv 직접 GRL (v5, domain_inv_weight=0.5) | fold_500 AUROC 0.85, Acc 0.77 (심각한 하락) |
| **n_sigma=0.0 (support_mean만 사용)** | **Avg Acc 0.8683, F1 0.9250 → 현재 best** |
