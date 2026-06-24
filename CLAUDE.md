# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Goal

**도메인 불변성을 고려한 정상 특징 메모리 기반 퓨샷 이상탐지 (Few-shot Anomaly Detection)**

미학습 RPM 도메인(예: 800/802/804)에서도 정상 샘플 몇 개만으로 이상 탐지가 가능하도록 학습.

- 원래 논문: AUROC + cosine similarity + 5개 coarse domain (400/500/600/700/800)
- 저널 확장: **15개 fine-grained domain + Acc/F1 (현장 적용)** + z_inv 도메인 불변성 강화

## Setup

```bash
pip install -r requirements.txt
pip install PyWavelets  # CWT 전처리용
```

Python 환경: `/home/smai9/anaconda3/envs/torch/bin/python`

Large artifacts (`.pth`, `processed/`, `processed_cwt/`, `processed_fine/`, `HUST bearing dataset/`, `results/`)는 git-ignored.

---

## Architecture

### 데이터 표현

**CWT + RPM 정규화** (`processed/make_cwt.py`) ← **현재 권장**
- frequency axis를 shaft frequency로 정규화 → order domain 변환
- 같은 결함 유형이 RPM에 관계없이 동일 position에 나타남 (BPFI → row 28)
- 출력: `processed_cwt/{domain}/{normal,anomaly}/{stem}_{start}.png`

### 도메인 구분

| 모드 | 도메인 수 | 디렉토리 |
|------|----------|---------|
| coarse | 5 (400/500/600/700/800) | `processed/` |
| **fine** (권장) | **15** (400/402/404/.../804) | `processed_cwt/` |

LOO 전략: speed group 단위 hold-out (예: 800+802+804 전체를 test)

### 클래스 레이블

```
fault_type: N=0, I=1(inner race), O=2(outer race), B=3(ball), IB/IO/OB=4(compound)
label:      0=normal, 1=anomaly  ← 평가 시 binary
```
- 학습: `binary_label = (fault_type > 0)` 으로 ClassClassifier 훈련
  - 이유: 5-class mc는 N/I/O/B/compound를 구분하는 gradient를 선택 → z_inv에 anomaly 정보 소실
  - binary mc는 "정상 vs 이상" gradient를 선택 → z_inv에 anomaly 정보 보존

### MaskDecompositionModel (`models/mask_decomposition_model.py`)

XdomainMix (Liu et al., 2024)의 feature decomposition 아이디어를 few-shot anomaly detection에 맞게 수정한 구조.

```
입력 x [B,3,224,224]
  → FeatureExtractor(ResNet50 up to layer3) → z [B,1024,14,14]
  → ClassClassifier(GAP → Linear) → class_logits [B,2]   (binary: normal/anomaly)
  → DomainClassifier(GAP → Linear) → domain_logits [B,15] (via GRL)

Gradient-based masks (XdomainMix 방식에서 soft mask로 수정):
  mc = ReLU(∂binary_anomaly_logit/∂z) / max  ← class-relevant (anomaly 정보 보존)
  md = ReLU(∂domain_score/∂z) / max          ← domain-relevant (GRL 없이 계산)

Feature decomposition:
  z_inv    = z ⊙ mc ⊙ (1-md)  ← class-relevant, domain-invariant
  z_notc_d = z ⊙ (1-mc) ⊙ md  ← domain-relevant

Adversarial: GRL(z) → domain_logits (domain loss용)
```

**현재 한계 및 개선 방향:**
- 현재 GRL이 `z` 전체에만 작용 → `z_inv`에 domain 정보가 잔존
- 개선: `z_inv`에 직접 GRL + domain adversarial loss 추가 필요
  ```
  GRL(z_inv) → domain classifier → domain loss  (z_inv가 domain 예측 불가하도록 직접 강제)
  ```

### Training Loss

```
L = CE(classifier(z_inv), binary_label)
  + λ_d · CE(domain_cls(GRL(z)), domain)
  + λ_m_eff · mean(mc ⊙ md)
```

### Inference (Few-shot)

1. support 정상 샘플 → z_inv 추출 → prototype (mean)
2. LedoitWolf shrinkage 공분산으로 Mahalanobis distance 계산
3. calib domain 정상 샘플 score 분포 (mean + 2σ)로 threshold 결정
4. query domain에서 Acc, F1 평가 (AUROC도 함께 보고)

---

## Commands

### 1. 전처리

```bash
python processed/make_cwt.py \
  --data_dir "HUST bearing dataset" \
  --save_dir processed_cwt \
  --domain_mode fine \
  --num_workers 8
```

### 2. 학습 (fine-grained, binary label)

```bash
# 800 RPM group hold-out
python experiments/train_mask_decomposition.py \
  --root processed_cwt \
  --train_domains 400 402 404 500 502 504 600 602 604 700 702 704 \
  --all_domains 400 402 404 500 502 504 600 602 604 700 702 704 800 802 804 \
  --epochs 20 --num_classes 2 --warmup_epochs 3 \
  --mask_weight 0.1 --domain_weight 1.0 \
  --save_path mask_decomposition.pth
```

### 3. 분해 품질 진단

```bash
python experiments/check_disentangle.py \
  --ckpt mask_decomposition.pth \
  --root processed_cwt \
  --all_domains 400 402 404 500 502 504 600 602 604 700 702 704 800 802 804

# 기대값:
#   z_inv domain acc    < 0.07  (chance = 1/15)
#   z_notc_d domain acc > 0.70
#   z_inv anomaly acc   > 0.80
```

### 4. t-SNE 시각화

```bash
python experiments/visualize_tsne_decomposition.py \
  --root processed_cwt \
  --domains 400 600 700 800 802 804 \
  --feature zinv \
  --ckpt mask_decomposition.pth \
  --num_classes 2 --num_domains 15 \
  --save_prefix results/tsne_binary
```

### 5. 전체 LOO 평가

```bash
python experiments/eval_all_folds.py \
  --root processed_cwt \
  --mode fine \
  --num_classes 2
```

---

## Checkpoints (현재)

| 파일 | Fold | num_classes | 날짜 |
|------|------|------------|------|
| `mask_decomposition.pth` | 800 hold-out | 2 (binary) | 2026-06-22 |
| `mask_decomposition_fold_700.pth` | 700 hold-out | 2 (binary) | 2026-06-22 |
| `mask_decomposition_fold_600.pth` | 600 hold-out | 2 (binary) | 2026-06-22 |
| `mask_decomposition_fold_500.pth` | 500 hold-out | 2 (binary) | 2026-06-22 |

---

## HUST 데이터셋 구조 (중요)

파일명의 숫자가 실제 RPM이 아님:

| 파일 그룹 | fs (sampling freq) | 의미 |
|----------|-------------------|------|
| N400, N500, ..., N800 | 24.93 kHz | 측정 배치 1 |
| N402, N502, ..., N802 | 24.22 kHz | 측정 배치 2 |
| N404, N504, ..., N804 | ~23.0 kHz | 측정 배치 3 |

→ **실제 구조**: 5개 RPM 조건 × 3회 반복 측정 = 15개 파일 그룹
→ N402는 "402 RPM"이 아니라 "400 RPM 조건의 두 번째 측정 배치"
→ **15개 도메인 처리 유지**: 배치 간 fs가 달라 실질적 분포 차이 존재
→ B400.mat, IB400.mat 없음 (ball/compound fault는 400 RPM 조건 미수집)

---

## Key Design Decisions

| 결정 | 이유 |
|------|------|
| CWT + RPM 정규화 | 결함 주파수는 RPM에 비례 → order domain 변환으로 class-domain 분리 가능 |
| Fine-grained 15 domains | 배치별 fs 차이 = 실질적 분포 차이 → 더 세밀한 domain invariance 강제 |
| Binary class label | 5-class mc는 anomaly 정보를 보존 못 함 → binary로 "정상 vs 이상" gradient 직접 선택 |
| Soft mask (XdomainMix 수정) | XdomainMix의 threshold binary mask 대신 soft mask → 연속적 분리로 안정적 학습 |
| GT label로 mc 계산 | max logit 사용 시 오분류 샘플에서 mc gradient 불안정 → GT logit 사용으로 안정화 |
| md 계산 시 GRL 제거 | GRL 통과 z는 adversarial-confused 상태 → GRL 없이 실제 domain 정보 반영 |
| Mask weight warm-up | 초기 mc/md 미정제 상태에서 z_inv ≈ 0 → CE 발산 방지 |
| Mahalanobis + LedoitWolf | 고차원 feature 공분산 추정 → cosine보다 분포 기반 거리로 정확도 향상 |
