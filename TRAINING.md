# 학습 (Training)

## 스크립트
`experiments/train_mask_decomposition.py`  
`models/mask_decomposition_model.py`

## 4-Fold LOO 구조
| Fold | Test 도메인 | Train 도메인 |
|------|-----------|------------|
| fold_500 | 500, 502, 504 | 나머지 12개 |
| fold_600 | 600, 602, 604 | 나머지 12개 |
| fold_700 | 700, 702, 704 | 나머지 12개 |
| fold_800 | 800, 802, 804 | 나머지 12개 |

## 모델 구조

### 1. Feature 추출
```
입력 x [B, 3, 224, 224]
  → ResNet50 (layer1~layer4, ImageNet pretrained)
  → z [B, 2048, 7, 7]
```

### 2. Gradient Mask 계산 (create_graph=True)
```
binary_label = (fault_type > 0)   # N=0, fault=1

class_score  = class_logits[GT label].sum()
domain_score = domain_logits.max(dim=1)[0].sum()   (GRL 통해 계산)

mc = ReLU(∂class_score / ∂z)  / max   ← class-relevant mask
md = ReLU(∂domain_score / ∂z) / max   ← domain-relevant mask
```

### 3. Feature 분해
```
z_inv = z ⊙ mc ⊙ (1 - md)   ← class-relevant, domain-invariant  ← 추론에 사용
z_c_notd = z_inv  (동일)

z_notc_d = z ⊙ (1-mc) ⊙ md  ← domain-relevant
```

## Loss 구성
```
L = CE(classifier(z_inv), binary_label)                      ← class loss
  + domain_weight   × CE(domain_cls(GRL(z)), domain)          ← domain adversarial
  + mask_weight_eff × mean(mc ⊙ md)                           ← mask orthogonality
  + supcon_weight   × SupCon(normalize(z_inv), binary_label)  ← normal clustering
  + proto_weight    × ProtoAlign(normalize(z_inv), domain)     ← cross-domain alignment
  + episodic_weight × EpisodicProto(z_inv, binary_label)       ← 4-shot 시뮬레이션
```

### EpisodicProto Loss (핵심)
추론 시나리오를 학습에서 직접 시뮬레이션:
```python
z_norm = F.normalize(z_inv_pool, dim=1)      # unit sphere

for _ in range(4):  # 4 episodes per batch
    support 4개 정상 샘플 랜덤 선택
    prototype = normalize(mean(z_norm[support]))
    dists = L2(z_norm, prototype)             # unit sphere 위 L2 거리
    thresh = mean(dists[support])             # = 추론 threshold와 동일
    loss += BCE(dists - thresh, binary_label)
```

### SupCon Loss
정상 샘플끼리 가깝게, 정상-이상 쌍 멀게 (anchor = 정상만):
```python
z = normalize(z_inv_pool)
# normal-normal: attract / normal-anomaly: repel / anomaly-anomaly: ignore
```

### ProtoAlign Loss
도메인별 정상 prototype이 z_inv 공간에서 가까워야 함:
```python
z = normalize(z_inv_pool)
protos = [mean(z[domain==d, label==0]) for d in unique_domains]
loss = mean(||proto - global_mean||^2)
```

### Mask Weight Warm-up
초기에 mc/md가 미정제 → z_inv ≈ 0 → class loss 발산 방지:
```
epoch < 3:  mask_weight_eff = 0
epoch >= 3: mask_weight_eff = mask_weight × min(1.0, linear_ramp)
```

## 현재 하이퍼파라미터 (v3, best)
```bash
--root processed_gadf_fine_4096
--epochs 20
--batch_size 16
--lr 1e-4
--num_classes 2
--warmup_epochs 3
--mask_weight 0.1
--domain_weight 1.0
--supcon_weight 0.5
--proto_weight 0.1
--episodic_weight 1.0
--seed 42
```

## 실행 명령 (fold_500 예시)
```bash
conda run -n torch python experiments/train_mask_decomposition.py \
  --root processed_gadf_fine_4096 \
  --train_domains 400 402 404 600 602 604 700 702 704 800 802 804 \
  --all_domains 400 402 404 500 502 504 600 602 604 700 702 704 800 802 804 \
  --epochs 20 --num_classes 2 --warmup_epochs 3 \
  --mask_weight 0.1 --domain_weight 1.0 \
  --supcon_weight 0.5 --proto_weight 0.1 --episodic_weight 1.0 \
  --seed 42 --save_path resnet50_l4_v3_fold_500.pth
```

## 체크포인트 (현재 best)
| 파일 | Fold |
|------|------|
| `resnet50_l4_v3_fold_500.pth` | 500 hold-out |
| `resnet50_l4_v3_fold_600.pth` | 600 hold-out |
| `resnet50_l4_v3_fold_700.pth` | 700 hold-out |
| `resnet50_l4_v3_fold_800.pth` | 800 hold-out |
