# Control experiment: shuffled domain labels (2026-09-25)

**목적**: "z_inv의 Acc/F1 개선이 진짜 도메인 적대적 학습(GRL) 때문인지, 아니면
class_gate/mask 구조 자체의 재가중 효과(도메인 라벨의 진위와 무관)인지" 구분.
성공/실패 프레임 없음 — 결과가 어느 쪽이든 논문에 그대로 쓸 수 있는 결론.

## 배경
- 기존 GRL 학습으로도 z_inv에서 도메인 정보가 유의하게 제거되지 않음(독립 프로브
  domain-acc 60.2%, chance 20%) — domain leakage 개선 4개 메커니즘/6개 설정 모두 실패로
  종료됨(`docs/generated/results.md` "MMD 도메인 정렬 시도" 절)
- 그럼에도 raw z 대비 z_inv의 Acc/F1은 유의하게 개선됨(실험 3/3b, p<0.001)
- 이 개선의 메커니즘이 "도메인 불변성"이 아니라면 무엇인지 아직 불명 — 이번 실험으로 직접 검증

## 구현
- `experiments/train_gated_mask_model.py`: `--shuffle_domain_labels` 플래그 추가(기본 off,
  하위호환). on이면 각 학습 샘플의 진짜 domain label을 seed 고정 permutation으로 무작위
  재배정(전체 학습 샘플 domain label 배열을 한 번 섞어서 고정 — epoch마다 다시 섞지
  않음, 즉 "일관되지만 틀린" 라벨). class_loss/mask 구조/class_gate는 전부 동일.
  domain_classifier+GRL만 가짜 라벨에 대해 학습됨.
- 구현 위치: `train()` 함수, `datasets` 리스트 생성 직후 `ConcatDataset` 이전 —
  `HUSTDataset.samples`의 domain_idx(tuple index 2)만 in-memory로 치환, 디스크 파일이나
  다른 스크립트의 HUSTDataset 인스턴스에는 영향 없음.

## 설정
- mmd_weight=0.0(끔), balanced_batch 끔 — 순수하게 domain label 진위 여부만 격리
- domain_weight=1.0, GRL 유지, epochs=10, batch_size=16, lr=1e-4, mask_weight=0.1,
  domain_disc_weight=1.0, `--no_domain_gate --shuffle_domain_labels`
- 1-seed(seed=0) x 4-fold 먼저 판정, 결과 애매하면 추가 seed 필요 여부만 판단(자동으로
  확장하지 않음)

## 체크포인트
`checkpoints/coarse5_fold*_gated_nodg_shuffled_s0.pth`

## 평가 (기존 wrapper 재사용 — 새 스크립트 불필요)
`eval_gated_folds_mmd.py`/`analyze_domain_invariance_tsne_mmd.py`의 `--mmd_tag` 인자는
이름과 달리 순수 checkpoint-path-tag 스위치이므로 그대로 재사용 가능:
```bash
conda run -n torch python experiments/eval_gated_folds_mmd.py --mmd_tag shuffled --beta 0.5 --n_sigma 2.0 --train_seeds 0
conda run -n torch python experiments/analyze_domain_invariance_tsne_mmd.py --mmd_tag shuffled --no_tsne
```

## 학습 커맨드 (fold별, calib 도메인은 held-out 제외 4개)
```bash
conda run -n torch python experiments/train_gated_mask_model.py \
  --root processed --train_domains <calib 4개> --all_domains 400 500 600 700 800 \
  --epochs 10 --batch_size 16 --lr 1e-4 --num_classes 2 \
  --mask_weight 0.1 --domain_weight 1.0 --domain_disc_weight 1.0 \
  --no_domain_gate --shuffle_domain_labels --seed 0 \
  --save_path checkpoints/coarse5_fold<fold>_gated_nodg_shuffled_s0.pth
```

## 비교표 (2026-09-25 결과, 1-seed x 4-fold)

`docs/generated/shuffled_domain_attempt/{train,eval,probe}.txt` 원본 로그.

| | AUROC | Acc | F1 |
|---|---|---|---|
| raw z (decomposition 없음) | 0.9449 | 0.8684 | 0.9174 |
| z_inv (진짜 도메인 라벨) | 0.9540 | 0.9053 | 0.9429 |
| **z_inv (shuffle 도메인 라벨)** | **0.9465** | **0.9041** | **0.9428** |

### Gain 분해 (raw z 대비)
| | AUROC Δ | Acc Δ | F1 Δ |
|---|---|---|---|
| 진짜 라벨 − raw z (전체 gain) | +0.91pp | +3.69pp | +2.55pp |
| shuffle − raw z (gate 구조만으로 얻는 gain) | +0.16pp | +3.57pp | +2.54pp |
| **shuffle이 전체 gain에서 차지하는 비율** | **17.6%** | **96.7%** | **99.6%** |
| 진짜 라벨 − shuffle (진짜 도메인 라벨이 추가로 기여하는 부분) | +0.75pp | +0.12pp | +0.01pp |

### fold별 AUROC/Acc/F1 (shuffle, 1-seed)
| Fold | AUROC | Acc | F1 |
|---|---|---|---|
| 500 | 0.9857±0.0018 | 0.9563±0.0046 | 0.9745±0.0029 |
| 600 | 0.9389±0.0028 | 0.8941±0.0151 | 0.9368±0.0106 |
| 700 | 0.9693±0.0037 | 0.9173±0.0079 | 0.9517±0.0044 |
| 800 | 0.8922±0.0027 | 0.8486±0.0189 | 0.9080±0.0144 |
| **Avg** | **0.9465** | **0.9041** | **0.9428** |

### 독립 도메인 프로브 (참고용 — 아래 해석 주의 참고)
| Fold | z dom-acc | z_inv dom-acc | z cls-acc | z_inv cls-acc |
|---|---|---|---|---|
| 500 | 0.5950 | 0.5550 | 0.9875 | 0.9925 |
| 600 | 0.5600 | 0.5925 | 0.9750 | 0.9850 |
| 700 | 0.5550 | 0.5925 | 1.0000 | 1.0000 |
| 800 | 0.6100 | 0.6450 | 0.9350 | 0.9425 |
| **Avg** | **0.5800** | **0.5962** | **0.9744** | **0.9800** |

(진짜 라벨 버전의 domain-acc=0.6017과 거의 동일 — 아래 해석 주의 참고. 사실상 노이즈
범위 내 동일값.)

## 해석 (사전 기준 적용)
결과는 **"shuffle ≈ 진짜 라벨 버전, 둘 다 raw z보다 확실히 높음"** 쪽에 명확히 위치함
(중간/애매함 아님 — 추가 seed 불필요):

- **Acc/F1**: shuffle이 raw z 대비 얻는 gain(Acc +3.57pp, F1 +2.54pp)이 진짜 라벨
  버전의 전체 gain(Acc +3.69pp, F1 +2.55pp)의 **96.7%/99.6%**를 이미 차지함 — 진짜
  라벨이 추가로 기여하는 부분은 사실상 0(Acc +0.12pp, F1 +0.01pp, 노이즈 범위).
  **Acc/F1 개선은 거의 전적으로 class_gate/mask 구조 자체의 재가중 효과이며, 도메인
  라벨이 진짜인지 가짜인지와 무관함.**
- **AUROC**: 반대로 gain의 대부분(82.4%, +0.75pp/+0.91pp)이 shuffle에는 없고 진짜
  라벨 버전에만 있음 — AUROC에 한해서는 진짜 도메인 적대적 학습이 의미 있게 기여함.
  다만 절대 크기 자체는 작음(raw z AUROC가 이미 0.9449로 높아 개선 여지가 원래 작았음).
- **독립 도메인 프로브**: shuffle 버전의 z_inv domain-acc(0.5962)가 진짜 라벨 버전
  (0.6017)과 사실상 동일 — 예상대로임(exec-plan에 명시한 해석 주의사항대로, encoder가
  진짜 라벨을 본 적이 없으므로 도메인 불변성 지표로 쓸 수 없고, raw feature가 원래
  갖고 있던 "자연적" 도메인 분리 가능성 수준을 그대로 반영할 뿐). 이 숫자로 도메인
  불변성에 대해 새로 결론 내릴 것은 없음 — 기존 결론(GRL이 도메인 정보를 유의하게
  제거하지 못함, 0.6017)과 일관될 뿐.

## 논문에 쓸 수 있는 결론
Acc/F1 개선은 "도메인 불변성" 때문이라는 서술은 근거가 약하고(도메인 프로브 결과가 이미
이를 시사했음), 이번 대조군 실험으로 **직접 확증**됨: 가짜 도메인 라벨로 학습해도 Acc/F1
개선의 96.7%/99.6%가 그대로 재현됨. 개선의 진짜 메커니즘은 class_gate 기반 채널
재가중(regularization/feature-selection 효과)이며, 도메인 라벨의 진위와 무관함 — 이는
흥미로운 독립적 발견으로 논문에 명시할 가치가 있음. 단, AUROC는 예외적으로 진짜 도메인
학습이 작지만 일관된 추가 기여(+0.75pp)를 하므로, "완전히 무관하다"고 과장하지 말고
지표별로 구분해서 서술할 것.
