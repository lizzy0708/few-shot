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

## 비교표
| | AUROC | Acc | F1 |
|---|---|---|---|
| raw z (decomposition 없음) | 0.9449 | 0.8684 | 0.9174 |
| z_inv (진짜 도메인 라벨) | 0.9539/0.9540* | 0.9054/0.9053* | 0.9430/0.9429* |
| z_inv (shuffle 도메인 라벨) | ? | ? | ? |

*소수점 4자리 rounding 차이(원본 실험 3 vs 이후 재확인 로그) — 실질적으로 동일값.

## 해석 기준 (결과 후 적용, 미리 결론 내지 않음)
- shuffle ≈ raw z (진짜 라벨 버전보다 확실히 낮음) → 진짜 도메인 적대적 학습이
  유의미하게 기여한다는 뜻
- shuffle ≈ 진짜 라벨 버전 (raw z보다 확실히 높음) → 개선은 라벨 진위와 무관한 gate
  구조 자체의 재가중 효과 — 논문에 명시적으로 추가할 가치 있는 발견
- 중간 어딘가면 애매함으로 보고, 추가 seed 필요 여부만 판단(자동 확장 없음)

## 독립 도메인 프로브 결과 해석 주의
shuffle 버전은 애초에 진짜 도메인 라벨을 학습하지 않으므로, 이 프로브의 domain-acc는
"도메인 불변성이 얼마나 되는지"의 지표로 해석할 수 없음(진짜 라벨을 본 적 없는
encoder에 진짜 라벨로 프로브를 걸면 필연적으로 낮게 나올 수 있음) — 참고용으로만 기록,
메인 해석은 AUROC/Acc/F1 3-way 비교표로 함.
