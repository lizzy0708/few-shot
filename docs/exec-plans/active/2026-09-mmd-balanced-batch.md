# MMD domain-alignment, attempt 4: domain-balanced batch (2026-09-24)

**정말 마지막 시도.** 실패하면 domain leakage 개선 방향(직교화·domain_weight·MMD 불균형
batch·MMD 균형 batch, 총 4개 메커니즘/6개 설정) 완전 최종 종료, 추가 시도 없음.

## 배경
mmd_weight 스윕 {10.0, 1.0, 0.1} 전부 실패 (`2026-09-gated-nodg-mmd.md` 참고):
monotonic trade-off — weight↑ ⟹ domain-acc 더 감소하지만 Acc/F1도 더 붕괴, weight↓ ⟹
반대. 어떤 weight도 두 조건(domain-acc -5pp 이상 & Acc/F1 -1pp 이내) 동시 만족 못함.

**가설**: batch_size=16이 4개 calib 도메인에 무작위 배정되면 도메인당 평균 4개뿐(분산도
큼) — utils/mmd.py의 multi-bandwidth RBF MMD 추정 자체가 편향/분산이 큰 상태였을 수
있음. 즉 "MMD loss"가 실제로는 노이즈에 가까운 신호를 최소화하고 있었을 가능성.

## 구현
- `utils/domain_balanced_sampler.py`: `DomainBalancedBatchSampler` — ConcatDataset의
  domain별 연속 index range(cumulative_sizes)를 이용해 매 배치가 정확히
  `batch_size // num_domains`개씩 각 도메인에서 뽑히도록 구성. 매 epoch 독립적으로
  재셔플.
- `experiments/train_gated_mask_model.py`: `--balanced_batch` 플래그 추가(기본 off,
  하위호환). on이면 DataLoader가 `batch_sampler=DomainBalancedBatchSampler(...)` 사용.
- `utils/mmd.py`, loss 공식은 변경 없음 — 오직 batch 구성 방식만 격리해서 테스트.

## 설정
- mmd_weight=1.0 고정 (스윕 결과 가장 균형 잡혔던 값 — 0.1은 효과 거의 없었고 10.0은
  과했음; 이번엔 batch 개선 효과만 순수하게 격리)
- batch_size=64 (GPU 메모리 프로브 결과: bs=64 peak 5.2GB 안전, bs=128 peak 10.3GB로
  12GB 카드에 너무 타이트해서 제외), num_domains=4(calib) → 도메인당 16개/batch
- domain_weight=1.0, GRL 유지, epochs=10, lr=1e-4, mask_weight=0.1, domain_disc_weight=1.0,
  `--no_domain_gate` (기존 mmd 스윕과 동일 조건, batch 구성만 다름)
- 1-seed(seed=0) x 4-fold 먼저 판정, 조짐 좋으면 3-seed 확장

## 체크포인트
`checkpoints/coarse5_fold*_gated_nodg_mmd1_balanced_s0.pth`

## 판정 기준 (사전 고정, 결과 후 변경 금지)
- 성공: domain-acc가 노이즈 범위를 넘어(5pp 이상) 감소 **AND** AUROC/Acc/F1 각각 -1pp
  이내 유지 → 3-seed로 확장해 최종 확정
- 실패: 위 조건 미충족 → domain leakage 개선 방향 완전 종료, 이후 사용자의 명시적
  요청 없이는 재언급 금지

## 비교 대상
| | domain-acc | class-acc | AUROC | Acc | F1 |
|---|---|---|---|---|---|
| 기존(GRL만, dw=1.0) | 0.6017 | 0.9790 | 0.9540 | 0.9053 | 0.9429 |
| MMD w=1.0 (불균형 batch) | 0.5756 | 0.9800 | 0.9409 | 0.8924 | 0.9337 |
| MMD w=1.0 + balanced batch | ? | ? | ? | ? | ? |
