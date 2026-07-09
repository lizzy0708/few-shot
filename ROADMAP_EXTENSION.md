# 저널 2차 확장 로드맵 (2026-07-09 수립)

fine15 확정 결과(AUROC 0.9239 / Acc 0.8683 / F1 0.9250, `fine15-baseline` 태그) 위에
방법 기여 2개를 쌓는 확장. 상세 배경은 노션 "저널 2차 확장 — 계획과 방향" 참고.

## 확장 기여 (계획)

### A. Patch Memory (진행 중)

**문제의식**: z_inv [2048,7,7]를 GAP으로 평균 → gradient mask가 만든 공간 선택성을 버림.
AnomalyDINO(WACV'25)의 patch-kNN 패러다임을 z_inv에 적용.

| 단계 | 상태 | 내용 |
|---|---|---|
| Phase 1: 평가측 파일럿 | ✅ 완료 | `eval_patch_folds.py`. fold_500: cosine-top5 AUROC 0.9503 ≈ GAP 0.9507, Acc는 열세 → 신호 존재하나 평가 교체만으론 부족 (학습-추론 불일치) |
| Phase 2: 학습 정렬 | 🔄 Gate 2 실행 중 | `episodic_patch_loss()` 추가 (train_mask_decomposition.py, `--patch_episodic_weight`). fold_500 재학습 → patch 평가가 0.9507/0.8847 초과하는지 판정 |
| Phase 3: 본 실험 | 대기 | 4 fold × 5 seeds, 조건표(GAP/patch평가만/patch학습정렬/hybrid), 거리·top-k ablation |

**Gate 2 실패 시**: weight 조합 1회 재시도 → 최종 실패면 ablation 소재로 강등, GAP 유지.

### B. Hierarchical Domain (다음 착수)

**문제의식**: 15개 도메인은 실제로 RPM(5) × 측정 배치(3)의 2계층인데
flat 15-way adversarial은 물리적 변동(RPM)과 측정 변동(배치)을 동일 취급.
테스트 축이 정확히 RPM hold-out이므로 두 변동의 분리 제거가 이론적으로 정합.

**점진 설계** (v5의 z_inv 직접 GRL 실패 전례 → 한 번에 안 감):
1. 1단계: 기존 GRL feature에 5-way RPM 헤드 추가 (multi-task adversarial). 라벨 공짜 (`rpm=idx//3, batch=idx%3`)
2. 2단계 (1단계 성공 시): md_rpm / md_batch 마스크 분리, z_inv = z ⊙ mc ⊙ (1-md_rpm) ⊙ (1-md_batch)
3. 분석 figure: md_rpm vs md_batch 시각화 — 채널별 변동 요인 해석

### C. DINOv2 — 베이스라인으로 강등

방법 기여 아님. Phase 3 비교표에 frozen DINOv2 + patch-kNN (AnomalyDINO 방식) 한 줄.
"자연 이미지 특징 vs 도메인 학습 z_inv on GADF" 논거용.

## 최종 그림

```
방법 = 계층적 도메인 마스크 (B) + 도메인 불변 patch memory (A)
비교 = GAP 기존 / AnomalyDINO(DINOv2) / coarse5
```

A와 B는 직교 (A=스코어링 방식, B=제거 대상) → 독립 검증 후 결합.

## 원칙

- 게이트 미통과 방향은 즉시 중단, ablation/분석 소재로 전환 (물성-shift 전례)
- `fine15_fold*.pth` 수정 금지, 새 체크포인트는 `patch15_*`, `hier15_*` 명명
- threshold 프로토콜 불변: 정상 support 4개만 사용 (Youden 금지)
- 실험은 순차 실행 (병렬 금지)
