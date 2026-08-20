# 저널 2차 확장 로드맵 (2026-07-09 수립 → 2026-07-10 탐색 종료)

> **최종 결론 (7/10)**: 4갈래 확장(A patch, B1/B2 hierarchical, DC) 전부 fine15 기준
> (AUROC 0.9239 / Acc 0.8683 / F1 0.9250) 미초과 → fine15가 이 설정의 천장으로 확정.
> 탐색 기록은 ablation/discussion 소재로 전환. 아래 각 섹션에 최종 수치 기록.
>
> **건진 것들**:
> 1. md_rpm/md_batch 채널 분석: 두 변동 요인이 거의 완전 분리된 채널 사용
>    (상관 r=-0.007, top-100 겹침 3% < 랜덤 5%) → `results/md_hierarchy_fold500.png`
> 2. mc/md 마스크는 순수 채널 마스크임을 규명 (GAP→Linear 분류기라 gradient가
>    공간 상수) → patch 확장이 동률에 그친 구조적 이유
> 3. patch vs GAP fold별 상호보완 패턴 → "GADF 이상은 전역 패턴" 분석
> 4. 4-shot 프로토타입은 DC 보정으로도 개선 불가 → 현 통계 추정이 이미 포화

fine15 확정 결과(AUROC 0.9239 / Acc 0.8683 / F1 0.9250, `fine15-baseline` 태그) 위에
방법 기여 2개를 쌓는 확장이었음. 상세 배경은 노션 "저널 2차 확장 — 계획과 방향" 참고.

## 확장 기여 (계획)

### A. Patch Memory (진행 중)

**문제의식**: z_inv [2048,7,7]를 GAP으로 평균 → gradient mask가 만든 공간 선택성을 버림.
AnomalyDINO(WACV'25)의 patch-kNN 패러다임을 z_inv에 적용.

| 단계 | 상태 | 내용 |
|---|---|---|
| Phase 1: 평가측 파일럿 | ✅ 완료 | fold_500: cosine-top5 AUROC 0.9503 ≈ GAP 0.9507, Acc 열세 → 신호 존재하나 학습-추론 불일치 |
| Phase 2: 학습 정렬 (Gate 2) | ✅ 통과 | fold_500 mahal-top5 0.9516/0.9023/0.9423 — 전 지표 초과 |
| Phase 3: 본 실험 | ❌ **최종 미달** | 4-fold avg: cosine-top5 0.9256/0.8681/0.9195 (동률), hybrid 0.9211(열세), mahal 0.9041(fold800 붕괴로 불안정). fold별 2승1무1패 — 평균 개선 없음 |

**판정: ablation 소재로 강등** (patch15_fold*.pth 4개는 재현용 보관).
사후 규명: mc/md가 순수 채널 마스크(공간 상수)라 "마스크의 공간 선택성 활용" 전제 자체가
성립하지 않았음 — patch의 공간 정보는 z 원본에서만 오고, 그 정보량이 GAP 대비 우위가 없음.

### B. Hierarchical Domain (다음 착수)

**문제의식**: 15개 도메인은 실제로 RPM(5) × 측정 배치(3)의 2계층인데
flat 15-way adversarial은 물리적 변동(RPM)과 측정 변동(배치)을 동일 취급.
테스트 축이 정확히 RPM hold-out이므로 두 변동의 분리 제거가 이론적으로 정합.

**결과 (7/10, 둘 다 fold_500 기준 0.9507/0.8847 미달):**
1. 1단계 (5-way RPM adversarial 헤드): ❌ w=1.0 → 0.9322/0.8706, w=0.5 → 0.9113/0.8808.
   encoder에 adversarial 압력 추가는 v5 포함 3연속 실패 — 패턴 확정
2. 2단계 (md_rpm/md_batch disc 분리, encoder 무영향): ❌ 0.9408/0.8789 — 최근접이나 미달.
   `--hier_md` 플래그로 코드 보존 (`hier15_md_fold500.pth`)
3. 분석 figure: ✅ **성과** — 채널 상관 r=-0.007, top-100 겹침 3/100 (랜덤 이하).
   물리(RPM) 변동과 측정(배치) 변동이 사실상 분리된 채널 집합에 인코딩됨.
   flat 15-way md가 이미 그 합집합을 커버하기에 분리가 성능 개선으로 이어지지 않은 것으로 해석.
   → `results/md_hierarchy_fold500.png`, `scripts/visualize/visualize_md_hierarchy.py`

### C. DINOv2 — 베이스라인으로 강등 (미실행)

방법 기여 아님. 필요 시 비교표에 frozen DINOv2 + patch-kNN 한 줄 추가 가능 (선택).

### D. Distribution Calibration — ❌ 종료 (7/10)

Tukey 변환 + 프로토타입 calib-통계 보정 (`scripts/eval/eval_dc_folds.py`, 평가만).
4-fold avg: baseline 0.9233 / Tukey 0.9117↓ / DC β=0.2 0.9222 / Tukey+DC 0.9096↓ —
전 조건 동률~열세. 4-shot 프로토타입 통계는 현 LedoitWolf+PCA 체계에서 이미 포화.

## 논문 구성 (탐색 종료 후 확정)

```
본편  = fine15 (z_inv, GAP, Mahalanobis) — 기존 확정 결과
비교  = coarse5, baseline(pretrained) 특징
분석  = ① md 채널 분해 figure (물리 vs 측정 변동의 채널 직교성)
        ② patch vs GAP (GADF 이상의 전역성)
        ③ 확장 4갈래 negative results 요약 (robustness of the recipe)
```

## 원칙

- 게이트 미통과 방향은 즉시 중단, ablation/분석 소재로 전환 (물성-shift 전례)
- `fine15_fold*.pth` 수정 금지, 새 체크포인트는 `patch15_*`, `hier15_*` 명명
- threshold 프로토콜 불변: 정상 support 4개만 사용 (Youden 금지)
- 실험은 순차 실행 (병렬 금지)
