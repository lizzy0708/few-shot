# QUALITY_SCORE.md — 결과 신뢰도 체크리스트

새 수치를 "확정 결과"로 `docs/generated/results.md`나 논문에 올리기 전에 아래를 확인한다.

## 체크리스트

- [ ] **몇 seed로 검증됐는가** — 단일 seed 수치는 잠정치로만 취급, 최소 5 eval-seed
      (4-shot support 재샘플링) 평균인가? fold 800/600은 학습 seed 자체의 분산이 커서
      (`RELIABILITY.md` §1) 3-seed 학습 앙상블 없이는 신뢰 불가.
- [ ] **threshold 프로토콜에 leakage가 없는가** — calib에 이상 라벨을 쓰는 Youden 방식이면
      "leakage-free"라고 표기하지 않는다 (`docs/design-docs/threshold-protocol.md`).
- [ ] **하이퍼파라미터가 held-out을 보기 전에 확정됐는가** — fold별로 다른 값을 쓰고 있다면
      그 선택 근거가 held-out 성능이 아닌지 확인.
- [ ] **코드 위치가 명시돼 있는가** — 어떤 스크립트/체크포인트로 재현하는지
      (`docs/exec-plans/`에 재현 명령이 있는가).
- [ ] **알려진 기준 수치와 대조했는가** — 파이프라인을 바꿨다면 안 바뀐 부분(예: raw baseline)이
      기존 확정 수치와 여전히 일치하는지 먼저 확인 (`RELIABILITY.md` §3 사례).
- [ ] **비교 대상과 프로토콜이 동일한가** — coarse5 legacy / fine15 / coarse5 gated 세
      트랙은 threshold 프로토콜이 달라 직접 비교 금지 (`ARCHITECTURE.md`).

## 현재 확정 결과의 등급

| 결과 | seed | leakage-free | 등급 |
|---|---|---|---|
| fine15 z_inv (AUROC 0.9239) | 5 eval-seed | ✅ (n_sigma=0.0) | **논문 인용 가능** |
| coarse5 legacy (AUROC 0.9450) | 미고정(랜덤) | ❌ (Youden) | 참고용, 논문 인용 금지 |
| coarse5 gated 3-seed 앙상블 (AUROC 0.9426) | 3 train-seed × 5 eval-seed | ✅ (n_sigma=2.0) | **현재 최고, 재현 명령 확보** — 논문 인용 후보 |
| coarse5 역대 최고(AUROC 0.9913, 6/15) | 미고정 | ❌ | **재현 불가, 체크포인트 소실 — 인용 금지** |
