# AGENTS.md

이 repo에서 작업하는 에이전트(Claude Code 등)가 지켜야 할 고정 규칙. 실험 방법론 자체는
`ARCHITECTURE.md`, 진행 중 계획은 `docs/exec-plans/`, 왜 그렇게 결정했는지는
`docs/design-docs/`를 본다.

## 환경

```bash
conda activate torch   # 모든 학습/평가는 이 환경에서
```
Python: `/home/smai9/anaconda3/envs/torch/bin/python`

## 프로토콜 고정값 (트랙별로 다름 — 섞어 쓰지 말 것)

| 트랙 | 데이터 root | Threshold | 비고 |
|---|---|---|---|
| coarse5 legacy (`original_mask_model.py`) | `processed/` | Youden's J (calib 이상 라벨 사용) | METHOD_615.md §6.2, label-leaking — 공식 비교엔 부적합 |
| fine15 확정 (`mask_decomposition_model.py`) | `processed_gadf_fine_4096/` | support_mean만 (n_sigma=0.0) | 논문 본편, 체크포인트 수정 금지 |
| coarse5 gated (`gated_mask_model.py`, 2026-08~) | `processed/` | support_mean + n_sigma·calib_std (n_sigma=2.0, leakage-free) | 현재 활성 개발 트랙 |

세 트랙의 수치는 threshold 프로토콜이 달라 직접 비교하지 않는다.

## 실험 규칙

- **GPU 동시 실행 지양**: 학습 job은 순차 실행이 기본 원칙(`ROADMAP_EXTENSION.md` 원칙 절 참고).
  eval-only(추론) job은 학습 job과 동시 실행 시 GPU 메모리 여유를 반드시 확인할 것 — 12GB 카드에서
  ResNet50 학습 1개 + 평가 1개 동시 실행 시 11.6GB까지 찬 사례 있음(OOM 위험, 2026-08-16).
- **모델 선택(하이퍼파라미터/weight 스윕)은 held-out test 도메인을 보지 않고 확정한다.**
  calib 도메인 내부 지표(도메인 프로브 F1 등)로 먼저 고르고, held-out 평가는 확정된 설정으로 **한 번만** 수행.
- 새 체크포인트는 기존 `*_best.pth`, `fine15_fold*.pth`를 덮어쓰지 않는다. 실험용은 별도 이름
  (`coarse5_fold{d}_gated_{variant}_s{seed}.pth`) 또는 `scratchpad/`에 저장.
- 효과가 없는 방향은 즉시 중단하고 ablation/negative-result로 기록한다(끝까지 다 돌리지 않음).

## 파일 규칙

- 실험 원본 데이터·중요 체크포인트는 `/tmp` 스크래치패드가 아니라 repo(`checkpoints/`, `results/`)에 저장한다.
- 유의미한 실험 결과는 요청 없이도 Notion(parent `35e20ebf59c4807c8451e6baa7054c90`)에 바로 정리한다.
- 체크포인트 명명: `{track}_fold{domain}_{variant}_s{train_seed}.pth`
  예) `coarse5_fold800_gated_noGRL_s1.pth`

## 문서 갱신 책임

| 무엇이 바뀌었나 | 갱신할 문서 |
|---|---|
| 모델 구조/손실 함수 변경 | `ARCHITECTURE.md` |
| "왜 이렇게 했는지" 의사결정 | `docs/design-docs/` 새 파일 |
| 여러 스텝짜리 실험 시작/종료 | `docs/exec-plans/active/` → 종료 시 `completed/`로 이동 |
| 재현성/불안정성 새로 발견 | `RELIABILITY.md` |
| 공식 확정 수치 | `docs/generated/results.md` + 해당 결과의 `QUALITY_SCORE.md` 체크리스트 |
