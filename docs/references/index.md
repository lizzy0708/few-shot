# References

## 데이터셋
- HUST bearing dataset 원 논문: https://pmc.ncbi.nlm.nih.gov/articles/PMC10327369/
  도메인 코드 정의(파일명 숫자 = [베어링 타입][부하조건], RPM 아님)의 출처.
  발견 경위: `PREPROCESSING.md` 2026-08-13 정정 항목.

## 방법
- LedoitWolf shrinkage covariance: `sklearn.covariance.LedoitWolf` — 고차원(1024~2048dim)에서
  샘플 대비 차원이 큰 상황의 공분산 추정 안정화. 대각 공분산(diag(1/z_var))보다 항상 우월,
  세션 내 재확인(2026-08-16).
- GRL(Gradient Reversal Layer) / DANN: `models/original_mask_model.py`의 `grad_reverse`.
  domain-adversarial 표준 기법(Ganin & Lempitsky, DANN).
- Mahalanobis distance 기반 OOD/이상탐지: `INFERENCE.md` 참고.

## 관련 시도했으나 폐기된 방향 (재시도 방지용)
- AnomalyDINO(WACV'25) 스타일 patch-kNN → `ROADMAP_EXTENSION.md` §A, 구조적으로 mc/md가
  순수 채널 마스크(공간 선택성 없음)라 patch 확장이 무의미했음.
- Distribution Calibration(Tukey 변환 + 통계 보정) → `ROADMAP_EXTENSION.md` §D, 전 조건 동률~열세.

## 사내(Notion) 기록
parent page: `35e20ebf59c4807c8451e6baa7054c90` ("멀티미디어학회 (KCI)") — 모든 세션 실험
결과가 순서대로 기록됨. 이 repo 문서는 "왜/구조"를, Notion은 "시간순 실험 로그"를 담당.
