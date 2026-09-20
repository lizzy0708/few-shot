# PLANS.md — 로드맵

세부 완료/진행 기록은 `docs/exec-plans/{active,completed}/`. 이 문서는 트랙 단위의
큰 그림만 담는다.

## 트랙 A: fine15 (논문 본편) — 종료, 확장 탐색도 종료

fine15 z_inv 확정(AUROC 0.9239)이 이 설정의 천장으로 판정됨(2026-07-10). Patch memory /
Hierarchical domain / DC calibration 4갈래 확장 전부 미초과. 상세: `ROADMAP_EXTENSION.md`.
남은 작업은 논문 집필뿐(본편=fine15, 비교=coarse5+baseline, 분석=md 채널 분해·patch vs
GAP·negative results).

## 트랙 B: coarse5 gated-model — 활성 개발 중

`docs/exec-plans/completed/2026-08-coarse5-gated-model-pipeline.md`에서 세션 최고
AUROC 0.9426 확립. 현재 진행: domain_gate weak-GRL 스윕
(`docs/exec-plans/active/2026-08-domain-gate-weak-grl-sweep.md`).

다음 후보 (우선순위 미정, `docs/research-specs/index.md` 참고):
1. domain_gate weak-GRL 스윕 마무리 → 채택 여부 결정
2. H6: envelope 변환 신호로 O-fault 재학습
3. H7: domain-centering + proto_beta 결합
4. fine15 트랙에 gated-mask 구조 이식 (트랙 A가 이미 천장을 찍었다고 판정된 상태라
   우선순위 낮음 — 재검토 필요)

## 원칙 (변경 없음, `ROADMAP_EXTENSION.md`에서 계승)

- 게이트 미통과 방향은 즉시 중단, ablation/분석 소재로 전환.
- 확정 체크포인트(`fine15_fold*.pth`, `coarse5_fold*_best.pth`)는 수정 금지.
- threshold 프로토콜은 트랙별로 고정, 섞지 않음.
- 실험은 원칙적으로 순차 실행.
