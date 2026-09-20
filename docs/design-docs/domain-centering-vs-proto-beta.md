# domain-centering이 왜 proto_beta보다 못했는가

## domain-centering

학습 없이 각 도메인 자신의 4-shot support 평균을 그 도메인의 모든 정상 샘플에서 빼는
기법(calib fit 전, test 도메인 query/support 모두). 대칭적으로 적용되어 leakage 없음을
별도로 재확인함(calib centering mean도 4-shot만 사용, 전체 정상셋 사용 아님).

raw feature(학습 없는 pretrained z)에서도, 학습된 z_inv에서도 유의미한 개선을 줌 — 특히
fold 800처럼 도메인 자체가 어려운 경우.

## proto_beta 블렌딩이 더 나은 이유(관찰)

domain-centering은 "각 도메인의 분포를 원점으로 이동시켜 도메인 간 scale/offset 차이를
지운다"는 강한 가정을 쓴다. proto_beta는 대신 "test 도메인 고유의 4-shot 추정치가
노이즈가 많으면, calib 도메인들의 안정적인 전역 centroid 쪽으로 살짝 당긴다"는 더 약한
가정을 쓴다 — 통계적으로는 축소추정(shrinkage estimator)에 가깝다.

fold별 패턴이 이 해석과 일치: 어려운 fold(600/800, 4-shot 추정이 노이즈에 취약)일수록
낮은 β(전역 centroid 쪽으로 더 당김)를 선호했고, 쉬운 fold(500/700)는 β=1.0(순수
test_proto)을 선호했다 — "노이즈가 클수록 축소추정 효과가 큼"이라는 통계적 직관과 부합.

## 결론

두 기법은 상호 배타적으로 취급했다(동시 적용 시도 안 함) — domain-centering은 이미
"도메인 원점 정렬"을 하고 있어, 그 위에 proto_beta의 "calib centroid로 당기기"를
얹으면 어떤 효과가 나는지는 미탐구 (미시도 후보 목록 참고, `docs/research-specs/index.md`).
