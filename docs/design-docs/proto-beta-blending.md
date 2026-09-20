# proto_beta: 미구현 플래그 발견과 구현

## 발견

`eval_all_folds.py`의 `--proto_beta` 인자는 argparse에 존재하고 `run_folds(...)` 호출에
전달까지 되지만, 함수 본문 어디에서도 사용되지 않는 죽은 플래그였다 —
`calib_centroid`라는 변수 자체가 파일에 없었다. 값을 바꿔도 결과가 전혀 변하지 않아
발견됨.

## 구현

```python
blended_prototype = beta * test_proto + (1 - beta) * calib_centroid
```
- `test_proto`: test 도메인 자신의 4-shot support 평균 (uncentered).
- `calib_centroid`: calib 도메인들의 (uncentered) 전체 정상 z_inv 평균.
- `beta=1.0`은 정확히 "기존 no-centering baseline"과 동일 — 구현 후 이 회귀 테스트로 정합성 확인.
- 적용 대상은 **no-centering** n_sigma 프로토콜(domain-centering과는 별개 기법, 병행 사용 안 함).

## 결과

β=0.5/0.7이 domain-centering(당시 최고)을 앞섬 → 이후 3-seed 학습 앙상블과 결합해
세션 최고 기록(AUROC 0.9426)의 기반이 됨. fold별 최적 β는 다르게 나왔으나(500/700은
β=1.0 선호, 600/800은 β=0.5 선호) 최종 채택은 **β=0.5 전체 고정**(fold별 최적화는
held-out 성능을 보고 고르는 것이므로 leakage 위험 — `docs/design-docs/threshold-protocol.md`
원칙과 동일한 이유로 배제).
