# Threshold 프로토콜: Youden vs n_sigma, 그리고 "held-out을 보고 고르지 않는다"는 원칙

## 두 프로토콜

| 프로토콜 | 공식 | 문제 |
|---|---|---|
| Youden's J (트랙 1, coarse5 legacy) | calib 도메인의 정상+**이상** 라벨로 ROC에서 TPR-FPR 최대 threshold 탐색 | calib에 이상 라벨이 있어야 함 — held-out 도메인과 동일 분포의 라벨 정보를 미리 사용하는 label leakage |
| n_sigma (트랙 2/3) | `support_mean + n_sigma * calib_std`, 정상 샘플만 사용 | leakage-free, 실제 배포 시나리오(정상 4개만 있음)와 일치 |

fine15 확정 결과는 n_sigma=0.0(σ 항 없이 support_mean만), coarse5 gated 트랙은
n_sigma=2.0을 표준으로 쓴다. **두 트랙 수치를 직접 비교하지 않는다** (`ARCHITECTURE.md`).

## "선택은 held-out을 보기 전에 확정한다" 원칙

proto_beta, domain_gate 억제 강도(α), 이번 domain_gate weak-GRL weight 등 모든
하이퍼파라미터 스윕에서 반복 적용한 원칙:

1. **calib 도메인 내부**(또는 fold별 최적값이 아닌 전체 통일값)에서 판단 기준을 정하고 확정.
2. 확정된 값으로 held-out test 도메인 평가를 **한 번만** 수행.
3. held-out 성능을 보고 값을 역선택하지 않는다 — 그렇게 하면 held-out이 사실상 검증셋이 아니라
   튜닝셋이 되어 leakage와 동치.

proto_beta는 예외적으로 fold별 최적값이 갈렸음을 관찰했지만(500/700 vs 600/800), 이
원칙에 따라 fold별 최적화 대신 β=0.5 전체 고정을 채택했다.
