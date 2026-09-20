# Research Specs — 검증 가능한 가설과 현재 상태

하네스의 "product-specs"에 대응. 여기서는 기능 스펙 대신 **검증/반증 가능한 연구 가설**과
그 근거 상태를 추적한다.

| ID | 가설 | 상태 | 근거 |
|---|---|---|---|
| H1 | mask decomposition(z_inv)은 raw pretrained feature 대비 class 신호를 유지하며 domain 정보를 억제한다 | ✅ 지지 | fine15 z_inv AUROC 0.9239 > base 0.8957; coarse5도 raw vs z_inv 정량 비교로 확인 |
| H2 | 학습 가능한 sigmoid gate가 gradient-threshold mask보다 안정적이다 | ✅ 지지(GRL on/off 둘 다 재확인) | GRL-off: gradient-threshold는 fold별 극단 bimodal 붕괴. **GRL-on(공정 비교, 2026-08-17)**: 둘 다 학습되지만 gradient-threshold는 fold 800 4-shot에서 5개 중 2개 eval-seed가 threshold 완전 붕괴, gate는 붕괴 없음 + AUROC도 근소 우위(0.9540 vs 0.9444) — `docs/design-docs/gate-vs-gradient-threshold.md`, `RELIABILITY.md` §9 |
| H2b | 도메인 불변성(z_inv)이 raw feature보다 나으려면 encoder가 GRL(domain-adversarial) 학습을 받아야 한다 | ✅ 지지 | GRL-off 계열: raw가 z_inv를 이김(0.9522 vs 0.9426). GRL-on 계열: z_inv가 raw를 이김(0.9486 vs 0.9338) — `docs/design-docs/gate-vs-gradient-threshold.md` |
| H3 | Outer race(O) 결함이 fold 600/800 저성능의 지배적 원인이다(도메인 전체가 어려운 게 아니라) | ✅ 지지 | z_inv Mahalanobis AUROC: O 0.61~0.70 vs I/B/Compound 0.84~1.00 (두 도메인에서 재현) |
| H4 | 4-shot 프로토타입을 calib 전역 centroid 쪽으로 축소추정(shrinkage)하면(`proto_beta`) 학습 없는 domain-centering보다 낫다 | ✅ 지지(현 세션 기준) | β=0.5, 4-fold avg AUROC 0.9151(단일seed) > centering 0.9114 |
| H5 | domain_gate를 adversarial(weak-GRL)로 학습시키면 detach+직접분류보다 domain-invariance가 강해진다 | ❌ 기각(구현 결함) | weight 무관하게 domain_gate가 전부 0.0으로 collapse — 가설 자체가 검증 불가한 상태로 실패. `RELIABILITY.md` §7, `docs/exec-plans/completed/2026-08-domain-gate-weak-grl-sweep.md` |
| H6 | envelope(Hilbert) 변환한 O-fault 신호로 재학습하면 O-fault 탐지가 개선된다 | ⏳ 미검증(부분 근거만) | envelope 자체의 kurtosis는 3~9배 개선 확인, 그러나 raw-학습 모델에 그대로 넣었을 때 전이는 불확실 — **재학습해야 검증 가능**, 미실행 |
| H7 | domain-centering과 proto_beta를 동시 적용하면 각각보다 낫다 | ⏳ 미검증 | 두 기법을 상호 배타적으로만 테스트함 — `docs/design-docs/domain-centering-vs-proto-beta.md` |

## 미시도 개선 후보 (2026-08-16 시점)

- prototype 가중치(4-shot 샘플 균등 평균이 아닌 거리 기반 가중 평균)
- gate 구조 변경(GateNet hidden dim, layer4 기반 gate)
- threshold n_sigma 재튜닝(현재 2.0 고정, calib 도메인별 분산이 다를 가능성)
- H6(envelope 재학습), H7(centering+proto_beta 결합)
- fine15 트랙에 gated-mask 구조 이식(현재 gated는 coarse5 전용)
