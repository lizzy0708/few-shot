# Design Docs — 의사결정 기록

"왜 이렇게 했는가"만 남긴다. "무엇을 했는가"는 `ARCHITECTURE.md`, 수치는
`docs/generated/results.md`를 본다.

| 문서 | 결정 |
|---|---|
| [gated-mask-architecture.md](gated-mask-architecture.md) | gradient-threshold mc/md → 학습 가능 sigmoid gate로 전환 |
| [domain-gate-training-signal.md](domain-gate-training-signal.md) | domain_gate를 detach+직접분류로 학습시킨 이유, weak-GRL 대안 도입 배경 |
| [proto-beta-blending.md](proto-beta-blending.md) | `--proto_beta` 미구현 플래그 발견 및 직접 구현 |
| [threshold-protocol.md](threshold-protocol.md) | Youden(label-leak) vs n_sigma(leakage-free) 프로토콜 선택 |
| [domain-centering-vs-proto-beta.md](domain-centering-vs-proto-beta.md) | training-free domain-centering이 왜/언제 proto_beta보다 못했는가 |
| [gate-vs-gradient-threshold.md](gate-vs-gradient-threshold.md) | gate가 레거시 gradient-threshold(mc/md)보다 나은가 — 직접 대조 결론(gate 승, 안정성 이유) |
