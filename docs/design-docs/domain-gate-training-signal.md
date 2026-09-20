# domain_gate를 어떤 신호로 학습시킬 것인가

## (A) detach + 직접분류 (default, 2026-08 초 도입)

```python
domain_logits_disc = domain_classifier_disc(z_pool.detach() * domain_gate)
domain_disc_loss = CE(domain_logits_disc, true_domain)
```

`z_pool`을 detach하므로 이 branch의 gradient는 오직 `domain_gate_net`(과
`domain_classifier_disc`)에만 흐른다. loss는 "domain_gate로 걸러진 채널들로 도메인을
**잘 맞히게**" 학습시킨다 — 즉 domain_gate는 "어떤 채널이 도메인 정보를 담고 있는가"를
찾아내는 **탐지기**로 학습된다. 이 자체는 domain-invariance를 목표로 하지 않는다;
domain-invariance는 뒤이은 `z_inv = z*mc*(1-md)` 공식에서 "탐지된 채널을 빼버리는" 방식으로
간접적으로 달성된다.

**왜 detach가 필요한가**: detach 없이 (일반) adversarial(GRL) 신호를 이 branch에 직접
연결하면 domain_gate_net이 "모든 채널을 0으로 만들어 버리면 domain_classifier_disc가
아무것도 못 맞힌다"는 퇴화해(gate collapse to zero)를 학습해버리는 위험이 있다 —
class_gate 쪽에도 동일한 대칭 구조(직접분류, non-adversarial)를 적용해 이 위험을 피함.

## (B) weak-GRL (2026-08-16 도입, 스윕 중)

```python
masked = z_pool.detach() * domain_gate     # encoder는 여전히 detach로 보호
domain_logits_disc = domain_classifier_disc(grad_reverse(masked, alpha))
```

(A)와 마스킹 위치는 동일하지만 `domain_classifier_disc` 앞에 GRL을 추가로 삽입.
Forward는 동일(GRL은 순전파에서 identity), backward에서만 `domain_gate_net`이 받는
gradient가 반전된다 — 즉 domain_gate_net은 이제 "이 게이트를 곱하면 도메인 분류가
**어려워지도록**" 직접 adversarial하게 학습된다. (A)의 "탐지 후 별도 공식에서 억제"라는
2단계 구조 대신, gate 자체의 목적함수를 domain-invariance로 직접 맞추는 1단계 구조.

**collapse 위험 재검토**: (A)에서 detach를 도입한 이유였던 "전부 0으로 만드는 퇴화해"
위험이 (B)에도 동일하게 존재할 수 있음 — 다만 encoder는 여전히 detach로 보호되어 있어
붕괴가 domain_gate_net/domain_classifier_disc 두 작은 모듈 안에서만 일어나고 encoder
자체를 오염시키지는 않는다. 이 위험을 **작은 loss weight**(`domain_disc_weight ∈
{0.05, 0.1, 0.2, 0.3}`, "weak" GRL)로 억제할 수 있는지가 이번 스윕의 핵심 질문.

선택 절차와 결과: `docs/exec-plans/active/2026-08-domain-gate-weak-grl-sweep.md`.
