# 왜 gradient-threshold mc/md를 학습 가능한 sigmoid gate로 바꿨는가

## 배경

트랙 1/2(`original_mask_model.py`, `mask_decomposition_model.py`)의 mc/md는
`ReLU(∂score/∂z)/max`로 계산되는 gradient 기반 마스크다. `create_graph=True`가 필요한
2-pass autograd이고, end-to-end로 "이 마스크가 더 나은 z_inv를 만들도록" 직접 최적화되지
않는다 — score를 잘 내도록 encoder가 학습되면 마스크는 그 부산물로만 따라온다.

## domain_weight=0 실험에서 드러난 문제

`domain_weight`(GRL 강도)를 0으로 낮추는 ablation 중, fold별로 극단적인 bimodal 결과가
나왔다(500/700 fold는 양호, 600/800 fold는 붕괴). `mask_weight`를 0.1→0으로 바꿔도
패턴이 동일 → mask orthogonality loss는 원인이 아님을 확인(순수 class-loss-only ablation으로
재확인).

**근본 원인**: `z_inv = z ⊙ mc ⊙ (1-md)` 구조에서 `md`는 domain_classifier의 gradient에서
나온다. `domain_weight=0`이면 domain_classifier가 아예 학습되지 않아 md가 random-init
상태로 z_inv를 무조건 오염시킨다 — mask_weight와 무관한 구조적 결함.

## 결정

md를 gradient 유도값이 아니라 **자체 학습 신호를 가진 독립 네트워크**(`domain_gate_net`,
`GateNet: MLP→Sigmoid`)로 교체한다. 이러면 `domain_weight=0`이어도 domain_gate_net은
자기 자신의 loss(`domain_disc_weight * domain_disc_loss`)로 학습되므로 random-init
오염 문제가 원천적으로 사라진다. class_gate도 동일한 방식(`class_gate_net`)으로 교체해
구조를 대칭으로 유지.

부수 효과: `mask_loss = mean(class_gate * domain_gate)`가 완전히 미분 가능해져
2-pass autograd 트릭이 불필요해짐.

## 검증

`use_domain_gate=False`(domain_gate 항 자체를 z_inv에서 제거, class_gate만 사용)가
한 시점의 세션 최고였다가, `domain_gate` 억제 강도를 `z_inv = z*class_gate*(1-α·domain_gate)`로
스윕(α=0.3이 최적)한 버전이 이를 다시 앞섬 — domain_gate 자체는 유용한 신호이나 억제
강도를 완화해야 함을 시사. 이 α=0.3 발견이 `docs/exec-plans/completed/`의 여러 실험을
관통하는 공통 설정값이 됨.
