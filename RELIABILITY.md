# RELIABILITY.md — 재현성/불안정성 원인과 완화책

이 프로젝트에서 반복적으로 부딪힌 "결과가 흔들리는" 문제들과 근본 원인, 완화책을 모은다.
같은 삽질을 반복하지 않기 위한 문서.

## 1. fold 800(과 600)의 학습-재현성 문제

**증상**: 같은 설정(seed만 다름)으로 재학습하면 fold 800 성능이 다른 fold보다 훨씬 크게 흔들림.

**진단**: train-seed(0,1,2) × eval-seed(0~4) 분산을 분리 측정. 대부분의 fold는
eval-seed 분산(4-shot support 샘플링 노이즈)이 지배적이지만, fold 800은 train-seed
분산이 eval-seed 분산과 맞먹음 — 즉 4-shot 샘플링 문제가 아니라 **학습 자체의
재현성 문제**.

**완화책**: 3개 학습 seed의 이상 점수를 평균하는 **점수 앙상블**. 단일 seed 대비
fold 800 AUROC 0.7748±0.0431 → 앙상블 0.8478±0.0159(centering 적용 시) — 평균 개선과
분산 축소 동시 달성. 현재 4-fold 모두 3-seed 앙상블이 표준.

## 2. gradient-threshold mc/md의 구조적 붕괴 (domain_weight=0일 때)

**증상**: `domain_weight`(GRL)를 0으로 낮추면 fold별 극단적 bimodal(500/700 양호,
600/800 붕괴).

**진단**: `mask_weight`를 0으로 바꿔도 동일 패턴 재현 → mask orthogonality loss는
원인이 아님. 근본 원인은 `z_inv = z⊙mc⊙(1-md)`에서 `md`가 미학습 domain_classifier의
gradient에서 나오기 때문(`domain_weight=0`이면 domain_classifier 자체가 학습 안 됨 →
md가 random-init 상태로 z_inv를 무조건 오염).

**완화책**: `GatedMaskModel`로 전환, domain_gate_net에 독립적 학습 신호 부여
(`docs/design-docs/gated-mask-architecture.md`). 구조적 원인을 해결한 것이므로
mask_weight 재튜닝 같은 우회책보다 근본적.

## 3. Envelope-GADF 재평가 시 도메인 집계 버그

**증상**: envelope-GADF로 재생성한 raw-GADF baseline AUROC가 기존 확정 수치와
불일치(600: 0.8185 vs 기준 0.6980, 800: 0.6468 vs 기준 0.6089).

**원인**: coarse 도메인 "800"은 실제로 3개 부하조건 sub-file(O800+O802+O804.mat)을
합친 것인데, 최초 생성 스크립트가 주 파일(O800.mat) 하나만 처리해 샘플 수가
499개(정답 1497개)로 나왔음.

**완화책 겸 교훈**: 재현 검증 시 **알려진 기준 수치와 반드시 대조**한다 — 불일치가
발견된 것 자체가 버그를 잡은 신호였음. `make_gadf.py`의 `get_domain()`
(`number[:1]+'00'`)이 실제 집계 규칙의 단일 소스.

## 4. Envelope kurtosis band 탐색 방법론 오류 (2회 반복)

**증상**: 밴드 선택 기준을 바꿀 때마다 "envelope가 원본보다 나아졌다"는 결론이
바뀜(에너지 집중 기준 → 약한 개선; 전체-신호 kurtosis 최대화 기준 → 오히려 악화).

**원인**: 평가 시점의 실제 지표(1024-sample 윈도우 단위 평균 kurtosis)와 다른 지표로
밴드를 선택했기 때문. 전체-신호 kurtosis 최대화는 희귀한 outlier transient에
좌우되어, 윈도우 단위로 보면 오히려 나쁜 밴드를 고름.

**교훈**: 밴드/하이퍼파라미터 선택 기준은 **최종 평가에 쓰이는 것과 동일한 지표**로
해야 한다. 대리 지표(proxy metric)로 최적화하면 실제 목적함수에서 역효과가 날 수 있음.

## 5. GPU 동시 실행 시 메모리 여유 부족

**증상**: 학습 job과 평가 job을 동시에 백그라운드 실행했을 때 GPU 메모리가
11.6GB/12GB까지 참(2026-08-16).

**완화책**: 원칙적으로 순차 실행(`ROADMAP_EXTENSION.md`의 기존 원칙과 일치). 부득이
동시 실행 시 `nvidia-smi`로 여유 메모리를 먼저 확인.

## 7. sigmoid gate에 직접 GRL을 걸면 saturation collapse로 죽는다

**증상**: `domain_gate_net`(Sigmoid 출력)의 학습 신호를 detach+직접분류 대신 GRL(adversarial)로
바꾸자, weight를 0.05~0.3(6배 차이)로 스윕해도 결과가 완전히 동일 — `domain_gate`가 4개 weight
전부에서 정확히 0.0(mean=std=min=max=0)으로 collapse. `docs/exec-plans/completed/2026-08-domain-gate-weak-grl-sweep.md`.

**원인**: Sigmoid는 입력이 커질수록 gradient가 0에 수렴하는 saturating 함수. adversarial
압력이 게이트를 0 쪽으로 밀기 시작하면, 0 근처에서 gradient가 사라져 더 못 빠져나오는 흡수
상태(absorbing state)가 된다. loss weight는 "얼마나 세게 미는가"만 조절할 뿐 "그 지점에서
gradient가 사라진다"는 정성적 문제 자체는 못 건드리므로, weight를 낮춰도 collapse가 사라지지
않는다 — 이번처럼 weight 무관하게 100% 재현되는 패턴이 나오면 loss weight 튜닝이 아니라
**gate 구조 자체**(saturation 여부)를 의심해야 한다.

**진단 팁**: 학습 로그에서 서로 다른 weight/seed 조합의 정확도·loss가 소수점까지 완전히
동일하게 수렴하면(우연이 아니라 매 조합 반복됨) 최적화가 실제로 진행 중인 게 아니라 동일한
퇴화해(degenerate solution)에 흡수된 것을 의심할 것. `mean/std/min/max`를 직접 찍어 확인하는
게 F1 같은 요약 지표만 보는 것보다 훨씬 빠르게 잡아낸다(F1은 여전히 그럴듯한 숫자를 보여줄 수
있음 — 이번 경우 F1=0.1081이 "domain 정보가 잘 지워졌다"는 낙관적 해석을 유도할 뻔함).

**완화책**: 재시도한다면 gate 출력에 하한을 강제하거나(예: `0.05 + 0.9*sigmoid(...)`),
adversarial loss 대신 entropy/L2 정규화로 collapse를 억제하는 방향. 미시도.

## 9. gradient-threshold(레거시 mc/md)는 4-shot에서 threshold가 통째로 붕괴할 수 있다

**증상**: `OriginalMaskModel`(gradient-threshold mc/md, GRL-on) 3-seed 앙상블을 fold 800에서
평가했더니, 5개 eval-seed(4-shot support 재추출) 중 2개에서 Acc=0.155, F1=0.000으로 완전히
무너짐 — 나머지 3개는 정상(Acc~0.88). AUROC은 5개 전부 0.87~0.88대로 정상이라 순위(ranking)는
멀쩡한데 threshold만 완전히 잘못 잡힘.

**1차 원인(수정됨)**: 평가 스크립트가 calib/support 추출 시 `class_label=None`(max-logit)을
썼다 — `CLAUDE.md`의 기존 설계 결정("GT label로 mc 계산, max logit 사용 시 오분류 샘플
불안정")을 재확인 없이 어긴 것. calib/support는 `only_normal=True`로 이미 라벨이 100%
확정(정상)이라 `class_label=0`을 명시해도 누수가 아닌데, 불필요하게 label-free 경로를 써서
불안정한 mc가 섞였다. 이 부분을 고치자 fold 500/600/700의 이상 현상은 전부 사라짐.

**2차 원인(진짜 모델 취약성, 미해결)**: 수정 후에도 fold 800만 5개 중 2개 eval-seed에서
동일한 붕괴가 남았다. gradient-threshold의 mc/md는 `torch.autograd.grad`로 **샘플별로**
계산되는 값이라, 4-shot(표본 4개)처럼 극소 표본에 이상치 gradient 하나가 섞이면 prototype과
threshold 전체가 쉽게 왜곡된다. 학습된 gate(`class_gate_net`, 매끄러운 고정 함수)는 같은
상황에서 이런 붕괴가 없었다 — `docs/design-docs/gate-vs-gradient-threshold.md`.

**교훈**: (a) legacy 모델을 새 프로토콜로 재평가할 때는 그 모델의 기존 설계 문서(왜 GT label이
필요한지 등)를 먼저 확인할 것 — 새로 짜는 평가 스크립트가 과거에 이미 발견된 함정을 다시
밟기 쉽다. (b) AUROC만 보고 "괜찮다"고 판단하면 안 된다 — threshold 의존 지표(Acc/F1)가
따로 붕괴할 수 있으므로 항상 같이 확인. (c) 극소 표본(4-shot) 상황에서는 "학습된 매끄러운
함수" 기반 방법이 "샘플별 gradient" 기반 방법보다 구조적으로 더 강건할 수 있다.

## 10. 하이퍼파라미터 선택과 held-out leakage

**원칙 위반 시 발생하는 문제**: proto_beta처럼 fold별 최적값이 다르게 나오는 경우,
held-out 성능을 보고 fold마다 다른 값을 고르면 held-out이 사실상 튜닝셋이 되어
보고된 수치가 낙관적으로 편향된다.

**완화책**: `docs/design-docs/threshold-protocol.md`의 "held-out을 보기 전에 확정"
원칙. 모든 스윕(α, β, 이번 domain_gate weight)에 일관 적용.
