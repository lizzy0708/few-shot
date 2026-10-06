# GRL-off (domain_weight=0) ablation on gated_nodg, seed 0 (2026-10-06)

## 목적
Table 2(GatedMaskModel, nodg, domain_weight=1.0, GRL-on)의 z_inv Acc/F1 이득이 GRL
자체(domain-adversarial 학습)에 의존하는지, 단일 seed로 빠르게 재확인. 이미 2026-08-17에
GRL-off(`coarse5_fold*_gated_noGRL_s{0,1,2}.pth`, `use_domain_gate=True` 구조, 단일seed
AUROC 0.9338 z_inv 대 raw 0.9486 비교)가 있었으나, **구조(use_domain_gate=True, mask_loss
실제 작동)가 현재 nodg 트랙과 다름** — 이번엔 nodg 구조(`--no_domain_gate`)를 그대로
유지한 채 `--domain_weight 0`만 바꿔 순수하게 GRL 유무만 격리.

## 코드 변경
**없음** — `experiments/train_gated_mask_model.py`는 `--domain_weight`를 이미 CLI 인자로
지원함(기본값 1.0, `--domain_weight 0`으로 바로 끌 수 있음). 코드 수정이 필요 없어 커밋할
변경사항 없음(이 exec-plan 파일 자체만 커밋).

## 학습 커맨드 (fold별, calib 도메인은 held-out 제외 4개, seed=0 고정)
```bash
conda run -n torch python experiments/train_gated_mask_model.py \
  --root processed --train_domains <calib 4개> --all_domains 400 500 600 700 800 \
  --epochs 10 --batch_size 16 --lr 1e-4 --num_classes 2 \
  --mask_weight 0.1 --domain_weight 0.0 --domain_disc_weight 1.0 \
  --no_domain_gate --seed 0 \
  --save_path checkpoints/coarse5_fold<fold>_gated_nodg_nogrl_s0.pth
```
fold=500: calib=400,600,700,800 / fold=600: calib=400,500,700,800 /
fold=700: calib=400,500,600,800 / fold=800: calib=400,500,600,700

## 체크포인트
`checkpoints/coarse5_fold{500,600,700,800}_gated_nodg_nogrl_s0.pth` — 기존
`coarse5_fold*_gated_nodg_s{0,1,2}.pth`(GRL-on)나 `coarse5_fold*_gated_noGRL_s{0,1,2}.pth`
(2026-08-16, use_domain_gate=True 구조)와 이름이 달라 덮어쓰지 않음.

## 평가 (재사용, 새 eval 스크립트 최소화)
- z_inv: 기존 `experiments/eval_gated_folds_mmd.py --mmd_tag nogrl --train_seeds 0`
  (이름은 "mmd"이지만 순수 checkpoint-path-tag 스위치라 그대로 재사용 가능 —
  `checkpoints/coarse5_fold*_gated_nodg_nogrl_s0.pth` 패턴과 정확히 일치)
- raw z: 새 파일 `experiments/eval_gated_folds_rawz_tagged.py`(신규, `eval_gated_folds_rawz.py`의
  `score_one_model` monkey-patch와 `eval_gated_folds_mmd.py`의 `--ckpt_tag` 패턴을 결합 —
  기존 `eval_gated_folds_rawz.py`는 태그를 지원하지 않아 그대로는 재사용 불가하므로 최소 래퍼 추가)
  `--ckpt_tag nogrl --train_seeds 0`

## 결과 (2026-10-06, 학습·평가 완료)

| | AUROC | Acc | F1 |
|---|---|---|---|
| GRL-on seed0 z_inv | 0.9486 | 0.8846 | 0.9278 |
| GRL-on seed0 raw z | 0.9337 | 0.8508 | 0.9055 |
| **GRL-off seed0 z_inv** | **0.9438** | **0.9089** | **0.9457** |
| **GRL-off seed0 raw z** | **0.9503** | **0.8830** | **0.9265** |

### fold별
| Fold | GRL-off z_inv (AUROC/Acc/F1) | GRL-off raw z (AUROC/Acc/F1) |
|---|---|---|
| 500 | 0.9880/0.9645/0.9794 | 0.9869/0.9552/0.9736 |
| 600 | 0.9320/0.8905/0.9342 | 0.9384/0.8662/0.9170 |
| 700 | 0.9753/0.9277/0.9578 | 0.9630/0.9127/0.9484 |
| 800 | 0.8797/0.8529/0.9114 | 0.9129/0.7981/0.8670 |

### 해석
- **GRL-off z_inv vs GRL-on z_inv**: AUROC는 GRL-off가 소폭 낮음(−0.48pp)이지만 **Acc(+2.43pp)/F1(+1.79pp)는 GRL-off가 오히려 더 높음** — 단일 seed라 노이즈 가능성 있으나, 적어도 "GRL이 Acc/F1에 필수적"이라는 가정과는 반대 방향.
- **GRL-off raw z vs GRL-on raw z**: 세 지표 모두 **GRL-off가 더 높음**(AUROC+1.66pp, Acc+3.22pp, F1+2.10pp) — encoder 자체도 GRL 없이 더 잘 나온 단일-seed 결과.
- **GRL-off 안에서 z_inv vs raw z**: AUROC는 raw z가 근소 우위(+0.65pp)지만 **Acc(+2.59pp)/F1(+1.92pp)는 z_inv가 우위** — GRL을 완전히 꺼도 decomposition(gate)의 Acc/F1 기여는 그대로 남음. 이는 2026-09-25 "도메인 라벨 shuffle" 대조실험 결론(Acc/F1 개선이 거의 전적으로 class_gate 구조 자체의 재가중 효과이며 GRL/도메인 라벨 진위와 무관)과 **독립적인 방법(label shuffle vs domain_weight=0)으로 같은 결론에 도달** — 교차검증됨.
- 전부 **단일 seed(seed=0)** 결과이므로 해석에 주의 — fold 600/800은 이미 알려진 고분산 fold라 단일 seed 결과만으로 일반화하기엔 약함(3-seed 확장은 이번 요청 범위 밖).

원본 로그: `docs/generated/nogrl_ablation/{driver,eval_zinv,eval_rawz}.txt`.
