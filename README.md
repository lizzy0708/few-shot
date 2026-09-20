# Few-shot HUST Bearing Anomaly Detection

도메인 불변 특징(z_inv) 기반 퓨샷 이상탐지. 미학습 베어링 타입 도메인에서 **정상 샘플 4개만으로** 베어링 고장을 탐지한다.

> **도메인 정의 (2026-08-13 정정)**: 파일 코드(예: `I402`)의 숫자는 RPM이 아니라 **[베어링 타입 자리][부하조건 자리]**다 — 첫 자리(4~8)는 베어링 모델 6204~6208, 둘째 자리(0/2/4)는 부하조건 0W/200W/400W ([HUST bearing dataset 원 논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC10327369/) 확인). 즉:
> - **coarse5** (400/500/600/700/800): 순수 **베어링 타입** 도메인 (부하조건은 섞음)
> - **fine15** (400/402/404/.../804): **베어링 타입 × 부하조건** 조합 도메인 — "5 RPM × 3 측정 배치"가 아니라 "5 베어링 타입 × 3 부하조건"

- 방법: gradient mask 분해로 class-관련·domain-불변 특징 추출 + Mahalanobis 거리
- 메인 설정 **fine15**: 15개 세분화 도메인 (5 베어링 타입 × 3 부하조건), 4-fold LOO
- 비교 설정 **coarse5**: 5개 베어링 타입 도메인 (원 논문 세팅)

## 결과 (fine15, 4-fold 평균)

| | AUROC | Acc | F1 |
|---|---|---|---|
| Baseline (z) | 0.8957 | 0.8459 | 0.9073 |
| **z_inv (ours)** | **0.9239** | **0.8683** | **0.9250** |

세부 수치와 설계 결정은 `CLAUDE.md`, 학습/추론 절차는 `TRAINING.md` / `INFERENCE.md`, coarse5 기록은 `METHOD_615.md` 참고.

## 문서 지도

에이전트 작업 규칙은 `AGENTS.md`, 모델 구조 단일 소스는 `ARCHITECTURE.md`, 로드맵은
`PLANS.md`. "왜 이렇게 했는가"는 `docs/design-docs/`, 진행 중/완료 실험은
`docs/exec-plans/{active,completed}/`, 검증 가능한 가설 목록은 `docs/research-specs/`,
확정 수치는 `docs/generated/results.md`, 외부 근거는 `docs/references/`, 재현성
이슈 모음은 `RELIABILITY.md`, 결과 신뢰도 체크리스트는 `QUALITY_SCORE.md`.

## Setup

```bash
conda activate torch
pip install -r requirements.txt
```

이 저장소는 소스코드와 함께 GADF 변환 데이터(`processed_gadf_fine_4096/`)를 포함한다.
제외 대상 (로컬 준비 필요): 원본 `HUST bearing dataset/`, `checkpoints/`(*.pth), `results/`, `processed_gadf_coarse_4096/`(→ `experiments/make_coarse_dataset.py`로 생성).

## 주요 파일

| 경로 | 역할 |
|---|---|
| `models/mask_decomposition_model.py` | 메인 모델 (ResNet50 layer4 + gradient mask 분해) |
| `experiments/train_mask_decomposition.py` | fine15 학습 |
| `experiments/eval_all_folds.py` | 4-fold 평가 (fine/coarse 모드) |
| `experiments/eval_coarse_folds.py` | coarse5 전용 평가 (sub-batch 균등 support) |
| `experiments/train_original_mask.py` 등 | coarse5 재현·ablation 변형들 |
| `checkpoints/fine15_fold{500..800}.pth` | best 체크포인트 (git 미포함) |

## 학습 / 평가

```bash
# 학습 (fold_500 예시)
python experiments/train_mask_decomposition.py \
  --root processed_gadf_fine_4096 \
  --train_domains 400 402 404 600 602 604 700 702 704 800 802 804 \
  --all_domains 400 402 404 500 502 504 600 602 604 700 702 704 800 802 804 \
  --epochs 20 --num_classes 2 --warmup_epochs 3 \
  --mask_weight 0.1 --domain_weight 1.0 \
  --supcon_weight 0.5 --proto_weight 0.1 --episodic_weight 1.0 \
  --seed 42 --save_path checkpoints/fine15_fold500.pth

# 평가 (전체 4 fold)
python experiments/eval_all_folds.py \
  --root processed_gadf_fine_4096 --mode fine --num_classes 2 \
  --ckpt_fold_500 checkpoints/fine15_fold500.pth \
  --ckpt_fold_600 checkpoints/fine15_fold600.pth \
  --ckpt_fold_700 checkpoints/fine15_fold700.pth \
  --ckpt_fold_800 checkpoints/fine15_fold800.pth
```
