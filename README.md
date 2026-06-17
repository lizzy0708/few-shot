# Few-shot HUST Bearing Anomaly Detection

This project studies few-shot anomaly detection on the HUST bearing dataset using domain-invariant normal features and Mahalanobis scoring.

## Setup

```bash
pip install -r requirements.txt
```

The repository tracks source code only. Large files are intentionally excluded:

- `HUST bearing dataset/`
- `processed/`
- `checkpoints/`
- `results/`
- `*.pth`

Place the dataset, processed images, and checkpoints in the same relative paths on each machine before running experiments.

## Preprocess

```bash
python processed/make_gadf.py
```

## Train Decomposition Model

Example leave-one-domain-out training:

```bash
python experiments/train_mask_decomposition.py \
  --train_domains 400 500 600 700 \
  --save_path mask_decomposition.pth
```

## Calibrated Few-shot Evaluation

Example target domain `800`:

```bash
python experiments/run_fewshot_calibrated_threshold.py \
  --support_domain 800 \
  --calib_domain 400,500,600,700 \
  --query_domain 800 \
  --shot 4 \
  --ckpt mask_decomposition.pth \
  --seed 0 \
  --feature zinv \
  --score_method mahalanobis \
  --cov_mode diag \
  --cov_source support_calib_normal \
  --select_metric youden
```

The decision threshold is selected on calibration domains by maximizing Youden's J statistic, then applied to the target query domain.
