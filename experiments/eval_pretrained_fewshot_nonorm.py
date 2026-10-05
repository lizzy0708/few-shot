"""
eval_pretrained_fewshot_nonorm.py — 2026-10-05. Duplicate of experiments/eval_pretrained_fewshot.py
(variant (a): current transform, NO ImageNet Normalize), created alongside
eval_pretrained_fewshot_imagenet_norm.py (variant (b): + ImageNet mean/std Normalize)
for a paired, side-by-side fairness comparison of the pretrained-ResNet50 baseline.
Does NOT modify any existing script -- new file only, no retraining.

This variant is functionally IDENTICAL to eval_pretrained_fewshot.py (same transform:
datasets/hust_image.py's default Resize((224,224))+ToTensor(), no Normalize). Since
the encoder is a fixed, never-trained ImageNet ResNet50 run in eval() mode (fully
deterministic, cudnn.deterministic=True set at eval_all_folds.py import time), its
output on an unchanged input pipeline is guaranteed byte-identical to a prior run --
so this script's results are NOT independently re-executed; they are the same
already-recorded numbers as eval_pretrained_fewshot.py's run
(docs/generated/pretrained_baseline/eval.txt, Avg AUROC=0.8615 Acc=0.7694 F1=0.8473).
This file exists as the requested explicit "(a)" artifact for direct comparison
against variant (b)'s own file, not to recompute what cannot change.

Usage (would reproduce the existing cached numbers exactly, not re-run by this audit):
  conda run -n torch python experiments/eval_pretrained_fewshot_nonorm.py --beta 0.5 --n_sigma 2.0
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

import experiments.eval_gated_folds as base

device = base.device


class PretrainedRawEncoder(nn.Module):
    """ImageNet-pretrained ResNet50 truncated at layer3 (1024-dim), GAP-pooled.
    NEVER trained/fine-tuned on HUST data. No Normalize applied to inputs (uses
    base.transform as-is -- Resize((224,224)) + ToTensor() only, see
    experiments/eval_all_folds.py:74-77)."""
    def __init__(self):
        super().__init__()
        backbone = resnet50(pretrained=True)
        self.encoder = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3,
        )

    def forward(self, x, alpha=1.0):
        z = self.encoder(x)
        z_pool = F.adaptive_avg_pool2d(z, 1).flatten(1)
        return {"z": z, "z_pool": z_pool, "z_c_notd": z, "z_inv": z,
                "domain_logits_disc": None}


def load_pretrained_model(_path):
    model = PretrainedRawEncoder().to(device)
    model.eval()
    return model


if __name__ == "__main__":
    base.load_gated_model = load_pretrained_model
    base.TRAIN_SEEDS = [0]
    print("[eval_pretrained_fewshot_nonorm] variant (a): current transform, NO Normalize. "
          "Deterministic -- identical to the already-recorded eval_pretrained_fewshot.py run.")
    base.main()
