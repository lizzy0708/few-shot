"""
eval_pretrained_fewshot.py — 2026-09-28: untrained-encoder baseline for Table 3's
"ResNet50" row candidate (see docs/generated/table2-pipeline-audit findings).

Does NOT modify experiments/eval_gated_folds.py, experiments/eval_all_folds.py, or
any other existing script. New file only.

Uses a fixed, NEVER trained/fine-tuned ImageNet-pretrained ResNet50 truncated at
layer3 + GAP (1024-dim) as the "encoder" -- exactly matching GatedMaskModel's
FeatureExtractor architecture/output dim (models/original_mask_model.py:33-46), but
with its ImageNet weights left completely untouched (no HUST training at all, no
class_gate, no domain classifier). Everything else in the pipeline is byte-for-byte
identical to Table 2's actual protocol, achieved by importing and reusing
experiments.eval_gated_folds's run_fold() / build_support_query() / build_calib_normal()
/ score_one_model() / GATED_FOLDS / ROOT / ALL_DOMAINS / SHOT unchanged, and
monkey-patching only the module-level `load_gated_model` function (never touching
the file itself) plus `TRAIN_SEEDS` (a fixed untrained encoder has no seed axis --
ImageNet weights are exactly reproducible with zero randomness at eval() time, so a
3-seed "ensemble" of this model would just average 3 byte-identical score arrays;
using train_seeds=[0] is the efficient, semantically correct choice, not a
protocol deviation).

Same beta=0.5 blend, same LedoitWolf full-covariance fit on calib-domain NORMAL-ONLY
features (4 domains, ~5988 samples per fold), same n_sigma=2.0 threshold, same
exclude_paths leakage fix, same 4-fold x 5-eval-seed loop as eval_gated_folds.py.

Usage:
  conda run -n torch python experiments/eval_pretrained_fewshot.py --beta 0.5 --n_sigma 2.0
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
    NEVER trained/fine-tuned on HUST data -- weights are exactly torchvision's
    ImageNet checkpoint (resnet50(pretrained=True)), frozen at eval() throughout,
    identical construction to models/original_mask_model.py:33-46's FeatureExtractor
    (encoder_layer='layer3') minus the HUST training. No gate, no domain classifier:
    returns the same raw pooled feature under both "z" and "z_c_notd" keys so it is a
    drop-in replacement for GatedMaskModel's output dict, consumed unchanged by
    experiments.eval_all_folds.get_features(feature_key=...).
    """
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
    """Signature-compatible replacement for base.load_gated_model -- ignores the
    checkpoint path entirely (there is no checkpoint; the encoder is fixed ImageNet
    weights) and always returns a fresh, deterministic, eval()-mode encoder."""
    model = PretrainedRawEncoder().to(device)
    model.eval()
    return model


if __name__ == "__main__":
    # Fixed pretrained encoder has no train-seed axis (see module docstring) --
    # a single "pseudo-seed" avoids wastefully re-running byte-identical forward
    # passes 3x for zero ensemble benefit.
    base.load_gated_model = load_pretrained_model
    base.TRAIN_SEEDS = [0]
    print("[eval_pretrained_fewshot] Using fixed ImageNet-pretrained ResNet50 "
          "(layer3, GAP, 1024-dim) -- NOT trained on HUST data. train_seeds=[0] "
          "(no ensemble axis for an untrained, deterministic encoder).")
    base.main()
