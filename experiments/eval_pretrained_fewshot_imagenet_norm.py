"""
eval_pretrained_fewshot_imagenet_norm.py — 2026-10-05. Variant (b) of the pretrained-
ResNet50 baseline fairness check: same as eval_pretrained_fewshot_nonorm.py (variant
(a)) except the input transform additionally applies the standard ImageNet
Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) that an ImageNet-
pretrained ResNet50 is conventionally fed with and that the current HUST pipeline
does NOT apply anywhere (confirmed in the prior audit: datasets/hust_image.py:35-38
and experiments/eval_all_folds.py:74-77 both stop at ToTensor(), no Normalize step).

Does NOT modify any existing script -- new file only, no retraining. Reuses
experiments.eval_gated_folds's run_fold()/build_support_query()/build_calib_normal()/
score_one_model()/GATED_FOLDS unchanged. Two monkey-patches beyond the (a) variant's:
`load_gated_model` (fixed pretrained encoder, as in (a)) AND `base.transform` (the
module-global imported into eval_gated_folds.py's namespace from
experiments.eval_all_folds -- build_support_query/build_calib_normal reference this
name as a late-bound global, so patching the attribute on the imported module object
redirects it without touching the file, same pattern already used for ckpt_path/
load_gated_model in eval_gated_folds_mmd.py and eval_gated_folds_rawz.py).

Usage:
  conda run -n torch python experiments/eval_pretrained_fewshot_imagenet_norm.py --beta 0.5 --n_sigma 2.0
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet50

import experiments.eval_gated_folds as base

device = base.device

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PretrainedRawEncoder(nn.Module):
    """Identical architecture to eval_pretrained_fewshot_nonorm.py's -- the only
    difference between (a) and (b) is the input transform (patched at module level
    below), not this class."""
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
    # Add ImageNet Normalize on top of the existing Resize+ToTensor transform --
    # base.transform is the same object imported from experiments.eval_all_folds
    # (eval_gated_folds.py:61-63); reassigning the attribute here only affects THIS
    # process's view of that name, never the source file.
    base.transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    print("[eval_pretrained_fewshot_imagenet_norm] variant (b): current transform + "
          "ImageNet Normalize(mean=%s, std=%s)." % (IMAGENET_MEAN, IMAGENET_STD))
    base.main()
