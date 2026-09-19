"""
eval_gated_folds_orth.py — thin wrapper around eval_gated_folds.py, retargeted at the
Stage-2 anti-domain-orthogonalization checkpoints
(`checkpoints/coarse5_fold*_gated_nodg_orth_s{0,1,2}.pth`).

Does NOT modify experiments/eval_gated_folds.py (which is the script already verified
to reproduce paper Table 2 exactly for the non-orth gated_nodg checkpoints, and should
stay untouched). Instead this monkey-patches its module-level `ckpt_path` and
`load_gated_model` names to point at the new checkpoint family + construct
GatedMaskModel with use_gate_orth=True, then reuses eval_gated_folds.run_fold() /
sanity_check() / main() completely unchanged -- identical beta blending, n_sigma
threshold, LedoitWolf full-covariance, leakage-fix, and 3-train-seed score-ensemble
logic as the already-verified script.

Usage:
  conda run -n torch python experiments/eval_gated_folds_orth.py --beta 0.5 --n_sigma 2.0
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.gated_mask_model import GatedMaskModel
import experiments.eval_gated_folds as base


def ckpt_path_orth(fold, train_seed):
    return f"checkpoints/coarse5_fold{fold}_gated_nodg_orth_s{train_seed}.pth"


def load_gated_model_orth(path):
    sd = torch.load(path, map_location=base.device)
    assert "domain_classifier_disc.weight" not in sd and "domain_gate_net.net.0.weight" not in sd, (
        f"{path}: checkpoint has domain_gate keys — this is not a 'nodg' checkpoint, refusing to load"
    )
    num_domains = sd["domain_classifier.weight"].shape[0]
    in_dim = sd["classifier.weight"].shape[1]
    encoder_layer = "layer3" if in_dim == 1024 else "layer4"
    model = GatedMaskModel(
        num_classes=2, num_domains=num_domains,
        encoder_layer=encoder_layer, use_domain_gate=False, use_gate_orth=True,
    ).to(base.device)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, f"{path}: state_dict mismatch missing={missing} unexpected={unexpected}"
    model.eval()
    return model


# Monkey-patch: base.run_fold() / base.sanity_check() look these up as module-level
# globals at call time, so patching the module's attributes here redirects them
# without touching eval_gated_folds.py's source.
base.ckpt_path = ckpt_path_orth
base.load_gated_model = load_gated_model_orth


if __name__ == "__main__":
    base.main()
