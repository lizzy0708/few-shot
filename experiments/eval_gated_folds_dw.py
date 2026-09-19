"""
eval_gated_folds_dw.py — thin wrapper around eval_gated_folds.py, retargeted at the
domain_weight-sweep checkpoints (`checkpoints/coarse5_fold*_gated_nodg_dw{2,5}_s{0,1,2}.pth`).

Does NOT modify experiments/eval_gated_folds.py. Monkey-patches its module-level
`ckpt_path` and `load_gated_model` to point at the requested dw-tag checkpoint family
(plain nodg architecture, use_gate_orth=False -- this sweep only changes domain_weight
at training time, not the model class), then reuses run_fold()/sanity_check()/main()
unchanged -- identical beta/n_sigma/LedoitWolf/leakage-fix logic as the already-verified
script.

Usage:
  conda run -n torch python experiments/eval_gated_folds_dw.py --dw_tag dw2 --beta 0.5 --n_sigma 2.0
  conda run -n torch python experiments/eval_gated_folds_dw.py --dw_tag dw5 --beta 0.5 --n_sigma 2.0
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch

from models.gated_mask_model import GatedMaskModel
import experiments.eval_gated_folds as base


def make_ckpt_path_fn(dw_tag):
    def ckpt_path_dw(fold, train_seed):
        return f"checkpoints/coarse5_fold{fold}_gated_nodg_{dw_tag}_s{train_seed}.pth"
    return ckpt_path_dw


def load_gated_model_dw(path):
    sd = torch.load(path, map_location=base.device)
    assert "domain_classifier_disc.weight" not in sd and "domain_gate_net.net.0.weight" not in sd, (
        f"{path}: checkpoint has domain_gate keys — this is not a 'nodg' checkpoint, refusing to load"
    )
    num_domains = sd["domain_classifier.weight"].shape[0]
    in_dim = sd["classifier.weight"].shape[1]
    encoder_layer = "layer3" if in_dim == 1024 else "layer4"
    model = GatedMaskModel(
        num_classes=2, num_domains=num_domains,
        encoder_layer=encoder_layer, use_domain_gate=False, use_gate_orth=False,
    ).to(base.device)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, f"{path}: state_dict mismatch missing={missing} unexpected={unexpected}"
    model.eval()
    return model


if __name__ == "__main__":
    pre_ap = argparse.ArgumentParser(add_help=False)
    pre_ap.add_argument("--dw_tag", type=str, required=True, choices=["dw2", "dw5"])
    pre_args, remaining_argv = pre_ap.parse_known_args()

    base.ckpt_path = make_ckpt_path_fn(pre_args.dw_tag)
    base.load_gated_model = load_gated_model_dw

    sys.argv = [sys.argv[0]] + remaining_argv
    base.main()
