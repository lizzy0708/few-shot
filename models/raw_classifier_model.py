"""
raw_classifier_model.py — 특징분해(gate) 전혀 없이 raw z_pool을 그대로 classifier에
입력하는 최소 모델. 2x2 매트릭스(분해 여부 x few-shot 여부)의 "분해 안 함" 축을
gate 계열 체크포인트와 동일 학습 레시피(GRL on)로 채우기 위한 대조군.

구조:
  z = FeatureExtractor(x)
  z_pool = GAP(z)
  class_logits = classifier(z_pool)              <- gate 없이 바로
  domain_logits = domain_classifier(GRL(z_pool))  <- 표준 domain-adversarial (encoder 전체)
"""
import torch.nn as nn
import torch.nn.functional as F
from models.original_mask_model import FeatureExtractor, grad_reverse


class RawClassifierModel(nn.Module):
    def __init__(self, num_classes=2, num_domains=5, encoder_layer='layer3'):
        super().__init__()
        self.feature_extractor = FeatureExtractor(encoder_layer=encoder_layer)
        in_dim = self.feature_extractor.out_dim
        self.classifier = nn.Linear(in_dim, num_classes)
        self.domain_classifier = nn.Linear(in_dim, num_domains)

    def forward(self, x, alpha=1.0):
        z = self.feature_extractor(x)
        z_pool = F.adaptive_avg_pool2d(z, 1).flatten(1)
        class_logits = self.classifier(z_pool)
        domain_logits = self.domain_classifier(grad_reverse(z_pool, alpha))
        return {
            "z": z,
            "z_pool": z_pool,
            "class_logits": class_logits,
            "domain_logits": domain_logits,
        }
