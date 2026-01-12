"""
Model definitions for satellite image classification and captioning.
"""

from .vit_classifier import ViTClassifier
from .vit_captioner import ViTCaptioner

__all__ = ['ViTClassifier', 'ViTCaptioner']
