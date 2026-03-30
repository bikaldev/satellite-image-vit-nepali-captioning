"""
Vision Transformer (ViT) based image classifier for satellite images.
Classifies images into 8 categories with Nepali labels.
"""

import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
from typing import Dict, List, Optional


class ViTClassifier(nn.Module):
    """
    Vision Transformer classifier for satellite image classification.
    """
    
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        num_classes: int = 8,
        dropout: float = 0.1,
        freeze_backbone: bool = False
    ):
        """
        Initialize ViT classifier.
        
        Args:
            model_name: Pretrained ViT model name from HuggingFace
            num_classes: Number of output classes
            dropout: Dropout rate for classification head
            freeze_backbone: Whether to freeze the ViT backbone
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Load pretrained ViT model
        self.vit = ViTModel.from_pretrained(model_name)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
        
        # Classification head
        hidden_size = self.vit.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            pixel_values: Input images [batch_size, 3, 224, 224]
            
        Returns:
            logits: Class logits [batch_size, num_classes]
        """
        # Get ViT outputs
        outputs = self.vit(pixel_values=pixel_values)
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0]
        
        # Classification
        logits = self.classifier(cls_output)
        
        return logits
    
    def unfreeze_backbone(self):
        """Unfreeze the ViT backbone for fine-tuning."""
        for param in self.vit.parameters():
            param.requires_grad = True
    
    def freeze_backbone(self):
        """Freeze the ViT backbone."""
        for param in self.vit.parameters():
            param.requires_grad = False
    
    def predict(
        self,
        pixel_values: torch.Tensor,
        return_probs: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions on input images.
        
        Args:
            pixel_values: Input images [batch_size, 3, 224, 224]
            return_probs: Whether to return probabilities (softmax)
            
        Returns:
            Dictionary with 'logits', 'probs' (if requested), and 'predictions'
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(pixel_values)
            predictions = torch.argmax(logits, dim=-1)
            
            result = {
                'logits': logits,
                'predictions': predictions
            }
            
            if return_probs:
                probs = torch.softmax(logits, dim=-1)
                result['probs'] = probs
            
            return result
    
    def save_checkpoint(self, path: str, epoch: int, optimizer_state: Optional[dict] = None):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'model_name': self.model_name,
            'num_classes': self.num_classes,
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    @classmethod
    def load_checkpoint(cls, path: str, device: str = 'cuda'):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        model = cls(
            model_name=checkpoint['model_name'],
            num_classes=checkpoint['num_classes']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return model, checkpoint.get('epoch', 0)
