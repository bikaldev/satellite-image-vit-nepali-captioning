"""
Vision Transformer (ViT) based image captioning model.
Uses ViT encoder and GPT-2 decoder for generating Nepali captions.
"""

import torch
import torch.nn as nn
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    GenerationConfig
)
from typing import Dict, List, Optional


class ViTCaptioner(nn.Module):
    """
    Vision Transformer based image captioning model.
    Encoder: ViT, Decoder: GPT-2
    """
    
    def __init__(
        self,
        encoder_model: str = "google/vit-base-patch16-224",
        decoder_model: str = "gpt2",
        max_length: int = 128,
        num_beams: int = 5
    ):
        """
        Initialize ViT captioning model.
        
        Args:
            encoder_model: Pretrained ViT model name
            decoder_model: Pretrained decoder model name (GPT-2)
            max_length: Maximum caption length
            num_beams: Number of beams for beam search
        """
        super().__init__()
        
        self.encoder_model_name = encoder_model
        self.decoder_model_name = decoder_model
        self.max_length = max_length
        self.num_beams = num_beams
        
        # Initialize Vision Encoder-Decoder model
        self.model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            encoder_model,
            decoder_model
        )
        
        # Initialize tokenizer first
        self.tokenizer = AutoTokenizer.from_pretrained(decoder_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set model config tokens
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        
        # Generation config
        self.generation_config = GenerationConfig(
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=2.0,
            decoder_start_token_id=self.model.config.decoder_start_token_id,
            pad_token_id=self.model.config.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            pixel_values: Input images [batch_size, 3, 224, 224]
            labels: Target caption token IDs [batch_size, seq_len]
            
        Returns:
            Dictionary with 'loss' and 'logits'
        """
        outputs = self.model(
            pixel_values=pixel_values,
            labels=labels
        )
        
        return {
            'loss': outputs.loss,
            'logits': outputs.logits
        }
    
    def generate_caption(
        self,
        pixel_values: torch.Tensor,
        class_label: Optional[str] = None,
        num_beams: Optional[int] = None,
        max_length: Optional[int] = None
    ) -> List[str]:
        """
        Generate captions for input images.
        
        Args:
            pixel_values: Input images [batch_size, 3, 224, 224]
            class_label: Optional class label to condition caption
            num_beams: Number of beams (overrides default)
            max_length: Max caption length (overrides default)
            
        Returns:
            List of generated captions
        """
        self.eval()
        
        # Update generation config if specified
        gen_config = self.generation_config
        if num_beams is not None:
            gen_config.num_beams = num_beams
        if max_length is not None:
            gen_config.max_length = max_length
        
        with torch.no_grad():
            # Generate caption token IDs
            generated_ids = self.model.generate(
                pixel_values=pixel_values,
                generation_config=gen_config
            )
            
            # Decode to text
            captions = self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
            
            # Prepend class label if provided
            if class_label is not None:
                captions = [f"{class_label}: {cap}" for cap in captions]
        
        return captions
    
    def prepare_labels(self, captions: List[str], class_labels: Optional[List[str]] = None) -> torch.Tensor:
        """
        Prepare caption labels for training.
        
        Args:
            captions: List of caption strings
            class_labels: Optional list of class labels to prepend
            
        Returns:
            Tokenized and padded labels tensor
        """
        # Prepend class labels if provided
        if class_labels is not None:
            captions = [f"{label}: {cap}" for label, cap in zip(class_labels, captions)]
        
        # Tokenize
        encoded = self.tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        labels = encoded['input_ids']
        
        # Replace padding token id with -100 (ignored in loss)
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return labels
    
    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        optimizer_state: Optional[dict] = None,
        metrics: Optional[dict] = None
    ):
        """Save model checkpoint."""
        # Save model
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Save additional info
        checkpoint_info = {
            'epoch': epoch,
            'encoder_model': self.encoder_model_name,
            'decoder_model': self.decoder_model_name,
            'max_length': self.max_length,
            'num_beams': self.num_beams,
        }
        
        if optimizer_state is not None:
            checkpoint_info['optimizer_state_dict'] = optimizer_state
        
        if metrics is not None:
            checkpoint_info['metrics'] = metrics
        
        torch.save(checkpoint_info, f"{path}/checkpoint_info.pt")
        print(f"Checkpoint saved to {path}")
    
    @classmethod
    def load_checkpoint(cls, path: str, device: str = 'cuda'):
        """Load model from checkpoint."""
        # Load checkpoint info
        checkpoint_info = torch.load(f"{path}/checkpoint_info.pt", map_location=device, weights_only=False)
        
        # Create model instance
        model = cls(
            encoder_model=checkpoint_info['encoder_model'],
            decoder_model=checkpoint_info['decoder_model'],
            max_length=checkpoint_info['max_length'],
            num_beams=checkpoint_info['num_beams']
        )
        
        # Load pretrained weights
        model.model = VisionEncoderDecoderModel.from_pretrained(path)
        model.tokenizer = AutoTokenizer.from_pretrained(path)
        model.tokenizer.pad_token = model.tokenizer.eos_token
        
        model.to(device)
        
        return model, checkpoint_info.get('epoch', 0)
