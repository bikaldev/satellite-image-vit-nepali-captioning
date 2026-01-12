"""
PyTorch datasets for satellite image classification and captioning.
"""

import os
import ast
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Tuple, Optional, Callable
from pathlib import Path


class SatelliteClassificationDataset(Dataset):
    """
    Dataset for satellite image classification task.
    """
    
    def __init__(
        self,
        csv_file: str,
        root_dir: str,
        transform: Optional[Callable] = None,
        class_names_nepali: Optional[List[str]] = None
    ):
        """
        Args:
            csv_file: Path to CSV file with annotations
            root_dir: Root directory containing images
            transform: Optional transform to be applied on images
            class_names_nepali: List of class names in Nepali
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Get unique classes and create mapping
        if class_names_nepali is not None:
            self.class_names = class_names_nepali
        else:
            self.class_names = sorted(self.data['class'].unique().tolist())
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        print(f"Loaded {len(self.data)} samples")
        print(f"Classes: {self.class_names}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Returns:
            image: Transformed image tensor
            label: Class index
            class_name: Class name (Nepali)
        """
        row = self.data.iloc[idx]
        
        # Load image
        img_path = self.root_dir / row['filepath']
        image = Image.open(img_path).convert('RGB')
        
        # Get class label
        class_name = row['class']
        label = self.class_to_idx[class_name]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label, class_name


class SatelliteCaptioningDataset(Dataset):
    """
    Dataset for satellite image captioning task.
    """
    
    def __init__(
        self,
        csv_file: str,
        root_dir: str,
        transform: Optional[Callable] = None,
        use_nepali_captions: bool = True,
        class_names_nepali: Optional[List[str]] = None
    ):
        """
        Args:
            csv_file: Path to CSV file with annotations
            root_dir: Root directory containing images
            transform: Optional transform to be applied on images
            use_nepali_captions: Whether to use Nepali captions (if available)
            class_names_nepali: List of class names in Nepali
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.use_nepali_captions = use_nepali_captions
        
        # Check if Nepali captions are available
        self.has_nepali_captions = 'captions_nepali' in self.data.columns
        
        if use_nepali_captions and not self.has_nepali_captions:
            print("Warning: Nepali captions not found in dataset. Using English captions.")
            self.use_nepali_captions = False
        
        # Get unique classes
        if class_names_nepali is not None:
            self.class_names = class_names_nepali
        else:
            self.class_names = sorted(self.data['class'].unique().tolist())
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        
        print(f"Loaded {len(self.data)} samples for captioning")
        print(f"Using {'Nepali' if self.use_nepali_captions else 'English'} captions")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def _parse_captions(self, caption_str: str) -> List[str]:
        """Parse caption string (list format) to extract individual captions."""
        try:
            captions = ast.literal_eval(caption_str)
            if isinstance(captions, list):
                return captions
            return [str(caption_str)]
        except:
            return [str(caption_str)]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str]:
        """
        Returns:
            image: Transformed image tensor
            caption: Single caption (randomly selected from list)
            class_name: Class name (Nepali)
        """
        row = self.data.iloc[idx]
        
        # Load image
        img_path = self.root_dir / row['filepath']
        image = Image.open(img_path).convert('RGB')
        
        # Get caption
        if self.use_nepali_captions and self.has_nepali_captions:
            captions = self._parse_captions(row['captions_nepali'])
        else:
            captions = self._parse_captions(row['captions'])
        
        # Randomly select one caption from the list
        import random
        caption = random.choice(captions)
        
        # Get class name
        class_name = row['class']
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, caption, class_name


def collate_fn_captioning(batch, tokenizer, max_length=128):
    """
    Custom collate function for captioning dataset.
    
    Args:
        batch: List of (image, caption, class_name) tuples
        tokenizer: Tokenizer for captions
        max_length: Maximum caption length
        
    Returns:
        Dictionary with batched images and tokenized captions
    """
    images, captions, class_names = zip(*batch)
    
    # Stack images
    images = torch.stack(images)
    
    # Prepend class names to captions
    captions_with_class = [f"{cls}: {cap}" for cls, cap in zip(class_names, captions)]
    
    # Tokenize captions
    encoded = tokenizer(
        captions_with_class,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    labels = encoded['input_ids'].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    
    return {
        'pixel_values': images,
        'labels': labels,
        'captions': captions,
        'class_names': class_names
    }
