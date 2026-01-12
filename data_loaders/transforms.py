"""
Image transformations for satellite images.
Includes ViT-compatible transforms and augmentation strategies.
"""

import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple


# ImageNet normalization (used by ViT)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224):
    """
    Get training transforms with data augmentation.
    
    Args:
        image_size: Target image size (default 224 for ViT)
        
    Returns:
        Composed transforms
    """
    return T.Compose([
        T.Resize((256, 256)),
        T.RandomCrop(image_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_val_transforms(image_size: int = 224):
    """
    Get validation/test transforms (no augmentation).
    
    Args:
        image_size: Target image size (default 224 for ViT)
        
    Returns:
        Composed transforms
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Denormalize image tensor for visualization.
    
    Args:
        tensor: Normalized image tensor [C, H, W] or [B, C, H, W]
        
    Returns:
        Denormalized tensor
    """
    mean = torch.tensor(IMAGENET_MEAN).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(-1, 1, 1)
    
    if tensor.dim() == 4:  # Batch
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    return tensor * std + mean


def visualize_augmentations(
    image_path: str,
    num_samples: int = 8,
    save_path: str = None
):
    """
    Visualize data augmentation effects.
    
    Args:
        image_path: Path to input image
        num_samples: Number of augmented samples to generate
        save_path: Optional path to save visualization
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Get transforms
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    
    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Validation transform (no augmentation)
    val_img = val_transform(image)
    val_img_denorm = denormalize(val_img).permute(1, 2, 0).numpy()
    val_img_denorm = np.clip(val_img_denorm, 0, 1)
    axes[0, 1].imshow(val_img_denorm)
    axes[0, 1].set_title('Val Transform\n(No Augmentation)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Hide unused subplot
    axes[0, 2].axis('off')
    
    # Augmented samples
    for i in range(6):
        row = (i // 3) + 1
        col = i % 3
        
        aug_img = train_transform(image)
        aug_img_denorm = denormalize(aug_img).permute(1, 2, 0).numpy()
        aug_img_denorm = np.clip(aug_img_denorm, 0, 1)
        
        axes[row, col].imshow(aug_img_denorm)
        axes[row, col].set_title(f'Augmented {i+1}', fontsize=10)
        axes[row, col].axis('off')
    
    plt.suptitle('Data Augmentation Visualization', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Augmentation visualization saved to {save_path}")
    
    plt.show()


def visualize_batch(
    images: torch.Tensor,
    labels: List[str],
    predictions: List[str] = None,
    save_path: str = None
):
    """
    Visualize a batch of images with labels.
    
    Args:
        images: Batch of images [B, C, H, W]
        labels: Ground truth labels
        predictions: Optional predicted labels
        save_path: Optional path to save visualization
    """
    batch_size = min(len(images), 16)
    grid_size = int(np.ceil(np.sqrt(batch_size)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()
    
    for i in range(batch_size):
        img = denormalize(images[i]).permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        
        title = f"GT: {labels[i]}"
        if predictions is not None:
            title += f"\nPred: {predictions[i]}"
            color = 'green' if labels[i] == predictions[i] else 'red'
        else:
            color = 'black'
        
        axes[i].set_title(title, fontsize=8, color=color)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(batch_size, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Batch visualization saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Demo: visualize augmentations
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        save_path = "outputs/preprocessing/augmentation_demo.png" if len(sys.argv) > 2 else None
        visualize_augmentations(image_path, save_path=save_path)
    else:
        print("Usage: python transforms.py <image_path> [save_path]")
