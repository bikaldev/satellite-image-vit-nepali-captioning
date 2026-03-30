"""
Training script for satellite image classification using ViT.
Includes comprehensive monitoring, visualization, and checkpointing.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import local modules
from models.vit_classifier import ViTClassifier
from data_loaders.dataset import SatelliteClassificationDataset
from data_loaders.transforms import get_train_transforms, get_val_transforms, visualize_batch


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves(history, save_path):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Accuracy curves
    axes[0, 1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    axes[0, 1].plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy Curves')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Learning rate
    if 'lr' in history:
        axes[1, 0].plot(history['lr'], linewidth=2, color='orange')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].grid(alpha=0.3)
        axes[1, 0].set_yscale('log')
    
    # Per-class F1 scores (last epoch)
    if 'val_f1_per_class' in history and len(history['val_f1_per_class']) > 0:
        f1_scores = history['val_f1_per_class'][-1]
        class_names = history.get('class_names', [f'Class {i}' for i in range(len(f1_scores))])
        
        axes[1, 1].barh(range(len(f1_scores)), f1_scores, color='skyblue', edgecolor='black')
        axes[1, 1].set_yticks(range(len(f1_scores)))
        axes[1, 1].set_yticklabels(class_names, fontsize=8)
        axes[1, 1].set_xlabel('F1 Score')
        axes[1, 1].set_title('Per-Class F1 Scores (Last Epoch)')
        axes[1, 1].invert_yaxis()
        axes[1, 1].grid(axis='x', alpha=0.3)
    
    plt.suptitle('Training Progress', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels, _ in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, class_names):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    # Detailed metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels
    }


def main(args):
    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Set device
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\n[INFO] CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] CUDA Version: {torch.version.cuda}")
        print(f"[INFO] CuDNN Version: {torch.backends.cudnn.version()}")
        torch.backends.cudnn.benchmark = True
        print("[INFO] Enabled CuDNN benchmark for performance")
    else:
        device = torch.device('cpu')
        print("\n[WARNING] CUDA is NOT available. Using device: cpu")
        print("[INFO] Please check your PyTorch installation and NVIDIA drivers if you have a GPU.")

    print(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'checkpoints').mkdir(exist_ok=True)
    (output_dir / 'training_curves').mkdir(exist_ok=True)
    (output_dir / 'confusion_matrices').mkdir(exist_ok=True)
    (output_dir / 'sample_predictions').mkdir(exist_ok=True)
    
    # Class names
    class_names = config['classes']['names_nepali']
    
    # Create datasets
    print("\n=== Loading Datasets ===")
    train_dataset = SatelliteClassificationDataset(
        csv_file=f"{config['data']['processed_dir']}/train.csv",
        root_dir=config['data']['raw_dir'],
        transform=get_train_transforms(config['image']['size']),
        class_names_nepali=class_names
    )
    
    val_dataset = SatelliteClassificationDataset(
        csv_file=f"{config['data']['processed_dir']}/valid.csv",
        root_dir=config['data']['raw_dir'],
        transform=get_val_transforms(config['image']['size']),
        class_names_nepali=class_names
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train_classifier']['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Create model
    print("\n=== Creating Model ===")
    model = ViTClassifier(
        model_name=config['classifier']['model_name'],
        num_classes=config['classifier']['num_classes'],
        dropout=config['classifier']['dropout'],
        freeze_backbone=True  # Start with frozen backbone
    )
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['train_classifier']['learning_rate'],
        weight_decay=config['train_classifier']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['train_classifier']['epochs'],
        eta_min=1e-6
    )
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if config['train_classifier']['mixed_precision'] else None
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=output_dir / 'logs')
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1_per_class': [],
        'lr': [],
        'class_names': class_names
    }
    
    best_val_acc = 0.0
    
    print("\n=== Starting Training ===")
    for epoch in range(config['train_classifier']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['train_classifier']['epochs']}")
        print("-" * 50)
        
        # Unfreeze backbone after specified epochs
        if epoch == config['classifier']['freeze_backbone_epochs']:
            print("Unfreezing backbone for fine-tuning...")
            model.unfreeze_backbone()
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, class_names)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1_per_class'].append(val_metrics['f1'].tolist())
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"Val F1 (avg): {np.mean(val_metrics['f1']):.4f}")
        
        # TensorBoard logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Save confusion matrix
        if (epoch + 1) % 5 == 0:
            cm_path = output_dir / 'confusion_matrices' / f'epoch_{epoch+1}.png'
            plot_confusion_matrix(val_metrics['confusion_matrix'], class_names, cm_path)
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            model.save_checkpoint(
                output_dir / 'checkpoints' / 'best_model.pth',
                epoch=epoch,
                optimizer_state=optimizer.state_dict()
            )
            print(f"✓ Saved best model (val_acc: {best_val_acc:.4f})")
        
        # Save checkpoint every N epochs
        if (epoch + 1) % config['logging']['save_interval'] == 0:
            model.save_checkpoint(
                output_dir / 'checkpoints' / f'epoch_{epoch+1}.pth',
                epoch=epoch,
                optimizer_state=optimizer.state_dict()
            )
        
        # Update learning rate
        scheduler.step()
    
    # Save final model
    model.save_checkpoint(
        output_dir / 'checkpoints' / 'last_epoch.pth',
        epoch=config['train_classifier']['epochs'] - 1,
        optimizer_state=optimizer.state_dict()
    )
    
    # Save training history
    history_df = pd.DataFrame({
        'epoch': range(1, len(history['train_loss']) + 1),
        'train_loss': history['train_loss'],
        'train_acc': history['train_acc'],
        'val_loss': history['val_loss'],
        'val_acc': history['val_acc'],
        'lr': history['lr']
    })
    history_df.to_csv(output_dir / 'training_history.csv', index=False)
    
    # Plot training curves
    plot_training_curves(history, output_dir / 'training_curves' / 'training_progress.png')
    
    # Close writer
    writer.close()
    
    print("\n=== Training Complete ===")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train satellite image classifier")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='outputs/classification', help='Output directory')
    
    args = parser.parse_args()
    main(args)
