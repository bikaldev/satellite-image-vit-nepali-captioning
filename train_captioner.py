"""
Training script for satellite image captioning using ViT encoder and GPT-2 decoder.
Includes comprehensive monitoring, BLEU score tracking, and sample caption generation.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
import json
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.meteor_score import meteor_score
import nltk

# Download NLTK data
# Force fresh download to avoid corrupt zip file issues
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data (force=True to overwrite corrupt files)
nltk.download('punkt', quiet=True, force=True)
nltk.download('punkt_tab', quiet=True, force=True)
nltk.download('wordnet', quiet=True, force=True)
nltk.download('omw-1.4', quiet=True, force=True)

# Import local modules
from models.vit_captioner import ViTCaptioner
from data_loaders.dataset import SatelliteCaptioningDataset, collate_fn_captioning
from data_loaders.transforms import get_train_transforms, get_val_transforms
import matplotlib.pyplot as plt


def plot_training_curves(history, save_path):
    """Plot captioning training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # BLEU scores
    for i, bleu_key in enumerate(['bleu1', 'bleu2', 'bleu3', 'bleu4']):
        if bleu_key in history:
            axes[0, 1].plot(history[bleu_key], label=f'BLEU-{i+1}', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('BLEU Score')
    axes[0, 1].set_title('BLEU Scores')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # METEOR score
    if 'meteor' in history:
        axes[1, 0].plot(history['meteor'], linewidth=2, color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('METEOR Score')
        axes[1, 0].set_title('METEOR Score')
        axes[1, 0].grid(alpha=0.3)
    
    # Learning rate
    if 'lr' in history:
        axes[1, 1].plot(history['lr'], linewidth=2, color='orange')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].grid(alpha=0.3)
        axes[1, 1].set_yscale('log')
    
    plt.suptitle('Captioning Training Progress', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def compute_bleu_scores(references, hypotheses):
    """Compute BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores."""
    # Tokenize
    refs_tokenized = [[ref.split()] for ref in references]
    hyps_tokenized = [hyp.split() for hyp in hypotheses]
    
    bleu1 = corpus_bleu(refs_tokenized, hyps_tokenized, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(refs_tokenized, hyps_tokenized, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(refs_tokenized, hyps_tokenized, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = corpus_bleu(refs_tokenized, hyps_tokenized, weights=(0.25, 0.25, 0.25, 0.25))
    
    return bleu1, bleu2, bleu3, bleu4


def compute_meteor_score(references, hypotheses):
    """Compute average METEOR score."""
    scores = []
    for ref, hyp in zip(references, hypotheses):
        try:
            score = meteor_score([ref.split()], hyp.split())
            scores.append(score)
        except:
            scores.append(0.0)
    return np.mean(scores)


def train_one_epoch(model, dataloader, optimizer, device, scaler=None):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs['loss']
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item() * pixel_values.size(0)
        pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate(model, dataloader, device, class_names):
    """Validate model and generate sample captions."""
    model.eval()
    running_loss = 0.0
    all_references = []
    all_hypotheses = []
    sample_results = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc='Validation')):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            # Compute loss
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs['loss']
            running_loss += loss.item() * pixel_values.size(0)
            
            # Generate captions
            generated_captions = model.generate_caption(pixel_values)
            
            # Store for metrics
            for ref, hyp, cls in zip(batch['captions'], generated_captions, batch['class_names']):
                all_references.append(ref)
                all_hypotheses.append(hyp)
                
                # Save first 10 samples
                if len(sample_results) < 10:
                    sample_results.append({
                        'class': cls,
                        'reference': ref,
                        'generated': hyp
                    })
    
    epoch_loss = running_loss / len(dataloader.dataset)
    
    # Compute metrics
    bleu1, bleu2, bleu3, bleu4 = compute_bleu_scores(all_references, all_hypotheses)
    meteor = compute_meteor_score(all_references, all_hypotheses)
    
    return {
        'loss': epoch_loss,
        'bleu1': bleu1,
        'bleu2': bleu2,
        'bleu3': bleu3,
        'bleu4': bleu4,
        'meteor': meteor,
        'sample_results': sample_results
    }


def main(args):
    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'checkpoints').mkdir(exist_ok=True)
    (output_dir / 'training_curves').mkdir(exist_ok=True)
    (output_dir / 'sample_captions').mkdir(exist_ok=True)
    
    # Class names
    class_names = config['classes']['names_nepali']
    
    # Create datasets
    print("\n=== Loading Datasets ===")
    train_dataset = SatelliteCaptioningDataset(
        csv_file=f"{config['data']['processed_dir']}/train.csv",  # Use new dataset split
        root_dir=config['data']['raw_dir'],
        transform=get_train_transforms(config['image']['size']),
        use_nepali_captions=True,
        class_names_nepali=class_names
    )
    
    val_dataset = SatelliteCaptioningDataset(
        csv_file=f"{config['data']['processed_dir']}/valid.csv",
        root_dir=config['data']['raw_dir'],
        transform=get_val_transforms(config['image']['size']),
        use_nepali_captions=True,
        class_names_nepali=class_names
    )
    
    # Check if resuming from checkpoint
    start_epoch = 0
    checkpoint_info = None
    
    if args.resume_from:
        print(f"\n=== Resuming from Checkpoint: {args.resume_from} ===")
        checkpoint_path = output_dir / 'checkpoints' / args.resume_from
        
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load model from checkpoint
        model, loaded_epoch = ViTCaptioner.load_checkpoint(str(checkpoint_path), device=str(device))
        checkpoint_info = torch.load(checkpoint_path / "checkpoint_info.pt", map_location=device, weights_only=False)
        
        start_epoch = loaded_epoch + 1
        print(f"✓ Loaded checkpoint from epoch {loaded_epoch}")
        print(f"✓ Will resume training from epoch {start_epoch + 1}")
        
        if 'metrics' in checkpoint_info:
            metrics = checkpoint_info['metrics']
            print(f"  Previous BLEU-4: {metrics.get('bleu4', 'N/A')}")
            print(f"  Previous Loss: {metrics.get('loss', 'N/A')}")
    else:
        # Create model from scratch
        print("\n=== Creating Model ===")
        model = ViTCaptioner(
            encoder_model=config['captioner']['encoder_model'],
            decoder_model=config['captioner']['decoder_model'],
            max_length=config['captioner']['max_length'],
            num_beams=config['captioner']['num_beams']
        )
        model = model.to(device)
    
    # Create dataloaders with custom collate function
    from functools import partial
    collate_fn = partial(collate_fn_captioning, tokenizer=model.tokenizer, max_length=config['captioner']['max_length'])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train_captioner']['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['train_captioner']['learning_rate'],
        weight_decay=config['train_captioner']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['train_captioner']['epochs'],
        eta_min=1e-6
    )
    
    # Load optimizer and scheduler states if resuming
    if checkpoint_info is not None and 'optimizer_state_dict' in checkpoint_info:
        optimizer.load_state_dict(checkpoint_info['optimizer_state_dict'])
        print("✓ Loaded optimizer state")
        
        # Step scheduler to the correct position
        for _ in range(start_epoch):
            scheduler.step()
        print(f"✓ Advanced scheduler to epoch {start_epoch}")
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if config['train_captioner']['mixed_precision'] else None
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=output_dir / 'logs')
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'bleu1': [],
        'bleu2': [],
        'bleu3': [],
        'bleu4': [],
        'meteor': [],
        'lr': []
    }
    
    # Load existing training history if resuming
    history_file = output_dir / 'training_history.csv'
    if args.resume_from and history_file.exists():
        print("\n=== Loading Training History ===")
        history_df = pd.read_csv(history_file)
        
        # Convert back to dictionary format
        for key in history.keys():
            if key in history_df.columns:
                history[key] = history_df[key].tolist()
        
        print(f"✓ Loaded {len(history['train_loss'])} epochs of history")
    
    # Track best BLEU-4 (from history if resuming)
    best_bleu4 = max(history['bleu4']) if history['bleu4'] else 0.0
    
    print("\n=== Starting Training ===")
    if args.resume_from:
        print(f"Resuming from epoch {start_epoch + 1}/{config['train_captioner']['epochs']}")
        print(f"Best BLEU-4 so far: {best_bleu4:.4f}")
    
    for epoch in range(start_epoch, config['train_captioner']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['train_captioner']['epochs']}")
        print("-" * 50)
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
        
        # Validate
        val_metrics = validate(model, val_loader, device, class_names)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['bleu1'].append(val_metrics['bleu1'])
        history['bleu2'].append(val_metrics['bleu2'])
        history['bleu3'].append(val_metrics['bleu3'])
        history['bleu4'].append(val_metrics['bleu4'])
        history['meteor'].append(val_metrics['meteor'])
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"BLEU-1: {val_metrics['bleu1']:.4f}, BLEU-2: {val_metrics['bleu2']:.4f}")
        print(f"BLEU-3: {val_metrics['bleu3']:.4f}, BLEU-4: {val_metrics['bleu4']:.4f}")
        print(f"METEOR: {val_metrics['meteor']:.4f}")
        
        # TensorBoard logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('BLEU/bleu1', val_metrics['bleu1'], epoch)
        writer.add_scalar('BLEU/bleu2', val_metrics['bleu2'], epoch)
        writer.add_scalar('BLEU/bleu3', val_metrics['bleu3'], epoch)
        writer.add_scalar('BLEU/bleu4', val_metrics['bleu4'], epoch)
        writer.add_scalar('METEOR', val_metrics['meteor'], epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Save sample captions
        sample_file = output_dir / 'sample_captions' / f'epoch_{epoch+1}.json'
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(val_metrics['sample_results'], f, ensure_ascii=False, indent=2)
        
        # Save best model (based on BLEU-4)
        if val_metrics['bleu4'] > best_bleu4:
            best_bleu4 = val_metrics['bleu4']
            model.save_checkpoint(
                str(output_dir / 'checkpoints' / 'best_bleu_model'),
                epoch=epoch,
                optimizer_state=optimizer.state_dict(),
                metrics=val_metrics
            )
            print(f"✓ Saved best model (BLEU-4: {best_bleu4:.4f})")
        
        # Save checkpoint every N epochs
        if (epoch + 1) % config['logging']['save_interval'] == 0:
            model.save_checkpoint(
                str(output_dir / 'checkpoints' / f'epoch_{epoch+1}'),
                epoch=epoch,
                optimizer_state=optimizer.state_dict(),
                metrics=val_metrics
            )
        
        # Update learning rate
        scheduler.step()
    
    # Save final model
    model.save_checkpoint(
        str(output_dir / 'checkpoints' / 'last_epoch'),
        epoch=config['train_captioner']['epochs'] - 1,
        optimizer_state=optimizer.state_dict()
    )
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_df['epoch'] = range(1, len(history['train_loss']) + 1)
    history_df.to_csv(output_dir / 'training_history.csv', index=False)
    
    # Plot training curves
    plot_training_curves(history, output_dir / 'training_curves' / 'training_progress.png')
    
    # Close writer
    writer.close()
    
    print("\n=== Training Complete ===")
    print(f"Best BLEU-4 score: {best_bleu4:.4f}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train satellite image captioning model")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='outputs/captioning', help='Output directory')
    parser.add_argument('--resume_from', type=str, default=None, help='Checkpoint directory name to resume from (e.g., "epoch_8" or "best_bleu_model")')
    
    args = parser.parse_args()
    main(args)
