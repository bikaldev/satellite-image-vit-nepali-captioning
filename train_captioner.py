"""
Refined training script for satellite image captioning.
Features:
- Comprehensive metrics: BLEU 1-4, METEOR, ROUGE-L.
- Storage efficiency: Keeps only the 'best' and 'last' checkpoint tensors.
- Environment-aware execution.
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
import shutil
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import nltk

# Download required NLTK data
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Import local modules
from models.vit_captioner import ViTCaptioner
from data_loaders.dataset import SatelliteCaptioningDataset, collate_fn_captioning
from data_loaders.transforms import get_train_transforms, get_val_transforms
import matplotlib.pyplot as plt

def compute_metrics(references, hypotheses):
    """Compute BLEU 1-4, METEOR, and ROUGE-L."""
    # Tokenize for BLEU
    refs_tokenized = [[ref.split()] for ref in references]
    hyps_tokenized = [hyp.split() for hyp in hypotheses]
    
    bleu1 = corpus_bleu(refs_tokenized, hyps_tokenized, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(refs_tokenized, hyps_tokenized, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(refs_tokenized, hyps_tokenized, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = corpus_bleu(refs_tokenized, hyps_tokenized, weights=(0.25, 0.25, 0.25, 0.25))
    
    # METEOR
    meteor_scores = []
    for ref, hyp in zip(references, hypotheses):
        try:
            m_score = meteor_score([ref.split()], hyp.split())
            meteor_scores.append(m_score)
        except:
            meteor_scores.append(0.0)
    avg_meteor = np.mean(meteor_scores)
    
    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = []
    for ref, hyp in zip(references, hypotheses):
        scores = scorer.score(ref, hyp)
        rouge_scores.append(scores['rougeL'].fmeasure)
    avg_rouge = np.mean(rouge_scores)
    
    return {
        'bleu1': bleu1,
        'bleu2': bleu2,
        'bleu3': bleu3,
        'bleu4': bleu4,
        'meteor': avg_meteor,
        'rougeL': avg_rouge
    }

def train_one_epoch(model, dataloader, optimizer, device, scaler=None, resume_batch=0, save_callback=None):
    model.train()
    running_loss = 0.0
    pbar = tqdm(dataloader, desc='Training')
    
    for i, batch in enumerate(pbar):
        if i < resume_batch:
            continue
            
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        if scaler is not None:
            # Use torch.amp.autocast('cuda') for modern PyTorch
            with torch.amp.autocast('cuda'):
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
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Periodic check for saving (e.g., every 50 batches)
        if save_callback and (i + 1) % 50 == 0:
            save_callback(i + 1)
    
    return running_loss / len(dataloader.dataset)

def validate(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    all_refs = []
    all_hyps = []
    
    # Clear cache before generation
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(pixel_values=pixel_values, labels=labels)
            running_loss += outputs['loss'].item() * pixel_values.size(0)
            
            generated = model.generate_caption(pixel_values)
            all_refs.extend(batch['captions'])
            all_hyps.extend(generated)
            
    val_loss = running_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_refs, all_hyps)
    metrics['loss'] = val_loss
    return metrics

def main(args):
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Model
    model = ViTCaptioner(
        encoder_model=config['captioner']['encoder_model'],
        decoder_model=config['captioner']['decoder_model'],
        max_length=config['captioner']['max_length']
    ).to(device)
    
    # Dataset
    from functools import partial
    collate_fn = partial(collate_fn_captioning, tokenizer=model.tokenizer, max_length=config['captioner']['max_length'])
    
    train_ds = SatelliteCaptioningDataset(
        csv_file=f"{config['data']['processed_dir']}/train.csv",
        root_dir=config['data']['raw_dir'],
        transform=get_train_transforms(config['image']['size']),
        use_nepali_captions=True,
        class_names_nepali=config['classes']['names_nepali']
    )
    
    val_ds = SatelliteCaptioningDataset(
        csv_file=f"{config['data']['processed_dir']}/valid.csv",
        root_dir=config['data']['raw_dir'],
        transform=get_val_transforms(config['image']['size']),
        use_nepali_captions=True,
        class_names_nepali=config['classes']['names_nepali']
    )
    
    train_loader = DataLoader(train_ds, batch_size=config['train_captioner']['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=config['num_workers'])
    val_loader = DataLoader(val_ds, batch_size=config['evaluation']['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=config['num_workers'])
    
    optimizer = optim.AdamW(model.parameters(), lr=config['train_captioner']['learning_rate'])
    # Use torch.amp.GradScaler('cuda') for modern PyTorch
    scaler = torch.amp.GradScaler('cuda') if config['train_captioner'].get('mixed_precision', True) else None
    
    start_epoch = 0
    start_batch = 0
    checkpoint_info = None
    
    # Handle resuming
    if args.resume:
        checkpoint_path = args.resume_from if args.resume_from else str(ckpt_dir / 'checkpoint_latest')
        if os.path.exists(checkpoint_path):
            print(f"Resuming from checkpoint: {checkpoint_path}")
            # Modified load_checkpoint logic to also get batch and optimizer state
            # We'll use a local load helper if the class method doesn't support everything
            checkpoint_info = torch.load(os.path.join(checkpoint_path, 'checkpoint_info.pt'), map_location=device, weights_only=False)
            
            # Load model weights
            from transformers import VisionEncoderDecoderModel
            model.model = VisionEncoderDecoderModel.from_pretrained(checkpoint_path).to(device)
            
            start_epoch = checkpoint_info.get('epoch', 0)
            start_batch = checkpoint_info.get('batch_idx', 0)
            print(f"Resuming from Epoch {start_epoch + 1}, Batch {start_batch}")
        else:
            print(f"Warning: --resume specified but checkpoint not found at {checkpoint_path}")

    # Overwrite with explicit args if provided
    if args.resume_epoch is not None:
        start_epoch = args.resume_epoch
    if args.resume_batch is not None:
        start_batch = args.resume_batch

    # IMPORTANT: Create optimizer AFTER potential model restoration
    # Otherwise, the optimizer will track orphaned parameters.
    optimizer = optim.AdamW(model.parameters(), lr=config['train_captioner']['learning_rate'])
    # Use torch.amp.GradScaler('cuda') for modern PyTorch
    scaler = torch.amp.GradScaler('cuda') if config['train_captioner'].get('mixed_precision', True) else None

    # Restore optimizer and scaler state if resuming
    if checkpoint_info:
        if 'optimizer_state_dict' in checkpoint_info:
            optimizer.load_state_dict(checkpoint_info['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint_info and scaler is not None:
            scaler.load_state_dict(checkpoint_info['scaler_state_dict'])

    writer = SummaryWriter(log_dir=output_dir / 'logs')
    best_bleu4 = 0.0
    
    # Define save callback for periodic saves
    def save_checkpoint_callback(epoch, batch_idx, metrics=None):
        latest_path = ckpt_dir / 'checkpoint_latest'
        # We need to pass more state to save_checkpoint
        # Since ViTCaptioner.save_checkpoint might not be flexible enough, 
        # we'll add optimizer and batch_idx to the info
        
        if latest_path.exists():
            shutil.rmtree(latest_path)
        
        model.save_checkpoint(str(latest_path), epoch=epoch, optimizer_state=optimizer.state_dict(), metrics=metrics)
        
        # Add batch_idx and scaler to the checkpoint_info.pt manually since model.save_checkpoint is limited
        info_path = latest_path / 'checkpoint_info.pt'
        info = torch.load(info_path, weights_only=False)
        info['batch_idx'] = batch_idx
        if scaler is not None:
            info['scaler_state_dict'] = scaler.state_dict()
        torch.save(info, info_path)
    
    print("\n=== Starting Training ===")
    for epoch in range(start_epoch, config['train_captioner']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['train_captioner']['epochs']}")
        
        current_resume_batch = start_batch if epoch == start_epoch else 0
        
        train_loss = train_one_epoch(
            model, 
            train_loader, 
            optimizer, 
            device, 
            scaler, 
            resume_batch=current_resume_batch,
            save_callback=lambda idx: save_checkpoint_callback(epoch, idx)
        )
        val_metrics = validate(model, val_loader, device)
        
        # Log metrics
        for k, v in val_metrics.items():
            writer.add_scalar(f'Val/{k}', v, epoch)
        writer.add_scalar('Train/Loss', train_loss, epoch)
        
        print(f"Metrics: {val_metrics}")
        
        # Storage Efficiency Strategy:
        # 1. Always save 'last_epoch'
        # 2. Save 'best_model' if improved
        # 3. Delete previous 'last_epoch' if it exists (handled by overwriting or explicit deletion)
        
        last_ckpt_path = ckpt_dir / 'last_epoch'
        best_ckpt_path = ckpt_dir / 'best_model'
        
        # Save Last
        if last_ckpt_path.exists():
            shutil.rmtree(last_ckpt_path) # Ensure clean save
        model.save_checkpoint(str(last_ckpt_path), epoch=epoch, metrics=val_metrics)
        
        # Save Best
        if val_metrics['bleu4'] > best_bleu4:
            best_bleu4 = val_metrics['bleu4']
            if best_ckpt_path.exists():
                shutil.rmtree(best_ckpt_path)
            model.save_checkpoint(str(best_ckpt_path), epoch=epoch, metrics=val_metrics)
            print(f"New Best BLEU-4: {best_bleu4:.4f}")

    writer.close()
    print(f"Training complete. Best BLEU-4: {best_bleu4:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--output_dir', type=str, default='outputs/captioning_refined')
    parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint')
    parser.add_argument('--resume_from', type=str, default=None, help='Specific checkpoint directory to resume from')
    parser.add_argument('--resume_epoch', type=int, default=None, help='Explicit epoch to start from (0-indexed)')
    parser.add_argument('--resume_batch', type=int, default=None, help='Explicit batch to start from (0-indexed)')
    args = parser.parse_args()
    main(args)
