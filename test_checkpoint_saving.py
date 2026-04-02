import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import shutil
import yaml
import traceback
import sys

# Import local modules
from models.vit_captioner import ViTCaptioner
from data_loaders.dataset import SatelliteCaptioningDataset, collate_fn_captioning
from data_loaders.transforms import get_train_transforms, get_val_transforms

def main():
    try:
        # Load config
        config_path = "configs/config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Test output directory
        output_dir = Path("outputs/test_dummy")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        ckpt_dir = output_dir / 'checkpoints'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        # Model (Use existing architecture)
        print("Loading model (this might take a while and lot of RAM)...")
        # Monitor RAM if possible
        import psutil
        print(f"RAM before load: {psutil.virtual_memory().available / (1024**3):.2f} GB")
        
        model = ViTCaptioner(
            encoder_model=config['captioner']['encoder_model'],
            decoder_model=config['captioner']['decoder_model'],
            max_length=config['captioner']['max_length']
        ).to(device)
        
        print(f"RAM after load: {psutil.virtual_memory().available / (1024**3):.2f} GB")
        
        # Dataset (Subset of 10 images)
        print("Preparing dummy dataset (10 images)...")
        from functools import partial
        collate_fn = partial(collate_fn_captioning, tokenizer=model.tokenizer, max_length=config['captioner']['max_length'])
        
        full_train_ds = SatelliteCaptioningDataset(
            csv_file=f"{config['data']['processed_dir']}/train.csv",
            root_dir=config['data']['raw_dir'],
            transform=get_train_transforms(config['image']['size']),
            use_nepali_captions=True,
            class_names_nepali=config['classes']['names_nepali']
        )
        
        dummy_train_ds = Subset(full_train_ds, range(10))
        train_loader = DataLoader(dummy_train_ds, batch_size=2, shuffle=True, collate_fn=collate_fn)
        
        optimizer = optim.AdamW(model.parameters(), lr=1e-5)
        scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        
        def save_checkpoint_callback(epoch, batch_idx, metrics=None):
            print(f"\n--- Saving Checkpoint: Epoch {epoch}, Batch {batch_idx} ---")
            print(f"Current RAM: {psutil.virtual_memory().available / (1024**3):.2f} GB")
            latest_path = ckpt_dir / 'checkpoint_latest'
            
            if latest_path.exists():
                shutil.rmtree(latest_path)
                
            try:
                print(f"Executing model.save_checkpoint to {latest_path}...")
                model.save_checkpoint(str(latest_path), epoch=epoch, optimizer_state=optimizer.state_dict(), metrics=metrics)
                
                info_path = latest_path / 'checkpoint_info.pt'
                if info_path.exists():
                    print("Updating checkpoint_info.pt with batch_idx...")
                    info = torch.load(info_path, weights_only=False)
                    info['batch_idx'] = batch_idx
                    if scaler is not None:
                        info['scaler_state_dict'] = scaler.state_dict()
                    torch.save(info, info_path)
                    print("Finalized checkpoint_info.pt")
                else:
                    print("Error: checkpoint_info.pt was NOT created!")
            except Exception as e:
                print(f"ERROR DURING SAVE: {e}")
                traceback.print_exc()

        print("\nStarting Training Loop...")
        for epoch in range(1):
            model.train()
            print(f"\nEpoch {epoch+1}")
            
            for i, batch in enumerate(train_loader):
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                
                optimizer.zero_grad()
                if scaler is not None:
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
                
                print(f"Batch {i+1}/5 - Loss: {loss.item():.4f}")
                
                # Save at batch 2
                if i + 1 == 2:
                    save_checkpoint_callback(epoch, i + 1)
        
        print("\nDummy training finished successfully.")
        
        # Test RESUME
        print("\n=== Testing Resume Logic ===")
        resume_from = str(ckpt_dir / 'checkpoint_latest')
        if os.path.exists(resume_from):
            print(f"Loading from: {resume_from}")
            # This replicates train_captioner.py lines 203-207
            checkpoint_info = torch.load(os.path.join(resume_from, 'checkpoint_info.pt'), map_location=device, weights_only=False)
            from transformers import VisionEncoderDecoderModel
            print("Running VisionEncoderDecoderModel.from_pretrained...")
            model.model = VisionEncoderDecoderModel.from_pretrained(resume_from).to(device)
            print("Model loaded. Testing forward pass...")
            
            # Forward pass
            batch = next(iter(train_loader))
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            with torch.no_grad():
                model.eval()
                outputs = model(pixel_values=pixel_values, labels=labels)
                print(f"Resume test SUCCESS. Test loss: {outputs['loss'].item():.4f}")
        else:
            print("Skip resume test: No checkpoint found.")

    except Exception as e:
        print(f"\nFATAL GLOBAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
