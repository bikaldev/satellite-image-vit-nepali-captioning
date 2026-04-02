import torch
checkpoint_path = r"c:\Users\User\Documents\Satellite Image\satellite-image-vit-nepali-captioning\outputs\captioning_refined\checkpoints\last_epoch\checkpoint_info.pt"
try:
    # Use weights_only=False because it likely contains training state (optimizer, epoch, etc.)
    checkpoint_info = torch.load(checkpoint_path, weights_only=False)
    print("Checkpoint info keys:", checkpoint_info.keys())
    if 'epoch' in checkpoint_info:
        print(f"Epoch: {checkpoint_info['epoch']}")
    else:
        print("Epoch key not found in checkpoint info.")
except Exception as e:
    print(f"Error loading checkpoint: {e}")
