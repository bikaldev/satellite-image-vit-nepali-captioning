import torch
for name in ["best_model", "last_epoch"]:
    checkpoint_path = f"r\"c:\\Users\\User\\Documents\\Satellite Image\\satellite-image-vit-nepali-captioning\\outputs\\captioning_refined\\checkpoints\\{name}\\checkpoint_info.pt\""
    # Using raw string correctly in the loop
    checkpoint_path = f"c:\\Users\\User\\Documents\\Satellite Image\\satellite-image-vit-nepali-captioning\\outputs\\captioning_refined\\checkpoints\\{name}\\checkpoint_info.pt"
    try:
        checkpoint_info = torch.load(checkpoint_path, weights_only=False)
        print(f"{name} Epoch: {checkpoint_info.get('epoch', 'N/A')}")
    except Exception as e:
        print(f"Error loading {name} checkpoint: {e}")
