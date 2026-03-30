import torch
import sys

print(f"Python: {sys.version}")
print(f"Torch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"Device Count: {torch.cuda.device_count()}")
    print(f"Current Device: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("No CUDA device detected.")

try:
    if torch.cuda.is_available():
        print("Testing GPU execution...")
        x = torch.tensor([1.0, 2.0, 3.0]).cuda()
        y = x * 2
        print(f"Success! Result: {y}")
except Exception as e:
    print(f"FATAL ERROR: GPU execution failed: {e}")
    sys.exit(1)
