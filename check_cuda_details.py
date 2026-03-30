import sys
import subprocess
import torch

def check_cuda():
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version: {sys.version}")
    print("-" * 30)
    
    try:
        print(f"PyTorch Version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_available}")
        
        if cuda_available:
            print(f"CUDA Version (Torch): {torch.version.cuda}")
            print(f"CuDNN Version: {torch.backends.cudnn.version()}")
            print(f"Device Count: {torch.cuda.device_count()}")
            print(f"Current Device: {torch.cuda.current_device()}")
            print(f"Device Name: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA is NOT available.")
            print("Possible reasons: CPU-only PyTorch installed, missing drivers, or unsupported hardware.")
            
    except ImportError:
        print("PyTorch is not installed.")
    
    print("-" * 30)
    print("Installed torch packages:")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "list", "--format=freeze"])
    except Exception as e:
        print(f"Could not list packages: {e}")

if __name__ == "__main__":
    check_cuda()
