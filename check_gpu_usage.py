"""
GPU Memory Monitoring Script
Checks GPU availability, memory usage, and whether training can proceed safely.
"""

import torch
import subprocess
import yaml
from pathlib import Path


def get_gpu_info():
    """Get detailed GPU information."""
    if not torch.cuda.is_available():
        return None
    
    gpu_info = []
    for i in range(torch.cuda.device_count()):
        device = torch.device(f'cuda:{i}')
        props = torch.cuda.get_device_properties(i)
        
        # Memory in GB
        total_memory = props.total_memory / (1024 ** 3)
        allocated_memory = torch.cuda.memory_allocated(i) / (1024 ** 3)
        reserved_memory = torch.cuda.memory_reserved(i) / (1024 ** 3)
        free_memory = total_memory - reserved_memory
        
        info = {
            'device_id': i,
            'name': props.name,
            'compute_capability': f"{props.major}.{props.minor}",
            'total_memory_gb': total_memory,
            'allocated_memory_gb': allocated_memory,
            'reserved_memory_gb': reserved_memory,
            'free_memory_gb': free_memory,
            'memory_utilization': (reserved_memory / total_memory) * 100
        }
        gpu_info.append(info)
    
    return gpu_info


def get_gpu_processes():
    """Get list of processes using GPU via nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            processes = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        try:
                            processes.append({
                                'pid': parts[0].strip(),
                                'name': parts[1].strip(),
                                'memory_mb': float(parts[2].strip())
                            })
                        except (ValueError, IndexError):
                            # Skip lines that don't parse correctly
                            continue
            return processes if processes else None
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return None


def estimate_training_memory(config_path='configs/config.yaml'):
    """Estimate memory requirements based on config."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        batch_size = config['train_captioner']['batch_size']
        image_size = config['image']['size']
        
        # Rough estimates (in GB)
        # ViT encoder: ~0.5 GB
        # GPT-2 decoder: ~0.5 GB
        # Per image in batch: ~0.01 GB (224x224x3)
        # Gradients and optimizer states: 2x model size
        # Overhead and activations: 1.5x
        
        model_size = 1.0  # encoder + decoder
        batch_memory = batch_size * 0.01
        optimizer_memory = model_size * 2
        overhead = 1.5
        
        estimated_memory = (model_size + batch_memory + optimizer_memory) * overhead
        
        return {
            'batch_size': batch_size,
            'image_size': image_size,
            'estimated_memory_gb': estimated_memory
        }
    except Exception as e:
        print(f"Warning: Could not estimate memory requirements: {e}")
        return None


def check_training_safety(gpu_info, memory_estimate):
    """Determine if training can proceed safely."""
    if not gpu_info:
        return {
            'can_train': False,
            'reason': 'No GPU available',
            'recommendation': 'Training will use CPU (very slow for this model)'
        }
    
    # Check first GPU (default training device)
    gpu = gpu_info[0]
    free_memory = gpu['free_memory_gb']
    
    if memory_estimate:
        required_memory = memory_estimate['estimated_memory_gb']
        safety_margin = 1.2  # 20% extra for safety
        
        if free_memory >= required_memory * safety_margin:
            return {
                'can_train': True,
                'reason': f'Sufficient memory: {free_memory:.2f} GB free, ~{required_memory:.2f} GB required',
                'recommendation': 'Safe to proceed with training'
            }
        elif free_memory >= required_memory:
            return {
                'can_train': True,
                'reason': f'Tight memory: {free_memory:.2f} GB free, ~{required_memory:.2f} GB required',
                'recommendation': 'Training may proceed but monitor for OOM errors. Consider reducing batch size if issues occur.'
            }
        else:
            return {
                'can_train': False,
                'reason': f'Insufficient memory: {free_memory:.2f} GB free, ~{required_memory:.2f} GB required',
                'recommendation': f'Reduce batch size to ~{int(memory_estimate["batch_size"] * free_memory / required_memory)} or free up GPU memory'
            }
    
    # Fallback check without memory estimate
    if gpu['memory_utilization'] > 80:
        return {
            'can_train': False,
            'reason': f'GPU memory highly utilized: {gpu["memory_utilization"]:.1f}%',
            'recommendation': 'Free up GPU memory before training'
        }
    else:
        return {
            'can_train': True,
            'reason': f'GPU memory available: {free_memory:.2f} GB free ({100-gpu["memory_utilization"]:.1f}% available)',
            'recommendation': 'Should be safe to train, but monitor memory usage'
        }


def print_report():
    """Print comprehensive GPU usage report."""
    print("=" * 70)
    print("GPU MEMORY MONITORING REPORT")
    print("=" * 70)
    print()
    
    # CUDA availability
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    print()
    
    # GPU details
    gpu_info = get_gpu_info()
    if gpu_info:
        for gpu in gpu_info:
            print(f"--- GPU {gpu['device_id']}: {gpu['name']} ---")
            print(f"  Compute Capability: {gpu['compute_capability']}")
            print(f"  Total Memory: {gpu['total_memory_gb']:.2f} GB")
            print(f"  Allocated Memory: {gpu['allocated_memory_gb']:.2f} GB")
            print(f"  Reserved Memory: {gpu['reserved_memory_gb']:.2f} GB")
            print(f"  Free Memory: {gpu['free_memory_gb']:.2f} GB")
            print(f"  Memory Utilization: {gpu['memory_utilization']:.1f}%")
            print()
    else:
        print("No GPU detected!")
        print()
    
    # GPU processes
    processes = get_gpu_processes()
    if processes:
        print("--- GPU Processes ---")
        print(f"{'PID':<10} {'Memory (MB)':<15} {'Process Name'}")
        print("-" * 60)
        for proc in processes:
            print(f"{proc['pid']:<10} {proc['memory_mb']:<15.1f} {proc['name']}")
        print()
    elif processes is None:
        print("--- GPU Processes ---")
        print("(nvidia-smi not available or no processes found)")
        print()
    
    # Memory estimate
    memory_estimate = estimate_training_memory()
    if memory_estimate:
        print("--- Training Configuration ---")
        print(f"  Batch Size: {memory_estimate['batch_size']}")
        print(f"  Image Size: {memory_estimate['image_size']}x{memory_estimate['image_size']}")
        print(f"  Estimated Memory Required: ~{memory_estimate['estimated_memory_gb']:.2f} GB")
        print()
    
    # Safety assessment
    safety = check_training_safety(gpu_info, memory_estimate)
    print("=" * 70)
    print("SAFETY ASSESSMENT")
    print("=" * 70)
    print(f"Can Train: {'[YES]' if safety['can_train'] else '[NO]'}")
    print(f"Reason: {safety['reason']}")
    print(f"Recommendation: {safety['recommendation']}")
    print("=" * 70)
    
    return safety['can_train']


if __name__ == "__main__":
    can_train = print_report()
    exit(0 if can_train else 1)
