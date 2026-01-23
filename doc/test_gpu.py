import torch

def test_torch_gpu():
    """
    Test PyTorch GPU availability and functionality.
    """
    print("=" * 50)
    print("PyTorch GPU Test")
    print("=" * 50)
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if not cuda_available:
        print("WARNING: CUDA is not available. GPU tests cannot be performed.")
        return False
    
    # Get CUDA device information
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    # Get current device
    current_device = torch.cuda.current_device()
    print(f"Current device: {current_device}")
    
    # Get device name
    device_name = torch.cuda.get_device_name(current_device)
    print(f"Device name: {device_name}")
    
    # Test basic tensor operations on GPU
    print("\n" + "-" * 50)
    print("Testing GPU operations...")
    print("-" * 50)
    
    try:
        # Create tensors on GPU
        device = torch.device("cuda:0")
        print(f"Using device: {device}")
        
        # Create random tensors
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        print(f"Created tensors on GPU: {a.shape}")
        
        # Perform matrix multiplication
        print("Performing matrix multiplication on GPU...")
        c = torch.matmul(a, b)
        print(f"Result shape: {c.shape}")
        
        # Perform element-wise operations
        print("Performing element-wise operations...")
        d = a * b + c
        print(f"Result shape: {d.shape}")
        
        # Compute sum to ensure computation is done
        result_sum = d.sum().item()
        print(f"Sum of result tensor: {result_sum:.4f}")
        
        # Test memory allocation
        print("\nGPU Memory Info:")
        print(f"Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        print(f"Reserved: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
        
        print("\n" + "=" * 50)
        print("✓ GPU test PASSED!")
        print("=" * 50)
        return True
        
    except Exception as e:
        print(f"\n✗ GPU test FAILED with error: {e}")
        print("=" * 50)
        return False


if __name__ == "__main__":
    test_torch_gpu()