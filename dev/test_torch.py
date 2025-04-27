import torch
import time

def test_torch_installation():
    print("=== Torch Installation Test ===")
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print()

def test_matrix_multiplication(device):
    print(f"=== Matrix Multiplication on {device} ===")
    size = 1000  # Small size for quick test
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize() if device == 'cuda' else None
    end = time.time()

    print(f"Time taken: {end - start:.6f} seconds")
    print()

def main():
    test_torch_installation()

    # Test on CPU
    test_matrix_multiplication(device="cpu")

    # Test on GPU if available
    if torch.cuda.is_available():
        test_matrix_multiplication(device="cuda")
    else:
        print("No GPU available, skipping GPU test.")

if __name__ == "__main__":
    main()
