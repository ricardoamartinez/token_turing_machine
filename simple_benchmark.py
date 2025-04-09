"""
Simple benchmark script for comparing CPU vs CUDA performance.
"""

import torch
import time

def benchmark_matrix_multiplication(device, size=100, iterations=100):
    """Benchmark matrix multiplication."""
    print(f"Benchmarking matrix multiplication on {device}...")

    # Create random matrices
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # Warm-up
    for _ in range(5):
        _ = torch.matmul(a, b)

    # Synchronize if using CUDA
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Measure time
    start_time = time.time()

    for _ in range(iterations):
        _ = torch.matmul(a, b)

    # Synchronize if using CUDA
    if device.type == 'cuda':
        torch.cuda.synchronize()

    end_time = time.time()

    # Calculate average time per operation
    avg_time = (end_time - start_time) * 1000 / iterations

    print(f"Average time per operation: {avg_time:.4f} ms")

    return avg_time

def main():
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # CPU device
    cpu_device = torch.device('cpu')

    # Benchmark matrix multiplication on CPU
    cpu_time = benchmark_matrix_multiplication(device=cpu_device)

    # Benchmark on CUDA if available
    if cuda_available:
        # CUDA device
        cuda_device = torch.device('cuda')

        # Benchmark matrix multiplication on CUDA
        cuda_time = benchmark_matrix_multiplication(device=cuda_device)

        # Calculate speedup
        speedup = cpu_time / cuda_time if cuda_time > 0 else float('inf')
        print(f"CUDA speedup: {speedup:.2f}x")

    # Save results to file
    with open('results/matrix_benchmark.txt', 'w') as f:
        f.write("Matrix Multiplication Benchmark Results\n")
        f.write("=====================================\n\n")
        f.write(f"CPU: {cpu_time:.4f} ms\n")

        if cuda_available:
            f.write(f"CUDA: {cuda_time:.4f} ms\n")
            f.write(f"Speedup: {speedup:.2f}x\n")

    print("Results saved to results/matrix_benchmark.txt")

if __name__ == '__main__':
    main()
