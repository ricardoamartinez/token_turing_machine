"""
Benchmark script for comparing CPU vs CUDA performance for TTM operations.

This script measures the performance of matrix multiplication, token embedding,
and transformer operations on both CPU and CUDA devices.
"""

import torch
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def benchmark_matrix_multiplication(device, size=1000, iterations=1000):
    """Benchmark matrix multiplication.

    Args:
        device: PyTorch device (CPU or CUDA)
        size: Size of matrices
        iterations: Number of iterations

    Returns:
        Average time per operation in milliseconds
    """
    print(f"Benchmarking matrix multiplication on {device}...")

    # Create random matrices
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # Warm-up
    for _ in range(10):
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

def benchmark_token_embedding(device, vocab_size=10000, embedding_dim=128, seq_len=100, batch_size=32, iterations=1000):
    """Benchmark token embedding operations.

    Args:
        device: PyTorch device (CPU or CUDA)
        vocab_size: Size of vocabulary
        embedding_dim: Embedding dimension
        seq_len: Sequence length
        batch_size: Batch size
        iterations: Number of iterations

    Returns:
        Average time per operation in milliseconds
    """
    print(f"Benchmarking token embedding on {device}...")

    # Create embedding layer
    embedding = torch.nn.Embedding(vocab_size, embedding_dim).to(device)

    # Create random token indices
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Warm-up
    for _ in range(10):
        _ = embedding(tokens)

    # Synchronize if using CUDA
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Measure time
    start_time = time.time()

    for _ in range(iterations):
        _ = embedding(tokens)

    # Synchronize if using CUDA
    if device.type == 'cuda':
        torch.cuda.synchronize()

    end_time = time.time()

    # Calculate average time per operation
    avg_time = (end_time - start_time) * 1000 / iterations

    print(f"Average time per operation: {avg_time:.4f} ms")

    return avg_time

def benchmark_transformer(device, dim=128, num_heads=4, seq_len=100, batch_size=32, iterations=100):
    """Benchmark transformer operations.

    Args:
        device: PyTorch device (CPU or CUDA)
        dim: Model dimension
        num_heads: Number of attention heads
        seq_len: Sequence length
        batch_size: Batch size
        iterations: Number of iterations

    Returns:
        Average time per operation in milliseconds
    """
    print(f"Benchmarking transformer on {device}...")

    # Create transformer layer
    transformer_layer = torch.nn.TransformerEncoderLayer(
        d_model=dim,
        nhead=num_heads,
        dim_feedforward=512,
        batch_first=True
    ).to(device)

    # Create random input
    x = torch.randn(batch_size, seq_len, dim, device=device)

    # Warm-up
    for _ in range(5):
        _ = transformer_layer(x)

    # Synchronize if using CUDA
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Measure time
    start_time = time.time()

    for _ in range(iterations):
        _ = transformer_layer(x)

    # Synchronize if using CUDA
    if device.type == 'cuda':
        torch.cuda.synchronize()

    end_time = time.time()

    # Calculate average time per operation
    avg_time = (end_time - start_time) * 1000 / iterations

    print(f"Average time per operation: {avg_time:.4f} ms")

    return avg_time

def plot_results(cpu_times, cuda_times, operation_names, output_path):
    """Plot benchmark results.

    Args:
        cpu_times: List of CPU times
        cuda_times: List of CUDA times
        operation_names: List of operation names
        output_path: Path to save the plot
    """
    # Calculate speedup factors
    speedup_factors = [cpu / cuda if cuda > 0 else float('inf') for cpu, cuda in zip(cpu_times, cuda_times)]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot times
    x = np.arange(len(operation_names))
    width = 0.35

    ax1.bar(x - width/2, cpu_times, width, label='CPU')
    ax1.bar(x + width/2, cuda_times, width, label='CUDA')

    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Operation Time (lower is better)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(operation_names)
    ax1.legend()

    # Plot speedup factors
    ax2.bar(x, speedup_factors, width)

    ax2.set_ylabel('Speedup Factor (CPU / CUDA)')
    ax2.set_title('CUDA Speedup (higher is better)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(operation_names)

    # Add speedup values on top of bars
    for i, v in enumerate(speedup_factors):
        ax2.text(i, v + 0.1, f"{v:.1f}x", ha='center')

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Benchmark CPU vs CUDA performance')
    parser.add_argument('--matrix_size', type=int, default=1000, help='Size of matrices for multiplication benchmark')
    parser.add_argument('--matrix_iterations', type=int, default=1000, help='Number of iterations for matrix multiplication benchmark')
    parser.add_argument('--embedding_iterations', type=int, default=1000, help='Number of iterations for token embedding benchmark')
    parser.add_argument('--transformer_iterations', type=int, default=100, help='Number of iterations for transformer benchmark')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # CPU device
    cpu_device = torch.device('cpu')

    # CUDA device
    cuda_device = torch.device('cuda') if cuda_available else None

    # Benchmark matrix multiplication on CPU
    cpu_matrix_time = benchmark_matrix_multiplication(
        device=cpu_device,
        size=args.matrix_size,
        iterations=args.matrix_iterations
    )

    # Benchmark token embedding on CPU
    cpu_embedding_time = benchmark_token_embedding(
        device=cpu_device,
        iterations=args.embedding_iterations
    )

    # Benchmark transformer on CPU
    cpu_transformer_time = benchmark_transformer(
        device=cpu_device,
        iterations=args.transformer_iterations
    )

    # Initialize CUDA times
    cuda_matrix_time = 0
    cuda_embedding_time = 0
    cuda_transformer_time = 0

    # Benchmark on CUDA if available
    if cuda_available:
        # Benchmark matrix multiplication on CUDA
        cuda_matrix_time = benchmark_matrix_multiplication(
            device=cuda_device,
            size=args.matrix_size,
            iterations=args.matrix_iterations
        )

        # Benchmark token embedding on CUDA
        cuda_embedding_time = benchmark_token_embedding(
            device=cuda_device,
            iterations=args.embedding_iterations
        )

        # Benchmark transformer on CUDA
        cuda_transformer_time = benchmark_transformer(
            device=cuda_device,
            iterations=args.transformer_iterations
        )

    # Collect results
    cpu_times = [cpu_matrix_time, cpu_embedding_time, cpu_transformer_time]
    cuda_times = [cuda_matrix_time, cuda_embedding_time, cuda_transformer_time]
    operation_names = ['Matrix Mult', 'Token Embedding', 'Transformer']

    # Plot results if CUDA is available
    if cuda_available:
        plot_results(
            cpu_times=cpu_times,
            cuda_times=cuda_times,
            operation_names=operation_names,
            output_path=output_dir / 'benchmark_results.png'
        )

    # Save results to file
    with open(output_dir / 'benchmark_results.txt', 'w') as f:
        f.write("Benchmark Results\n")
        f.write("=================\n\n")

        f.write("Matrix Multiplication:\n")
        f.write(f"  CPU: {cpu_matrix_time:.4f} ms\n")
        if cuda_available:
            f.write(f"  CUDA: {cuda_matrix_time:.4f} ms\n")
            f.write(f"  Speedup: {cpu_matrix_time / cuda_matrix_time:.2f}x\n")
        f.write("\n")

        f.write("Token Embedding:\n")
        f.write(f"  CPU: {cpu_embedding_time:.4f} ms\n")
        if cuda_available:
            f.write(f"  CUDA: {cuda_embedding_time:.4f} ms\n")
            f.write(f"  Speedup: {cpu_embedding_time / cuda_embedding_time:.2f}x\n")
        f.write("\n")

        f.write("Transformer:\n")
        f.write(f"  CPU: {cpu_transformer_time:.4f} ms\n")
        if cuda_available:
            f.write(f"  CUDA: {cuda_transformer_time:.4f} ms\n")
            f.write(f"  Speedup: {cpu_transformer_time / cuda_transformer_time:.2f}x\n")

    print(f"Results saved to {output_dir / 'benchmark_results.txt'}")

if __name__ == '__main__':
    main()
