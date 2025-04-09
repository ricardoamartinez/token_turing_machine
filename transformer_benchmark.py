"""
Benchmark script for transformer operations.
"""

import torch
import time

def benchmark_transformer(device, dim=128, num_heads=4, seq_len=100, batch_size=32, iterations=10):
    """Benchmark transformer operations."""
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
    for _ in range(3):
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

def main():
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # CPU device
    cpu_device = torch.device('cpu')
    
    # Benchmark transformer on CPU
    cpu_time = benchmark_transformer(device=cpu_device)
    
    # Benchmark on CUDA if available
    if cuda_available:
        # CUDA device
        cuda_device = torch.device('cuda')
        
        # Benchmark transformer on CUDA
        cuda_time = benchmark_transformer(device=cuda_device)
        
        # Calculate speedup
        speedup = cpu_time / cuda_time if cuda_time > 0 else float('inf')
        print(f"CUDA speedup: {speedup:.2f}x")
    
    # Save results to file
    with open('results/transformer_benchmark.txt', 'w') as f:
        f.write("Transformer Benchmark Results\n")
        f.write("============================\n\n")
        f.write(f"CPU: {cpu_time:.4f} ms\n")
        
        if cuda_available:
            f.write(f"CUDA: {cuda_time:.4f} ms\n")
            f.write(f"Speedup: {speedup:.2f}x\n")
    
    print("Results saved to results/transformer_benchmark.txt")

if __name__ == '__main__':
    main()
