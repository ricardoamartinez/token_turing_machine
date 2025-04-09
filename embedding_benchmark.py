"""
Benchmark script for token embedding operations.
"""

import torch
import time

def benchmark_token_embedding(device, vocab_size=10000, embedding_dim=128, seq_len=100, batch_size=32, iterations=100):
    """Benchmark token embedding operations."""
    print(f"Benchmarking token embedding on {device}...")
    
    # Create embedding layer
    embedding = torch.nn.Embedding(vocab_size, embedding_dim).to(device)
    
    # Create random token indices
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # Warm-up
    for _ in range(5):
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

def main():
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # CPU device
    cpu_device = torch.device('cpu')
    
    # Benchmark token embedding on CPU
    cpu_time = benchmark_token_embedding(device=cpu_device)
    
    # Benchmark on CUDA if available
    if cuda_available:
        # CUDA device
        cuda_device = torch.device('cuda')
        
        # Benchmark token embedding on CUDA
        cuda_time = benchmark_token_embedding(device=cuda_device)
        
        # Calculate speedup
        speedup = cpu_time / cuda_time if cuda_time > 0 else float('inf')
        print(f"CUDA speedup: {speedup:.2f}x")
    
    # Save results to file
    with open('results/embedding_benchmark.txt', 'w') as f:
        f.write("Token Embedding Benchmark Results\n")
        f.write("================================\n\n")
        f.write(f"CPU: {cpu_time:.4f} ms\n")
        
        if cuda_available:
            f.write(f"CUDA: {cuda_time:.4f} ms\n")
            f.write(f"Speedup: {speedup:.2f}x\n")
    
    print("Results saved to results/embedding_benchmark.txt")

if __name__ == '__main__':
    main()
