"""
Performance optimization utilities for the Token Turing Machine (TTM) model.

This module provides utilities for measuring and optimizing the performance
of the TTM model, including FLOPS counting, memory usage tracking, and
benchmarking.
"""

import torch
import torch.nn as nn
import torch.jit
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
import os
import psutil
import gc
from functools import wraps
from contextlib import contextmanager

from ..models.ttm_model import TokenTuringMachine


class FLOPSCounter:
    """Count floating point operations (FLOPS) for a PyTorch model."""

    def __init__(self):
        """Initialize the FLOPS counter."""
        self.flops = 0
        self.hooks = []

    def count_conv2d(self, module, input, output):
        """Count FLOPS for Conv2D operation."""
        batch_size = input[0].shape[0]
        input_channels = module.in_channels
        output_channels = module.out_channels
        kernel_size = module.kernel_size[0] * module.kernel_size[1]
        output_size = output.shape[2] * output.shape[3]

        # Each output element requires (input_channels * kernel_size) multiplications and additions
        flops_per_element = 2 * input_channels * kernel_size
        total_flops = batch_size * output_channels * output_size * flops_per_element

        self.flops += total_flops

    def count_linear(self, module, input, output):
        """Count FLOPS for Linear operation."""
        batch_size = input[0].shape[0]
        input_features = module.in_features
        output_features = module.out_features

        # Each output element requires input_features multiplications and additions
        flops_per_element = 2 * input_features
        total_flops = batch_size * output_features * flops_per_element

        self.flops += total_flops

    def count_matmul(self, module, input, output):
        """Count FLOPS for matrix multiplication."""
        # For attention: Q * K^T, softmax, and V
        if len(input) == 2:
            batch_size = input[0].shape[0]
            m, k1 = input[0].shape[-2:]
            k2, n = input[1].shape[-2:]

            # Each output element requires k multiplications and additions
            flops_per_element = 2 * k1
            total_flops = batch_size * m * n * flops_per_element

            self.flops += total_flops

    def count_layernorm(self, module, input, output):
        """Count FLOPS for LayerNorm operation."""
        batch_size = input[0].shape[0]
        features = input[0].shape[-1]
        sequence_length = input[0].shape[1] if len(input[0].shape) > 2 else 1

        # Each element requires 2 additions (mean, variance), 1 sqrt, 1 division, 1 multiplication, 1 addition
        flops_per_element = 6
        total_flops = batch_size * sequence_length * features * flops_per_element

        self.flops += total_flops

    def count_attention(self, module, input, output):
        """Count FLOPS for self-attention operation."""
        # This is a custom hook for the attention mechanism
        # Assuming input is (batch_size, seq_len, embedding_dim)
        batch_size = input[0].shape[0]
        seq_len = input[0].shape[1]
        embedding_dim = input[0].shape[2]

        # Q, K, V projections
        projection_flops = 3 * (2 * embedding_dim * embedding_dim)

        # Q * K^T
        qk_flops = 2 * batch_size * seq_len * seq_len * embedding_dim

        # Softmax
        softmax_flops = batch_size * seq_len * seq_len * 4  # exp, sum, div

        # Attention * V
        av_flops = 2 * batch_size * seq_len * seq_len * embedding_dim

        # Output projection
        output_flops = 2 * batch_size * seq_len * embedding_dim * embedding_dim

        total_flops = projection_flops + qk_flops + softmax_flops + av_flops + output_flops

        self.flops += total_flops

    def register_hooks(self, model):
        """Register hooks for all modules in the model."""
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                self.hooks.append(module.register_forward_hook(self.count_conv2d))
            elif isinstance(module, nn.Linear):
                self.hooks.append(module.register_forward_hook(self.count_linear))
            elif isinstance(module, nn.LayerNorm):
                self.hooks.append(module.register_forward_hook(self.count_layernorm))
            elif "MultiheadAttention" in module.__class__.__name__:
                self.hooks.append(module.register_forward_hook(self.count_attention))

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def reset(self):
        """Reset the FLOPS counter."""
        self.flops = 0


def measure_flops(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: Optional[torch.device] = None
) -> int:
    """Measure the number of floating point operations (FLOPS) for a model.

    Args:
        model: The model to measure
        input_shape: The shape of the input tensor
        device: The device to run the model on

    Returns:
        Number of FLOPS
    """
    # Set device
    if device is None:
        device = next(model.parameters()).device

    # Create dummy input
    if isinstance(model, TokenTuringMachine):
        # For TTM, input should be integer tokens within vocab_size range
        vocab_size = model.vocab_size if hasattr(model, 'vocab_size') else 100
        dummy_input = torch.randint(0, vocab_size - 1, input_shape, device=device)
    else:
        # For other models, input can be float tensors
        dummy_input = torch.randn(*input_shape, device=device)

    # Create FLOPS counter
    counter = FLOPSCounter()

    # Register hooks
    counter.register_hooks(model)

    # Run model
    with torch.no_grad():
        model(dummy_input)

    # Get FLOPS
    flops = counter.flops

    # Remove hooks
    counter.remove_hooks()

    return flops


@contextmanager
def track_memory_usage():
    """Context manager to track memory usage.

    Yields:
        Dictionary with memory usage statistics
    """
    # Get initial memory usage
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    process = psutil.Process(os.getpid())
    initial_cpu_memory = process.memory_info().rss / (1024 * 1024)  # MB

    memory_stats = {}

    try:
        yield memory_stats
    finally:
        # Get final memory usage
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        # CPU memory
        final_cpu_memory = process.memory_info().rss / (1024 * 1024)  # MB
        cpu_memory_diff = final_cpu_memory - initial_cpu_memory

        # GPU memory
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 * 1024)  # MB
            gpu_max_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
        else:
            gpu_memory_allocated = 0
            gpu_memory_reserved = 0
            gpu_max_memory = 0

        # Update memory usage statistics
        memory_stats.update({
            'cpu_memory_initial': initial_cpu_memory,
            'cpu_memory_final': final_cpu_memory,
            'cpu_memory_diff': cpu_memory_diff,
            'gpu_memory_allocated': gpu_memory_allocated,
            'gpu_memory_reserved': gpu_memory_reserved,
            'gpu_max_memory': gpu_max_memory
        })


def measure_memory(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: Optional[torch.device] = None,
    backward: bool = True
) -> Dict[str, float]:
    """Measure memory usage for a model.

    Args:
        model: The model to measure
        input_shape: The shape of the input tensor
        device: The device to run the model on
        backward: Whether to perform backward pass

    Returns:
        Dictionary with memory usage statistics
    """
    # Set device
    if device is None:
        device = next(model.parameters()).device

    # Create dummy input
    if isinstance(model, TokenTuringMachine):
        # For TTM, input should be integer tokens within vocab_size range
        vocab_size = model.vocab_size if hasattr(model, 'vocab_size') else 100
        dummy_input = torch.randint(0, vocab_size - 1, input_shape, device=device)
    else:
        # For other models, input can be float tensors
        dummy_input = torch.randn(*input_shape, device=device)

    # Clear cache
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()

    # Measure memory usage
    with track_memory_usage() as memory_stats:
        # Forward pass
        output = model(dummy_input)

        # Backward pass
        if backward:
            if isinstance(output, tuple):
                output = output[0]

            if output.requires_grad:
                loss = output.sum()
                loss.backward()

    return memory_stats


def benchmark_forward(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: Optional[torch.device] = None,
    num_iterations: int = 100,
    warmup_iterations: int = 10
) -> Dict[str, float]:
    """Benchmark forward pass for a model.

    Args:
        model: The model to benchmark
        input_shape: The shape of the input tensor
        device: The device to run the model on
        num_iterations: Number of iterations to run
        warmup_iterations: Number of warmup iterations

    Returns:
        Dictionary with benchmark results
    """
    # Set device
    if device is None:
        device = next(model.parameters()).device

    # Create dummy input
    if isinstance(model, TokenTuringMachine):
        # For TTM, input should be integer tokens within vocab_size range
        vocab_size = model.vocab_size if hasattr(model, 'vocab_size') else 100
        dummy_input = torch.randint(0, vocab_size - 1, input_shape, device=device)
    else:
        # For other models, input can be float tensors
        dummy_input = torch.randn(*input_shape, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iterations):
            model(dummy_input)

    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_iterations):
            model(dummy_input)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()

    # Calculate results
    total_time = end_time - start_time
    avg_time = total_time / num_iterations

    return {
        'total_time': total_time,
        'avg_time': avg_time,
        'iterations': num_iterations
    }


def benchmark_sequence_length(
    model: nn.Module,
    batch_size: int,
    seq_lengths: List[int],
    device: Optional[torch.device] = None,
    num_iterations: int = 10,
    warmup_iterations: int = 2
) -> Dict[int, Dict[str, float]]:
    """Benchmark model performance for different sequence lengths.

    Args:
        model: The model to benchmark
        batch_size: Batch size
        seq_lengths: List of sequence lengths to benchmark
        device: The device to run the model on
        num_iterations: Number of iterations to run
        warmup_iterations: Number of warmup iterations

    Returns:
        Dictionary mapping sequence length to benchmark results
    """
    # Set device
    if device is None:
        device = next(model.parameters()).device

    # Benchmark for each sequence length
    results = {}

    for seq_len in seq_lengths:
        # Create input shape
        if isinstance(model, TokenTuringMachine):
            input_shape = (batch_size, seq_len)
        else:
            # Assume standard Transformer with embedding dimension
            embedding_dim = model.config.hidden_size if hasattr(model, 'config') else 512
            input_shape = (batch_size, seq_len, embedding_dim)

        # Benchmark
        result = benchmark_forward(
            model=model,
            input_shape=input_shape,
            device=device,
            num_iterations=num_iterations,
            warmup_iterations=warmup_iterations
        )

        # Add FLOPS
        result['flops'] = measure_flops(model, input_shape, device)

        # Add memory usage
        memory_stats = measure_memory(model, input_shape, device, backward=False)
        result.update(memory_stats)

        # Store result
        results[seq_len] = result

    return results


def plot_sequence_length_benchmark(
    results: Dict[int, Dict[str, float]],
    output_dir: str = './outputs',
    filename: str = 'sequence_length_benchmark.png'
) -> None:
    """Plot benchmark results for different sequence lengths.

    Args:
        results: Dictionary mapping sequence length to benchmark results
        output_dir: Directory to save the plot
        filename: Filename for the plot
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Extract data
    seq_lengths = sorted(results.keys())
    avg_times = [results[seq_len]['avg_time'] for seq_len in seq_lengths]
    flops = [results[seq_len]['flops'] for seq_len in seq_lengths]
    memory = [results[seq_len]['gpu_memory_allocated'] for seq_len in seq_lengths]

    # Create figure with multiple subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # Plot average time
    axs[0].plot(seq_lengths, avg_times, 'o-')
    axs[0].set_xlabel('Sequence Length')
    axs[0].set_ylabel('Average Time (s)')
    axs[0].set_title('Average Forward Pass Time vs Sequence Length')
    axs[0].grid(True)

    # Plot FLOPS
    axs[1].plot(seq_lengths, flops, 'o-')
    axs[1].set_xlabel('Sequence Length')
    axs[1].set_ylabel('FLOPS')
    axs[1].set_title('FLOPS vs Sequence Length')
    axs[1].grid(True)

    # Plot memory usage
    axs[2].plot(seq_lengths, memory, 'o-')
    axs[2].set_xlabel('Sequence Length')
    axs[2].set_ylabel('GPU Memory (MB)')
    axs[2].set_title('GPU Memory vs Sequence Length')
    axs[2].grid(True)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def compare_cpu_cuda(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    num_iterations: int = 10,
    warmup_iterations: int = 2
) -> Dict[str, Dict[str, float]]:
    """Compare model performance on CPU vs CUDA.

    Args:
        model: The model to benchmark
        input_shape: The shape of the input tensor
        num_iterations: Number of iterations to run
        warmup_iterations: Number of warmup iterations

    Returns:
        Dictionary with benchmark results for CPU and CUDA
    """
    # Check if CUDA is available
    if not torch.cuda.is_available():
        return {'cpu': benchmark_forward(model, input_shape, torch.device('cpu'), num_iterations, warmup_iterations)}

    # Benchmark on CPU
    model.to('cpu')
    cpu_results = benchmark_forward(model, input_shape, torch.device('cpu'), num_iterations, warmup_iterations)

    # Benchmark on CUDA
    model.to('cuda')
    cuda_results = benchmark_forward(model, input_shape, torch.device('cuda'), num_iterations, warmup_iterations)

    # Calculate speedup
    if cuda_results['avg_time'] > 0:
        speedup = cpu_results['avg_time'] / cuda_results['avg_time']
    else:
        speedup = 1.0
    cuda_results['speedup'] = speedup

    return {
        'cpu': cpu_results,
        'cuda': cuda_results
    }


def compare_ttm_transformer(
    ttm_model: TokenTuringMachine,
    transformer_model: nn.Module,
    batch_size: int,
    seq_lengths: List[int],
    device: Optional[torch.device] = None,
    num_iterations: int = 10,
    warmup_iterations: int = 2
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Compare TTM and standard Transformer performance for different sequence lengths.

    Args:
        ttm_model: The TTM model
        transformer_model: The standard Transformer model
        batch_size: Batch size
        seq_lengths: List of sequence lengths to benchmark
        device: The device to run the models on
        num_iterations: Number of iterations to run
        warmup_iterations: Number of warmup iterations

    Returns:
        Dictionary with benchmark results for TTM and Transformer
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move models to device
    ttm_model.to(device)
    transformer_model.to(device)

    # Benchmark TTM
    ttm_results = benchmark_sequence_length(
        model=ttm_model,
        batch_size=batch_size,
        seq_lengths=seq_lengths,
        device=device,
        num_iterations=num_iterations,
        warmup_iterations=warmup_iterations
    )

    # Benchmark Transformer
    transformer_results = benchmark_sequence_length(
        model=transformer_model,
        batch_size=batch_size,
        seq_lengths=seq_lengths,
        device=device,
        num_iterations=num_iterations,
        warmup_iterations=warmup_iterations
    )

    return {
        'ttm': ttm_results,
        'transformer': transformer_results
    }


def plot_ttm_transformer_comparison(
    results: Dict[str, Dict[int, Dict[str, float]]],
    output_dir: str = './outputs',
    filename: str = 'ttm_transformer_comparison.png'
) -> None:
    """Plot comparison of TTM and standard Transformer.

    Args:
        results: Dictionary with benchmark results for TTM and Transformer
        output_dir: Directory to save the plot
        filename: Filename for the plot
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Extract data
    seq_lengths = sorted(results['ttm'].keys())
    ttm_times = [results['ttm'][seq_len]['avg_time'] for seq_len in seq_lengths]
    transformer_times = [results['transformer'][seq_len]['avg_time'] for seq_len in seq_lengths]

    ttm_flops = [results['ttm'][seq_len]['flops'] for seq_len in seq_lengths]
    transformer_flops = [results['transformer'][seq_len]['flops'] for seq_len in seq_lengths]

    ttm_memory = [results['ttm'][seq_len]['gpu_memory_allocated'] for seq_len in seq_lengths]
    transformer_memory = [results['transformer'][seq_len]['gpu_memory_allocated'] for seq_len in seq_lengths]

    # Create figure with multiple subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # Plot average time
    axs[0].plot(seq_lengths, ttm_times, 'o-', label='TTM')
    axs[0].plot(seq_lengths, transformer_times, 's-', label='Transformer')
    axs[0].set_xlabel('Sequence Length')
    axs[0].set_ylabel('Average Time (s)')
    axs[0].set_title('Average Forward Pass Time vs Sequence Length')
    axs[0].legend()
    axs[0].grid(True)

    # Plot FLOPS
    axs[1].plot(seq_lengths, ttm_flops, 'o-', label='TTM')
    axs[1].plot(seq_lengths, transformer_flops, 's-', label='Transformer')
    axs[1].set_xlabel('Sequence Length')
    axs[1].set_ylabel('FLOPS')
    axs[1].set_title('FLOPS vs Sequence Length')
    axs[1].legend()
    axs[1].grid(True)

    # Plot memory usage
    axs[2].plot(seq_lengths, ttm_memory, 'o-', label='TTM')
    axs[2].plot(seq_lengths, transformer_memory, 's-', label='Transformer')
    axs[2].set_xlabel('Sequence Length')
    axs[2].set_ylabel('GPU Memory (MB)')
    axs[2].set_title('GPU Memory vs Sequence Length')
    axs[2].legend()
    axs[2].grid(True)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def optimize_batch_size(
    model: nn.Module,
    seq_len: int,
    batch_sizes: List[int],
    device: Optional[torch.device] = None,
    num_iterations: int = 10,
    warmup_iterations: int = 2
) -> Dict[int, Dict[str, float]]:
    """Optimize batch size for a model.

    Args:
        model: The model to benchmark
        seq_len: Sequence length
        batch_sizes: List of batch sizes to benchmark
        device: The device to run the model on
        num_iterations: Number of iterations to run
        warmup_iterations: Number of warmup iterations

    Returns:
        Dictionary mapping batch size to benchmark results
    """
    # Set device
    if device is None:
        device = next(model.parameters()).device

    # Benchmark for each batch size
    results = {}

    for batch_size in batch_sizes:
        try:
            # Create input shape
            if isinstance(model, TokenTuringMachine):
                input_shape = (batch_size, seq_len)
            else:
                # Assume standard Transformer with embedding dimension
                embedding_dim = model.config.hidden_size if hasattr(model, 'config') else 512
                input_shape = (batch_size, seq_len, embedding_dim)

            # Benchmark
            result = benchmark_forward(
                model=model,
                input_shape=input_shape,
                device=device,
                num_iterations=num_iterations,
                warmup_iterations=warmup_iterations
            )

            # Add memory usage
            memory_stats = measure_memory(model, input_shape, device, backward=True)
            result.update(memory_stats)

            # Calculate throughput (examples per second)
            result['throughput'] = batch_size / result['avg_time']

            # Store result
            results[batch_size] = result
        except RuntimeError as e:
            # Out of memory
            print(f"Batch size {batch_size} failed with error: {e}")
            break

    return results


def plot_batch_size_optimization(
    results: Dict[int, Dict[str, float]],
    output_dir: str = './outputs',
    filename: str = 'batch_size_optimization.png'
) -> None:
    """Plot batch size optimization results.

    Args:
        results: Dictionary mapping batch size to benchmark results
        output_dir: Directory to save the plot
        filename: Filename for the plot
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Extract data
    batch_sizes = sorted(results.keys())
    avg_times = [results[batch_size]['avg_time'] for batch_size in batch_sizes]
    throughputs = [results[batch_size]['throughput'] for batch_size in batch_sizes]
    memory = [results[batch_size]['gpu_memory_allocated'] for batch_size in batch_sizes]

    # Create figure with multiple subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # Plot average time
    axs[0].plot(batch_sizes, avg_times, 'o-')
    axs[0].set_xlabel('Batch Size')
    axs[0].set_ylabel('Average Time (s)')
    axs[0].set_title('Average Forward Pass Time vs Batch Size')
    axs[0].grid(True)

    # Plot throughput
    axs[1].plot(batch_sizes, throughputs, 'o-')
    axs[1].set_xlabel('Batch Size')
    axs[1].set_ylabel('Throughput (examples/s)')
    axs[1].set_title('Throughput vs Batch Size')
    axs[1].grid(True)

    # Plot memory usage
    axs[2].plot(batch_sizes, memory, 'o-')
    axs[2].set_xlabel('Batch Size')
    axs[2].set_ylabel('GPU Memory (MB)')
    axs[2].set_title('GPU Memory vs Batch Size')
    axs[2].grid(True)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


@torch.jit.script
def jit_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """JIT-compiled attention mechanism.

    Args:
        query: Query tensor of shape [batch_size, num_heads, seq_len, head_dim]
        key: Key tensor of shape [batch_size, num_heads, seq_len, head_dim]
        value: Value tensor of shape [batch_size, num_heads, seq_len, head_dim]
        mask: Optional mask tensor of shape [batch_size, 1, seq_len, seq_len]

    Returns:
        Output tensor of shape [batch_size, num_heads, seq_len, head_dim]
    """
    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(query.size(-1), dtype=query.dtype))

    # Apply mask
    if mask is not None:
        scores = scores + mask

    # Apply softmax
    attention_weights = torch.softmax(scores, dim=-1)

    # Apply attention to values
    output = torch.matmul(attention_weights, value)

    return output


@torch.jit.script
def jit_memory_read(memory: torch.Tensor, read_weights: torch.Tensor) -> torch.Tensor:
    """JIT-compiled memory read operation.

    Args:
        memory: Memory tensor of shape [batch_size, memory_size, embedding_dim]
        read_weights: Read weights of shape [batch_size, r, memory_size]

    Returns:
        Read vectors of shape [batch_size, r, embedding_dim]
    """
    # Compute read vectors
    read_vectors = torch.bmm(read_weights, memory)

    return read_vectors


@torch.jit.script
def jit_memory_write(memory: torch.Tensor, write_weights: torch.Tensor, write_vectors: torch.Tensor) -> torch.Tensor:
    """JIT-compiled memory write operation.

    Args:
        memory: Memory tensor of shape [batch_size, memory_size, embedding_dim]
        write_weights: Write weights of shape [batch_size, r, memory_size]
        write_vectors: Write vectors of shape [batch_size, r, embedding_dim]

    Returns:
        Updated memory of shape [batch_size, memory_size, embedding_dim]
    """
    # Compute memory update
    batch_size, memory_size, embedding_dim = memory.shape
    r = write_weights.shape[1]

    # Reshape write weights for broadcasting
    write_weights_expanded = write_weights.transpose(1, 2).unsqueeze(-1)  # [batch_size, memory_size, r, 1]

    # Reshape write vectors for broadcasting
    write_vectors_expanded = write_vectors.unsqueeze(1)  # [batch_size, 1, r, embedding_dim]

    # Compute weighted write vectors
    weighted_write_vectors = write_weights_expanded * write_vectors_expanded  # [batch_size, memory_size, r, embedding_dim]

    # Sum over r dimension
    memory_update = weighted_write_vectors.sum(dim=2)  # [batch_size, memory_size, embedding_dim]

    # Update memory
    updated_memory = memory + memory_update

    return updated_memory
