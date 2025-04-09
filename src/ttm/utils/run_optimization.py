"""
Script to run optimization for the Token Turing Machine (TTM) model.
"""

import torch
import torch.nn as nn
import argparse
import os
import json
import logging
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
import matplotlib.pyplot as plt
from datetime import datetime

from ..models.ttm_model import TokenTuringMachine
from .performance import (
    measure_flops,
    measure_memory,
    benchmark_forward,
    benchmark_sequence_length,
    plot_sequence_length_benchmark,
    compare_cpu_cuda
)
from .optimize import (
    create_jit_model,
    compare_jit_performance,
    quantize_model,
    compare_quantization_performance,
    optimize_ttm_model,
    benchmark_optimized_models
)


def setup_logging(log_dir: str) -> logging.Logger:
    """Set up logging.
    
    Args:
        log_dir: Directory to save logs
        
    Returns:
        Logger
    """
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('ttm_optimization')
    logger.setLevel(logging.INFO)
    
    # Create file handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(os.path.join(log_dir, f'optimization_{timestamp}.log'))
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def create_ttm_model(
    vocab_size: int = 128,
    embedding_dim: int = 128,
    memory_size: int = 16,
    r: int = 4,
    num_layers: int = 2,
    num_heads: int = 4,
    hidden_dim: int = 256,
    dropout: float = 0.1,
    device: Optional[torch.device] = None
) -> TokenTuringMachine:
    """Create a TTM model for optimization.
    
    Args:
        vocab_size: Vocabulary size
        embedding_dim: Embedding dimension
        memory_size: Memory size
        r: Number of memory slots to read/write
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        hidden_dim: Hidden dimension
        dropout: Dropout rate
        device: Device to create the model on
        
    Returns:
        TTM model
    """
    # Create model
    model = TokenTuringMachine(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        memory_size=memory_size,
        r=r,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        dropout=dropout
    )
    
    # Move to device
    if device is not None:
        model.to(device)
    
    return model


def plot_optimization_results(
    results: Dict[str, Dict[str, float]],
    output_dir: str = './outputs',
    filename: str = 'optimization_results.png'
) -> None:
    """Plot optimization benchmark results.
    
    Args:
        results: Dictionary mapping optimization type to benchmark results
        output_dir: Directory to save the plot
        filename: Filename for the plot
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    optimization_types = list(results.keys())
    avg_times = [results[opt_type]['avg_time'] for opt_type in optimization_types]
    speedups = [results[opt_type].get('speedup', 1.0) for opt_type in optimization_types]
    
    # Create figure with multiple subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot average time
    axs[0].bar(optimization_types, avg_times)
    axs[0].set_xlabel('Optimization Type')
    axs[0].set_ylabel('Average Time (s)')
    axs[0].set_title('Average Forward Pass Time by Optimization Type')
    axs[0].grid(True)
    
    # Plot speedup
    axs[1].bar(optimization_types[1:], speedups[1:])
    axs[1].set_xlabel('Optimization Type')
    axs[1].set_ylabel('Speedup (x)')
    axs[1].set_title('Speedup Relative to Original Model')
    axs[1].grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def run_jit_optimization(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: Optional[torch.device] = None,
    output_dir: str = './outputs',
    logger: Optional[logging.Logger] = None
) -> Dict[str, Dict[str, float]]:
    """Run JIT optimization for a model.
    
    Args:
        model: The model to optimize
        input_shape: The shape of the input tensor
        device: The device to run the model on
        output_dir: Directory to save results
        logger: Optional logger
        
    Returns:
        Dictionary with benchmark results for original and JIT-compiled models
    """
    # Set device
    if device is None:
        device = next(model.parameters()).device
    
    # Log start
    if logger is not None:
        logger.info(f"Running JIT optimization...")
    
    # Run optimization
    results = compare_jit_performance(
        model=model,
        input_shape=input_shape,
        device=device
    )
    
    # Log results
    if logger is not None:
        logger.info(f"JIT optimization results:")
        logger.info(f"  Original average time: {results['original']['avg_time']:.6f} s")
        logger.info(f"  JIT average time: {results['jit']['avg_time']:.6f} s")
        logger.info(f"  Speedup: {results['jit']['speedup']:.2f}x")
    
    # Save results
    with open(os.path.join(output_dir, 'jit_optimization.json'), 'w') as f:
        # Convert results to JSON-serializable format
        json_results = {}
        for model_type, model_results in results.items():
            json_results[model_type] = {k: float(v) if isinstance(v, torch.Tensor) else v for k, v in model_results.items()}
        
        json.dump(json_results, f, indent=4)
    
    return results


def run_quantization_optimization(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: Optional[torch.device] = None,
    output_dir: str = './outputs',
    logger: Optional[logging.Logger] = None
) -> Dict[str, Dict[str, float]]:
    """Run quantization optimization for a model.
    
    Args:
        model: The model to optimize
        input_shape: The shape of the input tensor
        device: The device to run the model on
        output_dir: Directory to save results
        logger: Optional logger
        
    Returns:
        Dictionary with benchmark results for original and quantized models
    """
    # Set device
    if device is None:
        device = next(model.parameters()).device
    
    # Log start
    if logger is not None:
        logger.info(f"Running quantization optimization...")
    
    # Run optimization
    results = compare_quantization_performance(
        model=model,
        input_shape=input_shape,
        device=device
    )
    
    # Log results
    if logger is not None:
        logger.info(f"Quantization optimization results:")
        logger.info(f"  Original average time: {results['original']['avg_time']:.6f} s")
        logger.info(f"  Quantized average time: {results['quantized']['avg_time']:.6f} s")
        logger.info(f"  Speedup: {results['quantized']['speedup']:.2f}x")
    
    # Save results
    with open(os.path.join(output_dir, 'quantization_optimization.json'), 'w') as f:
        # Convert results to JSON-serializable format
        json_results = {}
        for model_type, model_results in results.items():
            json_results[model_type] = {k: float(v) if isinstance(v, torch.Tensor) else v for k, v in model_results.items()}
        
        json.dump(json_results, f, indent=4)
    
    return results


def run_all_optimizations(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: Optional[torch.device] = None,
    output_dir: str = './outputs',
    logger: Optional[logging.Logger] = None
) -> Dict[str, Dict[str, float]]:
    """Run all optimizations for a model.
    
    Args:
        model: The model to optimize
        input_shape: The shape of the input tensor
        device: The device to run the model on
        output_dir: Directory to save results
        logger: Optional logger
        
    Returns:
        Dictionary with benchmark results for all optimization types
    """
    # Set device
    if device is None:
        device = next(model.parameters()).device
    
    # Log start
    if logger is not None:
        logger.info(f"Running all optimizations...")
    
    # Optimize model
    optimized_models = optimize_ttm_model(
        model=model,
        input_shape=input_shape,
        device=device,
        jit=True,
        quantize=True,
        output_dir=output_dir
    )
    
    # Benchmark optimized models
    results = benchmark_optimized_models(
        optimized_models=optimized_models,
        input_shape=input_shape,
        device=device,
        output_dir=output_dir
    )
    
    # Log results
    if logger is not None:
        logger.info(f"Optimization results:")
        for name, result in results.items():
            logger.info(f"  {name.capitalize()} average time: {result['avg_time']:.6f} s")
            if name != 'original':
                logger.info(f"  {name.capitalize()} speedup: {result['speedup']:.2f}x")
    
    # Plot results
    plot_optimization_results(results, output_dir)
    
    return results


def main():
    """Run optimizations for the TTM model."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Optimize the TTM model')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Log directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run optimizations on')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=128, help='Sequence length')
    parser.add_argument('--vocab_size', type=int, default=128, help='Vocabulary size')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--memory_size', type=int, default=16, help='Memory size')
    parser.add_argument('--r', type=int, default=4, help='Number of memory slots to read/write')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--optimization', type=str, default='all', help='Optimization to run (jit, quantize, all)')
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    
    # Set up logging
    logger = setup_logging(args.log_dir)
    logger.info(f"Arguments: {args}")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create TTM model
    logger.info("Creating TTM model...")
    model = create_ttm_model(
        vocab_size=args.vocab_size,
        embedding_dim=args.embedding_dim,
        memory_size=args.memory_size,
        r=args.r,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        device=device
    )
    
    # Create input shape
    input_shape = (args.batch_size, args.seq_len)
    
    # Run optimizations
    if args.optimization == 'jit' or args.optimization == 'all':
        logger.info("Running JIT optimization...")
        jit_results = run_jit_optimization(model, input_shape, device, args.output_dir, logger)
    
    if args.optimization == 'quantize' or args.optimization == 'all':
        logger.info("Running quantization optimization...")
        quantization_results = run_quantization_optimization(model, input_shape, device, args.output_dir, logger)
    
    if args.optimization == 'all':
        logger.info("Running all optimizations...")
        all_results = run_all_optimizations(model, input_shape, device, args.output_dir, logger)
    
    logger.info("Optimizations complete!")


if __name__ == '__main__':
    main()
