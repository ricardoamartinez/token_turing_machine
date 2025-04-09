"""
Optimization utilities for the Token Turing Machine (TTM) model.

This module provides utilities for optimizing the TTM model using TorchScript,
quantization, and other techniques.
"""

import torch
import torch.nn as nn
import torch.jit
import torch.quantization
import time
import os
import json
import logging
from typing import Dict, Any, Optional, Union, List, Tuple, Callable

# Import TokenTuringMachine at runtime to avoid circular imports
TokenTuringMachine = None
from .performance import benchmark_forward


def create_jit_model(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: Optional[torch.device] = None,
    optimize: bool = True
) -> torch.jit.ScriptModule:
    """Create a JIT-compiled model.

    Args:
        model: The model to compile
        input_shape: The shape of the input tensor
        device: The device to run the model on
        optimize: Whether to optimize the model

    Returns:
        JIT-compiled model
    """
    # Import TokenTuringMachine at runtime to avoid circular imports
    from ..models.ttm_model import TokenTuringMachine as TTM
    global TokenTuringMachine
    TokenTuringMachine = TTM

    # Set device
    if device is None:
        device = next(model.parameters()).device

    # Create dummy input
    if isinstance(model, TokenTuringMachine):
        # For TTM, input should be integer tokens
        vocab_size = model.vocab_size if hasattr(model, 'vocab_size') else 100
        dummy_input = torch.randint(0, vocab_size - 1, input_shape, device=device)
    else:
        # For other models, input can be float tensors
        dummy_input = torch.randn(*input_shape, device=device)

    # Set model to evaluation mode
    model.eval()

    # Trace or script the model
    with torch.no_grad():
        try:
            # Try scripting first
            jit_model = torch.jit.script(model)
        except Exception as e:
            print(f"Scripting failed with error: {e}")
            print("Falling back to tracing...")

            # Fall back to tracing
            jit_model = torch.jit.trace(model, dummy_input)

    # Optimize the model
    if optimize:
        jit_model = torch.jit.optimize_for_inference(jit_model)

    return jit_model


def compare_jit_performance(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: Optional[torch.device] = None,
    num_iterations: int = 100,
    warmup_iterations: int = 10
) -> Dict[str, Dict[str, float]]:
    """Compare performance of original and JIT-compiled models.

    Args:
        model: The model to benchmark
        input_shape: The shape of the input tensor
        device: The device to run the model on
        num_iterations: Number of iterations to run
        warmup_iterations: Number of warmup iterations

    Returns:
        Dictionary with benchmark results for original and JIT-compiled models
    """
    # Import TokenTuringMachine at runtime to avoid circular imports
    from ..models.ttm_model import TokenTuringMachine as TTM
    global TokenTuringMachine
    TokenTuringMachine = TTM

    # Set device
    if device is None:
        device = next(model.parameters()).device

    # Create dummy input
    if isinstance(model, TokenTuringMachine):
        # For TTM, input should be integer tokens
        vocab_size = model.vocab_size if hasattr(model, 'vocab_size') else 100
        dummy_input = torch.randint(0, vocab_size - 1, input_shape, device=device)
    else:
        # For other models, input can be float tensors
        dummy_input = torch.randn(*input_shape, device=device)

    # Set model to evaluation mode
    model.eval()

    # Benchmark original model
    original_results = benchmark_forward(
        model=model,
        input_shape=input_shape,
        device=device,
        num_iterations=num_iterations,
        warmup_iterations=warmup_iterations
    )

    # Create JIT-compiled model
    jit_model = create_jit_model(model, input_shape, device)

    # Benchmark JIT-compiled model
    jit_results = benchmark_forward(
        model=jit_model,
        input_shape=input_shape,
        device=device,
        num_iterations=num_iterations,
        warmup_iterations=warmup_iterations
    )

    # Calculate speedup
    if jit_results['avg_time'] > 0:
        speedup = original_results['avg_time'] / jit_results['avg_time']
    else:
        speedup = 1.0
    jit_results['speedup'] = speedup

    return {
        'original': original_results,
        'jit': jit_results
    }


def quantize_model(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.qint8
) -> nn.Module:
    """Quantize a model to reduce memory usage and improve performance.

    Args:
        model: The model to quantize
        input_shape: The shape of the input tensor
        device: The device to run the model on
        dtype: The quantization data type

    Returns:
        Quantized model
    """
    # Import TokenTuringMachine at runtime to avoid circular imports
    from ..models.ttm_model import TokenTuringMachine as TTM
    global TokenTuringMachine
    TokenTuringMachine = TTM

    # Set device
    if device is None:
        device = next(model.parameters()).device

    # Create dummy input
    if isinstance(model, TokenTuringMachine):
        # For TTM, input should be integer tokens
        vocab_size = model.vocab_size if hasattr(model, 'vocab_size') else 100
        dummy_input = torch.randint(0, vocab_size - 1, input_shape, device=device)
    else:
        # For other models, input can be float tensors
        dummy_input = torch.randn(*input_shape, device=device)

    # Set model to evaluation mode
    model.eval()

    # Create a copy of the model for quantization
    quantized_model = model

    # Fuse modules for quantization
    for module_name, module in quantized_model.named_children():
        if isinstance(module, nn.Sequential):
            for i in range(len(module) - 1):
                if isinstance(module[i], nn.Conv2d) and isinstance(module[i + 1], nn.BatchNorm2d):
                    torch.quantization.fuse_modules(module, [str(i), str(i + 1)], inplace=True)
                elif isinstance(module[i], nn.Linear) and isinstance(module[i + 1], nn.BatchNorm1d):
                    torch.quantization.fuse_modules(module, [str(i), str(i + 1)], inplace=True)

    # Prepare model for quantization
    quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(quantized_model, inplace=True)

    # Calibrate model with dummy input
    with torch.no_grad():
        quantized_model(dummy_input)

    # Convert model to quantized version
    torch.quantization.convert(quantized_model, inplace=True)

    return quantized_model


def compare_quantization_performance(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: Optional[torch.device] = None,
    num_iterations: int = 100,
    warmup_iterations: int = 10
) -> Dict[str, Dict[str, float]]:
    """Compare performance of original and quantized models.

    Args:
        model: The model to benchmark
        input_shape: The shape of the input tensor
        device: The device to run the model on
        num_iterations: Number of iterations to run
        warmup_iterations: Number of warmup iterations

    Returns:
        Dictionary with benchmark results for original and quantized models
    """
    # Set device
    if device is None:
        device = next(model.parameters()).device

    # Benchmark original model
    original_results = benchmark_forward(
        model=model,
        input_shape=input_shape,
        device=device,
        num_iterations=num_iterations,
        warmup_iterations=warmup_iterations
    )

    # Quantize model
    quantized_model = quantize_model(model, input_shape, device)

    # Benchmark quantized model
    quantized_results = benchmark_forward(
        model=quantized_model,
        input_shape=input_shape,
        device=device,
        num_iterations=num_iterations,
        warmup_iterations=warmup_iterations
    )

    # Calculate speedup
    if quantized_results['avg_time'] > 0:
        speedup = original_results['avg_time'] / quantized_results['avg_time']
    else:
        speedup = 1.0
    quantized_results['speedup'] = speedup

    return {
        'original': original_results,
        'quantized': quantized_results
    }


def optimize_ttm_model(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: Optional[torch.device] = None,
    jit: bool = True,
    quantize: bool = False,
    output_dir: str = './outputs'
) -> Dict[str, nn.Module]:
    """Optimize a TTM model using various techniques.

    Args:
        model: The TTM model to optimize
        input_shape: The shape of the input tensor
        device: The device to run the model on
        jit: Whether to create a JIT-compiled version
        quantize: Whether to create a quantized version
        output_dir: Directory to save optimized models

    Returns:
        Dictionary mapping optimization type to optimized model
    """
    # Import TokenTuringMachine at runtime to avoid circular imports
    from ..models.ttm_model import TokenTuringMachine as TTM
    global TokenTuringMachine
    TokenTuringMachine = TTM
    # Set device
    if device is None:
        device = next(model.parameters()).device

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize result dictionary
    optimized_models = {'original': model}

    # Create JIT-compiled model
    if jit:
        jit_model = create_jit_model(model, input_shape, device)
        optimized_models['jit'] = jit_model

        # Save JIT-compiled model
        torch.jit.save(jit_model, os.path.join(output_dir, 'ttm_jit.pt'))

    # Create quantized model
    if quantize:
        quantized_model = quantize_model(model, input_shape, device)
        optimized_models['quantized'] = quantized_model

        # Save quantized model
        torch.save(quantized_model.state_dict(), os.path.join(output_dir, 'ttm_quantized.pt'))

    return optimized_models


def benchmark_optimized_models(
    optimized_models: Dict[str, nn.Module],
    input_shape: Tuple[int, ...],
    device: Optional[torch.device] = None,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    output_dir: str = './outputs'
) -> Dict[str, Dict[str, float]]:
    """Benchmark optimized models.

    Args:
        optimized_models: Dictionary mapping optimization type to optimized model
        input_shape: The shape of the input tensor
        device: The device to run the model on
        num_iterations: Number of iterations to run
        warmup_iterations: Number of warmup iterations
        output_dir: Directory to save benchmark results

    Returns:
        Dictionary mapping optimization type to benchmark results
    """
    # Set device
    if device is None:
        device = next(iter(optimized_models.values())).device

    # Initialize result dictionary
    results = {}

    # Benchmark each model
    for name, model in optimized_models.items():
        # Benchmark model
        result = benchmark_forward(
            model=model,
            input_shape=input_shape,
            device=device,
            num_iterations=num_iterations,
            warmup_iterations=warmup_iterations
        )

        # Store result
        results[name] = result

    # Calculate speedups
    original_time = results['original']['avg_time']
    for name, result in results.items():
        if name != 'original' and result['avg_time'] > 0:
            result['speedup'] = original_time / result['avg_time']
        else:
            result['speedup'] = 1.0

    # Save results
    with open(os.path.join(output_dir, 'optimization_benchmark.json'), 'w') as f:
        # Convert results to JSON-serializable format
        json_results = {}
        for name, result in results.items():
            json_results[name] = {k: float(v) if isinstance(v, torch.Tensor) else v for k, v in result.items()}

        json.dump(json_results, f, indent=4)

    return results
