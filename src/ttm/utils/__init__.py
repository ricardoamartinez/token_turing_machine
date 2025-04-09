"""Utility functions for TTM."""

from .masking import (
    create_causal_mask,
    create_combined_mask,
    mask_after_eos,
    EOSCrossEntropyLoss
)

from .losses import (
    TTMLoss,
    LabelSmoothingLoss,
    FocalLoss,
    create_loss_function
)

from .performance import (
    measure_flops,
    measure_memory,
    benchmark_forward,
    benchmark_sequence_length,
    plot_sequence_length_benchmark,
    compare_cpu_cuda,
    compare_ttm_transformer,
    plot_ttm_transformer_comparison,
    optimize_batch_size,
    plot_batch_size_optimization,
    jit_attention,
    jit_memory_read,
    jit_memory_write
)

from .optimize import (
    create_jit_model,
    compare_jit_performance,
    quantize_model,
    compare_quantization_performance,
    optimize_ttm_model,
    benchmark_optimized_models
)
