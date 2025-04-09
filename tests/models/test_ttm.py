"""Tests for TTM model.

This module contains tests for the Token Turing Machine implementation.
"""

import jax
import jax.numpy as jnp
import pytest
import time
from typing import Dict, Any

from src.ttm.models.ttm import TokenTuringMachine, TTMEncoder, TTMMemorylessEncoder
from src.ttm.models.token_summarization import (
    MLPTokenSummarization,
    QueryTokenSummarization,
    PoolingTokenSummarization
)
from src.ttm.models.memory_operations import (
    MemoryReadOperation,
    MemoryWriteOperation,
    EraseAddWriteOperation,
    ConcatenationWriteOperation
)


def test_token_summarization_mlp():
    """Test MLP-based token summarization."""
    rng = jax.random.PRNGKey(0)
    
    # Create model
    model = MLPTokenSummarization(k=5)
    
    # Initialize model
    params = model.init(rng, jnp.ones((2, 10, 128)))
    
    # Test forward pass
    tokens = jnp.ones((2, 10, 128))
    output = model.apply(params, tokens)
    
    # Check output shape
    assert output.shape == (2, 5, 128)


def test_token_summarization_query():
    """Test query-based token summarization."""
    rng = jax.random.PRNGKey(0)
    
    # Create model
    model = QueryTokenSummarization(k=5)
    
    # Initialize model
    params = model.init(rng, jnp.ones((2, 10, 128)))
    
    # Test forward pass
    tokens = jnp.ones((2, 10, 128))
    output = model.apply(params, tokens)
    
    # Check output shape
    assert output.shape == (2, 5, 128)


def test_token_summarization_pooling():
    """Test pooling-based token summarization."""
    rng = jax.random.PRNGKey(0)
    
    # Create model
    model = PoolingTokenSummarization(k=5)
    
    # Initialize model
    params = model.init(rng, jnp.ones((2, 10, 128)))
    
    # Test forward pass
    tokens = jnp.ones((2, 10, 128))
    output = model.apply(params, tokens)
    
    # Check output shape
    assert output.shape == (2, 5, 128)


def test_memory_read_operation():
    """Test memory read operation."""
    rng = jax.random.PRNGKey(0)
    
    # Create model
    model = MemoryReadOperation(r=16)
    
    # Initialize model
    params = model.init(rng, jnp.ones((2, 96, 128)), jnp.ones((2, 10, 128)))
    
    # Test forward pass
    memory = jnp.ones((2, 96, 128))
    input_tokens = jnp.ones((2, 10, 128))
    output = model.apply(params, memory, input_tokens)
    
    # Check output shape
    assert output.shape == (2, 16, 128)


def test_memory_write_operation():
    """Test memory write operation."""
    rng = jax.random.PRNGKey(0)
    
    # Create model
    model = MemoryWriteOperation(m=96)
    
    # Initialize model
    params = model.init(
        rng,
        jnp.ones((2, 96, 128)),
        jnp.ones((2, 16, 128)),
        jnp.ones((2, 10, 128))
    )
    
    # Test forward pass
    memory = jnp.ones((2, 96, 128))
    output_tokens = jnp.ones((2, 16, 128))
    input_tokens = jnp.ones((2, 10, 128))
    output = model.apply(params, memory, output_tokens, input_tokens)
    
    # Check output shape
    assert output.shape == (2, 96, 128)


def test_erase_add_write_operation():
    """Test erase-add write operation."""
    rng = jax.random.PRNGKey(0)
    
    # Create model
    model = EraseAddWriteOperation()
    
    # Initialize model
    params = model.init(
        rng,
        jnp.ones((2, 96, 128)),
        jnp.ones((2, 16, 128))
    )
    
    # Test forward pass
    memory = jnp.ones((2, 96, 128))
    output_tokens = jnp.ones((2, 16, 128))
    output = model.apply(params, memory, output_tokens)
    
    # Check output shape
    assert output.shape == (2, 96, 128)


def test_concatenation_write_operation():
    """Test concatenation write operation."""
    rng = jax.random.PRNGKey(0)
    
    # Create model
    model = ConcatenationWriteOperation(m=96)
    
    # Initialize model
    params = model.init(
        rng,
        jnp.ones((2, 96, 128)),
        jnp.ones((2, 16, 128))
    )
    
    # Test forward pass
    memory = jnp.ones((2, 96, 128))
    output_tokens = jnp.ones((2, 16, 128))
    output = model.apply(params, memory, output_tokens)
    
    # Check output shape
    assert output.shape == (2, 96, 128)


def test_token_turing_machine():
    """Test Token Turing Machine."""
    rng = jax.random.PRNGKey(0)
    
    # Create model
    model = TokenTuringMachine()
    
    # Initialize model
    params = model.init(rng, jnp.ones((2, 10, 128)))
    
    # Test forward pass
    input_tokens = jnp.ones((2, 10, 128))
    memory, output_tokens = model.apply(params, input_tokens)
    
    # Check output shapes
    assert memory.shape == (2, 96, 128)
    assert output_tokens.shape == (2, 16, 128)


def test_ttm_encoder():
    """Test TTM encoder."""
    rng = jax.random.PRNGKey(0)
    
    # Create model
    model = TTMEncoder()
    
    # Initialize model
    params = model.init(rng, jnp.ones((2, 5, 10, 128)))
    
    # Test forward pass
    input_tokens = jnp.ones((2, 5, 10, 128))
    output_tokens = model.apply(params, input_tokens)
    
    # Check output shape
    assert output_tokens.shape == (2, 5, 16, 128)


def test_ttm_memoryless_encoder():
    """Test memory-less TTM encoder."""
    rng = jax.random.PRNGKey(0)
    
    # Create model
    model = TTMMemorylessEncoder()
    
    # Initialize model
    params = model.init(rng, jnp.ones((2, 5, 10, 128)))
    
    # Test forward pass
    input_tokens = jnp.ones((2, 5, 10, 128))
    output_tokens = model.apply(params, input_tokens)
    
    # Check output shape
    assert output_tokens.shape == (2, 5, 16, 128)


def test_memory_broadcast():
    """Test memory broadcasting."""
    rng = jax.random.PRNGKey(0)
    
    # Create model
    model = TokenTuringMachine()
    
    # Initialize model
    params = model.init(rng, jnp.ones((2, 10, 128)))
    
    # Test memory broadcasting
    memory = model.apply(params, method=model._broadcast_memory, batch_size=4)
    
    # Check output shape
    assert memory.shape == (4, 96, 128)


def test_performance_comparison():
    """Test performance comparison between CPU and GPU/TPU."""
    rng = jax.random.PRNGKey(0)
    
    # Create model
    model = TokenTuringMachine()
    
    # Initialize model
    params = model.init(rng, jnp.ones((2, 10, 128)))
    
    # Prepare input
    input_tokens = jnp.ones((2, 10, 128))
    
    # Warm-up
    for _ in range(5):
        model.apply(params, input_tokens)
    
    # Measure CPU performance
    start_time = time.time()
    for _ in range(10):
        model.apply(params, input_tokens)
    cpu_time = (time.time() - start_time) / 10
    
    print(f"Average CPU time: {cpu_time * 1000:.2f} ms")
    
    # Note: GPU/TPU comparison would be done in a separate environment
    # This test just measures CPU performance as a baseline
