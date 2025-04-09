"""Training utilities for TTM.

This module implements training and evaluation functions for TTM.
"""

import jax
import jax.numpy as jnp
import optax
import flax
from flax.training import train_state
from typing import Dict, Any, Tuple, Callable, Optional

from src.ttm.models.ttm import TTMEncoder


class TrainState(train_state.TrainState):
    """Train state with additional attributes."""
    
    batch_stats: Optional[Dict[str, Any]] = None


def create_train_state(
    model: TTMEncoder,
    rng: jnp.ndarray,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4
) -> TrainState:
    """Create initial training state.
    
    Args:
        model: TTM encoder model
        rng: Random number generator key
        learning_rate: Learning rate
        weight_decay: Weight decay
        
    Returns:
        Initial training state
    """
    # Initialize model
    dummy_input = jnp.ones((1, 1, 10, model.dim))
    params = model.init(rng, dummy_input, train=True)
    
    # Create optimizer
    tx = optax.adamw(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        b1=0.9,
        b2=0.999,
        eps=1e-8
    )
    
    # Create train state
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )


def create_learning_rate_schedule(
    base_learning_rate: float = 1e-4,
    warmup_steps: int = 1000,
    decay_steps: int = 100000
) -> Callable[[int], float]:
    """Create learning rate schedule.
    
    Args:
        base_learning_rate: Base learning rate
        warmup_steps: Number of warmup steps
        decay_steps: Number of decay steps
        
    Returns:
        Learning rate schedule function
    """
    def schedule(step: int) -> float:
        """Learning rate schedule.
        
        Args:
            step: Current step
            
        Returns:
            Learning rate
        """
        # Linear warmup
        warmup_factor = jnp.minimum(step / warmup_steps, 1.0)
        
        # Cosine decay
        decay_factor = 0.5 * (1.0 + jnp.cos(
            jnp.pi * jnp.minimum(step - warmup_steps, decay_steps) / decay_steps
        ))
        
        # Combine
        return base_learning_rate * warmup_factor * decay_factor
    
    return schedule


def compute_metrics(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    pad_token: int = 12
) -> Dict[str, float]:
    """Compute evaluation metrics.
    
    Args:
        logits: Logits of shape [batch_size, seq_len, vocab_size]
        targets: Target tokens of shape [batch_size, seq_len]
        pad_token: Padding token ID
        
    Returns:
        Dictionary of metrics
    """
    # Get predictions
    predictions = jnp.argmax(logits, axis=-1)
    
    # Create mask for non-padding tokens
    mask = targets != pad_token
    
    # Compute position-wise accuracy
    position_correct = (predictions == targets) * mask
    position_accuracy = jnp.sum(position_correct) / jnp.maximum(jnp.sum(mask), 1)
    
    # Compute sequence-level accuracy
    seq_correct = jnp.all(position_correct + ~mask, axis=-1)
    seq_accuracy = jnp.mean(seq_correct)
    
    return {
        'position_accuracy': position_accuracy,
        'sequence_accuracy': seq_accuracy
    }


def format_number_sequence(
    sequence: jnp.ndarray,
    times_token: int = 10,
    eos_token: int = 11,
    pad_token: int = 12
) -> str:
    """Format a number sequence for display.
    
    Args:
        sequence: Token sequence
        times_token: Multiplication symbol token ID
        eos_token: End of sequence token ID
        pad_token: Padding token ID
        
    Returns:
        Formatted string
    """
    result = []
    for token in sequence:
        if token == times_token:
            result.append('×')
        elif token == eos_token:
            break
        elif token == pad_token:
            break
        else:
            result.append(str(token))
    
    return ''.join(result)
