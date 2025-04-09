"""Token summarization methods for TTM.

This module implements various token summarization methods as described in the TTM paper:
- MLP-based token summarization (default)
- Query-based token summarization
- Pooling-based token summarization
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional


class MLPTokenSummarization(nn.Module):
    """MLP-based token summarization as described in the TTM paper.
    
    This implements Equation 1 from the paper:
    w_i = α_i(V) = softmax(MLP(V))
    """
    
    k: int  # Number of output tokens
    hidden_dim: int = 128
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, tokens: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """Apply MLP-based token summarization.
        
        Args:
            tokens: Input tokens of shape [batch_size, n_tokens, dim]
            train: Whether in training mode
            
        Returns:
            Summarized tokens of shape [batch_size, k, dim]
        """
        batch_size, n_tokens, dim = tokens.shape
        
        # Apply layer normalization
        x = nn.LayerNorm()(tokens)
        
        # MLP to compute importance weights
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        x = nn.Dense(self.k)(x)
        
        # Compute importance weights
        weights = jnp.transpose(x, [0, 2, 1])  # [batch_size, k, n_tokens]
        weights = jax.nn.softmax(weights, axis=-1)
        
        # Apply weighted summation
        output = jnp.einsum('bkn,bnd->bkd', weights, tokens)
        
        return output


class QueryTokenSummarization(nn.Module):
    """Query-based token summarization as described in the TTM paper.
    
    This implements Equation 2 from the paper:
    w_i = α_i(V) = softmax(q_i·V^T/√d)
    """
    
    k: int  # Number of output tokens
    num_heads: int = 8
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, tokens: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """Apply query-based token summarization.
        
        Args:
            tokens: Input tokens of shape [batch_size, n_tokens, dim]
            train: Whether in training mode
            
        Returns:
            Summarized tokens of shape [batch_size, k, dim]
        """
        batch_size, n_tokens, dim = tokens.shape
        
        # Initialize learnable query vectors
        query_init = nn.initializers.normal(stddev=0.02)
        queries = self.param('queries', query_init, (1, self.k, dim))
        queries = jnp.broadcast_to(queries, (batch_size, self.k, dim))
        
        # Apply multi-head attention
        output = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate
        )(
            inputs_q=queries,
            inputs_kv=tokens,
            deterministic=not train
        )
        
        return output


class PoolingTokenSummarization(nn.Module):
    """Pooling-based token summarization as described in the TTM paper.
    
    This implements a simple pooling-based approach without learning.
    """
    
    k: int  # Number of output tokens
    
    @nn.compact
    def __call__(self, tokens: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """Apply pooling-based token summarization.
        
        Args:
            tokens: Input tokens of shape [batch_size, n_tokens, dim]
            train: Whether in training mode
            
        Returns:
            Summarized tokens of shape [batch_size, k, dim]
        """
        batch_size, n_tokens, dim = tokens.shape
        
        # Determine pooling size
        pool_size = max(1, n_tokens // self.k)
        
        # Pad if necessary
        pad_size = pool_size * self.k - n_tokens
        if pad_size > 0:
            tokens = jnp.pad(tokens, ((0, 0), (0, pad_size), (0, 0)))
        
        # Reshape for pooling
        tokens = tokens.reshape(batch_size, self.k, pool_size, dim)
        
        # Apply average pooling
        output = jnp.mean(tokens, axis=2)
        
        return output


def get_token_summarizer(method: str, k: int, **kwargs):
    """Factory function to get a token summarization module.
    
    Args:
        method: One of 'mlp', 'query', or 'pooling'
        k: Number of output tokens
        **kwargs: Additional arguments to pass to the module
        
    Returns:
        A token summarization module
    """
    if method == 'mlp':
        return MLPTokenSummarization(k=k, **kwargs)
    elif method == 'query':
        return QueryTokenSummarization(k=k, **kwargs)
    elif method == 'pooling':
        return PoolingTokenSummarization(k=k, **kwargs)
    else:
        raise ValueError(f"Unknown token summarization method: {method}")
