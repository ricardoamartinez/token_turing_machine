"""Transformer processing unit for TTM.

This module implements the transformer-based processing unit as described in the TTM paper.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional


class FeedForward(nn.Module):
    """Feed-forward network used in Transformer.
    
    This is a standard MLP with two dense layers and a GELU activation.
    """
    
    dim: int
    hidden_dim: int = 512
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """Apply feed-forward network.
        
        Args:
            x: Input of shape [batch_size, seq_len, dim]
            train: Whether in training mode
            
        Returns:
            Output of shape [batch_size, seq_len, dim]
        """
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        x = nn.Dense(self.dim)(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism.
    
    This implements the standard multi-head attention as described in "Attention is All You Need".
    """
    
    dim: int
    num_heads: int = 4
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, 
                 x: jnp.ndarray, 
                 mask: Optional[jnp.ndarray] = None, 
                 train: bool = False) -> jnp.ndarray:
        """Apply multi-head attention.
        
        Args:
            x: Input of shape [batch_size, seq_len, dim]
            mask: Optional attention mask of shape [batch_size, num_heads, seq_len, seq_len]
            train: Whether in training mode
            
        Returns:
            Output of shape [batch_size, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape
        head_dim = self.dim // self.num_heads
        
        # Linear projections
        q = nn.Dense(self.dim, name='query')(x)
        k = nn.Dense(self.dim, name='key')(x)
        v = nn.Dense(self.dim, name='value')(x)
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, head_dim)
        
        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        
        # Compute attention scores
        scale = jnp.sqrt(head_dim)
        scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) / scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask
        
        # Apply attention weights
        weights = jax.nn.softmax(scores, axis=-1)
        weights = nn.Dropout(rate=self.dropout_rate)(weights, deterministic=not train)
        
        # Compute weighted sum
        output = jnp.matmul(weights, v)
        
        # Transpose back to [batch_size, seq_len, num_heads, head_dim]
        output = jnp.transpose(output, (0, 2, 1, 3))
        
        # Reshape back to [batch_size, seq_len, dim]
        output = output.reshape(batch_size, seq_len, self.dim)
        
        # Final linear projection
        output = nn.Dense(self.dim)(output)
        output = nn.Dropout(rate=self.dropout_rate)(output, deterministic=not train)
        
        return output


class TransformerBlock(nn.Module):
    """Transformer block.
    
    This implements a standard transformer block with pre-norm architecture.
    """
    
    dim: int
    num_heads: int = 4
    hidden_dim: int = 512
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """Apply transformer block.
        
        Args:
            x: Input of shape [batch_size, seq_len, dim]
            train: Whether in training mode
            
        Returns:
            Output of shape [batch_size, seq_len, dim]
        """
        # Pre-norm for attention
        norm1 = nn.LayerNorm()(x)
        
        # Multi-head attention
        attn_output = MultiHeadAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate
        )(norm1, train=train)
        
        # Residual connection
        x = x + attn_output
        
        # Pre-norm for feed-forward
        norm2 = nn.LayerNorm()(x)
        
        # Feed-forward network
        ff_output = FeedForward(
            dim=self.dim,
            hidden_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate
        )(norm2, train=train)
        
        # Residual connection
        x = x + ff_output
        
        return x


class TransformerStack(nn.Module):
    """Stack of transformer blocks.
    
    This implements a stack of transformer blocks as specified in the TTM paper.
    """
    
    dim: int
    depth: int = 4
    num_heads: int = 4
    hidden_dim: int = 512
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """Apply stack of transformer blocks.
        
        Args:
            x: Input of shape [batch_size, seq_len, dim]
            train: Whether in training mode
            
        Returns:
            Output of shape [batch_size, seq_len, dim]
        """
        for i in range(self.depth):
            x = TransformerBlock(
                dim=self.dim,
                num_heads=self.num_heads,
                hidden_dim=self.hidden_dim,
                dropout_rate=self.dropout_rate
            )(x, train=train)
        
        # Final layer normalization
        x = nn.LayerNorm()(x)
        
        return x
