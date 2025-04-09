"""Memory operations for Token Turing Machine.

This module implements the memory read and write operations as described in the TTM paper.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Optional

from src.ttm.models.token_summarization import get_token_summarizer


class MemoryReadOperation(nn.Module):
    """Memory read operation as described in the TTM paper.
    
    This implements the unified memory-input reading strategy from Section 3.1.2.
    """
    
    r: int  # Number of tokens to read
    summarization_method: str = 'mlp'
    use_positional_embedding: bool = True
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, 
                 memory: jnp.ndarray, 
                 input_tokens: jnp.ndarray, 
                 train: bool = False) -> jnp.ndarray:
        """Apply memory read operation.
        
        Args:
            memory: Memory tokens of shape [batch_size, m, dim]
            input_tokens: Input tokens of shape [batch_size, n, dim]
            train: Whether in training mode
            
        Returns:
            Read tokens of shape [batch_size, r, dim]
        """
        batch_size, m, dim = memory.shape
        _, n, _ = input_tokens.shape
        
        # Concatenate memory and input tokens
        combined_tokens = jnp.concatenate([memory, input_tokens], axis=1)
        
        # Add positional embeddings if enabled
        if self.use_positional_embedding:
            # Create position indices
            pos_indices = jnp.arange(m + n)[None, :]
            pos_indices = jnp.broadcast_to(pos_indices, (batch_size, m + n))
            
            # Create positional embeddings
            pos_embedding = nn.Embed(
                num_embeddings=m + n,
                features=dim,
                embedding_init=nn.initializers.normal(stddev=0.02)
            )(pos_indices)
            
            # Add positional embeddings
            combined_tokens = combined_tokens + pos_embedding
        
        # Apply token summarization to get read tokens
        summarizer = get_token_summarizer(
            method=self.summarization_method,
            k=self.r,
            dropout_rate=self.dropout_rate
        )
        
        read_tokens = summarizer(combined_tokens, train=train)
        
        return read_tokens


class MemoryWriteOperation(nn.Module):
    """Memory write operation as described in the TTM paper.
    
    This implements the token summarization-based write from Section 3.1.4.
    """
    
    m: int  # Memory size
    summarization_method: str = 'mlp'
    use_positional_embedding: bool = True
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, 
                 memory: jnp.ndarray, 
                 output_tokens: jnp.ndarray, 
                 input_tokens: jnp.ndarray, 
                 train: bool = False) -> jnp.ndarray:
        """Apply memory write operation.
        
        Args:
            memory: Memory tokens of shape [batch_size, m, dim]
            output_tokens: Output tokens from processing unit of shape [batch_size, r, dim]
            input_tokens: Input tokens of shape [batch_size, n, dim]
            train: Whether in training mode
            
        Returns:
            Updated memory of shape [batch_size, m, dim]
        """
        batch_size, _, dim = memory.shape
        
        # Concatenate memory, output, and input tokens
        combined_tokens = jnp.concatenate([memory, output_tokens, input_tokens], axis=1)
        
        # Add positional embeddings if enabled
        if self.use_positional_embedding:
            # Create position indices
            total_len = combined_tokens.shape[1]
            pos_indices = jnp.arange(total_len)[None, :]
            pos_indices = jnp.broadcast_to(pos_indices, (batch_size, total_len))
            
            # Create positional embeddings
            pos_embedding = nn.Embed(
                num_embeddings=total_len,
                features=dim,
                embedding_init=nn.initializers.normal(stddev=0.02)
            )(pos_indices)
            
            # Add positional embeddings
            combined_tokens = combined_tokens + pos_embedding
        
        # Apply token summarization to get new memory
        summarizer = get_token_summarizer(
            method=self.summarization_method,
            k=self.m,
            dropout_rate=self.dropout_rate
        )
        
        new_memory = summarizer(combined_tokens, train=train)
        
        return new_memory


class EraseAddWriteOperation(nn.Module):
    """NTM-style erase-and-add memory write operation.
    
    This implements the alternative write mechanism from the NTM paper,
    used for comparison in the TTM paper.
    """
    
    hidden_dim: int = 128
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, 
                 memory: jnp.ndarray, 
                 output_tokens: jnp.ndarray, 
                 input_tokens: Optional[jnp.ndarray] = None, 
                 train: bool = False) -> jnp.ndarray:
        """Apply erase-and-add memory write operation.
        
        Args:
            memory: Memory tokens of shape [batch_size, m, dim]
            output_tokens: Output tokens from processing unit of shape [batch_size, r, dim]
            input_tokens: Optional input tokens (not used in this implementation)
            train: Whether in training mode
            
        Returns:
            Updated memory of shape [batch_size, m, dim]
        """
        batch_size, m, dim = memory.shape
        
        # Compute erase vector
        erase = nn.LayerNorm()(memory)
        erase = nn.Dense(self.hidden_dim)(erase)
        erase = nn.gelu(erase)
        erase = nn.Dropout(rate=self.dropout_rate)(erase, deterministic=not train)
        erase = nn.Dense(output_tokens.shape[1])(erase)
        erase = jnp.transpose(erase, [0, 2, 1])  # [batch_size, r, m]
        erase = jax.nn.softmax(erase, axis=-1)
        
        # Compute add vector
        add = nn.LayerNorm()(output_tokens)
        add = nn.Dense(self.hidden_dim)(add)
        add = nn.gelu(add)
        add = nn.Dropout(rate=self.dropout_rate)(add, deterministic=not train)
        add = nn.Dense(dim)(add)
        
        # Erase from memory
        erase_expanded = jnp.expand_dims(erase, -1)  # [batch_size, r, m, 1]
        add_expanded = jnp.expand_dims(add, 2)  # [batch_size, r, 1, dim]
        erase_matrix = erase_expanded * add_expanded  # [batch_size, r, m, dim]
        erase_matrix = 1 - jnp.sum(erase_matrix, axis=1)  # [batch_size, m, dim]
        
        erased_memory = memory * erase_matrix
        
        # Add to memory
        add_matrix = jnp.einsum('brm,brd->bmd', erase, add)
        
        new_memory = erased_memory + add_matrix
        
        return new_memory


class ConcatenationWriteOperation(nn.Module):
    """Concatenation-based memory write operation.
    
    This implements the alternative write mechanism that concatenates every observed
    input token, used for comparison in the TTM paper.
    """
    
    m: int  # Memory size
    
    @nn.compact
    def __call__(self, 
                 memory: jnp.ndarray, 
                 output_tokens: jnp.ndarray, 
                 input_tokens: Optional[jnp.ndarray] = None, 
                 train: bool = False) -> jnp.ndarray:
        """Apply concatenation-based memory write operation.
        
        Args:
            memory: Memory tokens of shape [batch_size, m, dim]
            output_tokens: Output tokens from processing unit of shape [batch_size, r, dim]
            input_tokens: Optional input tokens (not used in this implementation)
            train: Whether in training mode
            
        Returns:
            Updated memory of shape [batch_size, m, dim]
        """
        batch_size, _, dim = memory.shape
        
        # Concatenate memory and output tokens
        combined = jnp.concatenate([output_tokens, memory], axis=1)
        
        # Take the most recent m tokens
        new_memory = combined[:, :self.m, :]
        
        return new_memory
