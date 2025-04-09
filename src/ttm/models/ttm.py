"""Token Turing Machine implementation.

This module implements the Token Turing Machine as described in the paper:
"Token Turing Machines" (https://arxiv.org/abs/2211.09119)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Optional, Dict, Any, Union

from src.ttm.models.transformer import TransformerStack
from src.ttm.models.memory_operations import (
    MemoryReadOperation,
    MemoryWriteOperation,
    EraseAddWriteOperation,
    ConcatenationWriteOperation
)


class TokenTuringMachine(nn.Module):
    """Token Turing Machine implementation.
    
    This implements the TTM as described in the paper, with a transformer as the processing unit.
    
    Attributes:
        memory_size: Number of memory tokens (m in the paper)
        process_size: Number of tokens for the processing unit to process (r in the paper)
        dim: Dimension of the token embeddings
        num_layers: Number of layers in the transformer
        num_heads: Number of attention heads in the transformer
        hidden_dim: Hidden dimension in the transformer feed-forward network
        summarization_method: Method for token summarization ('mlp', 'query', or 'pooling')
        write_method: Method for memory write ('summarize', 'erase_add', or 'concatenate')
        use_positional_embedding: Whether to use positional embeddings
        dropout_rate: Dropout rate
    """
    
    memory_size: int = 96  # m in the paper
    process_size: int = 16  # r in the paper
    dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    hidden_dim: int = 512
    summarization_method: str = 'mlp'
    write_method: str = 'summarize'
    use_positional_embedding: bool = True
    dropout_rate: float = 0.1
    
    def setup(self):
        """Set up the TTM components."""
        # Memory initialization
        self.memory_init = self.param(
            'memory_init',
            nn.initializers.normal(stddev=0.01),
            (1, self.memory_size, self.dim)
        )
        
        # Memory read operation
        self.read_operation = MemoryReadOperation(
            r=self.process_size,
            summarization_method=self.summarization_method,
            use_positional_embedding=self.use_positional_embedding,
            dropout_rate=self.dropout_rate
        )
        
        # Processing unit (Transformer)
        self.processing_unit = TransformerStack(
            dim=self.dim,
            depth=self.num_layers,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate
        )
        
        # Memory write operation
        if self.write_method == 'summarize':
            self.write_operation = MemoryWriteOperation(
                m=self.memory_size,
                summarization_method=self.summarization_method,
                use_positional_embedding=self.use_positional_embedding,
                dropout_rate=self.dropout_rate
            )
        elif self.write_method == 'erase_add':
            self.write_operation = EraseAddWriteOperation(
                hidden_dim=self.hidden_dim,
                dropout_rate=self.dropout_rate
            )
        elif self.write_method == 'concatenate':
            self.write_operation = ConcatenationWriteOperation(
                m=self.memory_size
            )
        else:
            raise ValueError(f"Unknown write method: {self.write_method}")
    
    def _broadcast_memory(self, batch_size: int) -> jnp.ndarray:
        """Broadcast memory initialization to batch size.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Memory of shape [batch_size, memory_size, dim]
        """
        return jnp.broadcast_to(self.memory_init, (batch_size, self.memory_size, self.dim))
    
    def __call__(self, 
                 inputs: jnp.ndarray, 
                 memory: Optional[jnp.ndarray] = None, 
                 train: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply Token Turing Machine.
        
        Args:
            inputs: Input tokens of shape [batch_size, n_tokens, dim]
            memory: Optional memory tokens of shape [batch_size, memory_size, dim]
            train: Whether in training mode
            
        Returns:
            Tuple of (new_memory, output_tokens)
        """
        batch_size = inputs.shape[0]
        
        # Initialize memory if not provided
        if memory is None:
            memory = self._broadcast_memory(batch_size)
        
        # Read from memory and input
        read_tokens = self.read_operation(memory, inputs, train=train)
        
        # Process through transformer
        output_tokens = self.processing_unit(read_tokens, train=train)
        
        # Write to memory
        new_memory = self.write_operation(memory, output_tokens, inputs, train=train)
        
        return new_memory, output_tokens


class TTMEncoder(nn.Module):
    """Token Turing Machine encoder for processing sequences.
    
    This implements the TTM encoder for processing sequences of tokens.
    """
    
    memory_size: int = 96
    process_size: int = 16
    dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    hidden_dim: int = 512
    summarization_method: str = 'mlp'
    write_method: str = 'summarize'
    use_positional_embedding: bool = True
    dropout_rate: float = 0.1
    
    def setup(self):
        """Set up the TTM encoder."""
        self.ttm = TokenTuringMachine(
            memory_size=self.memory_size,
            process_size=self.process_size,
            dim=self.dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
            summarization_method=self.summarization_method,
            write_method=self.write_method,
            use_positional_embedding=self.use_positional_embedding,
            dropout_rate=self.dropout_rate
        )
    
    def __call__(self, 
                 inputs: jnp.ndarray, 
                 train: bool = False) -> jnp.ndarray:
        """Apply TTM encoder to a sequence of inputs.
        
        Args:
            inputs: Input tokens of shape [batch_size, seq_len, n_tokens, dim]
            train: Whether in training mode
            
        Returns:
            Output tokens of shape [batch_size, seq_len, process_size, dim]
        """
        batch_size, seq_len, _, _ = inputs.shape
        
        # Initialize memory
        memory = self.ttm._broadcast_memory(batch_size)
        
        # Process each step
        output_tokens_list = []
        
        for i in range(seq_len):
            step_tokens = inputs[:, i]
            
            # Process through TTM
            memory, output_tokens = self.ttm(step_tokens, memory, train=train)
            
            # Add to output list
            output_tokens = jnp.expand_dims(output_tokens, axis=1)
            output_tokens_list.append(output_tokens)
        
        # Concatenate outputs
        output_tokens = jnp.concatenate(output_tokens_list, axis=1)
        
        return output_tokens


class TTMMemorylessEncoder(nn.Module):
    """Memory-less version of TTM encoder for comparison.
    
    This implements a memory-less version of the TTM encoder, where the memory
    is zeroed out after each step, as described in the paper for ablation studies.
    """
    
    memory_size: int = 96
    process_size: int = 16
    dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    hidden_dim: int = 512
    summarization_method: str = 'mlp'
    write_method: str = 'summarize'
    use_positional_embedding: bool = True
    dropout_rate: float = 0.1
    
    def setup(self):
        """Set up the memory-less TTM encoder."""
        self.ttm = TokenTuringMachine(
            memory_size=self.memory_size,
            process_size=self.process_size,
            dim=self.dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
            summarization_method=self.summarization_method,
            write_method=self.write_method,
            use_positional_embedding=self.use_positional_embedding,
            dropout_rate=self.dropout_rate
        )
    
    def __call__(self, 
                 inputs: jnp.ndarray, 
                 train: bool = False) -> jnp.ndarray:
        """Apply memory-less TTM encoder to a sequence of inputs.
        
        Args:
            inputs: Input tokens of shape [batch_size, seq_len, n_tokens, dim]
            train: Whether in training mode
            
        Returns:
            Output tokens of shape [batch_size, seq_len, process_size, dim]
        """
        batch_size, seq_len, _, _ = inputs.shape
        
        # Process each step
        output_tokens_list = []
        
        for i in range(seq_len):
            step_tokens = inputs[:, i]
            
            # Initialize memory (zeroed out each time)
            memory = self.ttm._broadcast_memory(batch_size)
            
            # Process through TTM
            _, output_tokens = self.ttm(step_tokens, memory, train=train)
            
            # Add to output list
            output_tokens = jnp.expand_dims(output_tokens, axis=1)
            output_tokens_list.append(output_tokens)
        
        # Concatenate outputs
        output_tokens = jnp.concatenate(output_tokens_list, axis=1)
        
        return output_tokens
