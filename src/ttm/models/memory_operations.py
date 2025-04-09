"""Memory operations module for the Token Turing Machine (TTM) model.

This module implements the memory operations described in the TTM paper,
including the unified memory-input reading strategy and token summarization-based
memory write operation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Any

from .token_summarization import token_summarize, MLPSummarizer, QuerySummarizer, PoolingSummarizer


class MemoryReadOperation(nn.Module):
    """Memory read operation for the Token Turing Machine.

    This implements the unified memory-input reading strategy described in the TTM paper,
    which concatenates memory and input tokens and applies token summarization to reduce
    the combined sequence to a fixed number of tokens.
    """

    def __init__(
        self,
        embedding_dim: int,
        r: int = 16,
        summarization_method: str = 'mlp',
        max_memory_tokens: int = 96,
        dropout: float = 0.1
    ):
        """Initialize the memory read operation.

        Args:
            embedding_dim: Dimension of token embeddings
            r: Number of tokens to produce after summarization (default: 16 as in TTM paper)
            summarization_method: Method for token summarization ('mlp', 'query', or 'pooling')
            max_memory_tokens: Maximum number of tokens in memory (default: 96 as in TTM paper)
            dropout: Dropout probability
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.r = r
        self.summarization_method = summarization_method
        self.max_memory_tokens = max_memory_tokens

        # Create token summarizer
        if summarization_method == 'mlp':
            self.summarizer = MLPSummarizer(embedding_dim, dropout=dropout)
        elif summarization_method == 'query':
            self.summarizer = QuerySummarizer(embedding_dim, dropout=dropout)
        elif summarization_method == 'pooling':
            self.summarizer = PoolingSummarizer(embedding_dim)
        else:
            raise ValueError(f"Unknown summarization method: {summarization_method}")

        # Create learnable positional embeddings to distinguish memory from input
        self.memory_pos_embedding = nn.Parameter(torch.randn(1, 1, embedding_dim) / (embedding_dim ** 0.5))
        self.input_pos_embedding = nn.Parameter(torch.randn(1, 1, embedding_dim) / (embedding_dim ** 0.5))

        # Layer normalization for the combined sequence
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def concat_memory_input(self, memory: torch.Tensor, input_tokens: torch.Tensor) -> torch.Tensor:
        """Concatenate memory and input tokens.

        Args:
            memory: Memory tokens of shape [batch_size, memory_size, embedding_dim]
            input_tokens: Input tokens of shape [batch_size, input_size, embedding_dim]

        Returns:
            Concatenated tokens of shape [batch_size, memory_size + input_size, embedding_dim]
        """
        return torch.cat([memory, input_tokens], dim=1)

    def add_positional_info(self, memory: torch.Tensor, input_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add positional embeddings to distinguish memory from input.

        Args:
            memory: Memory tokens of shape [batch_size, memory_size, embedding_dim]
            input_tokens: Input tokens of shape [batch_size, input_size, embedding_dim]

        Returns:
            Tuple of (memory_with_pos, input_with_pos) with positional embeddings added
        """
        batch_size, memory_size, embedding_dim = memory.shape
        _, input_size, _ = input_tokens.shape

        # Add positional embeddings
        memory_pos = self.memory_pos_embedding.expand(batch_size, memory_size, embedding_dim)
        input_pos = self.input_pos_embedding.expand(batch_size, input_size, embedding_dim)

        memory_with_pos = memory + memory_pos
        input_with_pos = input_tokens + input_pos

        return memory_with_pos, input_with_pos

    def forward(self, memory: torch.Tensor, input_tokens: torch.Tensor) -> torch.Tensor:
        """Apply memory read operation.

        Args:
            memory: Memory tokens of shape [batch_size, memory_size, embedding_dim]
            input_tokens: Input tokens of shape [batch_size, input_size, embedding_dim]

        Returns:
            Summarized tokens of shape [batch_size, r, embedding_dim]
        """
        # Add positional embeddings
        memory_with_pos, input_with_pos = self.add_positional_info(memory, input_tokens)

        # Concatenate memory and input
        combined = self.concat_memory_input(memory_with_pos, input_with_pos)

        # Apply layer normalization
        combined = self.layer_norm(combined)

        # Apply dropout
        combined = self.dropout(combined)

        # Apply token summarization to reduce to r tokens
        summarized = self.summarizer(combined, k=self.r)

        return summarized


class MemoryWriteOperation(nn.Module):
    """Memory write operation for the Token Turing Machine.

    This implements the token summarization-based memory write operation described in the TTM paper,
    which updates the memory based on the current memory and the new tokens to be written.
    """

    def __init__(
        self,
        embedding_dim: int,
        memory_size: int = 96,
        write_method: str = 'summarization',
        summarization_method: str = 'mlp',
        dropout: float = 0.1
    ):
        """Initialize the memory write operation.

        Args:
            embedding_dim: Dimension of token embeddings
            memory_size: Number of tokens in memory (default: 96 as in TTM paper)
            write_method: Method for memory write ('summarization', 'erase_add', or 'concat')
            summarization_method: Method for token summarization ('mlp', 'query', or 'pooling')
            dropout: Dropout probability
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.memory_size = memory_size
        self.write_method = write_method
        self.summarization_method = summarization_method

        # Create token summarizer for summarization-based write
        if summarization_method == 'mlp':
            self.summarizer = MLPSummarizer(embedding_dim, dropout=dropout)
        elif summarization_method == 'query':
            self.summarizer = QuerySummarizer(embedding_dim, dropout=dropout)
        elif summarization_method == 'pooling':
            self.summarizer = PoolingSummarizer(embedding_dim)
        else:
            raise ValueError(f"Unknown summarization method: {summarization_method}")

        # For erase-add method
        if write_method == 'erase_add':
            # Create erase and add gates
            self.erase_gate = nn.Linear(embedding_dim, embedding_dim)
            self.add_gate = nn.Linear(embedding_dim, embedding_dim)

            # Create attention mechanism for selective writing
            self.query_proj = nn.Linear(embedding_dim, embedding_dim)
            self.key_proj = nn.Linear(embedding_dim, embedding_dim)
            self.attn_proj = nn.Linear(embedding_dim, 1)

        # Layer normalization for the output memory
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def summarization_write(self, memory: torch.Tensor, write_tokens: torch.Tensor) -> torch.Tensor:
        """Update memory using token summarization.

        Args:
            memory: Memory tokens of shape [batch_size, memory_size, embedding_dim]
            write_tokens: Tokens to write of shape [batch_size, write_size, embedding_dim]

        Returns:
            Updated memory of shape [batch_size, memory_size, embedding_dim]
        """
        # Concatenate memory and write tokens
        combined = torch.cat([memory, write_tokens], dim=1)

        # Apply token summarization to reduce to memory_size tokens
        new_memory = self.summarizer(combined, k=self.memory_size)

        return new_memory

    def erase_add_write(self, memory: torch.Tensor, write_tokens: torch.Tensor) -> torch.Tensor:
        """Update memory using erase-add mechanism.

        Args:
            memory: Memory tokens of shape [batch_size, memory_size, embedding_dim]
            write_tokens: Tokens to write of shape [batch_size, write_size, embedding_dim]

        Returns:
            Updated memory of shape [batch_size, memory_size, embedding_dim]
        """
        batch_size, memory_size, embedding_dim = memory.shape
        _, write_size, _ = write_tokens.shape

        # Compute attention between write tokens and memory
        # This determines which memory locations to update
        queries = self.query_proj(write_tokens)  # [batch_size, write_size, embedding_dim]
        keys = self.key_proj(memory)  # [batch_size, memory_size, embedding_dim]

        # Compute attention scores
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (embedding_dim ** 0.5)  # [batch_size, write_size, memory_size]
        attn_weights = F.softmax(scores, dim=-1)  # [batch_size, write_size, memory_size]

        # Compute erase and add vectors for each write token
        erase_vectors = torch.sigmoid(self.erase_gate(write_tokens))  # [batch_size, write_size, embedding_dim]
        add_vectors = torch.tanh(self.add_gate(write_tokens))  # [batch_size, write_size, embedding_dim]

        # Initialize new memory as a copy of the old memory
        new_memory = memory.clone()

        # Apply erase and add operations for each write token
        for i in range(write_size):
            # Get attention weights for this write token
            attn_i = attn_weights[:, i, :].unsqueeze(-1)  # [batch_size, memory_size, 1]

            # Get erase and add vectors for this write token
            erase_i = erase_vectors[:, i, :].unsqueeze(1)  # [batch_size, 1, embedding_dim]
            add_i = add_vectors[:, i, :].unsqueeze(1)  # [batch_size, 1, embedding_dim]

            # Apply erase operation
            erase_mask = 1 - attn_i * erase_i  # [batch_size, memory_size, embedding_dim]
            new_memory = new_memory * erase_mask

            # Apply add operation
            add_values = attn_i * add_i  # [batch_size, memory_size, embedding_dim]
            new_memory = new_memory + add_values

        return new_memory

    def concat_write(self, memory: torch.Tensor, write_tokens: torch.Tensor) -> torch.Tensor:
        """Update memory by concatenating and truncating.

        Args:
            memory: Memory tokens of shape [batch_size, memory_size, embedding_dim]
            write_tokens: Tokens to write of shape [batch_size, write_size, embedding_dim]

        Returns:
            Updated memory of shape [batch_size, memory_size, embedding_dim]
        """
        # Concatenate write tokens to the end of memory
        combined = torch.cat([memory, write_tokens], dim=1)

        # Keep only the most recent memory_size tokens
        new_memory = combined[:, -self.memory_size:, :]

        return new_memory

    def forward(self, memory: torch.Tensor, write_tokens: torch.Tensor) -> torch.Tensor:
        """Apply memory write operation.

        Args:
            memory: Memory tokens of shape [batch_size, memory_size, embedding_dim]
            write_tokens: Tokens to write of shape [batch_size, write_size, embedding_dim]

        Returns:
            Updated memory of shape [batch_size, memory_size, embedding_dim]
        """
        # Apply the selected write method
        if self.write_method == 'summarization':
            new_memory = self.summarization_write(memory, write_tokens)
        elif self.write_method == 'erase_add':
            new_memory = self.erase_add_write(memory, write_tokens)
        elif self.write_method == 'concat':
            new_memory = self.concat_write(memory, write_tokens)
        else:
            raise ValueError(f"Unknown write method: {self.write_method}")

        # Apply layer normalization
        new_memory = self.layer_norm(new_memory)

        # Apply dropout
        new_memory = self.dropout(new_memory)

        return new_memory


class MemoryModule(nn.Module):
    """Memory module for the Token Turing Machine.

    This module combines the memory read and write operations into a single module
    that manages the memory state.
    """

    def __init__(
        self,
        embedding_dim: int,
        memory_size: int = 96,
        r: int = 16,
        summarization_method: str = 'mlp',
        write_method: str = 'summarization',
        dropout: float = 0.1
    ):
        """Initialize the memory module.

        Args:
            embedding_dim: Dimension of token embeddings
            memory_size: Number of tokens in memory (default: 96 as in TTM paper)
            r: Number of tokens to produce after summarization (default: 16 as in TTM paper)
            summarization_method: Method for token summarization ('mlp', 'query', or 'pooling')
            write_method: Method for memory write ('summarization', 'erase_add', or 'concat')
            dropout: Dropout probability
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.memory_size = memory_size
        self.r = r

        # Create memory read and write operations
        self.read_op = MemoryReadOperation(
            embedding_dim=embedding_dim,
            r=r,
            summarization_method=summarization_method,
            max_memory_tokens=memory_size,
            dropout=dropout
        )

        self.write_op = MemoryWriteOperation(
            embedding_dim=embedding_dim,
            memory_size=memory_size,
            write_method=write_method,
            summarization_method=summarization_method,
            dropout=dropout
        )

        # Initialize memory with zeros
        self.register_buffer('initial_memory', torch.zeros(1, memory_size, embedding_dim))

    def initialize_memory(self, batch_size: int) -> torch.Tensor:
        """Initialize memory for a new batch.

        Args:
            batch_size: Batch size

        Returns:
            Initialized memory of shape [batch_size, memory_size, embedding_dim]
        """
        return self.initial_memory.expand(batch_size, -1, -1)

    def read(self, memory: torch.Tensor, input_tokens: torch.Tensor) -> torch.Tensor:
        """Read from memory.

        Args:
            memory: Memory tokens of shape [batch_size, memory_size, embedding_dim]
            input_tokens: Input tokens of shape [batch_size, input_size, embedding_dim]

        Returns:
            Summarized tokens of shape [batch_size, r, embedding_dim]
        """
        return self.read_op(memory, input_tokens)

    def write(self, memory: torch.Tensor, write_tokens: torch.Tensor) -> torch.Tensor:
        """Write to memory.

        Args:
            memory: Memory tokens of shape [batch_size, memory_size, embedding_dim]
            write_tokens: Tokens to write of shape [batch_size, write_size, embedding_dim]

        Returns:
            Updated memory of shape [batch_size, memory_size, embedding_dim]
        """
        return self.write_op(memory, write_tokens)

    def forward(self, memory: torch.Tensor, input_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply memory read and write operations.

        Args:
            memory: Memory tokens of shape [batch_size, memory_size, embedding_dim]
            input_tokens: Input tokens of shape [batch_size, input_size, embedding_dim]

        Returns:
            Tuple of (read_tokens, new_memory) where:
                read_tokens: Summarized tokens of shape [batch_size, r, embedding_dim]
                new_memory: Updated memory of shape [batch_size, memory_size, embedding_dim]
        """
        # Read from memory
        read_tokens = self.read(memory, input_tokens)

        # Write to memory
        new_memory = self.write(memory, input_tokens)

        return read_tokens, new_memory
