"""
Transformer processing unit for the Token Turing Machine (TTM) model.

This module implements the transformer-based processing unit described in the TTM paper,
which processes the tokens read from memory and input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Any


class MultiHeadAttention(nn.Module):
    """Multi-head attention module.
    
    This implements the multi-head attention mechanism from the "Attention is All You Need" paper,
    with some modifications for the TTM model.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        qkv_bias: bool = True
    ):
        """Initialize the multi-head attention module.
        
        Args:
            dim: Dimension of the input and output embeddings
            num_heads: Number of attention heads
            dropout: Dropout probability
            qkv_bias: Whether to use bias in the query, key, and value projections
        """
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        self.scale = self.head_dim ** -0.5  # Scaling factor for attention scores
        
        # Create query, key, and value projections
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply multi-head attention.
        
        Args:
            q: Query tensor of shape [batch_size, q_len, dim]
            k: Key tensor of shape [batch_size, k_len, dim]
            v: Value tensor of shape [batch_size, v_len, dim]
            attn_mask: Optional attention mask of shape [q_len, k_len] or [batch_size, q_len, k_len]
            key_padding_mask: Optional mask of shape [batch_size, k_len] indicating which keys are padding
            
        Returns:
            Output tensor of shape [batch_size, q_len, dim]
        """
        batch_size, q_len, _ = q.shape
        _, k_len, _ = k.shape
        
        # Project query, key, and value
        q = self.q_proj(q)  # [batch_size, q_len, dim]
        k = self.k_proj(k)  # [batch_size, k_len, dim]
        v = self.v_proj(v)  # [batch_size, v_len, dim]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, q_len, head_dim]
        k = k.view(batch_size, k_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, k_len, head_dim]
        v = v.view(batch_size, k_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, v_len, head_dim]
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch_size, num_heads, q_len, k_len]
        
        # Apply attention mask if provided
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # [q_len, k_len] -> [1, 1, q_len, k_len]
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                # [batch_size, q_len, k_len] -> [batch_size, 1, q_len, k_len]
                attn_mask = attn_mask.unsqueeze(1)
            
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))
        
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            # [batch_size, k_len] -> [batch_size, 1, 1, k_len]
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(key_padding_mask, float('-inf'))
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch_size, num_heads, q_len, k_len]
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, q_len, head_dim]
        
        # Reshape back to original dimensions
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, q_len, self.dim)  # [batch_size, q_len, dim]
        
        # Apply output projection
        output = self.out_proj(attn_output)  # [batch_size, q_len, dim]
        
        return output


class FeedForward(nn.Module):
    """Feed-forward network module.
    
    This implements the feed-forward network from the "Attention is All You Need" paper,
    with some modifications for the TTM model.
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        """Initialize the feed-forward network module.
        
        Args:
            dim: Dimension of the input and output embeddings
            hidden_dim: Dimension of the hidden layer
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu', or 'swish')
        """
        super().__init__()
        
        self.dim = dim
        self.hidden_dim = hidden_dim
        
        # Create linear layers
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        
        # Activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'swish':
            self.activation = lambda x: x * torch.sigmoid(x)
        else:
            raise ValueError(f"Unknown activation function: {activation}")
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward network.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, dim]
        """
        # First linear layer
        h = self.fc1(x)  # [batch_size, seq_len, hidden_dim]
        
        # Apply activation function
        h = self.activation(h)  # [batch_size, seq_len, hidden_dim]
        
        # Apply dropout
        h = self.dropout(h)  # [batch_size, seq_len, hidden_dim]
        
        # Second linear layer
        output = self.fc2(h)  # [batch_size, seq_len, dim]
        
        return output


class TransformerLayer(nn.Module):
    """Transformer layer module.
    
    This implements a single transformer layer from the "Attention is All You Need" paper,
    with some modifications for the TTM model.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        hidden_dim: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu',
        norm_first: bool = True
    ):
        """Initialize the transformer layer module.
        
        Args:
            dim: Dimension of the input and output embeddings
            num_heads: Number of attention heads
            hidden_dim: Dimension of the feed-forward network hidden layer
            dropout: Dropout probability
            activation: Activation function for the feed-forward network
            norm_first: Whether to apply normalization before or after each sub-layer
        """
        super().__init__()
        
        self.dim = dim
        self.norm_first = norm_first
        
        # Multi-head self-attention
        self.self_attn = MultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Feed-forward network
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            activation=activation
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply transformer layer.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            attn_mask: Optional attention mask of shape [seq_len, seq_len] or [batch_size, seq_len, seq_len]
            key_padding_mask: Optional mask of shape [batch_size, seq_len] indicating which keys are padding
            
        Returns:
            Output tensor of shape [batch_size, seq_len, dim]
        """
        # Self-attention sub-layer
        if self.norm_first:
            # Apply normalization before self-attention
            attn_input = self.norm1(x)
            attn_output = self.self_attn(attn_input, attn_input, attn_input, attn_mask, key_padding_mask)
            x = x + self.dropout(attn_output)
            
            # Apply normalization before feed-forward
            ff_input = self.norm2(x)
            ff_output = self.feed_forward(ff_input)
            output = x + self.dropout(ff_output)
        else:
            # Apply self-attention
            attn_output = self.self_attn(x, x, x, attn_mask, key_padding_mask)
            x = self.norm1(x + self.dropout(attn_output))
            
            # Apply feed-forward
            ff_output = self.feed_forward(x)
            output = self.norm2(x + self.dropout(ff_output))
        
        return output


class TransformerEncoder(nn.Module):
    """Transformer encoder module.
    
    This implements the transformer encoder from the "Attention is All You Need" paper,
    with some modifications for the TTM model.
    """
    
    def __init__(
        self,
        dim: int,
        num_layers: int = 4,
        num_heads: int = 8,
        hidden_dim: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu',
        norm_first: bool = True
    ):
        """Initialize the transformer encoder module.
        
        Args:
            dim: Dimension of the input and output embeddings
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            hidden_dim: Dimension of the feed-forward network hidden layer
            dropout: Dropout probability
            activation: Activation function for the feed-forward network
            norm_first: Whether to apply normalization before or after each sub-layer
        """
        super().__init__()
        
        self.dim = dim
        self.num_layers = num_layers
        
        # Create transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(
                dim=dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first
            )
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.norm = nn.LayerNorm(dim)
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply transformer encoder.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            attn_mask: Optional attention mask of shape [seq_len, seq_len] or [batch_size, seq_len, seq_len]
            key_padding_mask: Optional mask of shape [batch_size, seq_len] indicating which keys are padding
            
        Returns:
            Output tensor of shape [batch_size, seq_len, dim]
        """
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, attn_mask, key_padding_mask)
        
        # Apply final normalization
        output = self.norm(x)
        
        return output


class TransformerProcessingUnit(nn.Module):
    """Transformer processing unit for the Token Turing Machine.
    
    This implements the transformer-based processing unit described in the TTM paper,
    which processes the tokens read from memory and input.
    """
    
    def __init__(
        self,
        dim: int,
        num_layers: int = 4,
        num_heads: int = 8,
        hidden_dim: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu',
        norm_first: bool = True
    ):
        """Initialize the transformer processing unit.
        
        Args:
            dim: Dimension of the input and output embeddings
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            hidden_dim: Dimension of the feed-forward network hidden layer
            dropout: Dropout probability
            activation: Activation function for the feed-forward network
            norm_first: Whether to apply normalization before or after each sub-layer
        """
        super().__init__()
        
        self.dim = dim
        
        # Create transformer encoder
        self.transformer = TransformerEncoder(
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply transformer processing unit.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            attn_mask: Optional attention mask of shape [seq_len, seq_len] or [batch_size, seq_len, seq_len]
            key_padding_mask: Optional mask of shape [batch_size, seq_len] indicating which keys are padding
            
        Returns:
            Output tensor of shape [batch_size, seq_len, dim]
        """
        # Apply dropout to input
        x = self.dropout(x)
        
        # Apply transformer encoder
        output = self.transformer(x, attn_mask, key_padding_mask)
        
        return output
