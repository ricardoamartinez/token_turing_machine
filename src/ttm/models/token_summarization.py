"""Token summarization module for the Token Turing Machine (TTM) model.

This module implements the token summarization methods described in the TTM paper,
which are used to maintain a compact memory representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Any


class MLPSummarizer(nn.Module):
    """MLP-based token summarization module.

    This module uses a multi-layer perceptron to compute importance weights
    for tokens, which are then used to create a weighted sum of the tokens.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        temperature: float = 1.0
    ):
        """Initialize the MLP summarizer.

        Args:
            embedding_dim: Dimension of token embeddings
            hidden_dim: Hidden dimension of the MLP
            num_layers: Number of layers in the MLP
            dropout: Dropout probability
            temperature: Temperature for softmax normalization
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.temperature = temperature

        # Create MLP layers
        layers = []
        input_dim = embedding_dim

        for i in range(num_layers - 1):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        # Final layer outputs a scalar importance weight for each token
        layers.append(nn.Linear(input_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def compute_importance_weights(self, tokens: torch.Tensor) -> torch.Tensor:
        """Compute importance weights for tokens.

        Args:
            tokens: Token embeddings of shape [batch_size, num_tokens, embedding_dim]

        Returns:
            Importance weights of shape [batch_size, num_tokens, 1]
        """
        # Apply MLP to each token
        weights = self.mlp(tokens)  # [batch_size, num_tokens, 1]
        return weights

    def normalize_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """Normalize importance weights using softmax.

        Args:
            weights: Importance weights of shape [batch_size, num_tokens, 1]

        Returns:
            Normalized weights of shape [batch_size, num_tokens, 1]
        """
        # Apply softmax along the token dimension
        # Use temperature to control the sharpness of the distribution
        weights = weights / self.temperature
        weights = F.softmax(weights, dim=1)
        return weights

    def weighted_sum(self, tokens: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Compute weighted sum of tokens.

        Args:
            tokens: Token embeddings of shape [batch_size, num_tokens, embedding_dim]
            weights: Normalized weights of shape [batch_size, num_tokens, 1]

        Returns:
            Weighted sum of tokens of shape [batch_size, 1, embedding_dim]
        """
        # Multiply tokens by weights and sum
        weighted_tokens = tokens * weights  # [batch_size, num_tokens, embedding_dim]
        summed_tokens = weighted_tokens.sum(dim=1, keepdim=True)  # [batch_size, 1, embedding_dim]
        return summed_tokens

    def forward(self, tokens: torch.Tensor, k: int = 1) -> torch.Tensor:
        """Summarize tokens into k tokens.

        Args:
            tokens: Token embeddings of shape [batch_size, num_tokens, embedding_dim]
            k: Number of summary tokens to produce

        Returns:
            Summary tokens of shape [batch_size, k, embedding_dim]
        """
        batch_size, num_tokens, embedding_dim = tokens.shape

        if num_tokens <= k:
            # If we already have fewer tokens than k, just return the original tokens
            # or pad with zeros if necessary
            if num_tokens < k:
                padding = torch.zeros(batch_size, k - num_tokens, embedding_dim, device=tokens.device)
                return torch.cat([tokens, padding], dim=1)
            return tokens

        # Compute importance weights
        weights = self.compute_importance_weights(tokens)  # [batch_size, num_tokens, 1]

        # For multiple summary tokens (k > 1), we need to compute k different sets of weights
        if k > 1:
            # Create k different linear projections
            if not hasattr(self, 'k_projections') or self.k_projections.shape[0] != k:
                self.k_projections = nn.Parameter(torch.randn(k, embedding_dim, 1) / (embedding_dim ** 0.5))

            # Compute k different sets of weights
            k_weights = []
            for i in range(k):
                # Project tokens to get different importance weights
                proj = torch.matmul(tokens, self.k_projections[i])  # [batch_size, num_tokens, 1]
                weights_i = weights + proj  # Add to base weights
                norm_weights_i = self.normalize_weights(weights_i)
                k_weights.append(norm_weights_i)

            # Stack weights and compute weighted sums
            stacked_weights = torch.cat(k_weights, dim=2)  # [batch_size, num_tokens, k]
            stacked_weights = stacked_weights.permute(0, 2, 1).unsqueeze(3)  # [batch_size, k, num_tokens, 1]

            # Expand tokens for batch matrix multiplication
            expanded_tokens = tokens.unsqueeze(1).expand(-1, k, -1, -1)  # [batch_size, k, num_tokens, embedding_dim]

            # Compute weighted sums
            weighted_tokens = expanded_tokens * stacked_weights  # [batch_size, k, num_tokens, embedding_dim]
            summary_tokens = weighted_tokens.sum(dim=2)  # [batch_size, k, embedding_dim]
        else:
            # Normalize weights
            norm_weights = self.normalize_weights(weights)

            # Compute weighted sum
            summary_tokens = self.weighted_sum(tokens, norm_weights)  # [batch_size, 1, embedding_dim]

        return summary_tokens


class QuerySummarizer(nn.Module):
    """Query-based token summarization module.

    This module uses learned query vectors to compute attention weights
    for tokens, which are then used to create a weighted sum of the tokens.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """Initialize the query summarizer.

        Args:
            embedding_dim: Dimension of token embeddings
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        assert self.head_dim * num_heads == embedding_dim, "embedding_dim must be divisible by num_heads"

        # Create query, key, and value projections
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        # Initialize learnable query vectors
        self.query_vectors = nn.Parameter(torch.randn(1, 1, embedding_dim) / (embedding_dim ** 0.5))

    def forward(self, tokens: torch.Tensor, k: int = 1) -> torch.Tensor:
        """Summarize tokens into k tokens using query-based attention.

        Args:
            tokens: Token embeddings of shape [batch_size, num_tokens, embedding_dim]
            k: Number of summary tokens to produce

        Returns:
            Summary tokens of shape [batch_size, k, embedding_dim]
        """
        batch_size, num_tokens, embedding_dim = tokens.shape

        if num_tokens <= k:
            # If we already have fewer tokens than k, just return the original tokens
            # or pad with zeros if necessary
            if num_tokens < k:
                padding = torch.zeros(batch_size, k - num_tokens, embedding_dim, device=tokens.device)
                return torch.cat([tokens, padding], dim=1)
            return tokens

        # Create k query vectors
        if k > 1:
            if not hasattr(self, 'k_query_vectors') or self.k_query_vectors.shape[1] != k:
                self.k_query_vectors = nn.Parameter(torch.randn(1, k, embedding_dim) / (embedding_dim ** 0.5))
            query = self.k_query_vectors.expand(batch_size, -1, -1)
        else:
            query = self.query_vectors.expand(batch_size, -1, -1)

        # Project query, key, and value
        q = self.q_proj(query)  # [batch_size, k, embedding_dim]
        k_proj = self.k_proj(tokens)  # [batch_size, num_tokens, embedding_dim]
        v = self.v_proj(tokens)  # [batch_size, num_tokens, embedding_dim]

        # Reshape for multi-head attention
        q = q.view(batch_size, k, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, k, head_dim]
        k_proj = k_proj.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, num_tokens, head_dim]
        v = v.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, num_tokens, head_dim]

        # Compute attention scores
        scores = torch.matmul(q, k_proj.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [batch_size, num_heads, k, num_tokens]

        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)  # [batch_size, num_heads, k, num_tokens]
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, k, head_dim]

        # Reshape back to original dimensions
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, k, embedding_dim)  # [batch_size, k, embedding_dim]

        # Apply output projection
        summary_tokens = self.out_proj(attn_output)  # [batch_size, k, embedding_dim]

        return summary_tokens


class PoolingSummarizer(nn.Module):
    """Pooling-based token summarization module.

    This module uses average pooling to summarize tokens.
    """

    def __init__(
        self,
        embedding_dim: int,
        pooling_type: str = 'avg'
    ):
        """Initialize the pooling summarizer.

        Args:
            embedding_dim: Dimension of token embeddings
            pooling_type: Type of pooling ('avg' or 'max')
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.pooling_type = pooling_type

        # Create a projection layer to transform pooled tokens
        self.projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, tokens: torch.Tensor, k: int = 1) -> torch.Tensor:
        """Summarize tokens into k tokens using pooling.

        Args:
            tokens: Token embeddings of shape [batch_size, num_tokens, embedding_dim]
            k: Number of summary tokens to produce

        Returns:
            Summary tokens of shape [batch_size, k, embedding_dim]
        """
        batch_size, num_tokens, embedding_dim = tokens.shape

        if num_tokens <= k:
            # If we already have fewer tokens than k, just return the original tokens
            # or pad with zeros if necessary
            if num_tokens < k:
                padding = torch.zeros(batch_size, k - num_tokens, embedding_dim, device=tokens.device)
                return torch.cat([tokens, padding], dim=1)
            return tokens

        # For k > 1, we need to divide the tokens into k groups
        if k > 1:
            # Compute number of tokens per group
            tokens_per_group = num_tokens // k
            remainder = num_tokens % k

            # Create summary tokens
            summary_tokens = []
            start_idx = 0

            for i in range(k):
                # Determine end index for this group
                end_idx = start_idx + tokens_per_group + (1 if i < remainder else 0)

                # Get tokens for this group
                group_tokens = tokens[:, start_idx:end_idx, :]

                # Apply pooling
                if self.pooling_type == 'avg':
                    pooled = group_tokens.mean(dim=1, keepdim=True)  # [batch_size, 1, embedding_dim]
                elif self.pooling_type == 'max':
                    pooled, _ = group_tokens.max(dim=1, keepdim=True)  # [batch_size, 1, embedding_dim]
                else:
                    raise ValueError(f"Unknown pooling type: {self.pooling_type}")

                # Apply projection
                projected = self.projection(pooled)  # [batch_size, 1, embedding_dim]

                summary_tokens.append(projected)
                start_idx = end_idx

            # Concatenate summary tokens
            summary_tokens = torch.cat(summary_tokens, dim=1)  # [batch_size, k, embedding_dim]
        else:
            # Apply pooling to all tokens
            if self.pooling_type == 'avg':
                pooled = tokens.mean(dim=1, keepdim=True)  # [batch_size, 1, embedding_dim]
            elif self.pooling_type == 'max':
                pooled, _ = tokens.max(dim=1, keepdim=True)  # [batch_size, 1, embedding_dim]
            else:
                raise ValueError(f"Unknown pooling type: {self.pooling_type}")

            # Apply projection
            summary_tokens = self.projection(pooled)  # [batch_size, 1, embedding_dim]

        return summary_tokens


def token_summarize(
    tokens: torch.Tensor,
    k: int = 5,
    method: str = 'mlp',
    summarizer: Optional[nn.Module] = None,
    **kwargs
) -> torch.Tensor:
    """Summarize tokens into k tokens.

    Args:
        tokens: Token embeddings of shape [batch_size, num_tokens, embedding_dim]
        k: Number of summary tokens to produce
        method: Summarization method ('mlp', 'query', or 'pooling')
        summarizer: Optional pre-initialized summarizer module
        **kwargs: Additional arguments to pass to the summarizer

    Returns:
        Summary tokens of shape [batch_size, k, embedding_dim]
    """
    batch_size, num_tokens, embedding_dim = tokens.shape

    if num_tokens <= k:
        # If we already have fewer tokens than k, just return the original tokens
        # or pad with zeros if necessary
        if num_tokens < k:
            padding = torch.zeros(batch_size, k - num_tokens, embedding_dim, device=tokens.device)
            return torch.cat([tokens, padding], dim=1)
        return tokens

    # Use provided summarizer or create a new one
    if summarizer is None:
        if method == 'mlp':
            summarizer = MLPSummarizer(embedding_dim, **kwargs)
        elif method == 'query':
            summarizer = QuerySummarizer(embedding_dim, **kwargs)
        elif method == 'pooling':
            summarizer = PoolingSummarizer(embedding_dim, **kwargs)
        else:
            raise ValueError(f"Unknown summarization method: {method}")

    # Apply summarizer
    summary_tokens = summarizer(tokens, k)

    return summary_tokens
