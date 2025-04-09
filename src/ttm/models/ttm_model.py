"""
Token Turing Machine (TTM) model implementation.

This module implements the complete TTM model as described in the paper,
integrating the token summarization, memory operations, and transformer processing unit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Any

from .token_summarization import token_summarize, MLPSummarizer, QuerySummarizer, PoolingSummarizer
from .memory_operations import MemoryModule
from .transformer_unit import TransformerProcessingUnit
from ..utils.masking import create_combined_mask, create_causal_mask, mask_after_eos, EOSCrossEntropyLoss
from ..utils.losses import TTMLoss, create_loss_function


class TokenEmbedding(nn.Module):
    """Token embedding module for the Token Turing Machine.

    This module converts token indices to embeddings and adds positional embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        max_seq_len: int = 128,
        dropout: float = 0.1
    ):
        """Initialize the token embedding module.

        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of the embeddings
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

        # Positional embedding (learnable)
        self.positional_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, embedding_dim) / (embedding_dim ** 0.5)
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert token indices to embeddings.

        Args:
            tokens: Token indices of shape [batch_size, seq_len]

        Returns:
            Token embeddings of shape [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len = tokens.shape

        # Get token embeddings
        embeddings = self.token_embedding(tokens)  # [batch_size, seq_len, embedding_dim]

        # Add positional embeddings
        positions = self.positional_embedding[:, :seq_len, :]  # [1, seq_len, embedding_dim]
        embeddings = embeddings + positions  # [batch_size, seq_len, embedding_dim]

        # Apply layer normalization
        embeddings = self.layer_norm(embeddings)  # [batch_size, seq_len, embedding_dim]

        # Apply dropout
        embeddings = self.dropout(embeddings)  # [batch_size, seq_len, embedding_dim]

        return embeddings


class OutputHead(nn.Module):
    """Output head for the Token Turing Machine.

    This module converts the output of the transformer processing unit to token probabilities.
    """

    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int,
        dropout: float = 0.1
    ):
        """Initialize the output head module.

        Args:
            embedding_dim: Dimension of the input embeddings
            vocab_size: Size of the vocabulary
            dropout: Dropout probability
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # Output projection
        self.output_projection = nn.Linear(embedding_dim, vocab_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert embeddings to token probabilities.

        Args:
            x: Input embeddings of shape [batch_size, seq_len, embedding_dim]

        Returns:
            Token logits of shape [batch_size, seq_len, vocab_size]
        """
        # Apply layer normalization
        x = self.layer_norm(x)  # [batch_size, seq_len, embedding_dim]

        # Apply dropout
        x = self.dropout(x)  # [batch_size, seq_len, embedding_dim]

        # Apply output projection
        logits = self.output_projection(x)  # [batch_size, seq_len, vocab_size]

        return logits


class TokenTuringMachine(nn.Module):
    """Token Turing Machine (TTM) model.

    This implements the complete TTM model as described in the paper,
    integrating the token summarization, memory operations, and transformer processing unit.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 512,
        memory_size: int = 96,
        r: int = 16,
        num_layers: int = 4,
        num_heads: int = 8,
        hidden_dim: int = 2048,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        summarization_method: str = 'mlp',
        write_method: str = 'summarization',
        memory_less: bool = False,
        padding_token: int = 0,
        eos_token: Optional[int] = None,
        causal_attention: bool = True
    ):
        """Initialize the Token Turing Machine model.

        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of the embeddings
            memory_size: Number of tokens in memory
            r: Number of tokens to produce after summarization
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            hidden_dim: Dimension of the feed-forward network hidden layer
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            summarization_method: Method for token summarization ('mlp', 'query', or 'pooling')
            write_method: Method for memory write ('summarization', 'erase_add', or 'concat')
            memory_less: Whether to use a memory-less version of the model
            padding_token: Token index used for padding
            eos_token: Optional token index used for end-of-sequence
            causal_attention: Whether to use causal attention masking
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.memory_size = memory_size
        self.r = r
        self.memory_less = memory_less
        self.padding_token = padding_token
        self.eos_token = eos_token
        self.causal_attention = causal_attention

        # Token embedding
        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_seq_len=max_seq_len,
            dropout=dropout
        )

        # Memory module
        if not memory_less:
            self.memory_module = MemoryModule(
                embedding_dim=embedding_dim,
                memory_size=memory_size,
                r=r,
                summarization_method=summarization_method,
                write_method=write_method,
                dropout=dropout
            )

        # Transformer processing unit
        self.transformer = TransformerProcessingUnit(
            dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        # Output head
        self.output_head = OutputHead(
            embedding_dim=embedding_dim,
            vocab_size=vocab_size,
            dropout=dropout
        )

        # Initialize memory with zeros
        if not memory_less:
            self.register_buffer('initial_memory', torch.zeros(1, memory_size, embedding_dim))

    def initialize_memory(self, batch_size: int) -> torch.Tensor:
        """Initialize memory for a new batch.

        Args:
            batch_size: Batch size

        Returns:
            Initialized memory of shape [batch_size, memory_size, embedding_dim]
        """
        if self.memory_less:
            return None

        return self.initial_memory.expand(batch_size, -1, -1)

    def forward(
        self,
        tokens: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        mask_eos: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply the Token Turing Machine model.

        Args:
            tokens: Token indices of shape [batch_size, seq_len]
            memory: Optional memory tokens of shape [batch_size, memory_size, embedding_dim]
            attn_mask: Optional attention mask of shape [seq_len, seq_len] or [batch_size, seq_len, seq_len]
            key_padding_mask: Optional mask of shape [batch_size, seq_len] indicating which keys are padding
            mask_eos: Whether to mask tokens after the first EOS token

        Returns:
            Tuple of (output_logits, new_memory) where:
                output_logits: Token logits of shape [batch_size, seq_len, vocab_size]
                new_memory: Updated memory of shape [batch_size, memory_size, embedding_dim] or None if memory_less
        """
        batch_size, seq_len = tokens.shape
        device = tokens.device

        # Mask tokens after EOS if needed
        if mask_eos and self.eos_token is not None:
            tokens = mask_after_eos(tokens, self.eos_token, self.padding_token)

        # Create attention masks if not provided
        if attn_mask is None and key_padding_mask is None:
            attn_mask, key_padding_mask = create_combined_mask(
                tokens,
                padding_token=self.padding_token,
                causal=self.causal_attention
            )

        # Initialize memory if not provided
        if memory is None and not self.memory_less:
            memory = self.initialize_memory(batch_size)

        # Convert tokens to embeddings
        token_embeddings = self.token_embedding(tokens)  # [batch_size, seq_len, embedding_dim]

        # Apply memory operations if not memory-less
        if not self.memory_less and memory is not None:
            # Read from memory
            read_tokens = self.memory_module.read(memory, token_embeddings)  # [batch_size, r, embedding_dim]

            # Concatenate read tokens with input tokens
            combined_tokens = torch.cat([read_tokens, token_embeddings], dim=1)  # [batch_size, r + seq_len, embedding_dim]

            # Update memory
            new_memory = self.memory_module.write(memory, token_embeddings)  # [batch_size, memory_size, embedding_dim]

            # Adjust attention masks for the combined tokens if needed
            if attn_mask is not None:
                # Create a new attention mask for the combined sequence
                combined_seq_len = combined_tokens.shape[1]
                if self.causal_attention:
                    # Create a causal mask for the combined sequence
                    combined_attn_mask = create_causal_mask(combined_seq_len, device=device)
                else:
                    # Create a full attention mask for the combined sequence
                    combined_attn_mask = torch.ones(combined_seq_len, combined_seq_len, device=device).bool()

                attn_mask = combined_attn_mask

            if key_padding_mask is not None:
                # Create a new key padding mask for the combined sequence
                # Assume no padding in the read tokens
                read_padding_mask = torch.zeros(batch_size, self.r, device=device, dtype=torch.bool)
                combined_key_padding_mask = torch.cat([read_padding_mask, key_padding_mask], dim=1)

                key_padding_mask = combined_key_padding_mask
        else:
            # Use only input tokens
            combined_tokens = token_embeddings  # [batch_size, seq_len, embedding_dim]
            new_memory = None

        # Apply transformer processing unit
        transformer_output, attention_weights = self.transformer(combined_tokens, attn_mask, key_padding_mask, return_attention=True)  # [batch_size, r + seq_len, embedding_dim] or [batch_size, seq_len, embedding_dim]

        # Store attention weights for visualization
        self.last_attention_weights = attention_weights

        # Extract the output tokens (excluding memory tokens if not memory-less)
        if not self.memory_less and memory is not None:
            output_tokens = transformer_output[:, self.r:, :]  # [batch_size, seq_len, embedding_dim]
        else:
            output_tokens = transformer_output  # [batch_size, seq_len, embedding_dim]

        # Apply output head
        output_logits = self.output_head(output_tokens)  # [batch_size, seq_len, vocab_size]

        return output_logits, new_memory

    def generate(
        self,
        tokens: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token: Optional[int] = None
    ) -> torch.Tensor:
        """Generate tokens autoregressively.

        Args:
            tokens: Initial token indices of shape [batch_size, seq_len]
            max_length: Maximum length of the generated sequence
            temperature: Temperature for sampling
            top_k: If specified, only sample from the top k most likely tokens
            top_p: If specified, only sample from the top tokens with cumulative probability >= top_p
            eos_token: If specified, stop generation when this token is generated

        Returns:
            Generated token indices of shape [batch_size, max_length]
        """
        batch_size, seq_len = tokens.shape
        device = tokens.device

        # Initialize memory
        memory = self.initialize_memory(batch_size) if not self.memory_less else None

        # Initialize generated tokens with input tokens
        generated_tokens = tokens.clone()

        # Generate tokens autoregressively
        for i in range(seq_len, max_length):
            # Get logits for the next token
            logits, memory = self(generated_tokens, memory)  # [batch_size, i, vocab_size]

            # Get logits for the last token
            next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Apply top-k sampling
            if top_k is not None:
                indices_to_remove = torch.topk(next_token_logits, top_k, dim=-1)[0][:, -1].unsqueeze(-1)
                next_token_logits = torch.where(
                    next_token_logits < indices_to_remove,
                    torch.ones_like(next_token_logits) * float('-inf'),
                    next_token_logits
                )

            # Apply top-p (nucleus) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p

                # Shift the indices to the right to keep the first token above the threshold
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0

                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                next_token_logits = torch.where(
                    indices_to_remove,
                    torch.ones_like(next_token_logits) * float('-inf'),
                    next_token_logits
                )

            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]

            # Append next token to generated tokens
            generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)

            # Check if all sequences have generated EOS token
            if eos_token is not None:
                eos_generated = (generated_tokens == eos_token).any(dim=-1)
                if eos_generated.all():
                    break

        return generated_tokens

    def create_loss_fn(
        self,
        loss_type: str = 'cross_entropy',
        ignore_index: int = -100,
        reduction: str = 'mean',
        label_smoothing: float = 0.0,
        include_eos: bool = True,
        memory_loss_weight: float = 0.0,
        attention_loss_weight: float = 0.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ) -> nn.Module:
        """Create a loss function for training.

        Args:
            loss_type: Type of loss function ('cross_entropy', 'label_smoothing', 'focal', or 'ttm')
            ignore_index: Target value to ignore in loss calculation
            reduction: Reduction method ('none', 'mean', or 'sum')
            label_smoothing: Label smoothing factor
            include_eos: Whether to include the EOS token in the loss calculation
            memory_loss_weight: Weight for the memory consistency loss
            attention_loss_weight: Weight for the attention entropy loss
            focal_alpha: Weighting factor for the rare class in focal loss
            focal_gamma: Focusing parameter in focal loss

        Returns:
            Loss function module
        """
        return create_loss_function(
            loss_type=loss_type,
            eos_token=self.eos_token,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
            include_eos=include_eos,
            memory_loss_weight=memory_loss_weight,
            attention_loss_weight=attention_loss_weight,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma
        )
