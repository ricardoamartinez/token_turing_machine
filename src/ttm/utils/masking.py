"""
Masking utilities for the Token Turing Machine (TTM) model.

This module provides utilities for handling end-of-sequence (EOS) tokens
and creating attention masks for the transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Any


def create_padding_mask(tokens: torch.Tensor, padding_token: int = 0) -> torch.Tensor:
    """Create a padding mask for attention.
    
    Args:
        tokens: Token indices of shape [batch_size, seq_len]
        padding_token: Token index used for padding
        
    Returns:
        Boolean mask of shape [batch_size, seq_len] where True indicates padding
    """
    # Create mask where True indicates padding
    padding_mask = tokens == padding_token  # [batch_size, seq_len]
    
    return padding_mask


def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """Create a causal mask for attention.
    
    Args:
        seq_len: Length of the sequence
        device: Device to create the mask on
        
    Returns:
        Boolean mask of shape [seq_len, seq_len] where True indicates allowed attention
    """
    # Create a lower triangular mask (1s in lower triangle, 0s elsewhere)
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    
    # Convert to boolean where True indicates allowed attention
    causal_mask = mask.bool()
    
    return causal_mask


def create_combined_mask(
    tokens: torch.Tensor,
    padding_token: int = 0,
    causal: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create combined padding and causal masks for attention.
    
    Args:
        tokens: Token indices of shape [batch_size, seq_len]
        padding_token: Token index used for padding
        causal: Whether to include causal masking
        
    Returns:
        Tuple of (attn_mask, key_padding_mask) where:
            attn_mask: Attention mask of shape [seq_len, seq_len] or None if not causal
            key_padding_mask: Padding mask of shape [batch_size, seq_len]
    """
    batch_size, seq_len = tokens.shape
    device = tokens.device
    
    # Create padding mask
    key_padding_mask = create_padding_mask(tokens, padding_token)
    
    # Create causal mask if needed
    attn_mask = create_causal_mask(seq_len, device) if causal else None
    
    return attn_mask, key_padding_mask


def find_eos_positions(tokens: torch.Tensor, eos_token: int) -> torch.Tensor:
    """Find the positions of the first EOS token in each sequence.
    
    Args:
        tokens: Token indices of shape [batch_size, seq_len]
        eos_token: Token index used for EOS
        
    Returns:
        Tensor of shape [batch_size] containing the position of the first EOS token
        or seq_len if no EOS token is found
    """
    batch_size, seq_len = tokens.shape
    
    # Find positions where tokens equal eos_token
    eos_positions = (tokens == eos_token).int()
    
    # Get the position of the first EOS token in each sequence
    first_eos_positions = eos_positions.argmax(dim=1)
    
    # If no EOS token is found, argmax returns 0, so we need to check if there's an EOS token
    has_eos = eos_positions.sum(dim=1) > 0
    
    # Set position to seq_len if no EOS token is found
    first_eos_positions = torch.where(
        has_eos,
        first_eos_positions,
        torch.tensor(seq_len, device=tokens.device)
    )
    
    return first_eos_positions


def mask_after_eos(
    tokens: torch.Tensor,
    eos_token: int,
    mask_token: int = 0
) -> torch.Tensor:
    """Mask tokens after the first EOS token in each sequence.
    
    Args:
        tokens: Token indices of shape [batch_size, seq_len]
        eos_token: Token index used for EOS
        mask_token: Token index to use for masking
        
    Returns:
        Masked tokens of shape [batch_size, seq_len]
    """
    batch_size, seq_len = tokens.shape
    device = tokens.device
    
    # Find positions of the first EOS token in each sequence
    first_eos_positions = find_eos_positions(tokens, eos_token)
    
    # Create a mask where True indicates positions after the first EOS token
    mask = torch.arange(seq_len, device=device).unsqueeze(0) > first_eos_positions.unsqueeze(1)
    
    # Apply the mask to the tokens
    masked_tokens = torch.where(mask, torch.tensor(mask_token, device=device), tokens)
    
    return masked_tokens


def create_eos_loss_mask(
    tokens: torch.Tensor,
    eos_token: int,
    include_eos: bool = True
) -> torch.Tensor:
    """Create a loss mask that excludes tokens after the first EOS token.
    
    Args:
        tokens: Token indices of shape [batch_size, seq_len]
        eos_token: Token index used for EOS
        include_eos: Whether to include the EOS token in the loss calculation
        
    Returns:
        Boolean mask of shape [batch_size, seq_len] where True indicates tokens to include in loss
    """
    batch_size, seq_len = tokens.shape
    device = tokens.device
    
    # Find positions of the first EOS token in each sequence
    first_eos_positions = find_eos_positions(tokens, eos_token)
    
    # Create a mask where True indicates positions before the first EOS token
    if include_eos:
        # Include positions up to and including the first EOS token
        mask = torch.arange(seq_len, device=device).unsqueeze(0) <= first_eos_positions.unsqueeze(1)
    else:
        # Include positions strictly before the first EOS token
        mask = torch.arange(seq_len, device=device).unsqueeze(0) < first_eos_positions.unsqueeze(1)
    
    return mask


def apply_eos_loss_mask(
    loss: torch.Tensor,
    tokens: torch.Tensor,
    eos_token: int,
    include_eos: bool = True,
    reduction: str = 'mean'
) -> torch.Tensor:
    """Apply a loss mask that excludes tokens after the first EOS token.
    
    Args:
        loss: Per-token loss of shape [batch_size, seq_len]
        tokens: Token indices of shape [batch_size, seq_len]
        eos_token: Token index used for EOS
        include_eos: Whether to include the EOS token in the loss calculation
        reduction: Reduction method ('none', 'mean', or 'sum')
        
    Returns:
        Masked loss of shape [batch_size, seq_len] if reduction='none',
        or a scalar if reduction='mean' or reduction='sum'
    """
    # Create loss mask
    loss_mask = create_eos_loss_mask(tokens, eos_token, include_eos)
    
    # Apply mask to loss
    masked_loss = loss * loss_mask.float()
    
    # Apply reduction
    if reduction == 'none':
        return masked_loss
    elif reduction == 'sum':
        return masked_loss.sum()
    elif reduction == 'mean':
        # Compute mean over non-masked elements
        return masked_loss.sum() / loss_mask.float().sum()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


class EOSCrossEntropyLoss(nn.Module):
    """Cross entropy loss that handles EOS tokens.
    
    This loss function excludes tokens after the first EOS token in each sequence.
    """
    
    def __init__(
        self,
        eos_token: int,
        include_eos: bool = True,
        ignore_index: int = -100,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        """Initialize the EOS cross entropy loss.
        
        Args:
            eos_token: Token index used for EOS
            include_eos: Whether to include the EOS token in the loss calculation
            ignore_index: Target value to ignore in loss calculation
            reduction: Reduction method ('none', 'mean', or 'sum')
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        
        self.eos_token = eos_token
        self.include_eos = include_eos
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute the EOS cross entropy loss.
        
        Args:
            logits: Predicted logits of shape [batch_size, seq_len, vocab_size]
            targets: Target token indices of shape [batch_size, seq_len]
            
        Returns:
            Loss value
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Compute standard cross entropy loss
        if self.reduction == 'none':
            # Compute per-token loss
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                targets.view(-1),
                ignore_index=self.ignore_index,
                reduction='none',
                label_smoothing=self.label_smoothing
            ).view(batch_size, seq_len)
            
            # Apply EOS loss mask
            masked_loss = apply_eos_loss_mask(
                loss,
                targets,
                self.eos_token,
                self.include_eos,
                reduction=self.reduction
            )
            
            return masked_loss
        else:
            # Create loss mask
            loss_mask = create_eos_loss_mask(targets, self.eos_token, self.include_eos)
            
            # Create a new target tensor with ignore_index for masked positions
            masked_targets = torch.where(
                loss_mask,
                targets,
                torch.tensor(self.ignore_index, device=targets.device)
            )
            
            # Compute loss with masked targets
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                masked_targets.view(-1),
                ignore_index=self.ignore_index,
                reduction=self.reduction,
                label_smoothing=self.label_smoothing
            )
            
            return loss
