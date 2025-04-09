"""
Loss functions for the Token Turing Machine (TTM) model.

This module provides loss functions for training the TTM model,
including cross-entropy loss, label smoothing, and auxiliary losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Any, Union

from .masking import create_eos_loss_mask, EOSCrossEntropyLoss


class TTMLoss(nn.Module):
    """Loss function for the Token Turing Machine.
    
    This loss function combines cross-entropy loss for token prediction
    with optional auxiliary losses for memory and attention.
    """
    
    def __init__(
        self,
        eos_token: Optional[int] = None,
        ignore_index: int = -100,
        reduction: str = 'mean',
        label_smoothing: float = 0.0,
        include_eos: bool = True,
        memory_loss_weight: float = 0.0,
        attention_loss_weight: float = 0.0
    ):
        """Initialize the TTM loss function.
        
        Args:
            eos_token: Optional token index used for end-of-sequence
            ignore_index: Target value to ignore in loss calculation
            reduction: Reduction method ('none', 'mean', or 'sum')
            label_smoothing: Label smoothing factor
            include_eos: Whether to include the EOS token in the loss calculation
            memory_loss_weight: Weight for the memory consistency loss
            attention_loss_weight: Weight for the attention entropy loss
        """
        super().__init__()
        
        self.eos_token = eos_token
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.include_eos = include_eos
        self.memory_loss_weight = memory_loss_weight
        self.attention_loss_weight = attention_loss_weight
        
        # Create the main token prediction loss
        if eos_token is not None:
            self.token_loss_fn = EOSCrossEntropyLoss(
                eos_token=eos_token,
                include_eos=include_eos,
                ignore_index=ignore_index,
                reduction=reduction,
                label_smoothing=label_smoothing
            )
        else:
            self.token_loss_fn = nn.CrossEntropyLoss(
                ignore_index=ignore_index,
                reduction=reduction,
                label_smoothing=label_smoothing
            )
    
    def compute_token_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute the token prediction loss.
        
        Args:
            logits: Predicted logits of shape [batch_size, seq_len, vocab_size]
            targets: Target token indices of shape [batch_size, seq_len]
            
        Returns:
            Token prediction loss
        """
        return self.token_loss_fn(logits, targets)
    
    def compute_memory_consistency_loss(
        self,
        memory_before: torch.Tensor,
        memory_after: torch.Tensor
    ) -> torch.Tensor:
        """Compute the memory consistency loss.
        
        This loss encourages the memory to be consistent across time steps,
        preventing it from changing too drastically.
        
        Args:
            memory_before: Memory before update of shape [batch_size, memory_size, embedding_dim]
            memory_after: Memory after update of shape [batch_size, memory_size, embedding_dim]
            
        Returns:
            Memory consistency loss
        """
        if self.memory_loss_weight <= 0.0:
            return torch.tensor(0.0, device=memory_before.device)
        
        # Compute L2 distance between memory states
        memory_diff = memory_after - memory_before
        memory_loss = torch.norm(memory_diff, p=2, dim=-1).mean()
        
        return memory_loss * self.memory_loss_weight
    
    def compute_attention_entropy_loss(
        self,
        attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """Compute the attention entropy loss.
        
        This loss encourages the attention weights to be more focused,
        by minimizing the entropy of the attention distribution.
        
        Args:
            attention_weights: Attention weights of shape [batch_size, num_heads, q_len, k_len]
            
        Returns:
            Attention entropy loss
        """
        if self.attention_loss_weight <= 0.0:
            return torch.tensor(0.0, device=attention_weights.device)
        
        # Compute entropy of attention weights
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        entropy = -torch.sum(
            attention_weights * torch.log(attention_weights + epsilon),
            dim=-1
        ).mean()
        
        return entropy * self.attention_loss_weight
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        memory_before: Optional[torch.Tensor] = None,
        memory_after: Optional[torch.Tensor] = None,
        attention_weights: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the combined loss.
        
        Args:
            logits: Predicted logits of shape [batch_size, seq_len, vocab_size]
            targets: Target token indices of shape [batch_size, seq_len]
            memory_before: Optional memory before update of shape [batch_size, memory_size, embedding_dim]
            memory_after: Optional memory after update of shape [batch_size, memory_size, embedding_dim]
            attention_weights: Optional attention weights of shape [batch_size, num_heads, q_len, k_len]
            
        Returns:
            Combined loss or dictionary of losses if reduction='none'
        """
        # Compute token prediction loss
        token_loss = self.compute_token_loss(logits, targets)
        
        # Compute memory consistency loss if memory states are provided
        memory_loss = torch.tensor(0.0, device=logits.device)
        if memory_before is not None and memory_after is not None:
            memory_loss = self.compute_memory_consistency_loss(memory_before, memory_after)
        
        # Compute attention entropy loss if attention weights are provided
        attention_loss = torch.tensor(0.0, device=logits.device)
        if attention_weights is not None:
            attention_loss = self.compute_attention_entropy_loss(attention_weights)
        
        # Combine losses
        total_loss = token_loss + memory_loss + attention_loss
        
        # Return combined loss or dictionary of losses
        if self.reduction == 'none':
            return {
                'total': total_loss,
                'token': token_loss,
                'memory': memory_loss,
                'attention': attention_loss
            }
        else:
            return total_loss


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss.
    
    This loss function applies label smoothing to the target distribution,
    which can help prevent the model from becoming overconfident.
    """
    
    def __init__(
        self,
        smoothing: float = 0.1,
        ignore_index: int = -100,
        reduction: str = 'mean'
    ):
        """Initialize the label smoothing loss.
        
        Args:
            smoothing: Label smoothing factor (0.0 means no smoothing)
            ignore_index: Target value to ignore in loss calculation
            reduction: Reduction method ('none', 'mean', or 'sum')
        """
        super().__init__()
        
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute the label smoothing loss.
        
        Args:
            logits: Predicted logits of shape [batch_size, seq_len, vocab_size]
            targets: Target token indices of shape [batch_size, seq_len]
            
        Returns:
            Label smoothing loss
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Create a mask for ignored indices
        ignore_mask = targets != self.ignore_index  # [batch_size, seq_len]
        
        # Create one-hot encoding of targets
        targets_one_hot = F.one_hot(
            torch.clamp(targets, min=0),  # Clamp to avoid negative indices
            num_classes=vocab_size
        ).float()  # [batch_size, seq_len, vocab_size]
        
        # Apply label smoothing
        targets_smooth = targets_one_hot * (1.0 - self.smoothing) + self.smoothing / vocab_size
        
        # Apply ignore mask
        ignore_mask = ignore_mask.unsqueeze(-1).expand_as(targets_smooth)
        targets_smooth = targets_smooth * ignore_mask.float()
        
        # Compute KL divergence loss
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -torch.sum(targets_smooth * log_probs, dim=-1)  # [batch_size, seq_len]
        
        # Apply ignore mask
        loss = loss * ignore_mask[:, :, 0].float()
        
        # Apply reduction
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.sum() / ignore_mask[:, :, 0].float().sum()
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


class FocalLoss(nn.Module):
    """Focal loss for imbalanced classification.
    
    This loss function applies a modulating factor to the standard cross-entropy loss,
    which reduces the relative loss for well-classified examples and focuses more on
    hard, misclassified examples.
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        ignore_index: int = -100,
        reduction: str = 'mean'
    ):
        """Initialize the focal loss.
        
        Args:
            alpha: Weighting factor for the rare class (0.0 means no weighting)
            gamma: Focusing parameter (0.0 means standard cross-entropy)
            ignore_index: Target value to ignore in loss calculation
            reduction: Reduction method ('none', 'mean', or 'sum')
        """
        super().__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute the focal loss.
        
        Args:
            logits: Predicted logits of shape [batch_size, seq_len, vocab_size]
            targets: Target token indices of shape [batch_size, seq_len]
            
        Returns:
            Focal loss
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Create a mask for ignored indices
        ignore_mask = targets != self.ignore_index  # [batch_size, seq_len]
        
        # Compute softmax probabilities
        probs = F.softmax(logits, dim=-1)  # [batch_size, seq_len, vocab_size]
        
        # Create one-hot encoding of targets
        targets_one_hot = F.one_hot(
            torch.clamp(targets, min=0),  # Clamp to avoid negative indices
            num_classes=vocab_size
        ).float()  # [batch_size, seq_len, vocab_size]
        
        # Compute probabilities of the target class
        pt = torch.sum(probs * targets_one_hot, dim=-1)  # [batch_size, seq_len]
        
        # Compute focal loss
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = self.alpha * targets_one_hot + (1 - self.alpha) * (1 - targets_one_hot)
        alpha_weight = torch.sum(alpha_weight, dim=-1)  # [batch_size, seq_len]
        
        # Compute cross-entropy loss
        ce_loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            targets.view(-1),
            ignore_index=self.ignore_index,
            reduction='none'
        ).view(batch_size, seq_len)  # [batch_size, seq_len]
        
        # Apply focal and alpha weighting
        loss = focal_weight * alpha_weight * ce_loss
        
        # Apply ignore mask
        loss = loss * ignore_mask.float()
        
        # Apply reduction
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.sum() / ignore_mask.float().sum()
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


def create_loss_function(
    loss_type: str = 'cross_entropy',
    eos_token: Optional[int] = None,
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
        eos_token: Optional token index used for end-of-sequence
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
    if loss_type == 'cross_entropy':
        if eos_token is not None:
            return EOSCrossEntropyLoss(
                eos_token=eos_token,
                include_eos=include_eos,
                ignore_index=ignore_index,
                reduction=reduction,
                label_smoothing=label_smoothing
            )
        else:
            return nn.CrossEntropyLoss(
                ignore_index=ignore_index,
                reduction=reduction,
                label_smoothing=label_smoothing
            )
    elif loss_type == 'label_smoothing':
        return LabelSmoothingLoss(
            smoothing=label_smoothing,
            ignore_index=ignore_index,
            reduction=reduction
        )
    elif loss_type == 'focal':
        return FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            ignore_index=ignore_index,
            reduction=reduction
        )
    elif loss_type == 'ttm':
        return TTMLoss(
            eos_token=eos_token,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
            include_eos=include_eos,
            memory_loss_weight=memory_loss_weight,
            attention_loss_weight=attention_loss_weight
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
