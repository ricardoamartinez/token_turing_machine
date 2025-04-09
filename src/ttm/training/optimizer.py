"""
Optimizer utilities for the Token Turing Machine (TTM) model.

This module provides utilities for creating and configuring optimizers
for training the TTM model.
"""

import torch
import torch.optim as optim
import math
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ReduceLROnPlateau
from typing import Dict, Any, Optional, Union, List, Tuple, Callable

from ..models.ttm_model import TokenTuringMachine


def create_optimizer(
    model: torch.nn.Module,
    optimizer_type: str = 'adam',
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.999),
    momentum: float = 0.9,
    separate_decay_parameters: bool = True
) -> torch.optim.Optimizer:
    """Create an optimizer for training.

    Args:
        model: The model to optimize
        optimizer_type: Type of optimizer ('adam', 'adamw', 'sgd', or 'rmsprop')
        learning_rate: Learning rate
        weight_decay: Weight decay factor
        betas: Adam betas parameters
        momentum: Momentum factor for SGD
        separate_decay_parameters: Whether to apply weight decay only to weight matrices

    Returns:
        Optimizer
    """
    if separate_decay_parameters and optimizer_type in ['adam', 'adamw']:
        # Separate parameters that should have weight decay from those that shouldn't
        decay_parameters = []
        no_decay_parameters = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                # Apply weight decay to weight matrices but not to biases and layer norms
                if 'bias' in name or 'layer_norm' in name or 'layernorm' in name:
                    no_decay_parameters.append(param)
                else:
                    decay_parameters.append(param)

        parameter_groups = [
            {'params': decay_parameters, 'weight_decay': weight_decay},
            {'params': no_decay_parameters, 'weight_decay': 0.0}
        ]
    else:
        # Use all parameters with the same weight decay
        parameter_groups = model.parameters()

    # Create optimizer
    if optimizer_type == 'adam':
        optimizer = optim.Adam(
            parameter_groups,
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(
            parameter_groups,
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(
            parameter_groups,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'rmsprop':
        optimizer = optim.RMSprop(
            parameter_groups,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = 'cosine',
    num_warmup_steps: int = 0,
    num_training_steps: Optional[int] = None,
    num_cycles: float = 0.5,
    min_lr: float = 0.0,
    factor: float = 0.1,
    patience: int = 10,
    threshold: float = 1e-4
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Create a learning rate scheduler.

    Args:
        optimizer: The optimizer to schedule
        scheduler_type: Type of scheduler ('linear', 'cosine', 'plateau', or None)
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        num_cycles: Number of cycles for cosine scheduler
        min_lr: Minimum learning rate
        factor: Factor by which to reduce learning rate for plateau scheduler
        patience: Number of epochs with no improvement for plateau scheduler
        threshold: Threshold for measuring improvement for plateau scheduler

    Returns:
        Learning rate scheduler or None if scheduler_type is None
    """
    if scheduler_type is None:
        return None

    if scheduler_type == 'linear':
        # Linear scheduler with warmup
        def lr_lambda(current_step: int) -> float:
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )

        return LambdaLR(optimizer, lr_lambda)

    elif scheduler_type == 'cosine':
        # Cosine scheduler with warmup
        def lr_lambda(current_step: int) -> float:
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(min_lr, 0.5 * (1.0 + torch.cos(torch.tensor(math.pi * num_cycles * 2.0 * progress)).item()))

        return LambdaLR(optimizer, lr_lambda)

    elif scheduler_type == 'cosine_annealing':
        # Cosine annealing scheduler
        return CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps,
            eta_min=min_lr
        )

    elif scheduler_type == 'plateau':
        # Reduce on plateau scheduler
        return ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=factor,
            patience=patience,
            threshold=threshold,
            min_lr=min_lr
        )

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def get_grouped_parameters(
    model: torch.nn.Module,
    weight_decay: float = 0.01,
    no_decay_name_list: List[str] = ['bias', 'layer_norm', 'layernorm']
) -> List[Dict[str, Any]]:
    """Group parameters for optimization with different weight decay.

    Args:
        model: The model to optimize
        weight_decay: Weight decay factor
        no_decay_name_list: List of parameter name patterns that should not have weight decay

    Returns:
        List of parameter groups
    """
    decay_parameters = []
    no_decay_parameters = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if any(nd in name for nd in no_decay_name_list):
                no_decay_parameters.append(param)
            else:
                decay_parameters.append(param)

    return [
        {'params': decay_parameters, 'weight_decay': weight_decay},
        {'params': no_decay_parameters, 'weight_decay': 0.0}
    ]


def get_parameter_names(
    model: torch.nn.Module,
    forbidden_layer_types: List[torch.nn.Module] = None
) -> List[str]:
    """Get names of parameters that should be optimized.

    Args:
        model: The model to optimize
        forbidden_layer_types: List of layer types that should not be optimized

    Returns:
        List of parameter names
    """
    if forbidden_layer_types is None:
        forbidden_layer_types = []

    result = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Check if the parameter belongs to a forbidden layer type
            is_forbidden = False
            for layer_type in forbidden_layer_types:
                for module in model.modules():
                    if isinstance(module, layer_type):
                        for param_name, _ in module.named_parameters():
                            if name.endswith(param_name):
                                is_forbidden = True
                                break
                        if is_forbidden:
                            break
                if is_forbidden:
                    break

            if not is_forbidden:
                result.append(name)

    return result


def get_ttm_optimizer(
    model: TokenTuringMachine,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    optimizer_type: str = 'adamw',
    betas: Tuple[float, float] = (0.9, 0.999),
    separate_decay: bool = True
) -> torch.optim.Optimizer:
    """Create an optimizer specifically for the TTM model.

    This function applies different learning rates to different components
    of the TTM model, which can improve training stability.

    Args:
        model: The TTM model to optimize
        learning_rate: Base learning rate
        weight_decay: Weight decay factor
        optimizer_type: Type of optimizer ('adam', 'adamw', 'sgd', or 'rmsprop')
        betas: Adam betas parameters
        separate_decay: Whether to apply weight decay only to weight matrices

    Returns:
        Optimizer
    """
    # Define parameter groups with different learning rates
    if separate_decay:
        # Memory module parameters
        memory_decay_params = []
        memory_no_decay_params = []

        # Transformer parameters
        transformer_decay_params = []
        transformer_no_decay_params = []

        # Output head parameters
        output_decay_params = []
        output_no_decay_params = []

        # Embedding parameters
        embedding_decay_params = []
        embedding_no_decay_params = []

        # Group parameters
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if 'memory_module' in name:
                if any(nd in name for nd in ['bias', 'layer_norm', 'layernorm']):
                    memory_no_decay_params.append(param)
                else:
                    memory_decay_params.append(param)
            elif 'transformer' in name:
                if any(nd in name for nd in ['bias', 'layer_norm', 'layernorm']):
                    transformer_no_decay_params.append(param)
                else:
                    transformer_decay_params.append(param)
            elif 'output_head' in name:
                if any(nd in name for nd in ['bias', 'layer_norm', 'layernorm']):
                    output_no_decay_params.append(param)
                else:
                    output_decay_params.append(param)
            elif 'token_embedding' in name:
                if any(nd in name for nd in ['bias', 'layer_norm', 'layernorm']):
                    embedding_no_decay_params.append(param)
                else:
                    embedding_decay_params.append(param)

        # Create parameter groups with different learning rates
        parameter_groups = [
            {'params': memory_decay_params, 'lr': learning_rate, 'weight_decay': weight_decay},
            {'params': memory_no_decay_params, 'lr': learning_rate, 'weight_decay': 0.0},
            {'params': transformer_decay_params, 'lr': learning_rate, 'weight_decay': weight_decay},
            {'params': transformer_no_decay_params, 'lr': learning_rate, 'weight_decay': 0.0},
            {'params': output_decay_params, 'lr': learning_rate, 'weight_decay': weight_decay},
            {'params': output_no_decay_params, 'lr': learning_rate, 'weight_decay': 0.0},
            {'params': embedding_decay_params, 'lr': learning_rate, 'weight_decay': weight_decay},
            {'params': embedding_no_decay_params, 'lr': learning_rate, 'weight_decay': 0.0}
        ]
    else:
        # Use all parameters with the same weight decay
        parameter_groups = model.parameters()

    # Create optimizer
    if optimizer_type == 'adam':
        optimizer = optim.Adam(
            parameter_groups,
            lr=learning_rate,
            betas=betas,
            weight_decay=0.0 if separate_decay else weight_decay
        )
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(
            parameter_groups,
            lr=learning_rate,
            betas=betas,
            weight_decay=0.0 if separate_decay else weight_decay
        )
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(
            parameter_groups,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'rmsprop':
        optimizer = optim.RMSprop(
            parameter_groups,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    return optimizer
