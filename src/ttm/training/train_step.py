"""
Training step functions for the Token Turing Machine (TTM) model.

This module provides functions for training and evaluation steps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
import numpy as np

from ..models.ttm_model import TokenTuringMachine


def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    clip_grad_norm: float = 1.0,
    device: Optional[torch.device] = None,
    memory: Optional[torch.Tensor] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    accumulation_steps: int = 1,
    current_step: int = 0
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, float]]:
    """Perform a single training step.

    Args:
        model: The model to train
        batch: Dictionary containing input_ids and labels
        optimizer: Optimizer for updating model parameters
        loss_fn: Loss function
        scheduler: Optional learning rate scheduler
        clip_grad_norm: Maximum gradient norm for gradient clipping
        device: Device to train on
        memory: Optional memory state from previous step
        scaler: Optional gradient scaler for mixed precision training
        accumulation_steps: Number of steps to accumulate gradients
        current_step: Current step in the accumulation cycle

    Returns:
        Tuple of (loss, memory, metrics)
    """
    # Move batch to device
    if device is not None:
        batch = {k: v.to(device) for k, v in batch.items()}

    # Get inputs and targets
    inputs = batch['input_ids']
    targets = batch['labels']

    # Forward pass with mixed precision if scaler is provided
    with torch.cuda.amp.autocast() if scaler is not None else torch.no_grad():
        # Get memory before update
        memory_before = memory

        # Forward pass
        if isinstance(model, TokenTuringMachine):
            logits, memory = model(inputs, memory)
        else:
            logits = model(inputs)
            memory = None

        # Compute loss
        if hasattr(loss_fn, 'compute_token_loss'):
            # TTMLoss with separate loss components
            loss_dict = loss_fn(
                logits,
                targets,
                memory_before,
                memory,
                None  # attention_weights
            )

            if isinstance(loss_dict, dict):
                loss = loss_dict['total']
                metrics = {
                    'token_loss': loss_dict['token'].item(),
                    'memory_loss': loss_dict['memory'].item(),
                    'attention_loss': loss_dict['attention'].item()
                }
            else:
                loss = loss_dict
                metrics = {'loss': loss.item()}
        else:
            # Standard loss function
            # Reshape logits for CrossEntropyLoss
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
            metrics = {'loss': loss.item()}

    # Scale loss for gradient accumulation
    if accumulation_steps > 1:
        loss = loss / accumulation_steps

    # Backward pass with mixed precision if scaler is provided
    if loss.requires_grad:
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

    # Update weights if accumulation steps are reached
    if (current_step + 1) % accumulation_steps == 0:
        # Clip gradients
        if clip_grad_norm > 0:
            if scaler is not None:
                scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        # Update weights
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # Zero gradients
        optimizer.zero_grad()

        # Update learning rate
        if scheduler is not None:
            scheduler.step()

    # Detach memory to avoid backpropagation through time
    if memory is not None:
        memory = memory.detach()

    # Add learning rate to metrics
    metrics['lr'] = optimizer.param_groups[0]['lr']

    return loss.detach(), memory, metrics


def eval_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    loss_fn: nn.Module,
    device: Optional[torch.device] = None,
    memory: Optional[torch.Tensor] = None,
    eos_token_id: Optional[int] = None,
    pad_token_id: int = 0
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, float]]:
    """Perform a single evaluation step.

    Args:
        model: The model to evaluate
        batch: Dictionary containing input_ids and labels
        loss_fn: Loss function
        device: Device to evaluate on
        memory: Optional memory state from previous step
        eos_token_id: Optional token ID for end-of-sequence
        pad_token_id: Token ID for padding

    Returns:
        Tuple of (loss, memory, metrics)
    """
    # Set model to evaluation mode
    model.eval()

    # Move batch to device
    if device is not None:
        batch = {k: v.to(device) for k, v in batch.items()}

    # Get inputs and targets
    inputs = batch['input_ids']
    targets = batch['labels']

    # Forward pass
    with torch.no_grad():
        # Get memory before update
        memory_before = memory

        # Forward pass
        if isinstance(model, TokenTuringMachine):
            logits, memory = model(inputs, memory)
        else:
            logits = model(inputs)
            memory = None

        # Compute loss
        if hasattr(loss_fn, 'compute_token_loss'):
            # TTMLoss with separate loss components
            loss_dict = loss_fn(
                logits,
                targets,
                memory_before,
                memory,
                None  # attention_weights
            )

            if isinstance(loss_dict, dict):
                loss = loss_dict['total']
                metrics = {
                    'token_loss': loss_dict['token'].item(),
                    'memory_loss': loss_dict['memory'].item(),
                    'attention_loss': loss_dict['attention'].item()
                }
            else:
                loss = loss_dict
                metrics = {'loss': loss.item()}
        else:
            # Standard loss function
            # Reshape logits for CrossEntropyLoss
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
            metrics = {'loss': loss.item()}

        # Calculate predictions
        predictions = logits.argmax(dim=-1)

        # Calculate position-wise accuracy
        valid_mask = targets != -100  # Ignore positions with -100 label

        if eos_token_id is not None:
            # Create mask for positions up to and including the first EOS token
            eos_mask = torch.zeros_like(targets, dtype=torch.bool)
            for i in range(targets.size(0)):
                eos_indices = (targets[i] == eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_indices) > 0:
                    eos_idx = eos_indices[0]
                    eos_mask[i, :eos_idx+1] = True
                else:
                    eos_mask[i, :] = True

            # Combine masks
            valid_mask = valid_mask & eos_mask

        # Calculate position-wise accuracy
        correct = (predictions == targets) & valid_mask
        position_accuracy = correct.sum().float() / valid_mask.sum().float()

        # Calculate sequence-level accuracy
        sequence_correct = torch.all(correct | ~valid_mask, dim=1)
        sequence_accuracy = sequence_correct.sum().float() / targets.size(0)

        # Add accuracy metrics
        metrics['position_accuracy'] = position_accuracy.item()
        metrics['sequence_accuracy'] = sequence_accuracy.item()

    # Detach memory to avoid backpropagation through time
    if memory is not None:
        memory = memory.detach()

    return loss.detach(), memory, metrics


def get_example_predictions(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    tokenizer: Optional[Callable] = None,
    device: Optional[torch.device] = None,
    memory: Optional[torch.Tensor] = None,
    eos_token_id: Optional[int] = None,
    pad_token_id: int = 0,
    num_examples: int = 3
) -> List[Dict[str, Any]]:
    """Get example predictions from the model.

    Args:
        model: The model to evaluate
        batch: Dictionary containing input_ids and labels
        tokenizer: Optional tokenizer for decoding tokens
        device: Device to evaluate on
        memory: Optional memory state from previous step
        eos_token_id: Optional token ID for end-of-sequence
        pad_token_id: Token ID for padding
        num_examples: Number of examples to return

    Returns:
        List of dictionaries with input, target, and prediction
    """
    # Set model to evaluation mode
    model.eval()

    # Move batch to device
    if device is not None:
        batch = {k: v.to(device) for k, v in batch.items()}

    # Get inputs and targets
    inputs = batch['input_ids']
    targets = batch['labels']

    # Limit to num_examples
    inputs = inputs[:num_examples]
    targets = targets[:num_examples]

    # Forward pass
    with torch.no_grad():
        # Forward pass
        if isinstance(model, TokenTuringMachine):
            logits, _ = model(inputs, memory)
        else:
            logits = model(inputs)

        # Calculate predictions
        predictions = logits.argmax(dim=-1)

    # Create examples
    examples = []
    for i in range(min(num_examples, inputs.size(0))):
        # Create valid mask
        valid_mask = targets[i] != -100

        if eos_token_id is not None:
            # Find first EOS token
            eos_indices = (targets[i] == eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_indices) > 0:
                eos_idx = eos_indices[0]
                valid_mask[eos_idx+1:] = False

        # Get valid tokens
        input_tokens = inputs[i][valid_mask].tolist()
        target_tokens = targets[i][valid_mask].tolist()
        prediction_tokens = predictions[i][valid_mask].tolist()

        # Decode tokens if tokenizer is provided
        if tokenizer is not None:
            input_text = tokenizer.decode(input_tokens)
            target_text = tokenizer.decode(target_tokens)
            prediction_text = tokenizer.decode(prediction_tokens)

            example = {
                'input_tokens': input_tokens,
                'target_tokens': target_tokens,
                'prediction_tokens': prediction_tokens,
                'input_text': input_text,
                'target_text': target_text,
                'prediction_text': prediction_text
            }
        else:
            example = {
                'input_tokens': input_tokens,
                'target_tokens': target_tokens,
                'prediction_tokens': prediction_tokens
            }

        # Add to examples
        examples.append(example)

    return examples
