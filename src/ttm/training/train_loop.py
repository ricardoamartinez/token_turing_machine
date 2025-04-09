"""
Training loop for the Token Turing Machine (TTM) model.

This module provides functions for training the TTM model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
import time
import math
import os
import json
import numpy as np
from tqdm import tqdm
import logging

from ..models.ttm_model import TokenTuringMachine
from ..utils.losses import create_loss_function
from .optimizer import create_optimizer, create_scheduler
from .train_step import train_step, eval_step, get_example_predictions


def train_loop(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    loss_fn: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
    num_epochs: int = 100,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    log_interval: int = 10,
    eval_interval: int = 10,
    checkpoint_dir: str = './checkpoints',
    checkpoint_interval: int = 1000,
    early_stopping_patience: int = 20,
    early_stopping_metric: str = 'loss',
    tokenizer: Optional[Callable] = None,
    eos_token_id: Optional[int] = None,
    pad_token_id: int = 0,
    use_amp: bool = False,
    logger: Optional[logging.Logger] = None
) -> Dict[str, List[float]]:
    """Train the model.
    
    Args:
        model: The model to train
        train_dataloader: DataLoader for training data
        val_dataloader: Optional DataLoader for validation data
        optimizer: Optional optimizer for training
        scheduler: Optional learning rate scheduler
        loss_fn: Optional loss function
        device: Device to train on
        num_epochs: Number of epochs to train for
        max_grad_norm: Maximum gradient norm for gradient clipping
        gradient_accumulation_steps: Number of steps to accumulate gradients
        log_interval: Number of steps between logging
        eval_interval: Number of epochs between evaluations
        checkpoint_dir: Directory to save checkpoints
        checkpoint_interval: Number of steps between checkpoints
        early_stopping_patience: Number of evaluations with no improvement before stopping
        early_stopping_metric: Metric to use for early stopping
        tokenizer: Optional tokenizer for decoding tokens
        eos_token_id: Optional token ID for end-of-sequence
        pad_token_id: Token ID for padding
        use_amp: Whether to use automatic mixed precision
        logger: Optional logger for logging
        
    Returns:
        Dictionary of training and validation metrics for each epoch
    """
    # Set up device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move model to device
    model.to(device)
    
    # Create optimizer if not provided
    if optimizer is None:
        optimizer = create_optimizer(
            model=model,
            optimizer_type='adamw',
            learning_rate=1e-4,
            weight_decay=0.01
        )
    
    # Create loss function if not provided
    if loss_fn is None:
        if isinstance(model, TokenTuringMachine):
            loss_fn = model.create_loss_fn(
                loss_type='ttm',
                memory_loss_weight=0.1,
                attention_loss_weight=0.1
            )
        else:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # Initialize metrics
    metrics = {
        'train_loss': [],
        'train_position_accuracy': [],
        'train_sequence_accuracy': [],
        'val_loss': [],
        'val_position_accuracy': [],
        'val_sequence_accuracy': []
    }
    
    # Initialize early stopping
    best_val_metric = float('inf') if early_stopping_metric == 'loss' else -float('inf')
    early_stopping_counter = 0
    
    # Initialize memory
    memory = None
    
    # Initialize global step
    global_step = 0
    
    # Train for num_epochs
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()
        
        # Initialize epoch metrics
        epoch_loss = 0.0
        epoch_position_accuracy = 0.0
        epoch_sequence_accuracy = 0.0
        
        # Create progress bar
        pbar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")
        
        # Train on batches
        for step, batch in enumerate(train_dataloader):
            # Perform training step
            loss, memory, step_metrics = train_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                loss_fn=loss_fn,
                scheduler=scheduler,
                clip_grad_norm=max_grad_norm,
                device=device,
                memory=memory,
                scaler=scaler,
                accumulation_steps=gradient_accumulation_steps,
                current_step=global_step % gradient_accumulation_steps
            )
            
            # Update epoch metrics
            epoch_loss += loss.item()
            if 'position_accuracy' in step_metrics:
                epoch_position_accuracy += step_metrics['position_accuracy']
            if 'sequence_accuracy' in step_metrics:
                epoch_sequence_accuracy += step_metrics['sequence_accuracy']
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{step_metrics['lr']:.6f}"
            })
            
            # Log metrics
            if (step + 1) % log_interval == 0:
                if logger is not None:
                    logger.info(f"Epoch {epoch+1}/{num_epochs}, Step {step+1}/{len(train_dataloader)}, "
                                f"Loss: {loss.item():.4f}, LR: {step_metrics['lr']:.6f}")
            
            # Save checkpoint
            if (global_step + 1) % checkpoint_interval == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    metrics=metrics,
                    checkpoint_dir=checkpoint_dir,
                    is_best=False
                )
            
            # Update global step
            global_step += 1
        
        # Close progress bar
        pbar.close()
        
        # Calculate epoch metrics
        epoch_loss /= len(train_dataloader)
        epoch_position_accuracy /= len(train_dataloader)
        epoch_sequence_accuracy /= len(train_dataloader)
        
        # Update metrics
        metrics['train_loss'].append(epoch_loss)
        metrics['train_position_accuracy'].append(epoch_position_accuracy)
        metrics['train_sequence_accuracy'].append(epoch_sequence_accuracy)
        
        # Log epoch metrics
        if logger is not None:
            logger.info(f"Epoch {epoch+1}/{num_epochs}, "
                        f"Train Loss: {epoch_loss:.4f}, "
                        f"Position Accuracy: {epoch_position_accuracy:.4f}, "
                        f"Sequence Accuracy: {epoch_sequence_accuracy:.4f}")
        
        # Evaluate if validation dataloader is provided and it's evaluation time
        if val_dataloader is not None and (epoch + 1) % eval_interval == 0:
            # Evaluate model
            val_metrics = evaluate(
                model=model,
                dataloader=val_dataloader,
                loss_fn=loss_fn,
                device=device,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id
            )
            
            # Update metrics
            metrics['val_loss'].append(val_metrics['loss'])
            metrics['val_position_accuracy'].append(val_metrics['position_accuracy'])
            metrics['val_sequence_accuracy'].append(val_metrics['sequence_accuracy'])
            
            # Log validation metrics
            if logger is not None:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, "
                            f"Val Loss: {val_metrics['loss']:.4f}, "
                            f"Position Accuracy: {val_metrics['position_accuracy']:.4f}, "
                            f"Sequence Accuracy: {val_metrics['sequence_accuracy']:.4f}")
            
            # Get example predictions
            if tokenizer is not None:
                examples = get_example_predictions(
                    model=model,
                    batch=next(iter(val_dataloader)),
                    tokenizer=tokenizer,
                    device=device,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    num_examples=3
                )
                
                # Log example predictions
                if logger is not None:
                    for i, example in enumerate(examples):
                        logger.info(f"Example {i+1}:")
                        logger.info(f"  Input: {example['input_text']}")
                        logger.info(f"  Target: {example['target_text']}")
                        logger.info(f"  Prediction: {example['prediction_text']}")
            
            # Check for early stopping
            val_metric = val_metrics[early_stopping_metric.replace('val_', '')]
            is_better = val_metric < best_val_metric if early_stopping_metric == 'loss' else val_metric > best_val_metric
            
            if is_better:
                # Update best validation metric
                best_val_metric = val_metric
                
                # Reset early stopping counter
                early_stopping_counter = 0
                
                # Save best model
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    metrics=metrics,
                    checkpoint_dir=checkpoint_dir,
                    is_best=True
                )
            else:
                # Increment early stopping counter
                early_stopping_counter += 1
                
                # Check if early stopping criteria is met
                if early_stopping_counter >= early_stopping_patience:
                    if logger is not None:
                        logger.info(f"Early stopping after {epoch+1} epochs")
                    break
        
        # Save checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            metrics=metrics,
            checkpoint_dir=checkpoint_dir,
            is_best=False
        )
    
    return metrics


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: Optional[torch.device] = None,
    eos_token_id: Optional[int] = None,
    pad_token_id: int = 0
) -> Dict[str, float]:
    """Evaluate the model on a dataset.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation data
        loss_fn: Loss function
        device: Device to evaluate on
        eos_token_id: Optional token ID for end-of-sequence
        pad_token_id: Token ID for padding
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Set model to evaluation mode
    model.eval()
    
    # Initialize metrics
    total_loss = 0.0
    total_position_accuracy = 0.0
    total_sequence_accuracy = 0.0
    
    # Initialize memory
    memory = None
    
    # Evaluate on batches
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Perform evaluation step
            loss, memory, step_metrics = eval_step(
                model=model,
                batch=batch,
                loss_fn=loss_fn,
                device=device,
                memory=memory,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id
            )
            
            # Update metrics
            total_loss += loss.item()
            total_position_accuracy += step_metrics['position_accuracy']
            total_sequence_accuracy += step_metrics['sequence_accuracy']
    
    # Calculate average metrics
    avg_loss = total_loss / len(dataloader)
    avg_position_accuracy = total_position_accuracy / len(dataloader)
    avg_sequence_accuracy = total_sequence_accuracy / len(dataloader)
    
    return {
        'loss': avg_loss,
        'position_accuracy': avg_position_accuracy,
        'sequence_accuracy': avg_sequence_accuracy
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    global_step: int,
    metrics: Dict[str, List[float]],
    checkpoint_dir: str,
    is_best: bool = False
) -> None:
    """Save a checkpoint of the model.
    
    Args:
        model: The model to save
        optimizer: The optimizer
        scheduler: Optional learning rate scheduler
        epoch: Current epoch
        global_step: Current global step
        metrics: Dictionary of metrics
        checkpoint_dir: Directory to save checkpoints
        is_best: Whether this is the best model so far
    """
    # Create checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'epoch': epoch,
        'global_step': global_step,
        'metrics': metrics
    }
    
    # Save checkpoint
    if is_best:
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_model.pt'))
    else:
        torch.save(checkpoint, os.path.join(checkpoint_dir, f'checkpoint_{global_step}.pt'))
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'latest_model.pt'))
    
    # Save metrics
    with open(os.path.join(checkpoint_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    checkpoint_path: str = 'checkpoints/best_model.pt',
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler._LRScheduler], Dict[str, Any]]:
    """Load a checkpoint of the model.
    
    Args:
        model: The model to load
        optimizer: Optional optimizer to load
        scheduler: Optional learning rate scheduler to load
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on
        
    Returns:
        Tuple of (model, optimizer, scheduler, checkpoint)
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return model, optimizer, scheduler, checkpoint


def train_with_curriculum(
    model: nn.Module,
    dataset_fn: Callable[[int], DataLoader],
    num_stages: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    loss_fn: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
    num_epochs_per_stage: int = 1000,
    accuracy_threshold: float = 0.9,
    accuracy_window: int = 5,
    **kwargs
) -> Dict[str, List[float]]:
    """Train the model with curriculum learning.
    
    Args:
        model: The model to train
        dataset_fn: Function that takes a stage index and returns a tuple of (train_dataloader, val_dataloader)
        num_stages: Number of curriculum stages
        optimizer: Optional optimizer for training
        scheduler: Optional learning rate scheduler
        loss_fn: Optional loss function
        device: Device to train on
        num_epochs_per_stage: Maximum number of epochs per stage
        accuracy_threshold: Accuracy threshold for progressing to the next stage
        accuracy_window: Number of recent evaluations to consider for progression
        **kwargs: Additional arguments to pass to train_loop
        
    Returns:
        Dictionary of training and validation metrics for each epoch
    """
    # Initialize metrics
    all_metrics = {
        'train_loss': [],
        'train_position_accuracy': [],
        'train_sequence_accuracy': [],
        'val_loss': [],
        'val_position_accuracy': [],
        'val_sequence_accuracy': [],
        'stage': []
    }
    
    # Initialize accuracy history
    accuracy_history = []
    
    # Train on each stage
    for stage in range(num_stages):
        # Get dataloaders for this stage
        train_dataloader, val_dataloader = dataset_fn(stage)
        
        # Log stage
        if 'logger' in kwargs and kwargs['logger'] is not None:
            kwargs['logger'].info(f"Starting stage {stage+1}/{num_stages}")
        
        # Train on this stage
        stage_metrics = train_loop(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            device=device,
            num_epochs=num_epochs_per_stage,
            **kwargs
        )
        
        # Update metrics
        for key, value in stage_metrics.items():
            all_metrics[key].extend(value)
        
        # Add stage information
        all_metrics['stage'].extend([stage] * len(stage_metrics['train_loss']))
        
        # Update accuracy history
        if 'val_sequence_accuracy' in stage_metrics and len(stage_metrics['val_sequence_accuracy']) > 0:
            accuracy_history.extend(stage_metrics['val_sequence_accuracy'])
        
        # Check if we should progress to the next stage
        if len(accuracy_history) >= accuracy_window:
            recent_accuracy = np.mean(accuracy_history[-accuracy_window:])
            if recent_accuracy >= accuracy_threshold:
                if 'logger' in kwargs and kwargs['logger'] is not None:
                    kwargs['logger'].info(f"Progressing to next stage with accuracy {recent_accuracy:.4f}")
                continue
        
        # If we've reached the maximum number of epochs for this stage, progress anyway
        if 'logger' in kwargs and kwargs['logger'] is not None:
            kwargs['logger'].info(f"Maximum epochs reached for stage {stage+1}, progressing to next stage")
    
    return all_metrics
