"""
Trainer for the Token Turing Machine (TTM) model.

This module provides a trainer class for training the TTM model.
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
from tqdm import tqdm

from ..models.ttm_model import TokenTuringMachine
from ..utils.losses import create_loss_function
from .optimizer import create_optimizer, create_scheduler
from ..visualization.state_tracker import TTMStateTracker


class TTMTrainer:
    """Trainer for the Token Turing Machine model."""

    def __init__(
        self,
        model: TokenTuringMachine,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        loss_fn: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        log_interval: int = 10,
        checkpoint_dir: str = './checkpoints',
        checkpoint_interval: int = 1000,
        use_amp: bool = False,
        enable_visualization: bool = False,
        visualization_dir: str = './visualization_data',
        visualization_sampling_rate: float = 0.1
    ):
        """Initialize the TTM trainer.

        Args:
            model: The TTM model to train
            train_dataloader: DataLoader for training data
            val_dataloader: Optional DataLoader for validation data
            optimizer: Optional optimizer for training
            scheduler: Optional learning rate scheduler
            loss_fn: Optional loss function
            device: Device to train on
            max_grad_norm: Maximum gradient norm for gradient clipping
            gradient_accumulation_steps: Number of steps to accumulate gradients
            log_interval: Number of steps between logging
            checkpoint_dir: Directory to save checkpoints
            checkpoint_interval: Number of steps between checkpoints
            use_amp: Whether to use automatic mixed precision
            enable_visualization: Whether to enable visualization with TTMStateTracker
            visualization_dir: Directory to save visualization data
            visualization_sampling_rate: Fraction of batches to sample for visualization
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Create optimizer if not provided
        self.optimizer = optimizer or create_optimizer(
            model=model,
            optimizer_type='adamw',
            learning_rate=1e-4,
            weight_decay=0.01
        )

        # Create scheduler if not provided
        self.scheduler = scheduler

        # Create loss function if not provided
        self.loss_fn = loss_fn or model.create_loss_fn(
            loss_type='ttm',
            memory_loss_weight=0.1,
            attention_loss_weight=0.1
        )

        # Set device
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Training parameters
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.log_interval = log_interval
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.use_amp = use_amp

        # Initialize training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Initialize visualization state tracker if enabled
        self.enable_visualization = enable_visualization
        self.visualization_dir = visualization_dir
        self.visualization_sampling_rate = visualization_sampling_rate

        if self.enable_visualization:
            # Create visualization directory
            os.makedirs(visualization_dir, exist_ok=True)

            # Initialize state tracker
            self.state_tracker = TTMStateTracker(
                model=self.model,
                sampling_rate=self.visualization_sampling_rate
            )
            print(f"Initialized TTMStateTracker with sampling rate {self.visualization_sampling_rate}")
        else:
            self.state_tracker = None

        # Initialize scaler for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None

    def train_epoch(self) -> Dict[str, float]:
        """Train the model for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_token_loss = 0.0
        total_memory_loss = 0.0
        total_attention_loss = 0.0
        start_time = time.time()

        # Initialize memory for the first batch
        memory = None

        # Create progress bar
        pbar = tqdm(total=len(self.train_dataloader), desc=f"Epoch {self.epoch+1}")

        # Update state tracker with current epoch
        if self.enable_visualization and self.state_tracker is not None:
            self.state_tracker.start_epoch(self.epoch)

        for step, batch in enumerate(self.train_dataloader):
            # Update state tracker with current batch
            if self.enable_visualization and self.state_tracker is not None:
                self.state_tracker.start_batch(step)
            # Move batch to device
            inputs = batch['input_ids'].to(self.device)
            targets = batch['labels'].to(self.device)

            # Forward pass with mixed precision if enabled
            with torch.cuda.amp.autocast() if self.use_amp else torch.no_grad():
                # Get memory before update
                memory_before = memory

                # Forward pass
                # Track token states if visualization is enabled
                if self.enable_visualization and self.state_tracker is not None:
                    # Process each token in the sequence
                    batch_size, seq_len = inputs.shape
                    all_logits = []

                    for token_idx in range(seq_len):
                        # Update state tracker with current token
                        self.state_tracker.start_token(token_idx)

                        # Get current token
                        token_input = inputs[:, :token_idx+1]

                        # Forward pass for this token
                        token_logits, memory = self.model(token_input, memory)

                        # Store logits
                        all_logits.append(token_logits)

                    # Use the final logits for loss computation
                    logits = all_logits[-1]
                else:
                    # Standard forward pass without token tracking
                    logits, memory = self.model(inputs, memory)

                # Compute loss
                if isinstance(self.loss_fn, dict):
                    # If loss_fn is a dictionary of loss functions
                    loss = 0.0
                    for loss_name, (loss_fn, weight) in self.loss_fn.items():
                        if loss_name == 'token':
                            loss += weight * loss_fn(logits, targets)
                        elif loss_name == 'memory' and memory_before is not None and memory is not None:
                            loss += weight * loss_fn(memory_before, memory)
                else:
                    # If loss_fn is a single loss function
                    if hasattr(self.loss_fn, 'compute_token_loss'):
                        # TTMLoss with separate loss components
                        loss_dict = self.loss_fn(
                            logits,
                            targets,
                            memory_before,
                            memory,
                            None  # attention_weights
                        )

                        if isinstance(loss_dict, dict):
                            loss = loss_dict['total']
                            token_loss = loss_dict['token']
                            memory_loss = loss_dict['memory']
                            attention_loss = loss_dict['attention']
                        else:
                            loss = loss_dict
                            token_loss = loss
                            memory_loss = torch.tensor(0.0, device=self.device)
                            attention_loss = torch.tensor(0.0, device=self.device)
                    else:
                        # Standard loss function
                        loss = self.loss_fn(logits, targets)
                        token_loss = loss
                        memory_loss = torch.tensor(0.0, device=self.device)
                        attention_loss = torch.tensor(0.0, device=self.device)

            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps

            # Backward pass with mixed precision if enabled
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights if gradient accumulation steps are reached
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Clip gradients
                if self.max_grad_norm > 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                # Update weights
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                # Zero gradients
                self.optimizer.zero_grad()

                # Update learning rate
                if self.scheduler is not None:
                    self.scheduler.step()

            # Update metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            total_token_loss += token_loss.item()
            total_memory_loss += memory_loss.item()
            total_attention_loss += attention_loss.item()

            # Detach memory to avoid backpropagation through time
            if memory is not None:
                memory = memory.detach()

            # Log progress
            if (step + 1) % self.log_interval == 0:
                ms_per_batch = (time.time() - start_time) * 1000 / self.log_interval
                cur_loss = total_loss / (step + 1)
                ppl = math.exp(cur_loss) if cur_loss < 30 else float('inf')

                pbar.set_postfix({
                    'loss': f'{cur_loss:.3f}',
                    'ppl': f'{ppl:.3f}',
                    'ms/batch': f'{ms_per_batch:.3f}'
                })

                start_time = time.time()

            # Save checkpoint
            if (self.global_step + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint()

            # Update global step
            self.global_step += 1

            # Update progress bar
            pbar.update(1)

        # Close progress bar
        pbar.close()

        # Compute average metrics
        avg_loss = total_loss / len(self.train_dataloader)
        avg_token_loss = total_token_loss / len(self.train_dataloader)
        avg_memory_loss = total_memory_loss / len(self.train_dataloader)
        avg_attention_loss = total_attention_loss / len(self.train_dataloader)
        ppl = math.exp(avg_loss) if avg_loss < 30 else float('inf')

        # Save visualization data if enabled
        if self.enable_visualization and self.state_tracker is not None:
            # Save state history
            vis_file = os.path.join(self.visualization_dir, f"state_history_epoch_{self.epoch}.pkl")
            self.state_tracker.save_state_history(vis_file)
            print(f"Saved visualization data to {vis_file}")

        # Return metrics
        return {
            'loss': avg_loss,
            'token_loss': avg_token_loss,
            'memory_loss': avg_memory_loss,
            'attention_loss': avg_attention_loss,
            'ppl': ppl
        }

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on the validation set.

        Returns:
            Dictionary of validation metrics
        """
        if self.val_dataloader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_token_loss = 0.0
        total_memory_loss = 0.0
        total_attention_loss = 0.0

        # Initialize memory for the first batch
        memory = None

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                # Move batch to device
                inputs = batch['input_ids'].to(self.device)
                targets = batch['labels'].to(self.device)

                # Get memory before update
                memory_before = memory

                # Forward pass
                logits, memory = self.model(inputs, memory)

                # Compute loss
                if isinstance(self.loss_fn, dict):
                    # If loss_fn is a dictionary of loss functions
                    loss = 0.0
                    for loss_name, (loss_fn, weight) in self.loss_fn.items():
                        if loss_name == 'token':
                            loss += weight * loss_fn(logits, targets)
                        elif loss_name == 'memory' and memory_before is not None and memory is not None:
                            loss += weight * loss_fn(memory_before, memory)
                else:
                    # If loss_fn is a single loss function
                    if hasattr(self.loss_fn, 'compute_token_loss'):
                        # TTMLoss with separate loss components
                        loss_dict = self.loss_fn(
                            logits,
                            targets,
                            memory_before,
                            memory,
                            None  # attention_weights
                        )

                        if isinstance(loss_dict, dict):
                            loss = loss_dict['total']
                            token_loss = loss_dict['token']
                            memory_loss = loss_dict['memory']
                            attention_loss = loss_dict['attention']
                        else:
                            loss = loss_dict
                            token_loss = loss
                            memory_loss = torch.tensor(0.0, device=self.device)
                            attention_loss = torch.tensor(0.0, device=self.device)
                    else:
                        # Standard loss function
                        loss = self.loss_fn(logits, targets)
                        token_loss = loss
                        memory_loss = torch.tensor(0.0, device=self.device)
                        attention_loss = torch.tensor(0.0, device=self.device)

                # Update metrics
                total_loss += loss.item()
                total_token_loss += token_loss.item()
                total_memory_loss += memory_loss.item()
                total_attention_loss += attention_loss.item()

                # Detach memory to avoid backpropagation through time
                if memory is not None:
                    memory = memory.detach()

        # Compute average metrics
        avg_loss = total_loss / len(self.val_dataloader)
        avg_token_loss = total_token_loss / len(self.val_dataloader)
        avg_memory_loss = total_memory_loss / len(self.val_dataloader)
        avg_attention_loss = total_attention_loss / len(self.val_dataloader)
        ppl = math.exp(avg_loss) if avg_loss < 30 else float('inf')

        # Return metrics
        return {
            'loss': avg_loss,
            'token_loss': avg_token_loss,
            'memory_loss': avg_memory_loss,
            'attention_loss': avg_attention_loss,
            'ppl': ppl
        }

    def train(self, num_epochs: int) -> Dict[str, List[float]]:
        """Train the model for multiple epochs.

        Args:
            num_epochs: Number of epochs to train for

        Returns:
            Dictionary of training and validation metrics for each epoch
        """
        # Initialize metrics
        metrics = {
            'train_loss': [],
            'train_ppl': [],
            'val_loss': [],
            'val_ppl': []
        }

        # Train for num_epochs
        for epoch in range(num_epochs):
            self.epoch = epoch

            # Train epoch
            train_metrics = self.train_epoch()
            metrics['train_loss'].append(train_metrics['loss'])
            metrics['train_ppl'].append(train_metrics['ppl'])

            # Evaluate
            if self.val_dataloader is not None:
                val_metrics = self.evaluate()
                metrics['val_loss'].append(val_metrics['loss'])
                metrics['val_ppl'].append(val_metrics['ppl'])

                # Save best model
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint(is_best=True)

                # Print metrics
                print(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {train_metrics['loss']:.4f}, "
                      f"Train PPL: {train_metrics['ppl']:.4f}, "
                      f"Val Loss: {val_metrics['loss']:.4f}, "
                      f"Val PPL: {val_metrics['ppl']:.4f}")
            else:
                # Print metrics
                print(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {train_metrics['loss']:.4f}, "
                      f"Train PPL: {train_metrics['ppl']:.4f}")

            # Save checkpoint
            self.save_checkpoint()

        return metrics

    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save a checkpoint of the model.

        Args:
            is_best: Whether this is the best model so far
        """
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Create checkpoint
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss
        }

        # Save checkpoint
        if is_best:
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'best_model.pt'))
        else:
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, f'checkpoint_{self.global_step}.pt'))

            # Save latest checkpoint
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'latest_model.pt'))

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load a checkpoint of the model.

        Args:
            checkpoint_path: Path to the checkpoint file
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state
        if checkpoint['scheduler_state_dict'] is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load training state
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Global step: {self.global_step}, Epoch: {self.epoch}, Best val loss: {self.best_val_loss:.4f}")

    def save_metrics(self, metrics: Dict[str, List[float]], filename: str = 'metrics.json') -> None:
        """Save training metrics to a file.

        Args:
            metrics: Dictionary of metrics
            filename: Name of the file to save metrics to
        """
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Save metrics
        with open(os.path.join(self.checkpoint_dir, filename), 'w') as f:
            json.dump(metrics, f, indent=4)

    def load_metrics(self, filename: str = 'metrics.json') -> Dict[str, List[float]]:
        """Load training metrics from a file.

        Args:
            filename: Name of the file to load metrics from

        Returns:
            Dictionary of metrics
        """
        # Load metrics
        with open(os.path.join(self.checkpoint_dir, filename), 'r') as f:
            metrics = json.load(f)

        return metrics
