"""
Script to run training for the Token Turing Machine (TTM) model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
import argparse
import os
import logging
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from ..models.ttm_model import TokenTuringMachine
from ..utils.losses import create_loss_function
from .optimizer import create_optimizer, create_scheduler
from .train_step import train_step, eval_step, get_example_predictions
from .train_loop import train_loop, evaluate, save_checkpoint, load_checkpoint, train_with_curriculum
from .curriculum import create_curriculum_dataloaders, MultiplicationDataset
from .data import create_dataloaders


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(log_dir: str) -> logging.Logger:
    """Set up logging.
    
    Args:
        log_dir: Directory to save logs
        
    Returns:
        Logger
    """
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('ttm')
    logger.setLevel(logging.INFO)
    
    # Create file handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(os.path.join(log_dir, f'training_{timestamp}.log'))
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def plot_metrics(metrics: Dict[str, List[float]], output_dir: str) -> None:
    """Plot training and validation metrics.
    
    Args:
        metrics: Dictionary of metrics
        output_dir: Directory to save plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['train_loss'], label='Train Loss')
    if 'val_loss' in metrics and len(metrics['val_loss']) > 0:
        # Plot validation loss at the correct epochs
        val_epochs = list(range(0, len(metrics['train_loss']), len(metrics['train_loss']) // len(metrics['val_loss'])))
        if len(val_epochs) > len(metrics['val_loss']):
            val_epochs = val_epochs[:len(metrics['val_loss'])]
        plt.plot(val_epochs, metrics['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss.png'))
    plt.close()
    
    # Plot accuracy
    if 'train_position_accuracy' in metrics and 'train_sequence_accuracy' in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['train_position_accuracy'], label='Train Position Accuracy')
        plt.plot(metrics['train_sequence_accuracy'], label='Train Sequence Accuracy')
        if 'val_position_accuracy' in metrics and 'val_sequence_accuracy' in metrics and len(metrics['val_position_accuracy']) > 0:
            # Plot validation accuracy at the correct epochs
            val_epochs = list(range(0, len(metrics['train_position_accuracy']), len(metrics['train_position_accuracy']) // len(metrics['val_position_accuracy'])))
            if len(val_epochs) > len(metrics['val_position_accuracy']):
                val_epochs = val_epochs[:len(metrics['val_position_accuracy'])]
            plt.plot(val_epochs, metrics['val_position_accuracy'], label='Validation Position Accuracy')
            plt.plot(val_epochs, metrics['val_sequence_accuracy'], label='Validation Sequence Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'accuracy.png'))
        plt.close()
    
    # Plot learning rate
    if 'learning_rate' in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['learning_rate'])
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'learning_rate.png'))
        plt.close()
    
    # Plot curriculum stages
    if 'stage' in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['stage'])
        plt.xlabel('Epoch')
        plt.ylabel('Stage')
        plt.title('Curriculum Learning Stages')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'stages.png'))
        plt.close()


def main():
    """Run training for the TTM model."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train the TTM model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device to train on')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum gradient norm')
    parser.add_argument('--warmup_steps', type=int, default=100, help='Number of warmup steps')
    parser.add_argument('--curriculum', action='store_true', help='Use curriculum learning')
    parser.add_argument('--num_stages', type=int, default=6, help='Number of curriculum stages')
    parser.add_argument('--accuracy_threshold', type=float, default=0.9, help='Accuracy threshold for curriculum progression')
    parser.add_argument('--max_epochs_per_stage', type=int, default=1000, help='Maximum epochs per curriculum stage')
    parser.add_argument('--vocab_size', type=int, default=128, help='Vocabulary size')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--memory_size', type=int, default=16, help='Memory size')
    parser.add_argument('--r', type=int, default=4, help='Number of memory slots to read/write')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set up logging
    logger = setup_logging(args.log_dir)
    logger.info(f"Arguments: {args}")
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    logger.info("Creating model...")
    model = TokenTuringMachine(
        vocab_size=args.vocab_size,
        embedding_dim=args.embedding_dim,
        memory_size=args.memory_size,
        r=args.r,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        eos_token=2  # Assuming EOS token is 2
    )
    
    # Create optimizer
    logger.info("Creating optimizer...")
    optimizer = create_optimizer(
        model=model,
        optimizer_type='adamw',
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create scheduler
    logger.info("Creating scheduler...")
    scheduler = create_scheduler(
        optimizer=optimizer,
        scheduler_type='cosine',
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.num_epochs * 1000  # Approximate number of steps
    )
    
    # Create loss function
    logger.info("Creating loss function...")
    loss_fn = model.create_loss_fn(
        loss_type='ttm',
        memory_loss_weight=0.1,
        attention_loss_weight=0.1
    )
    
    # Resume from checkpoint if requested
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.checkpoint_path}")
        model, optimizer, scheduler, checkpoint = load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_path=args.checkpoint_path or os.path.join(args.checkpoint_dir, 'latest_model.pt'),
            device=device
        )
        
        # Log checkpoint information
        logger.info(f"Resumed from epoch {checkpoint['epoch']} with global step {checkpoint['global_step']}")
    
    # Train with curriculum if requested
    if args.curriculum:
        logger.info("Training with curriculum learning...")
        
        # Define dataset function
        def dataset_fn(stage):
            logger.info(f"Creating dataloaders for stage {stage}...")
            return create_curriculum_dataloaders(
                stage=stage,
                batch_size=args.batch_size,
                max_digits=5,
                seq_len=32,
                pad_token_id=0,
                eos_token_id=2
            )
        
        # Train with curriculum
        metrics = train_with_curriculum(
            model=model,
            dataset_fn=dataset_fn,
            num_stages=args.num_stages,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            device=device,
            num_epochs_per_stage=args.max_epochs_per_stage,
            accuracy_threshold=args.accuracy_threshold,
            accuracy_window=5,
            max_grad_norm=args.max_grad_norm,
            gradient_accumulation_steps=1,
            log_interval=10,
            eval_interval=1,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_interval=1000,
            early_stopping_patience=20,
            early_stopping_metric='loss',
            eos_token_id=2,
            pad_token_id=0,
            use_amp=args.use_amp,
            logger=logger
        )
    else:
        logger.info("Training without curriculum...")
        
        # Create dataloaders
        logger.info("Creating dataloaders...")
        train_dataset = MultiplicationDataset(
            num_digits_a=2,
            num_digits_b=2,
            seq_len=32,
            pad_token_id=0,
            eos_token_id=2,
            num_examples=10000
        )
        
        val_dataset = MultiplicationDataset(
            num_digits_a=2,
            num_digits_b=2,
            seq_len=32,
            pad_token_id=0,
            eos_token_id=2,
            num_examples=1000
        )
        
        train_dataloader, val_dataloader = create_dataloaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        # Train model
        logger.info("Training model...")
        metrics = train_loop(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            device=device,
            num_epochs=args.num_epochs,
            max_grad_norm=args.max_grad_norm,
            gradient_accumulation_steps=1,
            log_interval=10,
            eval_interval=1,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_interval=1000,
            early_stopping_patience=20,
            early_stopping_metric='loss',
            eos_token_id=2,
            pad_token_id=0,
            use_amp=args.use_amp,
            logger=logger
        )
    
    # Save final metrics
    logger.info("Saving metrics...")
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Plot metrics
    logger.info("Plotting metrics...")
    plot_metrics(metrics, args.output_dir)
    
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
