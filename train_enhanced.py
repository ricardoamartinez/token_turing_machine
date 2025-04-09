"""
Enhanced training script for the TTM model with improved training strategies.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import argparse
from datetime import datetime
import random
from typing import Dict, List, Tuple, Optional

from src.ttm.models.ttm_model import TokenTuringMachine
from src.ttm.data.multiplication_dataset import MultiplicationDataset
from src.ttm.data.tokenization import TIMES_TOKEN, EOS_TOKEN, PAD_TOKEN


def setup_logging(log_dir='./logs'):
    """Set up logging."""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('ttm_enhanced_training')
    logger.setLevel(logging.INFO)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(os.path.join(log_dir, f'enhanced_training_{timestamp}.log'))
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def visualize_memory(model, inputs, step, output_dir='./outputs/memory_viz'):
    """Visualize memory content during inference.
    
    Args:
        model: The TTM model
        inputs: Input tensor
        step: Training step (for filename)
        output_dir: Output directory for visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Forward pass with memory tracking
    with torch.no_grad():
        memory_states = []
        
        def hook_fn(module, input, output):
            if hasattr(module, 'memory'):
                memory_states.append(module.memory.detach().cpu())
        
        # Register hook
        if hasattr(model, 'memory_module'):
            hook = model.memory_module.register_forward_hook(hook_fn)
            
            # Run model
            _ = model(inputs)
            
            # Remove hook
            hook.remove()
    
    # Create visualization if we have memory states
    if memory_states:
        plt.figure(figsize=(12, 8))
        
        # Get the final memory state
        memory = memory_states[-1][0]  # [memory_size, embedding_dim]
        
        # Plot as heatmap
        plt.imshow(memory.numpy(), cmap='viridis', aspect='auto')
        plt.colorbar(label='Value')
        plt.xlabel('Embedding Dimension')
        plt.ylabel('Memory Slot')
        plt.title(f'Memory Content (Step {step})')
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f'memory_step_{step}.png'))
        plt.close()


def create_mixed_batch(dataset, current_stage, batch_size, device):
    """Create a mixed batch with examples from current and previous stages.
    
    Args:
        dataset: The dataset
        current_stage: Current difficulty stage
        batch_size: Batch size
        device: Device to create tensors on
        
    Returns:
        Tuple of (inputs, targets)
    """
    # If we're at the first stage, just return a regular batch
    if current_stage == 0:
        return dataset.generate_batch()
    
    # Determine how many examples to take from each stage
    # 70% from current stage, 30% from previous stages
    current_count = int(batch_size * 0.7)
    previous_count = batch_size - current_count
    
    # Save current stage
    original_stage = dataset.current_stage
    
    # Generate examples from current stage
    dataset.current_stage = current_stage
    current_inputs, current_targets = dataset.generate_batch()
    current_inputs = current_inputs[:current_count]
    current_targets = current_targets[:current_count]
    
    # Generate examples from previous stages
    all_prev_inputs = []
    all_prev_targets = []
    
    # Distribute previous examples across all previous stages
    prev_per_stage = max(1, previous_count // current_stage)
    for stage in range(current_stage):
        dataset.current_stage = stage
        prev_inputs, prev_targets = dataset.generate_batch()
        
        # Take a subset of examples from this stage
        stage_count = min(prev_per_stage, previous_count - len(all_prev_inputs))
        if stage_count > 0:
            all_prev_inputs.append(prev_inputs[:stage_count])
            all_prev_targets.append(prev_targets[:stage_count])
    
    # Concatenate all previous examples
    if all_prev_inputs:
        prev_inputs = torch.cat(all_prev_inputs, dim=0)
        prev_targets = torch.cat(all_prev_targets, dim=0)
        
        # Ensure we have exactly previous_count examples
        prev_inputs = prev_inputs[:previous_count]
        prev_targets = prev_targets[:previous_count]
        
        # Combine current and previous examples
        inputs = torch.cat([current_inputs, prev_inputs], dim=0)
        targets = torch.cat([current_targets, prev_targets], dim=0)
    else:
        inputs = current_inputs
        targets = current_targets
    
    # Restore original stage
    dataset.current_stage = original_stage
    
    return inputs, targets


def train_model(
    model,
    dataset,
    num_epochs=200,
    initial_learning_rate=1e-3,
    weight_decay=1e-5,
    device='cpu',
    logger=None,
    checkpoint_dir='./checkpoints',
    output_dir='./outputs',
    save_every=10,
    patience=5,
    difficulty_increase_threshold=0.9,
    visualize_every=50
):
    """Train the TTM model with enhanced strategies.
    
    Args:
        model: The TTM model
        dataset: The dataset
        num_epochs: Number of epochs to train
        initial_learning_rate: Initial learning rate
        weight_decay: Weight decay for regularization
        device: Device to train on
        logger: Logger instance
        checkpoint_dir: Directory to save checkpoints
        output_dir: Directory to save outputs
        save_every: Save checkpoint every N epochs
        patience: Patience for learning rate reduction
        difficulty_increase_threshold: Accuracy threshold for increasing difficulty
        visualize_every: Visualize memory every N steps
        
    Returns:
        Dictionary of training metrics
    """
    # Create optimizer with weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate, weight_decay=weight_decay)
    
    # Create learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=patience, 
        verbose=True
    )
    
    # Create loss function
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    
    # Initialize metrics
    train_losses = []
    train_position_accuracies = []
    train_sequence_accuracies = []
    learning_rates = []
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Track best model
    best_loss = float('inf')
    best_model_path = None
    
    # Training loop
    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        epoch_position_accuracies = []
        epoch_sequence_accuracies = []
        
        # Train for 100 batches per epoch
        for i in range(100):
            # Generate mixed batch
            inputs, targets = create_mixed_batch(
                dataset, 
                dataset.current_stage, 
                dataset.batch_size, 
                device
            )
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits, _ = model(inputs)
            
            # Calculate loss
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # Backward pass
            loss.backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            # Calculate accuracy
            predictions = logits.argmax(dim=-1)
            
            # Position accuracy (token-level)
            mask = targets != PAD_TOKEN
            position_accuracy = (predictions == targets)[mask].float().mean().item()
            
            # Sequence accuracy (entire sequence correct)
            sequence_correct = torch.all((predictions == targets) | (targets == PAD_TOKEN), dim=1).float().mean().item()
            
            # Log metrics
            epoch_losses.append(loss.item())
            epoch_position_accuracies.append(position_accuracy)
            epoch_sequence_accuracies.append(sequence_correct)
            
            # Visualize memory periodically
            if global_step % visualize_every == 0:
                visualize_memory(model, inputs[:1], global_step, os.path.join(output_dir, 'memory_viz'))
            
            global_step += 1
            
            # Log progress
            if (i + 1) % 10 == 0 and logger is not None:
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {i+1}/100, Loss: {loss.item():.4f}, "
                           f"Position Accuracy: {position_accuracy:.4f}, Sequence Accuracy: {sequence_correct:.4f}, "
                           f"LR: {current_lr:.6f}, Stage: {dataset.current_stage + 1}")
        
        # Calculate epoch metrics
        epoch_loss = np.mean(epoch_losses)
        epoch_position_accuracy = np.mean(epoch_position_accuracies)
        epoch_sequence_accuracy = np.mean(epoch_sequence_accuracies)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update learning rate scheduler
        scheduler.step(epoch_loss)
        
        # Log epoch metrics
        if logger is not None:
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, "
                       f"Position Accuracy: {epoch_position_accuracy:.4f}, "
                       f"Sequence Accuracy: {epoch_sequence_accuracy:.4f}, "
                       f"LR: {current_lr:.6f}, Stage: {dataset.current_stage + 1}")
        
        # Store metrics
        train_losses.append(epoch_loss)
        train_position_accuracies.append(epoch_position_accuracy)
        train_sequence_accuracies.append(epoch_sequence_accuracy)
        learning_rates.append(current_lr)
        
        # Check if we should increase difficulty
        if dataset.should_increase_difficulty(epoch_sequence_accuracy, threshold=difficulty_increase_threshold):
            if dataset.increase_difficulty():
                if logger is not None:
                    logger.info(f"Increasing difficulty to stage {dataset.current_stage + 1}")
                
                # Reset learning rate when difficulty increases
                for param_group in optimizer.param_groups:
                    param_group['lr'] = initial_learning_rate
                
                if logger is not None:
                    logger.info(f"Reset learning rate to {initial_learning_rate}")
            else:
                if logger is not None:
                    logger.info("Already at maximum difficulty")
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"ttm_enhanced_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': epoch_loss,
                'dataset_stage': dataset.current_stage
            }, checkpoint_path)
            
            if logger is not None:
                logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_path = os.path.join(checkpoint_dir, "ttm_enhanced_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': epoch_loss,
                'dataset_stage': dataset.current_stage
            }, best_model_path)
            
            if logger is not None:
                logger.info(f"Saved best model with loss {best_loss:.4f}")
        
        # Print example predictions
        if (epoch + 1) % 5 == 0:
            print_examples(model, dataset, device, logger)
    
    # Plot training metrics
    plot_metrics({
        'train_loss': train_losses,
        'train_position_accuracy': train_position_accuracies,
        'train_sequence_accuracy': train_sequence_accuracies,
        'learning_rate': learning_rates
    }, output_dir)
    
    return {
        'train_loss': train_losses,
        'train_position_accuracy': train_position_accuracies,
        'train_sequence_accuracy': train_sequence_accuracies,
        'learning_rate': learning_rates,
        'best_model_path': best_model_path
    }


def print_examples(model, dataset, device, logger=None, num_examples=5):
    """Print example predictions from the model.
    
    Args:
        model: The TTM model
        dataset: The dataset
        device: Device to run on
        logger: Logger instance
        num_examples: Number of examples to print
    """
    model.eval()
    
    with torch.no_grad():
        # Generate batch
        inputs, targets = dataset.generate_batch()
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward pass
        logits, _ = model(inputs)
        
        # Get predictions
        predictions = logits.argmax(dim=-1)
        
        # Print examples
        if logger is not None:
            logger.info("\nExample predictions:")
        else:
            print("\nExample predictions:")
        
        for i in range(min(num_examples, inputs.size(0))):
            input_tokens = inputs[i].cpu().tolist()
            target_tokens = targets[i].cpu().tolist()
            pred_tokens = predictions[i].cpu().tolist()
            
            # Find the position of the multiplication symbol
            try:
                times_pos = input_tokens.index(TIMES_TOKEN)
                eos_pos = input_tokens.index(EOS_TOKEN)
            except ValueError:
                continue
            
            # Extract the operands
            num1_tokens = input_tokens[:times_pos]
            num2_tokens = input_tokens[times_pos+1:eos_pos]
            
            # Convert to strings
            num1_str = ''.join([str(t) for t in num1_tokens])
            num2_str = ''.join([str(t) for t in num2_tokens])
            
            # Find EOS in target and prediction
            try:
                target_eos_pos = target_tokens.index(EOS_TOKEN)
            except ValueError:
                target_eos_pos = len(target_tokens)
            
            try:
                pred_eos_pos = pred_tokens.index(EOS_TOKEN)
            except ValueError:
                pred_eos_pos = len(pred_tokens)
            
            # Extract the results
            target_result_tokens = target_tokens[:target_eos_pos]
            pred_result_tokens = pred_tokens[:pred_eos_pos]
            
            # Convert to strings
            target_result_str = ''.join([str(t) for t in target_result_tokens if t < 10])
            pred_result_str = ''.join([str(t) for t in pred_result_tokens if t < 10])
            
            # Format the example
            example_str = f"{num1_str} × {num2_str} = {target_result_str} (predicted: {pred_result_str})"
            
            if logger is not None:
                logger.info(f"  Example {i+1}: {example_str}")
            else:
                print(f"  Example {i+1}: {example_str}")


def plot_metrics(metrics, output_dir='./outputs'):
    """Plot training metrics.
    
    Args:
        metrics: Dictionary of metrics
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['train_loss'], label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'enhanced_loss.png'))
    plt.close()
    
    # Plot accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['train_position_accuracy'], label='Position Accuracy')
    plt.plot(metrics['train_sequence_accuracy'], label='Sequence Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'enhanced_accuracy.png'))
    plt.close()
    
    # Plot learning rate
    if 'learning_rate' in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['learning_rate'])
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
        plt.yscale('log')
        plt.savefig(os.path.join(output_dir, 'learning_rate.png'))
        plt.close()


def test_multiplication(model, num1, num2, device='cpu'):
    """Test the model on a single multiplication problem.
    
    Args:
        model: The TTM model
        num1: First number
        num2: Second number
        device: Device to run on
        
    Returns:
        Dictionary with test results
    """
    # Create input sequence
    input_tokens = []
    for digit in str(num1):
        input_tokens.append(int(digit))
    
    input_tokens.append(TIMES_TOKEN)
    
    for digit in str(num2):
        input_tokens.append(int(digit))
    
    input_tokens.append(EOS_TOKEN)
    
    # Pad to length 20
    input_tokens = input_tokens + [PAD_TOKEN] * (20 - len(input_tokens))
    
    # Convert to tensor
    inputs = torch.tensor([input_tokens], dtype=torch.long, device=device)
    
    # Forward pass
    with torch.no_grad():
        logits, _ = model(inputs)
    
    # Get predictions
    predictions = logits.argmax(dim=-1)[0].cpu().tolist()
    
    # Find EOS in prediction
    try:
        eos_pos = predictions.index(EOS_TOKEN)
    except ValueError:
        eos_pos = len(predictions)
    
    # Extract the result
    pred_result_tokens = predictions[:eos_pos]
    
    # Convert to string
    pred_result_str = ''.join([str(t) for t in pred_result_tokens if t < 10])
    
    # Check if prediction is correct
    expected_result = num1 * num2
    try:
        pred_result = int(pred_result_str)
        is_correct = pred_result == expected_result
    except ValueError:
        # Invalid prediction
        pred_result = None
        is_correct = False
    
    return {
        'num1': num1,
        'num2': num2,
        'expected': expected_result,
        'predicted': pred_result,
        'predicted_str': pred_result_str,
        'is_correct': is_correct
    }


def test_model(model, device, logger=None):
    """Test the model on multiplication tasks.
    
    Args:
        model: The TTM model
        device: Device to run on
        logger: Logger instance
        
    Returns:
        Dictionary of test results
    """
    model.eval()
    
    # Define test cases for different difficulty levels
    test_cases = [
        # Single-digit multiplication
        [(3, 4), (5, 7), (8, 9), (2, 6), (7, 8)],
        
        # Two-digit by one-digit multiplication
        [(12, 5), (34, 7), (56, 9), (78, 3), (90, 6)],
        
        # Two-digit by two-digit multiplication
        [(12, 34), (56, 78), (23, 45), (67, 89), (10, 20)],
        
        # Three-digit by two-digit multiplication
        [(123, 45), (678, 90), (234, 56), (789, 12), (345, 67)]
    ]
    
    results = {}
    
    for i, cases in enumerate(test_cases):
        correct = 0
        total = len(cases)
        
        if logger is not None:
            logger.info(f"\nTest set {i+1}:")
            if i == 0:
                logger.info("Single-digit multiplication:")
            elif i == 1:
                logger.info("Two-digit by one-digit multiplication:")
            elif i == 2:
                logger.info("Two-digit by two-digit multiplication:")
            elif i == 3:
                logger.info("Three-digit by two-digit multiplication:")
        
        for num1, num2 in cases:
            result = test_multiplication(model, num1, num2, device)
            
            if logger is not None:
                logger.info(f"  {num1} × {num2} = {result['expected']} (predicted: {result['predicted_str']})")
            
            if result['is_correct']:
                correct += 1
        
        accuracy = correct / total
        
        if logger is not None:
            logger.info(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
        
        # Store results
        if i == 0:
            results['single_digit'] = accuracy
        elif i == 1:
            results['two_by_one_digit'] = accuracy
        elif i == 2:
            results['two_by_two_digit'] = accuracy
        elif i == 3:
            results['three_by_two_digit'] = accuracy
    
    return results


def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train and test TTM with enhanced strategies')
    parser.add_argument('--output_dir', type=str, default='./outputs/enhanced', help='Output directory')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/enhanced', help='Checkpoint directory')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run on')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--memory_size', type=int, default=16, help='Memory size')
    parser.add_argument('--r', type=int, default=4, help='Number of memory slots to read/write')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Set device
    device = torch.device(args.device)
    
    # Set up logging
    logger = setup_logging(args.log_dir)
    logger.info(f"Arguments: {args}")
    
    # Create model with increased capacity
    logger.info("Creating model with increased capacity...")
    model = TokenTuringMachine(
        vocab_size=13,
        embedding_dim=args.embedding_dim,
        memory_size=args.memory_size,
        r=args.r,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    )
    model.to(device)
    
    # Create dataset
    logger.info("Creating dataset...")
    dataset = MultiplicationDataset(batch_size=args.batch_size, max_seq_len=20, device=device)
    
    # Train model with enhanced strategies
    logger.info("Training model with enhanced strategies...")
    metrics = train_model(
        model=model,
        dataset=dataset,
        num_epochs=args.num_epochs,
        initial_learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        logger=logger,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir
    )
    
    # Test model
    logger.info("Testing model...")
    results = test_model(model, device, logger)
    
    # Log results
    logger.info("\nTest results:")
    logger.info(f"  Single-digit multiplication: {results['single_digit']:.4f}")
    logger.info(f"  Two-digit by one-digit multiplication: {results['two_by_one_digit']:.4f}")
    logger.info(f"  Two-digit by two-digit multiplication: {results['two_by_two_digit']:.4f}")
    logger.info(f"  Three-digit by two-digit multiplication: {results['three_by_two_digit']:.4f}")


if __name__ == '__main__':
    main()
