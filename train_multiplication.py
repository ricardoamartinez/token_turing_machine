"""Train TTM on multiplication task.

This script trains a Token Turing Machine on the multiplication task.
"""

import jax
import jax.numpy as jnp
import flax
import optax
import numpy as np
from typing import Dict, Any, Tuple
import time
import argparse
import os

from src.ttm.models.ttm import TTMEncoder, TTMMemorylessEncoder
from src.ttm.data.multiplication_dataset import MultiplicationDataset
from src.ttm.utils.training import (
    create_train_state,
    create_learning_rate_schedule,
    compute_metrics,
    format_number_sequence
)


def create_model(args):
    """Create TTM model based on arguments.
    
    Args:
        args: Command-line arguments
        
    Returns:
        TTM encoder model
    """
    if args.memory_less:
        return TTMMemorylessEncoder(
            memory_size=args.memory_size,
            process_size=args.process_size,
            dim=args.dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            summarization_method=args.summarization_method,
            write_method=args.write_method,
            use_positional_embedding=args.use_positional_embedding,
            dropout_rate=args.dropout_rate
        )
    else:
        return TTMEncoder(
            memory_size=args.memory_size,
            process_size=args.process_size,
            dim=args.dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            summarization_method=args.summarization_method,
            write_method=args.write_method,
            use_positional_embedding=args.use_positional_embedding,
            dropout_rate=args.dropout_rate
        )


def create_embedding_layer(vocab_size, dim):
    """Create token embedding layer.
    
    Args:
        vocab_size: Vocabulary size
        dim: Embedding dimension
        
    Returns:
        Embedding layer
    """
    class EmbeddingLayer(flax.linen.Module):
        """Token embedding layer."""
        
        @flax.linen.compact
        def __call__(self, tokens):
            """Apply token embedding.
            
            Args:
                tokens: Input tokens of shape [batch_size, seq_len]
                
            Returns:
                Embedded tokens of shape [batch_size, seq_len, dim]
            """
            return flax.linen.Embed(
                num_embeddings=vocab_size,
                features=dim,
                embedding_init=flax.linen.initializers.normal(stddev=0.01)
            )(tokens)
    
    return EmbeddingLayer()


def create_output_layer(vocab_size, dim):
    """Create output layer.
    
    Args:
        vocab_size: Vocabulary size
        dim: Input dimension
        
    Returns:
        Output layer
    """
    class OutputLayer(flax.linen.Module):
        """Output layer."""
        
        @flax.linen.compact
        def __call__(self, x):
            """Apply output layer.
            
            Args:
                x: Input of shape [batch_size, seq_len, process_size, dim]
                
            Returns:
                Logits of shape [batch_size, seq_len, vocab_size]
            """
            batch_size, seq_len, process_size, dim = x.shape
            
            # Reshape to [batch_size * seq_len, process_size, dim]
            x = x.reshape(-1, process_size, dim)
            
            # Apply layer normalization
            x = flax.linen.LayerNorm()(x)
            
            # Average over process_size dimension
            x = jnp.mean(x, axis=1)  # [batch_size * seq_len, dim]
            
            # Apply final dense layer
            x = flax.linen.Dense(vocab_size)(x)
            
            # Reshape back to [batch_size, seq_len, vocab_size]
            x = x.reshape(batch_size, seq_len, vocab_size)
            
            return x
    
    return OutputLayer()


def train_step(state, batch, embedding_params, output_params, learning_rate_fn):
    """Perform a single training step.
    
    Args:
        state: Training state
        batch: Batch of data
        embedding_params: Token embedding parameters
        output_params: Output layer parameters
        learning_rate_fn: Learning rate schedule function
        
    Returns:
        Updated state and metrics
    """
    inputs, targets = batch
    
    def loss_fn(params):
        # Embed input tokens
        embedded_inputs = flax.linen.apply(embedding_params, inputs)
        
        # Reshape for TTM
        batch_size, seq_len = inputs.shape
        embedded_inputs = embedded_inputs.reshape(batch_size, seq_len, 1, -1)
        
        # Apply TTM
        outputs = state.apply_fn(params, embedded_inputs, train=True)
        
        # Apply output layer
        logits = flax.linen.apply(output_params, outputs)
        
        # Compute loss
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits,
            labels=targets
        )
        
        # Mask out padding tokens
        mask = targets != 12  # PAD_TOKEN
        loss = jnp.sum(loss * mask) / jnp.sum(mask)
        
        # Compute metrics
        metrics = compute_metrics(logits, targets)
        metrics['loss'] = loss
        
        return loss, metrics
    
    # Get current learning rate
    lr = learning_rate_fn(state.step)
    
    # Compute gradients
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, metrics), grads = grad_fn(state.params)
    
    # Update parameters
    state = state.apply_gradients(grads=grads)
    
    # Add learning rate to metrics
    metrics['learning_rate'] = lr
    
    return state, metrics


def eval_step(state, batch, embedding_params, output_params):
    """Perform a single evaluation step.
    
    Args:
        state: Training state
        batch: Batch of data
        embedding_params: Token embedding parameters
        output_params: Output layer parameters
        
    Returns:
        Metrics
    """
    inputs, targets = batch
    
    # Embed input tokens
    embedded_inputs = flax.linen.apply(embedding_params, inputs)
    
    # Reshape for TTM
    batch_size, seq_len = inputs.shape
    embedded_inputs = embedded_inputs.reshape(batch_size, seq_len, 1, -1)
    
    # Apply TTM
    outputs = state.apply_fn(state.params, embedded_inputs, train=False)
    
    # Apply output layer
    logits = flax.linen.apply(output_params, outputs)
    
    # Compute metrics
    metrics = compute_metrics(logits, targets)
    
    # Compute loss
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits,
        labels=targets
    )
    
    # Mask out padding tokens
    mask = targets != 12  # PAD_TOKEN
    loss = jnp.sum(loss * mask) / jnp.sum(mask)
    
    metrics['loss'] = loss
    
    return metrics, logits


def main(args):
    """Main training function.
    
    Args:
        args: Command-line arguments
    """
    # Set random seed
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)
    
    # Create dataset
    dataset = MultiplicationDataset(batch_size=args.batch_size, seq_len=args.seq_len)
    
    # Create model
    model = create_model(args)
    
    # Create embedding layer
    embedding_layer = create_embedding_layer(dataset.vocab_size, args.dim)
    rng, embed_rng = jax.random.split(rng)
    embedding_params = embedding_layer.init(embed_rng, jnp.ones((1, args.seq_len), dtype=jnp.int32))
    
    # Create output layer
    output_layer = create_output_layer(dataset.vocab_size, args.dim)
    rng, output_rng = jax.random.split(rng)
    output_params = output_layer.init(
        output_rng,
        jnp.ones((1, args.seq_len, args.process_size, args.dim))
    )
    
    # Create training state
    state = create_train_state(
        model=model,
        rng=init_rng,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create learning rate schedule
    learning_rate_fn = create_learning_rate_schedule(
        base_learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        decay_steps=args.decay_steps
    )
    
    # Training loop
    best_accuracy = 0.0
    consecutive_improvements = 0
    
    for epoch in range(args.max_epochs):
        # Training
        train_metrics = []
        
        for _ in range(args.steps_per_epoch):
            # Generate batch
            batch = dataset.generate_batch()
            
            # Apply data augmentation
            if args.use_augmentation:
                batch = dataset.augment_batch(*batch)
            
            # Perform training step
            state, metrics = train_step(
                state=state,
                batch=batch,
                embedding_params=embedding_params,
                output_params=output_params,
                learning_rate_fn=learning_rate_fn
            )
            
            train_metrics.append(metrics)
        
        # Compute average training metrics
        train_metrics = {
            k: np.mean([m[k] for m in train_metrics])
            for k in train_metrics[0]
        }
        
        # Evaluation
        eval_metrics = []
        
        for _ in range(args.eval_steps):
            # Generate batch
            batch = dataset.generate_batch()
            
            # Perform evaluation step
            metrics, logits = eval_step(
                state=state,
                batch=batch,
                embedding_params=embedding_params,
                output_params=output_params
            )
            
            eval_metrics.append(metrics)
        
        # Compute average evaluation metrics
        eval_metrics = {
            k: np.mean([m[k] for m in eval_metrics])
            for k in eval_metrics[0]
        }
        
        # Print metrics
        print(f"Epoch {epoch + 1}/{args.max_epochs}")
        print(f"  Train loss: {train_metrics['loss']:.4f}")
        print(f"  Train position accuracy: {train_metrics['position_accuracy']:.4f}")
        print(f"  Train sequence accuracy: {train_metrics['sequence_accuracy']:.4f}")
        print(f"  Eval loss: {eval_metrics['loss']:.4f}")
        print(f"  Eval position accuracy: {eval_metrics['position_accuracy']:.4f}")
        print(f"  Eval sequence accuracy: {eval_metrics['sequence_accuracy']:.4f}")
        print(f"  Learning rate: {train_metrics['learning_rate']:.6f}")
        print(f"  Current stage: {dataset.current_stage}")
        
        # Check for improvement
        current_accuracy = eval_metrics['sequence_accuracy']
        
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            consecutive_improvements += 1
            print(f"  New best accuracy: {best_accuracy:.4f}")
            print(f"  Consecutive improvements: {consecutive_improvements}")
        else:
            consecutive_improvements = 0
        
        # Progress to next stage if criteria met
        if consecutive_improvements >= 5 and current_accuracy >= 0.9:
            if dataset.increase_difficulty():
                print(f"  Progressing to stage {dataset.current_stage}")
                best_accuracy = 0.0
                consecutive_improvements = 0
            else:
                print("  Already at maximum difficulty")
        
        # Example predictions
        if epoch % args.print_examples_every == 0:
            # Generate batch
            inputs, targets = dataset.generate_batch()
            
            # Embed input tokens
            embedded_inputs = flax.linen.apply(embedding_params, inputs)
            
            # Reshape for TTM
            batch_size, seq_len = inputs.shape
            embedded_inputs = embedded_inputs.reshape(batch_size, seq_len, 1, -1)
            
            # Apply TTM
            outputs = state.apply_fn(state.params, embedded_inputs, train=False)
            
            # Apply output layer
            logits = flax.linen.apply(output_params, outputs)
            
            # Get predictions
            predictions = jnp.argmax(logits, axis=-1)
            
            # Print examples
            print("\nExamples:")
            for i in range(min(5, batch_size)):
                input_str = format_number_sequence(inputs[i])
                target_str = format_number_sequence(targets[i])
                pred_str = format_number_sequence(predictions[i])
                
                print(f"  Input: {input_str}")
                print(f"  Target: {target_str}")
                print(f"  Prediction: {pred_str}")
                print()
        
        # Save checkpoint
        if args.save_checkpoints and epoch % args.save_every == 0:
            checkpoint = {
                'model': state.params,
                'embedding': embedding_params,
                'output': output_params,
                'stage': dataset.current_stage,
                'epoch': epoch
            }
            
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            with open(os.path.join(args.checkpoint_dir, f"checkpoint_{epoch}.pkl"), 'wb') as f:
                flax.serialization.to_bytes(checkpoint)
            
            print(f"Saved checkpoint at epoch {epoch}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train TTM on multiplication task')
    
    # Model parameters
    parser.add_argument('--memory_size', type=int, default=96, help='Memory size (m)')
    parser.add_argument('--process_size', type=int, default=16, help='Process size (r)')
    parser.add_argument('--dim', type=int, default=128, help='Token dimension')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension in transformer')
    parser.add_argument('--summarization_method', type=str, default='mlp', choices=['mlp', 'query', 'pooling'], help='Token summarization method')
    parser.add_argument('--write_method', type=str, default='summarize', choices=['summarize', 'erase_add', 'concatenate'], help='Memory write method')
    parser.add_argument('--use_positional_embedding', action='store_true', help='Use positional embeddings')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--memory_less', action='store_true', help='Use memory-less version for comparison')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=12, help='Sequence length')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps')
    parser.add_argument('--decay_steps', type=int, default=100000, help='Decay steps')
    parser.add_argument('--max_epochs', type=int, default=1000, help='Maximum number of epochs')
    parser.add_argument('--steps_per_epoch', type=int, default=100, help='Steps per epoch')
    parser.add_argument('--eval_steps', type=int, default=20, help='Evaluation steps')
    parser.add_argument('--use_augmentation', action='store_true', help='Use data augmentation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Logging and checkpointing
    parser.add_argument('--print_examples_every', type=int, default=10, help='Print examples every N epochs')
    parser.add_argument('--save_checkpoints', action='store_true', help='Save checkpoints')
    parser.add_argument('--save_every', type=int, default=50, help='Save checkpoint every N epochs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    
    args = parser.parse_args()
    
    main(args)
