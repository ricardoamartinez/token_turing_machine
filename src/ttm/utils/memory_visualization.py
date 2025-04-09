"""
Memory visualization utilities for the Token Turing Machine (TTM) model.

This module provides utilities for visualizing the memory content and attention
patterns of the TTM model during training and inference.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List, Tuple, Dict, Optional, Union, Any
import math

from ..models.ttm_model import TokenTuringMachine
from ..data.tokenization import tokens_to_string, TIMES_TOKEN, EOS_TOKEN, PAD_TOKEN


class MemoryVisualizer:
    """Class for visualizing memory content and attention patterns in the TTM model."""
    
    def __init__(
        self,
        model: TokenTuringMachine,
        output_dir: str = './outputs/memory_viz',
        device: Optional[torch.device] = None
    ):
        """Initialize the memory visualizer.
        
        Args:
            model: The TTM model
            output_dir: Directory to save visualizations
            device: Device to run on
        """
        self.model = model
        self.output_dir = output_dir
        self.device = device if device is not None else torch.device('cpu')
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize hooks
        self.hooks = []
        self.memory_states = []
        self.attention_maps = {}
    
    def register_hooks(self):
        """Register hooks for collecting memory and attention information."""
        # Remove existing hooks
        self.remove_hooks()
        
        # Clear stored states
        self.memory_states = []
        self.attention_maps = {}
        
        # Register hook for memory module
        if hasattr(self.model, 'memory_module'):
            hook = self.model.memory_module.register_forward_hook(
                lambda module, input, output: self._memory_hook(module, input, output)
            )
            self.hooks.append(hook)
        
        # Register hooks for attention modules
        if hasattr(self.model, 'transformer'):
            for i, layer in enumerate(self.model.transformer.transformer.layers):
                if hasattr(layer, 'self_attn'):
                    hook = layer.self_attn.register_forward_hook(
                        lambda module, input, output, i=i: self._attention_hook(module, input, output, i)
                    )
                    self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def _memory_hook(self, module, input, output):
        """Hook for collecting memory content.
        
        Args:
            module: Memory module
            input: Input tensors
            output: Output tensors
        """
        if hasattr(module, 'initial_memory'):
            # Store initial memory
            self.memory_states.append(('initial', module.initial_memory.detach().cpu()))
        
        # For read and write operations, we need to capture the memory state
        # This depends on the specific implementation of the memory module
        if hasattr(module, 'read_op') and hasattr(module.read_op, 'memory'):
            self.memory_states.append(('read', module.read_op.memory.detach().cpu()))
        
        if hasattr(module, 'write_op') and hasattr(module.write_op, 'memory'):
            self.memory_states.append(('write', module.write_op.memory.detach().cpu()))
    
    def _attention_hook(self, module, input, output, layer_idx):
        """Hook for collecting attention patterns.
        
        Args:
            module: Attention module
            input: Input tensors
            output: Output tensors
            layer_idx: Layer index
        """
        # Extract attention weights
        if isinstance(output, tuple) and len(output) > 1:
            # Some implementations return attention weights as second element
            attention_weights = output[1]
        elif hasattr(module, 'attn_output_weights'):
            # PyTorch's MultiheadAttention stores attention weights
            attention_weights = module.attn_output_weights
        else:
            return
        
        # Store attention pattern
        key = f"layer_{layer_idx}"
        self.attention_maps[key] = attention_weights.detach().cpu()
    
    def visualize_memory(
        self,
        inputs: torch.Tensor,
        step: Optional[int] = None,
        save: bool = True,
        show: bool = False
    ):
        """Visualize memory content during inference.
        
        Args:
            inputs: Input tensor
            step: Training step (for filename)
            save: Whether to save the visualization
            show: Whether to show the visualization
        """
        # Register hooks
        self.register_hooks()
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(inputs)
        
        # Remove hooks
        self.remove_hooks()
        
        # Create visualization if we have memory states
        if self.memory_states:
            # Create figure
            plt.figure(figsize=(15, 10))
            
            # Determine number of memory states to visualize
            num_states = min(len(self.memory_states), 3)  # Initial, after read, after write
            
            # Create subplots
            for i in range(num_states):
                state_name, memory = self.memory_states[i]
                
                # Get the first batch item
                if memory.dim() > 2:
                    memory = memory[0]  # [memory_size, embedding_dim]
                
                # Create subplot
                plt.subplot(1, num_states, i + 1)
                
                # Plot as heatmap
                sns.heatmap(memory.numpy(), cmap='viridis', annot=False)
                
                # Add labels
                plt.xlabel('Embedding Dimension')
                plt.ylabel('Memory Slot')
                plt.title(f'Memory Content ({state_name})')
            
            # Add overall title
            step_str = f" (Step {step})" if step is not None else ""
            plt.suptitle(f'Memory Content Evolution{step_str}')
            plt.tight_layout()
            
            # Save figure
            if save:
                filename = f"memory_step_{step}.png" if step is not None else "memory.png"
                plt.savefig(os.path.join(self.output_dir, filename))
            
            # Show figure
            if show:
                plt.show()
            else:
                plt.close()
    
    def visualize_attention(
        self,
        inputs: torch.Tensor,
        step: Optional[int] = None,
        save: bool = True,
        show: bool = False
    ):
        """Visualize attention patterns.
        
        Args:
            inputs: Input tensor
            step: Training step (for filename)
            save: Whether to save the visualization
            show: Whether to show the visualization
        """
        # Register hooks
        self.register_hooks()
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(inputs)
        
        # Remove hooks
        self.remove_hooks()
        
        # Create visualization if we have attention maps
        if self.attention_maps:
            # Create figure
            num_layers = len(self.attention_maps)
            fig_height = 4 * num_layers
            plt.figure(figsize=(15, fig_height))
            
            # Create subplots for each layer
            for i, (key, attention) in enumerate(self.attention_maps.items()):
                # Get the first batch item and first head
                if attention.dim() >= 4:  # [batch_size, num_heads, seq_len, seq_len]
                    attention = attention[0, 0]  # [seq_len, seq_len]
                
                # Create subplot
                plt.subplot(num_layers, 1, i + 1)
                
                # Plot attention pattern
                sns.heatmap(attention.numpy(), cmap='viridis', annot=False)
                
                # Add labels
                plt.xlabel('Key Position')
                plt.ylabel('Query Position')
                plt.title(f'Attention Pattern ({key})')
            
            # Add overall title
            step_str = f" (Step {step})" if step is not None else ""
            plt.suptitle(f'Attention Patterns{step_str}')
            plt.tight_layout()
            
            # Save figure
            if save:
                filename = f"attention_step_{step}.png" if step is not None else "attention.png"
                plt.savefig(os.path.join(self.output_dir, filename))
            
            # Show figure
            if show:
                plt.show()
            else:
                plt.close()
    
    def visualize_memory_usage(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        step: Optional[int] = None,
        save: bool = True,
        show: bool = False
    ):
        """Visualize memory usage with input and target sequences.
        
        Args:
            inputs: Input tensor
            targets: Target tensor
            step: Training step (for filename)
            save: Whether to save the visualization
            show: Whether to show the visualization
        """
        # Register hooks
        self.register_hooks()
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Forward pass
        with torch.no_grad():
            logits, _ = self.model(inputs)
            predictions = logits.argmax(dim=-1)
        
        # Remove hooks
        self.remove_hooks()
        
        # Create visualization if we have memory states
        if self.memory_states:
            # Create figure
            plt.figure(figsize=(15, 12))
            
            # Get the first batch item
            input_tokens = inputs[0].cpu().numpy().tolist()
            target_tokens = targets[0].cpu().numpy().tolist()
            pred_tokens = predictions[0].cpu().numpy().tolist()
            
            # Convert to strings
            input_str = tokens_to_string(input_tokens)
            target_str = tokens_to_string(target_tokens)
            pred_str = tokens_to_string(pred_tokens)
            
            # Extract the problem
            try:
                times_pos = input_tokens.index(TIMES_TOKEN)
                eos_pos = input_tokens.index(EOS_TOKEN)
                
                num1_tokens = input_tokens[:times_pos]
                num2_tokens = input_tokens[times_pos+1:eos_pos]
                
                num1 = int(''.join([str(t) for t in num1_tokens]))
                num2 = int(''.join([str(t) for t in num2_tokens]))
                
                problem_str = f"{num1} × {num2} = {num1 * num2}"
            except (ValueError, IndexError):
                problem_str = "Unknown problem"
            
            # Plot memory states
            for i, (state_name, memory) in enumerate(self.memory_states):
                # Get the first batch item
                if memory.dim() > 2:
                    memory = memory[0]  # [memory_size, embedding_dim]
                
                # Create subplot
                plt.subplot(len(self.memory_states) + 1, 1, i + 1)
                
                # Plot as heatmap
                sns.heatmap(memory.numpy(), cmap='viridis', annot=False)
                
                # Add labels
                plt.xlabel('Embedding Dimension')
                plt.ylabel('Memory Slot')
                plt.title(f'Memory Content ({state_name})')
            
            # Add text information
            plt.subplot(len(self.memory_states) + 1, 1, len(self.memory_states) + 1)
            plt.axis('off')
            plt.text(0.1, 0.7, f"Problem: {problem_str}", fontsize=12)
            plt.text(0.1, 0.5, f"Input: {input_str}", fontsize=12)
            plt.text(0.1, 0.3, f"Target: {target_str}", fontsize=12)
            plt.text(0.1, 0.1, f"Prediction: {pred_str}", fontsize=12)
            
            # Add overall title
            step_str = f" (Step {step})" if step is not None else ""
            plt.suptitle(f'Memory Usage for Multiplication{step_str}')
            plt.tight_layout()
            
            # Save figure
            if save:
                filename = f"memory_usage_step_{step}.png" if step is not None else "memory_usage.png"
                plt.savefig(os.path.join(self.output_dir, filename))
            
            # Show figure
            if show:
                plt.show()
            else:
                plt.close()


def visualize_model_memory(
    model: TokenTuringMachine,
    inputs: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
    step: Optional[int] = None,
    output_dir: str = './outputs/memory_viz'
):
    """Visualize memory content and attention patterns for a TTM model.
    
    Args:
        model: The TTM model
        inputs: Input tensor
        targets: Target tensor (optional)
        step: Training step (for filename)
        output_dir: Directory to save visualizations
    """
    visualizer = MemoryVisualizer(model, output_dir)
    
    # Visualize memory content
    visualizer.visualize_memory(inputs, step)
    
    # Visualize attention patterns
    visualizer.visualize_attention(inputs, step)
    
    # Visualize memory usage with input and target
    if targets is not None:
        visualizer.visualize_memory_usage(inputs, targets, step)
