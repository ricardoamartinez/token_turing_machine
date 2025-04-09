"""
Training monitoring utilities for the Token Turing Machine (TTM) model.

This module provides utilities for monitoring and visualizing the training process
of the TTM model, including gradient flow, parameter distributions, and loss landscape.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
from datetime import datetime
import math

from ..models.ttm_model import TokenTuringMachine


class TrainingMonitor:
    """Class for monitoring and visualizing the training process of the TTM model."""
    
    def __init__(
        self,
        model: nn.Module,
        output_dir: str = './outputs/monitor',
        log_dir: str = './logs',
        device: Optional[torch.device] = None
    ):
        """Initialize the training monitor.
        
        Args:
            model: The model to monitor
            output_dir: Directory to save monitoring outputs
            log_dir: Directory to save logs
            device: Device to run on
        """
        self.model = model
        self.output_dir = output_dir
        self.log_dir = log_dir
        self.device = device if device is not None else torch.device('cpu')
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up logging
        self.logger = self._setup_logging()
        
        # Initialize monitoring metrics
        self.gradient_norms = {}
        self.parameter_stats = {}
        self.training_metrics = {
            'loss': [],
            'position_accuracy': [],
            'sequence_accuracy': [],
            'learning_rate': [],
            'difficulty_stage': []
        }
        
        # Register hooks
        self.hooks = []
        
        self.logger.info("Training monitor initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for monitoring.
        
        Returns:
            Logger instance
        """
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create logger
        logger = logging.getLogger('ttm_monitor')
        logger.setLevel(logging.INFO)
        
        # Create file handler
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_handler = logging.FileHandler(os.path.join(self.log_dir, f'monitor_{timestamp}.log'))
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
    
    def register_hooks(self):
        """Register hooks for collecting monitoring information."""
        # Remove existing hooks
        self.remove_hooks()
        
        # Register gradient hooks for all parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(lambda grad, name=name: self._gradient_hook(grad, name))
                self.hooks.append(hook)
        
        self.logger.info(f"Registered {len(self.hooks)} monitoring hooks")
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.logger.info("Removed all monitoring hooks")
    
    def _gradient_hook(self, grad: torch.Tensor, name: str):
        """Hook for collecting gradient information.
        
        Args:
            grad: Gradient tensor
            name: Parameter name
        """
        # Calculate gradient norm
        norm = grad.norm().item()
        
        # Store gradient norm
        if name not in self.gradient_norms:
            self.gradient_norms[name] = []
        
        self.gradient_norms[name].append(norm)
        
        # Check for gradient issues
        if math.isnan(norm) or math.isinf(norm):
            self.logger.warning(f"Gradient issue detected in {name}: {norm}")
        elif norm > 10.0:
            self.logger.warning(f"Large gradient detected in {name}: {norm}")
        elif norm < 1e-7 and norm > 0:
            self.logger.warning(f"Small gradient detected in {name}: {norm}")
    
    def collect_parameter_stats(self):
        """Collect statistics about model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Calculate statistics
                data = param.detach().cpu().numpy()
                stats = {
                    'mean': np.mean(data),
                    'std': np.std(data),
                    'min': np.min(data),
                    'max': np.max(data),
                    'norm': np.linalg.norm(data)
                }
                
                # Store statistics
                if name not in self.parameter_stats:
                    self.parameter_stats[name] = []
                
                self.parameter_stats[name].append(stats)
    
    def update_metrics(
        self,
        loss: float,
        position_accuracy: float,
        sequence_accuracy: float,
        learning_rate: float,
        difficulty_stage: int
    ):
        """Update training metrics.
        
        Args:
            loss: Loss value
            position_accuracy: Position accuracy
            sequence_accuracy: Sequence accuracy
            learning_rate: Learning rate
            difficulty_stage: Difficulty stage
        """
        self.training_metrics['loss'].append(loss)
        self.training_metrics['position_accuracy'].append(position_accuracy)
        self.training_metrics['sequence_accuracy'].append(sequence_accuracy)
        self.training_metrics['learning_rate'].append(learning_rate)
        self.training_metrics['difficulty_stage'].append(difficulty_stage)
    
    def visualize_metrics(self, step: Optional[int] = None, save: bool = True):
        """Visualize training metrics.
        
        Args:
            step: Training step (for filename)
            save: Whether to save the visualization
        """
        # Check if we have metrics data
        if not self.training_metrics['loss']:
            self.logger.warning("No metrics data available")
            return
        
        # Create figure
        fig, axs = plt.subplots(3, 1, figsize=(12, 15))
        
        # Plot loss
        axs[0].plot(self.training_metrics['loss'])
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')
        axs[0].set_title('Training Loss')
        axs[0].grid(True)
        
        # Plot accuracy
        axs[1].plot(self.training_metrics['position_accuracy'], label='Position Accuracy')
        axs[1].plot(self.training_metrics['sequence_accuracy'], label='Sequence Accuracy')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Accuracy')
        axs[1].set_title('Training Accuracy')
        axs[1].legend()
        axs[1].grid(True)
        
        # Plot learning rate and difficulty stage
        ax2 = axs[2].twinx()
        axs[2].plot(self.training_metrics['learning_rate'], 'g-', label='Learning Rate')
        ax2.plot(self.training_metrics['difficulty_stage'], 'r-', label='Difficulty Stage')
        axs[2].set_xlabel('Epoch')
        axs[2].set_ylabel('Learning Rate')
        ax2.set_ylabel('Difficulty Stage')
        axs[2].set_title('Learning Rate and Difficulty Stage')
        
        # Add both legends
        lines1, labels1 = axs[2].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        axs[2].grid(True)
        
        # Add overall title
        plt.suptitle('Training Metrics')
        plt.tight_layout()
        
        # Save figure
        if save:
            filename = f"metrics_{step}.png" if step is not None else "metrics.png"
            plt.savefig(os.path.join(self.output_dir, filename))
            self.logger.info(f"Saved metrics visualization to {filename}")
        
        plt.close()
    
    def visualize_gradients(self, step: Optional[int] = None, save: bool = True):
        """Visualize gradient flow in the model.
        
        Args:
            step: Training step (for filename)
            save: Whether to save the visualization
        """
        # Check if we have gradient data
        if not self.gradient_norms:
            self.logger.warning("No gradient data available")
            return
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot gradient norms for each parameter
        for name, norms in self.gradient_norms.items():
            if len(norms) > 0:
                plt.semilogy(norms, label=name)
        
        # Add labels and legend
        plt.xlabel('Training Step')
        plt.ylabel('Gradient Norm (log scale)')
        plt.title('Gradient Flow')
        plt.grid(True)
        
        # Adjust legend
        if len(self.gradient_norms) > 20:
            # Too many parameters, group by layer
            handles, labels = plt.gca().get_legend_handles_labels()
            by_layer = {}
            for i, label in enumerate(labels):
                layer = label.split('.')[0]
                if layer not in by_layer:
                    by_layer[layer] = i
            
            plt.legend([handles[by_layer[layer]] for layer in by_layer], 
                       [layer for layer in by_layer], 
                       loc='best')
        else:
            plt.legend(loc='best')
        
        # Save figure
        if save:
            filename = f"gradient_flow_{step}.png" if step is not None else "gradient_flow.png"
            plt.savefig(os.path.join(self.output_dir, filename))
            self.logger.info(f"Saved gradient flow visualization to {filename}")
        
        plt.close()
    
    def visualize_parameter_distributions(self, step: Optional[int] = None, save: bool = True):
        """Visualize parameter distributions.
        
        Args:
            step: Training step (for filename)
            save: Whether to save the visualization
        """
        # Check if we have parameter data
        if not self.parameter_stats:
            self.logger.warning("No parameter statistics available")
            return
        
        # Create figure
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract statistics
        names = list(self.parameter_stats.keys())
        means = [stats[-1]['mean'] for name, stats in self.parameter_stats.items()]
        stds = [stats[-1]['std'] for name, stats in self.parameter_stats.items()]
        mins = [stats[-1]['min'] for name, stats in self.parameter_stats.items()]
        maxs = [stats[-1]['max'] for name, stats in self.parameter_stats.items()]
        
        # Plot statistics
        axs[0, 0].bar(range(len(names)), means)
        axs[0, 0].set_title('Mean Values')
        axs[0, 0].set_xticks([])
        
        axs[0, 1].bar(range(len(names)), stds)
        axs[0, 1].set_title('Standard Deviations')
        axs[0, 1].set_xticks([])
        
        axs[1, 0].bar(range(len(names)), mins)
        axs[1, 0].set_title('Minimum Values')
        axs[1, 0].set_xticks([])
        
        axs[1, 1].bar(range(len(names)), maxs)
        axs[1, 1].set_title('Maximum Values')
        axs[1, 1].set_xticks([])
        
        # Add overall title
        plt.suptitle('Parameter Statistics')
        plt.tight_layout()
        
        # Save figure
        if save:
            filename = f"parameter_stats_{step}.png" if step is not None else "parameter_stats.png"
            plt.savefig(os.path.join(self.output_dir, filename))
            self.logger.info(f"Saved parameter statistics visualization to {filename}")
        
        plt.close()
    
    def generate_report(self, step: Optional[int] = None, save: bool = True):
        """Generate a comprehensive training report.
        
        Args:
            step: Training step (for filename)
            save: Whether to save the report
        """
        # Generate visualizations
        self.visualize_metrics(step, save)
        self.visualize_gradients(step, save)
        self.visualize_parameter_distributions(step, save)
        
        # Detect training issues
        issues = self.detect_training_issues()
        
        # Log issues
        if issues:
            self.logger.warning("Detected training issues:")
            for issue in issues:
                self.logger.warning(f"  - {issue}")
        else:
            self.logger.info("No training issues detected")
        
        # Generate report text
        report = []
        report.append("# Training Report")
        report.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Step: {step}" if step is not None else "")
        report.append("")
        
        report.append("## Training Metrics")
        if self.training_metrics['loss']:
            report.append(f"Current Loss: {self.training_metrics['loss'][-1]:.4f}")
            report.append(f"Current Position Accuracy: {self.training_metrics['position_accuracy'][-1]:.4f}")
            report.append(f"Current Sequence Accuracy: {self.training_metrics['sequence_accuracy'][-1]:.4f}")
            report.append(f"Current Learning Rate: {self.training_metrics['learning_rate'][-1]:.6f}")
            report.append(f"Current Difficulty Stage: {self.training_metrics['difficulty_stage'][-1]}")
        else:
            report.append("No metrics data available")
        report.append("")
        
        report.append("## Training Issues")
        if issues:
            for issue in issues:
                report.append(f"- {issue}")
        else:
            report.append("No training issues detected")
        report.append("")
        
        report.append("## Recommendations")
        if issues:
            if any("vanishing" in issue for issue in issues):
                report.append("- Consider increasing the learning rate")
                report.append("- Add skip connections or layer normalization")
            if any("exploding" in issue for issue in issues):
                report.append("- Consider decreasing the learning rate")
                report.append("- Add gradient clipping")
            if any("dead neurons" in issue for issue in issues):
                report.append("- Check activation functions")
                report.append("- Add weight initialization")
            if any("extreme" in issue for issue in issues):
                report.append("- Add weight decay or regularization")
        else:
            report.append("- Continue training with current parameters")
        
        # Save report
        if save:
            filename = f"report_{step}.md" if step is not None else "report.md"
            with open(os.path.join(self.output_dir, filename), 'w') as f:
                f.write('\n'.join(report))
            self.logger.info(f"Saved training report to {filename}")
        
        return '\n'.join(report)
    
    def detect_training_issues(self) -> List[str]:
        """Detect common training issues.
        
        Returns:
            List of detected issues
        """
        issues = []
        
        # Check for gradient issues
        for name, norms in self.gradient_norms.items():
            if len(norms) > 0:
                # Check for NaN or Inf
                if any(math.isnan(norm) or math.isinf(norm) for norm in norms):
                    issues.append(f"NaN or Inf gradients detected in {name}")
                
                # Check for vanishing gradients
                if any(0 < norm < 1e-7 for norm in norms):
                    issues.append(f"Vanishing gradients detected in {name}")
                
                # Check for exploding gradients
                if any(norm > 10.0 for norm in norms):
                    issues.append(f"Exploding gradients detected in {name}")
        
        # Check for parameter issues
        for name, stats_list in self.parameter_stats.items():
            if len(stats_list) > 0:
                stats = stats_list[-1]
                
                # Check for NaN or Inf
                if math.isnan(stats['mean']) or math.isinf(stats['mean']):
                    issues.append(f"NaN or Inf parameters detected in {name}")
                
                # Check for dead neurons (std = 0)
                if stats['std'] == 0:
                    issues.append(f"Dead neurons detected in {name} (std = 0)")
                
                # Check for extreme values
                if abs(stats['max']) > 100 or abs(stats['min']) > 100:
                    issues.append(f"Extreme parameter values detected in {name}")
        
        # Check for training metrics issues
        if self.training_metrics['loss']:
            # Check for loss not decreasing
            if len(self.training_metrics['loss']) > 10:
                recent_losses = self.training_metrics['loss'][-10:]
                if all(recent_losses[i] >= recent_losses[i-1] for i in range(1, len(recent_losses))):
                    issues.append("Loss not decreasing for 10 epochs")
            
            # Check for accuracy not increasing
            if len(self.training_metrics['sequence_accuracy']) > 10:
                recent_accs = self.training_metrics['sequence_accuracy'][-10:]
                if all(recent_accs[i] <= recent_accs[i-1] for i in range(1, len(recent_accs))):
                    issues.append("Sequence accuracy not increasing for 10 epochs")
        
        return issues


def create_training_monitor(
    model: nn.Module,
    output_dir: str = './outputs/monitor',
    log_dir: str = './logs',
    device: Optional[torch.device] = None
) -> TrainingMonitor:
    """Create a training monitor instance.
    
    Args:
        model: The model to monitor
        output_dir: Directory to save monitoring outputs
        log_dir: Directory to save logs
        device: Device to run on
        
    Returns:
        Training monitor instance
    """
    return TrainingMonitor(model, output_dir, log_dir, device)
