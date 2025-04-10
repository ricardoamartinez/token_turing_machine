"""
State Tracker for TTM model.

This module provides functionality to track and store model states during training and inference.
"""

import torch
import numpy as np
import os
import json
from typing import Dict, List, Tuple, Any, Optional, Union
import pickle
from datetime import datetime


class TTMStateTracker:
    """
    Tracks and stores internal states of the TTM model during training and inference.

    This class provides hooks to capture model states at different points in the
    training and inference process, and stores them for later visualization and analysis.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 storage_dir: str = './visualization_data',
                 max_epochs: int = 100,
                 max_batches_per_epoch: int = 100,
                 max_tokens_per_batch: int = 20,
                 sampling_rate: float = 0.1):
        """
        Initialize the state tracker.

        Args:
            model: The TTM model to track
            storage_dir: Directory to store state data
            max_epochs: Maximum number of epochs to track
            max_batches_per_epoch: Maximum number of batches per epoch to track
            max_tokens_per_batch: Maximum number of tokens per batch to track
            sampling_rate: Fraction of batches to sample (0.1 = 10%)
        """
        self.model = model
        self.storage_dir = storage_dir
        self.max_epochs = max_epochs
        self.max_batches_per_epoch = max_batches_per_epoch
        self.max_tokens_per_batch = max_tokens_per_batch
        self.sampling_rate = sampling_rate

        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)

        # Initialize state storage
        self.current_epoch = 0
        self.current_batch = 0
        self.current_token = 0
        self.state_history = {
            'epochs': [],
            'batches': {},
            'tokens': {},
            'states': {}
        }

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register hooks to capture model states during forward and backward passes."""
        # Define target modules to track
        self.target_modules = {
            'token_embedding': ['token_embedding'],
            'memory': ['memory_module'],
            'transformer': ['transformer'],
            'output_head': ['output_head']
        }

        # Store registered hooks for cleanup
        self.hooks = []

        # Register forward hooks on specific modules
        for module_type, module_names in self.target_modules.items():
            for name, module in self.model.named_modules():
                # Check if this module should be tracked
                if any(module_name in name for module_name in module_names):
                    # Register pre-forward hook
                    pre_hook = module.register_forward_pre_hook(
                        lambda mod, inp, module_name=name: self._capture_input(module_name, inp)
                    )
                    self.hooks.append(pre_hook)

                    # Register forward hook
                    hook = module.register_forward_hook(
                        lambda mod, inp, out, module_name=name: self._capture_output(module_name, inp, out)
                    )
                    self.hooks.append(hook)

                    print(f"Registered hooks for module: {name}")

        # Register backward hooks for parameter gradients
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(lambda grad, name=name: self._capture_gradient(name, grad))
                self.hooks.append(hook)

    def _capture_input(self, module_name: str, inputs: torch.Tensor):
        """Capture module input during forward pass.

        Args:
            module_name: Name of the module
            inputs: Input tensor(s)
        """
        # Skip if not sampling this batch
        if np.random.random() >= self.sampling_rate:
            return

        # Create state key
        state_key = (self.current_epoch, self.current_batch, self.current_token)

        # Initialize state dictionary if needed
        if state_key not in self.state_history['states']:
            self.state_history['states'][state_key] = {}

        # Initialize module dictionary if needed
        if 'modules' not in self.state_history['states'][state_key]:
            self.state_history['states'][state_key]['modules'] = {}

        # Initialize this module's dictionary if needed
        if module_name not in self.state_history['states'][state_key]['modules']:
            self.state_history['states'][state_key]['modules'][module_name] = {}

        # Store input tensor(s) in standardized format
        if isinstance(inputs, tuple) and len(inputs) > 0:
            # Handle tuple of tensors
            for i, inp in enumerate(inputs):
                if isinstance(inp, torch.Tensor):
                    # Create standardized state format
                    state = {
                        'name': f"{module_name}_input_{i}",
                        'type': 'tensor',
                        'shape': inp.shape,
                        'data': inp.detach().cpu().numpy(),
                        'metadata': {
                            'epoch': self.current_epoch,
                            'batch': self.current_batch,
                            'token': self.current_token,
                            'module': module_name,
                            'is_input': True,
                            'index': i
                        }
                    }
                    # Store in state history
                    if 'inputs' not in self.state_history['states'][state_key]['modules'][module_name]:
                        self.state_history['states'][state_key]['modules'][module_name]['inputs'] = []
                    self.state_history['states'][state_key]['modules'][module_name]['inputs'].append(state)
        elif isinstance(inputs, torch.Tensor):
            # Handle single tensor
            # Create standardized state format
            state = {
                'name': f"{module_name}_input",
                'type': 'tensor',
                'shape': inputs.shape,
                'data': inputs.detach().cpu().numpy(),
                'metadata': {
                    'epoch': self.current_epoch,
                    'batch': self.current_batch,
                    'token': self.current_token,
                    'module': module_name,
                    'is_input': True
                }
            }
            # Store in state history
            self.state_history['states'][state_key]['modules'][module_name]['inputs'] = state

    def _capture_output(self, module_name: str, inputs: torch.Tensor, outputs: torch.Tensor):
        """Capture module output during forward pass.

        Args:
            module_name: Name of the module
            inputs: Input tensor(s)
            outputs: Output tensor(s)
        """
        # Skip if not sampling this batch
        if np.random.random() >= self.sampling_rate:
            return

        # Create state key
        state_key = (self.current_epoch, self.current_batch, self.current_token)

        # Initialize state dictionary if needed
        if state_key not in self.state_history['states']:
            self.state_history['states'][state_key] = {}

        # Initialize module dictionary if needed
        if 'modules' not in self.state_history['states'][state_key]:
            self.state_history['states'][state_key]['modules'] = {}

        # Initialize this module's dictionary if needed
        if module_name not in self.state_history['states'][state_key]['modules']:
            self.state_history['states'][state_key]['modules'][module_name] = {}

        # Store output tensor(s) in standardized format
        if isinstance(outputs, tuple) and len(outputs) > 0:
            # Handle tuple of tensors
            for i, out in enumerate(outputs):
                if isinstance(out, torch.Tensor):
                    # Create standardized state format
                    state = {
                        'name': f"{module_name}_output_{i}",
                        'type': 'tensor',
                        'shape': out.shape,
                        'data': out.detach().cpu().numpy(),
                        'metadata': {
                            'epoch': self.current_epoch,
                            'batch': self.current_batch,
                            'token': self.current_token,
                            'module': module_name,
                            'is_input': False,
                            'index': i
                        }
                    }
                    # Store in state history
                    if 'outputs' not in self.state_history['states'][state_key]['modules'][module_name]:
                        self.state_history['states'][state_key]['modules'][module_name]['outputs'] = []
                    self.state_history['states'][state_key]['modules'][module_name]['outputs'].append(state)
        elif isinstance(outputs, torch.Tensor):
            # Handle single tensor
            # Create standardized state format
            state = {
                'name': f"{module_name}_output",
                'type': 'tensor',
                'shape': outputs.shape,
                'data': outputs.detach().cpu().numpy(),
                'metadata': {
                    'epoch': self.current_epoch,
                    'batch': self.current_batch,
                    'token': self.current_token,
                    'module': module_name,
                    'is_input': False
                }
            }
            # Store in state history
            self.state_history['states'][state_key]['modules'][module_name]['outputs'] = state

    def _capture_forward_states(self, inputs: torch.Tensor, outputs: torch.Tensor):
        """
        Capture model states during the forward pass.

        Args:
            inputs: Input tensor
            outputs: Output tensor from forward pass
        """
        # Create state key
        state_key = (self.current_epoch, self.current_batch, self.current_token)

        # Initialize state dictionary if needed
        if state_key not in self.state_history['states']:
            self.state_history['states'][state_key] = {}

        # Capture input and output
        self.state_history['states'][state_key]['inputs'] = inputs.detach().cpu().numpy()
        self.state_history['states'][state_key]['outputs'] = outputs.detach().cpu().numpy()

        # Capture memory if available
        if hasattr(self.model, 'memory_module') and hasattr(self.model.memory_module, 'memory'):
            self.state_history['states'][state_key]['memory'] = self.model.memory_module.memory.detach().cpu().numpy()

        # Capture attention weights if available
        if hasattr(self.model, 'last_attention_weights'):
            attention_weights = {}
            if isinstance(self.model.last_attention_weights, list):
                for i, layer_weights in enumerate(self.model.last_attention_weights):
                    attention_weights[f'layer_{i}'] = layer_weights.detach().cpu().numpy()
            else:
                attention_weights['combined'] = self.model.last_attention_weights.detach().cpu().numpy()

            self.state_history['states'][state_key]['attention'] = attention_weights

    def _capture_gradient(self, name: str, grad: torch.Tensor):
        """
        Capture parameter gradients during backward pass.

        Args:
            name: Parameter name
            grad: Gradient tensor
        """
        # Create state key
        state_key = (self.current_epoch, self.current_batch, self.current_token)

        # Initialize gradients dictionary if needed
        if state_key not in self.state_history['states']:
            self.state_history['states'][state_key] = {}

        if 'gradients' not in self.state_history['states'][state_key]:
            self.state_history['states'][state_key]['gradients'] = {}

        # Create standardized state format
        state = {
            'name': f"gradient_{name}",
            'type': 'tensor',
            'shape': grad.shape,
            'data': grad.detach().cpu().numpy(),
            'metadata': {
                'epoch': self.current_epoch,
                'batch': self.current_batch,
                'token': self.current_token,
                'parameter_name': name,
                'is_gradient': True
            }
        }

        # Store gradient
        self.state_history['states'][state_key]['gradients'][name] = state

        # Allow gradient to flow normally
        return grad

    def start_epoch(self, epoch_idx: int):
        """
        Start tracking a new epoch.

        Args:
            epoch_idx: Index of the current epoch
        """
        self.current_epoch = epoch_idx
        if epoch_idx not in self.state_history['epochs']:
            self.state_history['epochs'].append(epoch_idx)
            self.state_history['batches'][epoch_idx] = []

    def start_batch(self, batch_idx: int):
        """
        Start tracking a new batch.

        Args:
            batch_idx: Index of the current batch
        """
        self.current_batch = batch_idx
        if batch_idx not in self.state_history['batches'].get(self.current_epoch, []):
            self.state_history['batches'][self.current_epoch].append(batch_idx)
            self.state_history['tokens'][(self.current_epoch, batch_idx)] = []

    def start_token(self, token_idx: int):
        """
        Start tracking a new token.

        Args:
            token_idx: Index of the current token
        """
        self.current_token = token_idx
        key = (self.current_epoch, self.current_batch)
        if token_idx not in self.state_history['tokens'].get(key, []):
            self.state_history['tokens'][key].append(token_idx)

    def save_state_history(self, filename: Optional[str] = None):
        """
        Save the state history to disk.

        Args:
            filename: Optional filename, defaults to timestamp
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"state_history_{timestamp}.pkl"

        filepath = os.path.join(self.storage_dir, filename)

        with open(filepath, 'wb') as f:
            pickle.dump(self.state_history, f)

        print(f"State history saved to {filepath}")

    def load_state_history(self, filepath: str):
        """
        Load state history from disk.

        Args:
            filepath: Path to the state history file
        """
        with open(filepath, 'rb') as f:
            self.state_history = pickle.load(f)

        print(f"State history loaded from {filepath}")

    def get_state(self, epoch: int, batch: int, token: int) -> Dict[str, Any]:
        """
        Get the state for a specific epoch, batch, and token.

        Args:
            epoch: Epoch index
            batch: Batch index
            token: Token index

        Returns:
            Dictionary containing the state
        """
        state_key = (epoch, batch, token)
        return self.state_history['states'].get(state_key, {})

    def get_available_epochs(self) -> List[int]:
        """Get list of available epochs in the state history."""
        return self.state_history['epochs']

    def get_available_batches(self, epoch: int) -> List[int]:
        """Get list of available batches for a specific epoch."""
        return self.state_history['batches'].get(epoch, [])

    def get_available_tokens(self, epoch: int, batch: int) -> List[int]:
        """Get list of available tokens for a specific epoch and batch."""
        key = (epoch, batch)
        return self.state_history['tokens'].get(key, [])
