"""
Model Data Extractor for the TTM visualization engine.

This module provides functionality for extracting tensors and other data structures
from a model's computational graph for visualization.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union, Set
import numpy as np
import inspect
import re


class ModelDataExtractor:
    """Extracts data from a model's computational graph for visualization."""

    def __init__(self, model: nn.Module):
        """Initialize the model data extractor.

        Args:
            model: PyTorch model to extract data from
        """
        self.model = model
        self.hooks = []
        self.tensor_data = {}
        self.registered_modules = set()

    def register_module(self, module_name: str, module: nn.Module) -> None:
        """Register a module for data extraction.

        Args:
            module_name: Name of the module
            module: Module to register
        """
        if module_name in self.registered_modules:
            print(f"Module {module_name} already registered")
            return

        # Register forward hook
        handle = module.register_forward_hook(
            lambda m, inp, out: self._forward_hook(module_name, m, inp, out)
        )
        self.hooks.append(handle)
        self.registered_modules.add(module_name)
        print(f"Registered module: {module_name}")

    def register_all_modules(self, prefix: str = "") -> None:
        """Register all modules in the model for data extraction.

        Args:
            prefix: Prefix for module names
        """
        for name, module in self.model.named_modules():
            if name:  # Skip the root module
                full_name = f"{prefix}.{name}" if prefix else name
                self.register_module(full_name, module)

    def _forward_hook(self, name: str, module: nn.Module, inp: Tuple[torch.Tensor], 
                     out: Union[torch.Tensor, Tuple[torch.Tensor]]) -> None:
        """Forward hook for data extraction.

        Args:
            name: Name of the module
            module: Module
            inp: Input tensors
            out: Output tensors
        """
        # Store input tensors
        if isinstance(inp, tuple) and len(inp) > 0:
            for i, tensor in enumerate(inp):
                if isinstance(tensor, torch.Tensor):
                    self.tensor_data[f"{name}.input{i}"] = tensor.detach()

        # Store output tensors
        if isinstance(out, torch.Tensor):
            self.tensor_data[f"{name}.output"] = out.detach()
        elif isinstance(out, tuple):
            for i, tensor in enumerate(out):
                if isinstance(tensor, torch.Tensor):
                    self.tensor_data[f"{name}.output{i}"] = tensor.detach()

        # Store module parameters
        for param_name, param in module.named_parameters():
            self.tensor_data[f"{name}.{param_name}"] = param.detach()

        # Store module buffers
        for buffer_name, buffer in module.named_buffers():
            self.tensor_data[f"{name}.{buffer_name}"] = buffer.detach()

    def extract_attention(self) -> Dict[str, torch.Tensor]:
        """Extract attention matrices from the model.

        Returns:
            Dictionary mapping attention names to tensors
        """
        attention_tensors = {}
        
        # Look for attention-related tensors
        for name, tensor in self.tensor_data.items():
            if "attention" in name.lower() or "attn" in name.lower():
                attention_tensors[name] = tensor
        
        return attention_tensors

    def extract_memory(self) -> Dict[str, torch.Tensor]:
        """Extract memory matrices from the model.

        Returns:
            Dictionary mapping memory names to tensors
        """
        memory_tensors = {}
        
        # Look for memory-related tensors
        for name, tensor in self.tensor_data.items():
            if "memory" in name.lower() or "mem" in name.lower():
                memory_tensors[name] = tensor
        
        return memory_tensors

    def extract_embeddings(self) -> Dict[str, torch.Tensor]:
        """Extract embedding matrices from the model.

        Returns:
            Dictionary mapping embedding names to tensors
        """
        embedding_tensors = {}
        
        # Look for embedding-related tensors
        for name, tensor in self.tensor_data.items():
            if "embed" in name.lower():
                embedding_tensors[name] = tensor
        
        return embedding_tensors

    def extract_tensor_by_name(self, pattern: str) -> Dict[str, torch.Tensor]:
        """Extract tensors by name pattern.

        Args:
            pattern: Regex pattern to match tensor names

        Returns:
            Dictionary mapping tensor names to tensors
        """
        matching_tensors = {}
        
        # Look for tensors matching the pattern
        for name, tensor in self.tensor_data.items():
            if re.search(pattern, name):
                matching_tensors[name] = tensor
        
        return matching_tensors

    def extract_tensor_by_shape(self, shape: Tuple[int, ...]) -> Dict[str, torch.Tensor]:
        """Extract tensors by shape.

        Args:
            shape: Tensor shape to match

        Returns:
            Dictionary mapping tensor names to tensors
        """
        matching_tensors = {}
        
        # Look for tensors with the specified shape
        for name, tensor in self.tensor_data.items():
            if tensor.shape == shape:
                matching_tensors[name] = tensor
        
        return matching_tensors

    def extract_all_tensors(self) -> Dict[str, torch.Tensor]:
        """Extract all tensors from the model.

        Returns:
            Dictionary mapping tensor names to tensors
        """
        return self.tensor_data.copy()

    def clear_data(self) -> None:
        """Clear all extracted data."""
        self.tensor_data.clear()

    def remove_hooks(self) -> None:
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.registered_modules.clear()

    def cleanup(self) -> None:
        """Clean up resources."""
        self.remove_hooks()
        self.clear_data()
