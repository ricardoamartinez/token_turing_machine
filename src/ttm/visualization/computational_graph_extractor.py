"""
Computational Graph Extractor for PyTorch models.

This module provides functionality for extracting and visualizing the computational
graph of PyTorch models, including TTM and vanilla Transformer models.
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from typing import Dict, List, Tuple, Set, Any, Optional, Union
import networkx as nx
import re
import inspect


class ComputationalGraphExtractor:
    """Extracts the computational graph from PyTorch models."""

    def __init__(self):
        """Initialize the computational graph extractor."""
        self.graph = nx.DiGraph()
        self.node_types = {}
        self.node_shapes = {}
        self.node_params = {}
        self.node_positions = {}
        self.edge_data = {}
        self.module_names = {}

    def extract_graph(self, model, input_size: Tuple[int, ...] = None) -> nx.DiGraph:
        """Extract the computational graph from a model.

        Args:
            model: Model (PyTorch or custom)
            input_size: Size of the input tensor (optional)

        Returns:
            NetworkX directed graph representing the computational graph
        """
        # Reset graph
        self.graph = nx.DiGraph()
        self.node_types = {}
        self.node_shapes = {}
        self.node_params = {}
        self.node_positions = {}
        self.edge_data = {}
        self.module_names = {}

        # Check if this is a PyTorch model
        is_pytorch_model = hasattr(model, 'named_modules')

        if is_pytorch_model:
            # Register module names for PyTorch model
            for name, module in model.named_modules():
                if name:  # Skip the root module
                    module_type = type(module).__name__
                    self.module_names[id(module)] = (name, module_type)

            # Extract graph structure from PyTorch model
            self._extract_module_graph(model)

            # If input size is provided, trace the model to get more accurate graph
            if input_size is not None:
                self._trace_model(model, input_size)
        else:
            # For custom models, create a simple graph based on model attributes
            self._extract_custom_model_graph(model)

        # Compute node positions using force-directed layout
        self._compute_node_positions()

        return self.graph

    def _extract_module_graph(self, model: nn.Module, parent: Optional[str] = None) -> None:
        """Extract the graph structure from a module and its children.

        Args:
            model: PyTorch module
            parent: Name of the parent module (optional)
        """
        # Add node for this module
        module_id = str(id(model))
        module_type = type(model).__name__

        if module_id not in self.graph:
            self.graph.add_node(module_id)
            self.node_types[module_id] = module_type

            # Get number of parameters
            num_params = sum(p.numel() for p in model.parameters())
            self.node_params[module_id] = num_params

            # Store parent relationship
            if parent:
                self.graph.add_edge(parent, module_id)
                self.edge_data[(parent, module_id)] = {"type": "parent_child"}

        # Recursively process child modules
        for name, child in model.named_children():
            child_id = str(id(child))

            # Add edge from parent to child
            if child_id not in self.graph:
                self._extract_module_graph(child, module_id)
            elif (module_id, child_id) not in self.graph.edges:
                self.graph.add_edge(module_id, child_id)
                self.edge_data[(module_id, child_id)] = {"type": "parent_child"}

    def _trace_model(self, model: nn.Module, input_size: Tuple[int, ...]) -> None:
        """Trace the model with an example input to get the computational graph.

        Args:
            model: PyTorch model
            input_size: Size of the input tensor
        """
        try:
            # Create dummy input
            dummy_input = torch.zeros(input_size, requires_grad=True)

            # Trace the model
            with torch.no_grad():
                output = model(dummy_input)

            # If output is a tuple/list, take the first element
            if isinstance(output, (tuple, list)):
                output = output[0]

            # Get the graph from autograd
            if hasattr(output, 'grad_fn'):
                self._add_grad_fn_to_graph(output.grad_fn)
        except Exception as e:
            print(f"Error tracing model: {e}")

    def _extract_custom_model_graph(self, model) -> None:
        """Extract a graph from a custom (non-PyTorch) model.

        Args:
            model: Custom model
        """
        # Create a node for the model itself
        model_id = str(id(model))
        model_type = type(model).__name__

        self.graph.add_node(model_id)
        self.node_types[model_id] = model_type

        # Add nodes for model attributes
        for attr_name in dir(model):
            # Skip private attributes and methods
            if attr_name.startswith('_') or callable(getattr(model, attr_name)):
                continue

            try:
                attr = getattr(model, attr_name)

                # Skip None values
                if attr is None:
                    continue

                # Create node for attribute
                attr_id = f"{model_id}_{attr_name}"
                attr_type = type(attr).__name__

                self.graph.add_node(attr_id)
                self.node_types[attr_id] = attr_type

                # Add edge from model to attribute
                self.graph.add_edge(model_id, attr_id)
                self.edge_data[(model_id, attr_id)] = {"type": "has_attribute"}

                # For numpy arrays or tensors, add shape information
                if hasattr(attr, 'shape'):
                    shape_id = f"{attr_id}_shape"
                    shape_str = str(attr.shape)

                    self.graph.add_node(shape_id)
                    self.node_types[shape_id] = "Shape"

                    # Add edge from attribute to shape
                    self.graph.add_edge(attr_id, shape_id)
                    self.edge_data[(attr_id, shape_id)] = {"type": "has_shape"}
            except Exception as e:
                # Skip attributes that can't be accessed
                print(f"Skipping attribute {attr_name}: {e}")

        # Add nodes for model methods
        for method_name in dir(model):
            if not method_name.startswith('_') and callable(getattr(model, method_name)):
                try:
                    method = getattr(model, method_name)

                    # Create node for method
                    method_id = f"{model_id}_{method_name}"

                    self.graph.add_node(method_id)
                    self.node_types[method_id] = "Method"

                    # Add edge from model to method
                    self.graph.add_edge(model_id, method_id)
                    self.edge_data[(model_id, method_id)] = {"type": "has_method"}
                except Exception as e:
                    # Skip methods that can't be accessed
                    print(f"Skipping method {method_name}: {e}")

    def _add_grad_fn_to_graph(self, grad_fn, visited=None) -> None:
        """Add a grad_fn and its parents to the graph.

        Args:
            grad_fn: Autograd function
            visited: Set of visited nodes
        """
        if visited is None:
            visited = set()

        if grad_fn is None or id(grad_fn) in visited:
            return

        visited.add(id(grad_fn))

        # Add node for this grad_fn
        node_id = str(id(grad_fn))
        node_type = type(grad_fn).__name__.replace('Backward', '')

        if node_id not in self.graph:
            self.graph.add_node(node_id)
            self.node_types[node_id] = node_type

        # Process next functions (parents in the autograd graph)
        if hasattr(grad_fn, 'next_functions'):
            for parent_fn, _ in grad_fn.next_functions:
                if parent_fn is not None:
                    parent_id = str(id(parent_fn))
                    parent_type = type(parent_fn).__name__.replace('Backward', '')

                    if parent_id not in self.graph:
                        self.graph.add_node(parent_id)
                        self.node_types[parent_id] = parent_type

                    # Add edge from parent to child (reverse direction for visualization)
                    self.graph.add_edge(parent_id, node_id)
                    self.edge_data[(parent_id, node_id)] = {"type": "data_flow"}

                    # Recursively process parent
                    self._add_grad_fn_to_graph(parent_fn, visited)

    def _compute_node_positions(self) -> None:
        """Compute 3D positions for nodes using a force-directed layout algorithm."""
        # Use NetworkX's spring layout for initial 2D positions
        pos_2d = nx.spring_layout(self.graph, dim=2, seed=42)

        # Assign layers based on topological sorting
        layers = {}
        for i, node in enumerate(nx.topological_sort(self.graph)):
            # Get all predecessors
            predecessors = list(self.graph.predecessors(node))
            if not predecessors:
                # Root node
                layers[node] = 0
            else:
                # Node layer is one more than the maximum layer of its predecessors
                layers[node] = max(layers.get(p, 0) for p in predecessors) + 1

        # Normalize layer values to [0, 1]
        max_layer = max(layers.values()) if layers else 0
        if max_layer > 0:
            normalized_layers = {node: layer / max_layer for node, layer in layers.items()}
        else:
            normalized_layers = {node: 0 for node in self.graph.nodes}

        # Create 3D positions: (x, y) from spring layout, z from layers
        for node in self.graph.nodes:
            if node in pos_2d:
                x, y = pos_2d[node]
                z = normalized_layers[node] * 2 - 1  # Scale to [-1, 1]
                self.node_positions[node] = (x, y, z)
            else:
                # Fallback for nodes not in pos_2d
                self.node_positions[node] = (0, 0, 0)

    def get_graph_data(self) -> Dict[str, Any]:
        """Get the extracted graph data in a format suitable for visualization.

        Returns:
            Dictionary containing graph data
        """
        nodes = []
        edges = []

        # Process nodes
        for node in self.graph.nodes:
            node_type = self.node_types.get(node, "Unknown")
            position = self.node_positions.get(node, (0, 0, 0))
            num_params = self.node_params.get(node, 0)

            # Determine node size based on parameters
            size = 0.05
            if num_params > 0:
                # Logarithmic scaling for node size
                size = 0.05 + 0.02 * np.log1p(num_params / 1000)

            # Determine node color based on type
            color = self._get_color_for_node_type(node_type)

            nodes.append({
                "id": node,
                "type": node_type,
                "position": position,
                "size": size,
                "color": color,
                "params": num_params
            })

        # Process edges
        for src, dst in self.graph.edges:
            edge_type = self.edge_data.get((src, dst), {}).get("type", "default")

            # Determine edge color based on type
            color = (0.7, 0.7, 0.7, 0.5)  # Default gray
            if edge_type == "data_flow":
                color = (0.2, 0.6, 1.0, 0.7)  # Blue for data flow
            elif edge_type == "parent_child":
                color = (0.8, 0.4, 0.2, 0.5)  # Orange for parent-child

            edges.append({
                "source": src,
                "target": dst,
                "type": edge_type,
                "color": color
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "node_types": list(set(self.node_types.values())),
            "edge_types": list(set(edge_data.get("type", "default") for edge_data in self.edge_data.values()))
        }

    def _get_color_for_node_type(self, node_type: str) -> Tuple[float, float, float, float]:
        """Get a color for a node based on its type.

        Args:
            node_type: Type of the node

        Returns:
            RGBA color tuple
        """
        # Define colors for common node types
        type_colors = {
            # Transformer components
            "MultiheadAttention": (0.8, 0.2, 0.2, 1.0),  # Red
            "TransformerEncoderLayer": (0.9, 0.4, 0.0, 1.0),  # Orange
            "TransformerDecoderLayer": (0.9, 0.6, 0.0, 1.0),  # Light orange
            "LayerNorm": (0.6, 0.8, 0.2, 1.0),  # Green-yellow

            # TTM components
            "TTM": (0.2, 0.4, 0.8, 1.0),  # Blue
            "Memory": (0.4, 0.2, 0.8, 1.0),  # Purple
            "MemoryLayer": (0.5, 0.3, 0.9, 1.0),  # Light purple

            # Common layers
            "Linear": (0.2, 0.7, 0.2, 1.0),  # Green
            "Conv2d": (0.2, 0.8, 0.4, 1.0),  # Light green
            "ReLU": (0.7, 0.7, 0.2, 1.0),  # Yellow
            "Dropout": (0.5, 0.5, 0.5, 1.0),  # Gray
            "BatchNorm": (0.6, 0.6, 0.8, 1.0),  # Light blue-gray
            "Embedding": (0.8, 0.4, 0.6, 1.0),  # Pink

            # Operations
            "Add": (0.9, 0.9, 0.2, 1.0),  # Bright yellow
            "Mul": (0.9, 0.7, 0.3, 1.0),  # Gold
            "MatMul": (0.7, 0.3, 0.3, 1.0),  # Dark red
            "Tanh": (0.3, 0.7, 0.7, 1.0),  # Teal
            "Sigmoid": (0.7, 0.3, 0.7, 1.0),  # Purple
            "Softmax": (0.5, 0.3, 0.5, 1.0),  # Dark purple
        }

        # Check for partial matches
        for key, color in type_colors.items():
            if key in node_type:
                return color

        # Default color for unknown types
        return (0.4, 0.4, 0.4, 1.0)  # Gray
