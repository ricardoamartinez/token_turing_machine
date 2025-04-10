"""
Visualization manager for the TTM visualization engine.

This module provides functionality for managing visualizations of model states.
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import pickle

from .vis_mapper import VisMapper, create_mapper_for_state, TensorToVoxelMapper
from .voxel_renderer import VoxelRenderer


class VisualizationManager:
    """Manages visualizations of model states."""

    def __init__(self, voxel_renderer: VoxelRenderer, max_states: int = 100):
        """Initialize the visualization manager.

        Args:
            voxel_renderer: Voxel renderer
            max_states: Maximum number of states to visualize
        """
        self.voxel_renderer = voxel_renderer
        self.max_states = max_states

        # Initialize state storage
        self.states: Dict[str, Dict[str, Any]] = {}
        self.active_states: List[str] = []

        # Initialize mappers
        self.mappers: Dict[str, VisMapper] = {}

        # Initialize voxel mapping
        self.voxel_mapping: Dict[str, List[int]] = {}  # Maps state names to voxel indices
        self.next_voxel_index = 0

    def load_state_history(self, filepath: str) -> None:
        """Load state history from a file.

        Args:
            filepath: Path to the state history file
        """
        with open(filepath, 'rb') as f:
            state_history = pickle.load(f)

        print(f"Loaded state history from {filepath}")
        print(f"Number of states: {len(state_history['states'])}")

        # Process states
        for state_key, state_data in state_history['states'].items():
            # Extract epoch, batch, token
            epoch, batch, token = state_key

            # Process modules
            if 'modules' in state_data:
                for module_name, module_data in state_data['modules'].items():
                    # Process inputs
                    if 'inputs' in module_data:
                        inputs = module_data['inputs']
                        if isinstance(inputs, list):
                            for i, input_state in enumerate(inputs):
                                state_name = f"{module_name}_input_{i}_{epoch}_{batch}_{token}"
                                self.add_state(state_name, input_state)
                        elif isinstance(inputs, dict) and 'name' in inputs:
                            state_name = f"{inputs['name']}_{epoch}_{batch}_{token}"
                            self.add_state(state_name, inputs)

                    # Process outputs
                    if 'outputs' in module_data:
                        outputs = module_data['outputs']
                        if isinstance(outputs, list):
                            for i, output_state in enumerate(outputs):
                                state_name = f"{module_name}_output_{i}_{epoch}_{batch}_{token}"
                                self.add_state(state_name, output_state)
                        elif isinstance(outputs, dict) and 'name' in outputs:
                            state_name = f"{outputs['name']}_{epoch}_{batch}_{token}"
                            self.add_state(state_name, outputs)

            # Process gradients
            if 'gradients' in state_data:
                for grad_name, grad_data in state_data['gradients'].items():
                    if isinstance(grad_data, dict) and 'name' in grad_data:
                        state_name = f"{grad_data['name']}_{epoch}_{batch}_{token}"
                        self.add_state(state_name, grad_data)

    def add_state(self, name: str, state: Dict[str, Any]) -> None:
        """Add a state to the visualization manager.

        Args:
            name: State name
            state: State data
        """
        # Check if we've reached the maximum number of states
        if len(self.states) >= self.max_states:
            # Remove the oldest state
            oldest_state = self.active_states.pop(0)
            self.states.pop(oldest_state)

            # Free up voxel indices
            if oldest_state in self.voxel_mapping:
                self.voxel_mapping.pop(oldest_state)

        # Add the state
        self.states[name] = state
        self.active_states.append(name)

        # Create a mapper for the state if needed
        if name not in self.mappers:
            self.mappers[name] = create_mapper_for_state(state)

        # Map the state to voxels
        self._map_state_to_voxels(name)

    def _map_state_to_voxels(self, state_name: str) -> None:
        """Map a state to voxels.

        Args:
            state_name: State name
        """
        # Get the state
        state = self.states[state_name]

        # Get the mapper
        mapper = self.mappers[state_name]

        # Map the state to voxels
        voxel_data = mapper.map_to_voxels(state)

        # Get voxel positions, colors, and scales
        voxels = voxel_data['voxels']
        dimensions = voxel_data['dimensions']

        # Calculate the number of non-zero voxels
        non_zero_indices = np.nonzero(voxels)
        num_non_zero = len(non_zero_indices[0])

        # Allocate voxel indices
        voxel_indices = list(range(self.next_voxel_index, self.next_voxel_index + num_non_zero))
        self.next_voxel_index += num_non_zero

        # Store voxel mapping
        self.voxel_mapping[state_name] = voxel_indices

        # Set voxels in the renderer
        voxel_index = 0
        for x, y, z in zip(non_zero_indices[0], non_zero_indices[1], non_zero_indices[2]):
            # Get voxel value
            value = voxels[x, y, z]

            # Calculate position
            # Scale to [-1, 1] range and offset based on state index
            state_index = self.active_states.index(state_name)
            position = np.array([
                (x / dimensions[0] - 0.5) * 2.0,
                (y / dimensions[1] - 0.5) * 2.0,
                (z / dimensions[2] - 0.5) * 2.0 + state_index * 0.5
            ], dtype=np.float32)

            # Calculate color
            # Use a simple colormap: blue -> cyan -> green -> yellow -> red
            if 'color_map' in voxel_data['metadata']:
                color_map = voxel_data['metadata']['color_map']
                if color_map == 'viridis':
                    # Viridis colormap: dark blue -> blue -> green -> yellow
                    if value < 0.25:
                        # Dark blue to blue
                        t = value * 4.0
                        color = np.array([0.0, t * 0.5, 0.5 + t * 0.5, 1.0])
                    elif value < 0.5:
                        # Blue to green
                        t = (value - 0.25) * 4.0
                        color = np.array([0.0, 0.5 + t * 0.5, 1.0, 1.0])
                    elif value < 0.75:
                        # Green to yellow
                        t = (value - 0.5) * 4.0
                        color = np.array([t, 1.0, 1.0 - t, 1.0])
                    else:
                        # Yellow to red
                        t = (value - 0.75) * 4.0
                        color = np.array([1.0, 1.0 - t, 0.0, 1.0])
                elif color_map == 'plasma':
                    # Plasma colormap: dark blue -> purple -> red -> yellow
                    if value < 0.25:
                        # Dark blue to purple
                        t = value * 4.0
                        color = np.array([0.0, 0.0, 0.5 + t * 0.5, 1.0])
                    elif value < 0.5:
                        # Purple to magenta
                        t = (value - 0.25) * 4.0
                        color = np.array([t, 0.0, 1.0, 1.0])
                    elif value < 0.75:
                        # Magenta to red
                        t = (value - 0.5) * 4.0
                        color = np.array([1.0, 0.0, 1.0 - t, 1.0])
                    else:
                        # Red to yellow
                        t = (value - 0.75) * 4.0
                        color = np.array([1.0, t, 0.0, 1.0])
                elif color_map == 'inferno':
                    # Inferno colormap: black -> purple -> red -> yellow
                    if value < 0.25:
                        # Black to purple
                        t = value * 4.0
                        color = np.array([t * 0.5, 0.0, t, 1.0])
                    elif value < 0.5:
                        # Purple to red
                        t = (value - 0.25) * 4.0
                        color = np.array([0.5 + t * 0.5, 0.0, 1.0 - t * 0.5, 1.0])
                    elif value < 0.75:
                        # Red to orange
                        t = (value - 0.5) * 4.0
                        color = np.array([1.0, t * 0.5, 0.0, 1.0])
                    else:
                        # Orange to yellow
                        t = (value - 0.75) * 4.0
                        color = np.array([1.0, 0.5 + t * 0.5, 0.0, 1.0])
                else:
                    # Default colormap: grayscale
                    color = np.array([value, value, value, 1.0])
            else:
                # Default colormap: grayscale
                color = np.array([value, value, value, 1.0])

            # Calculate scale
            scale = np.array([0.05, 0.05, 0.05], dtype=np.float32)

            # Set voxel
            self.voxel_renderer.set_voxel(voxel_indices[voxel_index], position, scale, color, value)
            voxel_index += 1

    def get_state_name_for_voxel(self, voxel_index: int) -> Optional[str]:
        """Get the state name for a voxel index.

        Args:
            voxel_index: Voxel index

        Returns:
            State name if found, None otherwise
        """
        # Find the state that contains this voxel index
        for state_name, voxel_indices in self.voxel_mapping.items():
            if voxel_index in voxel_indices:
                return state_name

        return None

    def update(self) -> None:
        """Update the visualization."""
        # Update voxel renderer
        self.voxel_renderer.update_buffers()

    def clear(self) -> None:
        """Clear all visualizations."""
        # Clear state storage
        self.states.clear()
        self.active_states.clear()

        # Clear mappers
        self.mappers.clear()

        # Clear voxel mapping
        self.voxel_mapping.clear()

        # Reset voxel index
        self.next_voxel_index = 0
