"""
Visualization mapper for the Token Turing Machine (TTM) model.

This module provides a modular interface for mapping model states to visual representations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import torch


class VisMapper(ABC):
    """Abstract base class for visualization mappers.
    
    A VisMapper is responsible for converting model states (tensors, matrices, etc.)
    into visual representations (voxels, heatmaps, etc.) that can be rendered by
    visualization components.
    """
    
    def __init__(self, name: str):
        """Initialize the visualization mapper.
        
        Args:
            name: Name of the mapper
        """
        self.name = name
    
    @abstractmethod
    def map_to_voxels(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Map a model state to a voxel representation.
        
        Args:
            state: Model state dictionary
            
        Returns:
            Dictionary containing voxel data
        """
        pass
    
    @abstractmethod
    def get_voxel_layout(self) -> Dict[str, Any]:
        """Get the layout information for the voxel representation.
        
        Returns:
            Dictionary containing layout information (dimensions, scale, etc.)
        """
        pass
    
    @abstractmethod
    def get_color_map(self) -> Dict[str, Any]:
        """Get the color map for the voxel representation.
        
        Returns:
            Dictionary containing color map information
        """
        pass


class TensorToVoxelMapper(VisMapper):
    """Mapper for converting tensors to voxel representations."""
    
    def __init__(
        self,
        name: str,
        voxel_dimensions: Tuple[int, int, int] = (16, 16, 16),
        color_map: str = 'viridis'
    ):
        """Initialize the tensor to voxel mapper.
        
        Args:
            name: Name of the mapper
            voxel_dimensions: Dimensions of the voxel grid (x, y, z)
            color_map: Name of the color map to use
        """
        super().__init__(name)
        self.voxel_dimensions = voxel_dimensions
        self.color_map = color_map
    
    def map_to_voxels(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Map a tensor state to a voxel representation.
        
        Args:
            state: State dictionary containing a tensor
            
        Returns:
            Dictionary containing voxel data
        """
        # Extract tensor data
        if 'data' not in state:
            raise ValueError("State dictionary must contain 'data' key")
        
        data = state['data']
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        
        # Get tensor shape
        tensor_shape = data.shape
        
        # Determine mapping strategy based on tensor dimensions
        if len(tensor_shape) == 1:
            # 1D tensor -> map to a line of voxels
            voxels = self._map_1d_tensor(data)
        elif len(tensor_shape) == 2:
            # 2D tensor -> map to a plane of voxels
            voxels = self._map_2d_tensor(data)
        elif len(tensor_shape) == 3:
            # 3D tensor -> map directly to voxels
            voxels = self._map_3d_tensor(data)
        else:
            # Higher dimensional tensor -> flatten to 3D
            voxels = self._map_nd_tensor(data)
        
        # Create voxel data dictionary
        voxel_data = {
            'name': state.get('name', 'unnamed_tensor'),
            'voxels': voxels,
            'dimensions': self.voxel_dimensions,
            'metadata': {
                'original_shape': tensor_shape,
                'mapper': self.name,
                'color_map': self.color_map
            }
        }
        
        return voxel_data
    
    def get_voxel_layout(self) -> Dict[str, Any]:
        """Get the layout information for the voxel representation.
        
        Returns:
            Dictionary containing layout information
        """
        return {
            'dimensions': self.voxel_dimensions,
            'scale': (1.0, 1.0, 1.0),
            'origin': (0.0, 0.0, 0.0)
        }
    
    def get_color_map(self) -> Dict[str, Any]:
        """Get the color map for the voxel representation.
        
        Returns:
            Dictionary containing color map information
        """
        return {
            'name': self.color_map,
            'min_value': 0.0,
            'max_value': 1.0,
            'alpha': 1.0
        }
    
    def _map_1d_tensor(self, data: np.ndarray) -> np.ndarray:
        """Map a 1D tensor to voxels.
        
        Args:
            data: 1D tensor data
            
        Returns:
            3D array of voxel values
        """
        # Normalize data to [0, 1]
        data_min, data_max = data.min(), data.max()
        if data_min != data_max:
            normalized_data = (data - data_min) / (data_max - data_min)
        else:
            normalized_data = np.zeros_like(data)
        
        # Create voxel grid
        voxels = np.zeros(self.voxel_dimensions, dtype=np.float32)
        
        # Map 1D data to a line along the x-axis
        x_dim = min(len(data), self.voxel_dimensions[0])
        for i in range(x_dim):
            voxels[i, 0, 0] = normalized_data[i]
        
        return voxels
    
    def _map_2d_tensor(self, data: np.ndarray) -> np.ndarray:
        """Map a 2D tensor to voxels.
        
        Args:
            data: 2D tensor data
            
        Returns:
            3D array of voxel values
        """
        # Normalize data to [0, 1]
        data_min, data_max = data.min(), data.max()
        if data_min != data_max:
            normalized_data = (data - data_min) / (data_max - data_min)
        else:
            normalized_data = np.zeros_like(data)
        
        # Create voxel grid
        voxels = np.zeros(self.voxel_dimensions, dtype=np.float32)
        
        # Map 2D data to a plane in the x-y plane
        x_dim = min(data.shape[0], self.voxel_dimensions[0])
        y_dim = min(data.shape[1], self.voxel_dimensions[1])
        
        for i in range(x_dim):
            for j in range(y_dim):
                voxels[i, j, 0] = normalized_data[i, j]
        
        return voxels
    
    def _map_3d_tensor(self, data: np.ndarray) -> np.ndarray:
        """Map a 3D tensor to voxels.
        
        Args:
            data: 3D tensor data
            
        Returns:
            3D array of voxel values
        """
        # Normalize data to [0, 1]
        data_min, data_max = data.min(), data.max()
        if data_min != data_max:
            normalized_data = (data - data_min) / (data_max - data_min)
        else:
            normalized_data = np.zeros_like(data)
        
        # Create voxel grid
        voxels = np.zeros(self.voxel_dimensions, dtype=np.float32)
        
        # Map 3D data directly to voxels
        x_dim = min(data.shape[0], self.voxel_dimensions[0])
        y_dim = min(data.shape[1], self.voxel_dimensions[1])
        z_dim = min(data.shape[2], self.voxel_dimensions[2])
        
        for i in range(x_dim):
            for j in range(y_dim):
                for k in range(z_dim):
                    voxels[i, j, k] = normalized_data[i, j, k]
        
        return voxels
    
    def _map_nd_tensor(self, data: np.ndarray) -> np.ndarray:
        """Map a higher-dimensional tensor to voxels.
        
        Args:
            data: N-dimensional tensor data
            
        Returns:
            3D array of voxel values
        """
        # Flatten the tensor to 3D
        shape = data.shape
        if len(shape) < 3:
            raise ValueError("Tensor must have at least 3 dimensions")
        
        # Reshape to 3D by flattening all but the first 3 dimensions
        flattened_shape = (shape[0], shape[1], np.prod(shape[2:]).astype(int))
        flattened_data = data.reshape(flattened_shape)
        
        # Map the flattened 3D tensor
        return self._map_3d_tensor(flattened_data)


class MemoryToVoxelMapper(TensorToVoxelMapper):
    """Specialized mapper for converting memory matrices to voxel representations."""
    
    def __init__(
        self,
        name: str = "memory_mapper",
        voxel_dimensions: Tuple[int, int, int] = (32, 32, 16),
        color_map: str = 'plasma'
    ):
        """Initialize the memory to voxel mapper.
        
        Args:
            name: Name of the mapper
            voxel_dimensions: Dimensions of the voxel grid (x, y, z)
            color_map: Name of the color map to use
        """
        super().__init__(name, voxel_dimensions, color_map)
    
    def map_to_voxels(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Map a memory state to a voxel representation.
        
        Args:
            state: State dictionary containing memory data
            
        Returns:
            Dictionary containing voxel data
        """
        # Extract memory data
        if 'data' not in state:
            raise ValueError("State dictionary must contain 'data' key")
        
        data = state['data']
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        
        # Memory is typically a 2D matrix (r x d)
        # Map it to a 3D volume where the z-axis represents time/token position
        if len(data.shape) == 2:
            # Single memory matrix
            voxels = self._map_2d_tensor(data)
        elif len(data.shape) == 3:
            # Sequence of memory matrices
            voxels = self._map_3d_tensor(data)
        else:
            raise ValueError(f"Unexpected memory shape: {data.shape}")
        
        # Create voxel data dictionary
        voxel_data = {
            'name': state.get('name', 'memory'),
            'voxels': voxels,
            'dimensions': self.voxel_dimensions,
            'metadata': {
                'original_shape': data.shape,
                'mapper': self.name,
                'color_map': self.color_map,
                'is_memory': True
            }
        }
        
        return voxel_data


class AttentionToVoxelMapper(TensorToVoxelMapper):
    """Specialized mapper for converting attention matrices to voxel representations."""
    
    def __init__(
        self,
        name: str = "attention_mapper",
        voxel_dimensions: Tuple[int, int, int] = (32, 32, 16),
        color_map: str = 'inferno'
    ):
        """Initialize the attention to voxel mapper.
        
        Args:
            name: Name of the mapper
            voxel_dimensions: Dimensions of the voxel grid (x, y, z)
            color_map: Name of the color map to use
        """
        super().__init__(name, voxel_dimensions, color_map)
    
    def map_to_voxels(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Map an attention state to a voxel representation.
        
        Args:
            state: State dictionary containing attention data
            
        Returns:
            Dictionary containing voxel data
        """
        # Extract attention data
        if 'data' not in state:
            raise ValueError("State dictionary must contain 'data' key")
        
        data = state['data']
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        
        # Attention is typically a 3D tensor (batch x seq_len x seq_len)
        # or a 4D tensor (batch x num_heads x seq_len x seq_len)
        if len(data.shape) == 2:
            # Single attention matrix
            voxels = self._map_2d_tensor(data)
        elif len(data.shape) == 3:
            # Batch of attention matrices or multi-head attention
            if data.shape[0] == 1:
                # Single batch, multiple heads or sequence positions
                voxels = self._map_2d_tensor(data[0])
            else:
                # Multiple batches or heads
                voxels = self._map_3d_tensor(data)
        elif len(data.shape) == 4:
            # Batch x heads x seq_len x seq_len
            # Flatten heads dimension
            flattened_data = data.reshape(data.shape[0], data.shape[1] * data.shape[2], data.shape[3])
            voxels = self._map_3d_tensor(flattened_data)
        else:
            raise ValueError(f"Unexpected attention shape: {data.shape}")
        
        # Create voxel data dictionary
        voxel_data = {
            'name': state.get('name', 'attention'),
            'voxels': voxels,
            'dimensions': self.voxel_dimensions,
            'metadata': {
                'original_shape': data.shape,
                'mapper': self.name,
                'color_map': self.color_map,
                'is_attention': True
            }
        }
        
        return voxel_data


# Factory function to create the appropriate mapper for a given state
def create_mapper_for_state(state: Dict[str, Any]) -> VisMapper:
    """Create an appropriate mapper for the given state.
    
    Args:
        state: State dictionary
        
    Returns:
        VisMapper instance
    """
    # Determine the type of state
    name = state.get('name', '')
    metadata = state.get('metadata', {})
    
    if 'memory' in name.lower() or metadata.get('is_memory', False):
        return MemoryToVoxelMapper()
    elif 'attention' in name.lower() or metadata.get('is_attention', False):
        return AttentionToVoxelMapper()
    else:
        # Default to generic tensor mapper
        return TensorToVoxelMapper(name="tensor_mapper")
