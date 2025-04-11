"""
CUDA-OpenGL Voxel Renderer for the TTM visualization engine.

This module provides functionality for rendering voxels using CUDA-OpenGL interoperability,
allowing for efficient GPU-based processing and visualization of model data structures.
"""

import os
import pyglet
from pyglet.gl import *
import numpy as np
import torch
import ctypes
from typing import Dict, List, Tuple, Optional, Any, Union

# Define cuda as None if not available
cuda = None

from .voxel_renderer import VoxelRenderer
from .shader_manager import ShaderManager

# Check if CUDA is available
CUDA_AVAILABLE = torch.cuda.is_available()

# Import PyCUDA if CUDA is available
if CUDA_AVAILABLE:
    try:
        import pycuda.driver as cuda
        import pycuda.gl as cuda_gl
        from pycuda.gl import graphics_map_flags
        import pycuda.gpuarray as gpuarray
        import pycuda.autoinit
        PYCUDA_AVAILABLE = True
    except ImportError:
        print("PyCUDA not available. CUDA-OpenGL interoperability will be disabled.")
        PYCUDA_AVAILABLE = False
        # Define mock CUDA functions for fallback
        class MockCuda:
            def mem_alloc(self, size):
                return None
            def memcpy_htod(self, dst, src):
                pass
            def memcpy_dtod(self, dst, src, size):
                pass
        cuda = MockCuda()
        cuda_gl = None
else:
    PYCUDA_AVAILABLE = False
    print("CUDA not available. CUDA-OpenGL interoperability will be disabled.")
    # Define mock CUDA functions for fallback
    class MockCuda:
        def mem_alloc(self, size):
            return None
        def memcpy_htod(self, dst, src):
            pass
        def memcpy_dtod(self, dst, src, size):
            pass
    cuda = MockCuda()
    cuda_gl = None


class CudaVoxelRenderer(VoxelRenderer):
    """Renders voxels using CUDA-OpenGL interoperability for efficient GPU-based processing."""

    def __init__(self, max_voxels: int = 100000):
        """Initialize the CUDA voxel renderer.

        Args:
            max_voxels: Maximum number of voxels to render
        """
        # Call parent constructor
        super().__init__(max_voxels)

        # CUDA-specific attributes
        self.cuda_available = CUDA_AVAILABLE and PYCUDA_AVAILABLE
        self.cuda_gl_buffer = None
        self.cuda_mapped_buffer = None
        self.cuda_buffer_mapping = None

        # Create CUDA-OpenGL shared buffer if CUDA is available
        if self.cuda_available:
            self._init_cuda_gl_buffer()

    def _init_cuda_gl_buffer(self):
        """Initialize CUDA-OpenGL shared buffer."""
        try:
            # Register the instance VBO with CUDA
            self.cuda_gl_buffer = cuda_gl.RegisteredBuffer(int(self.vbo_instances))
            print(f"Successfully registered OpenGL buffer with CUDA: {self.vbo_instances}")
        except Exception as e:
            print(f"Failed to register OpenGL buffer with CUDA: {e}")
            self.cuda_available = False

    def update_from_tensor(self, tensor: torch.Tensor, dimensions: Tuple[int, int, int],
                          colormap: str = 'viridis'):
        """Update voxel data directly from a tensor using CUDA.

        Args:
            tensor: Input tensor (should be on CUDA device)
            dimensions: Dimensions of the voxel grid (x, y, z)
            colormap: Name of the colormap to use
        """
        if not self.cuda_available:
            print("CUDA-OpenGL interoperability not available. Falling back to CPU implementation.")
            # Convert tensor to numpy and use the standard update method
            tensor_cpu = tensor.detach().cpu().numpy()
            self._update_from_numpy(tensor_cpu, dimensions, colormap)
            return

        # Ensure tensor is on CUDA
        if not tensor.is_cuda:
            tensor = tensor.cuda()

        # Normalize tensor values to [0, 1]
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        if tensor_min != tensor_max:
            normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
        else:
            normalized_tensor = torch.zeros_like(tensor)

        # Map the tensor to voxels based on its dimensionality
        if len(tensor.shape) == 1:
            voxel_tensor = self._map_1d_tensor_cuda(normalized_tensor, dimensions)
        elif len(tensor.shape) == 2:
            voxel_tensor = self._map_2d_tensor_cuda(normalized_tensor, dimensions)
        elif len(tensor.shape) == 3:
            voxel_tensor = self._map_3d_tensor_cuda(normalized_tensor, dimensions)
        else:
            # Flatten higher dimensions to 3D
            voxel_tensor = self._map_nd_tensor_cuda(normalized_tensor, dimensions)

        # Map the OpenGL buffer to CUDA
        try:
            self.cuda_buffer_mapping = self.cuda_gl_buffer.map()
            self.cuda_mapped_buffer = self.cuda_buffer_mapping.device_ptr_and_size()

            # Get the number of voxels in the tensor
            num_voxels = torch.sum(voxel_tensor > 0).item()
            self.num_active_voxels = min(num_voxels, self.max_voxels)

            # Create instance data on GPU
            instance_data = self._create_instance_data_cuda(voxel_tensor, dimensions, colormap)

            # Copy instance data to mapped buffer
            cuda.memcpy_dtod(self.cuda_mapped_buffer[0], instance_data.ptr,
                            min(instance_data.nbytes, self.cuda_mapped_buffer[1]))

            # Unmap the buffer
            self.cuda_buffer_mapping.unmap()

            print(f"Updated {self.num_active_voxels} voxels using CUDA")
        except Exception as e:
            print(f"Error updating voxels with CUDA: {e}")
            import traceback
            traceback.print_exc()

            # Fall back to CPU implementation
            tensor_cpu = tensor.detach().cpu().numpy()
            self._update_from_numpy(tensor_cpu, dimensions, colormap)

    def _update_from_numpy(self, array: np.ndarray, dimensions: Tuple[int, int, int],
                          colormap: str = 'viridis'):
        """Update voxel data from a numpy array (CPU fallback).

        Args:
            array: Input numpy array
            dimensions: Dimensions of the voxel grid (x, y, z)
            colormap: Name of the colormap to use
        """
        # Normalize array values to [0, 1]
        array_min = array.min()
        array_max = array.max()
        if array_min != array_max:
            normalized_array = (array - array_min) / (array_max - array_min)
        else:
            normalized_array = np.zeros_like(array)

        # Create voxel grid
        voxels = np.zeros(dimensions, dtype=np.float32)

        # Map the array to voxels based on its dimensionality
        if len(array.shape) == 1:
            self._map_1d_array(normalized_array, voxels)
        elif len(array.shape) == 2:
            self._map_2d_array(normalized_array, voxels)
        elif len(array.shape) == 3:
            self._map_3d_array(normalized_array, voxels)
        else:
            # Flatten higher dimensions to 3D
            flattened_shape = (array.shape[0], array.shape[1], np.prod(array.shape[2:]).astype(int))
            flattened_array = array.reshape(flattened_shape)
            self._map_3d_array(flattened_array, voxels)

        # Reset instance data
        self.instance_data = np.zeros(self.max_voxels, dtype=[
            ('position', np.float32, 3),
            ('scale', np.float32, 3),
            ('color', np.float32, 4),
            ('value', np.float32)
        ])

        # Create instance data for non-zero voxels
        voxel_index = 0
        for x in range(dimensions[0]):
            for y in range(dimensions[1]):
                for z in range(dimensions[2]):
                    if voxels[x, y, z] > 0.0:
                        # Calculate position
                        position = np.array([
                            (x / dimensions[0] - 0.5) * 2.0,
                            (y / dimensions[1] - 0.5) * 2.0,
                            (z / dimensions[2] - 0.5) * 2.0
                        ], dtype=np.float32)

                        # Calculate color based on colormap
                        value = voxels[x, y, z]
                        color = self._get_color_from_colormap(value, colormap)

                        # Set voxel data
                        self.set_voxel(voxel_index, position, np.array([0.05, 0.05, 0.05]), color, value)
                        voxel_index += 1

                        if voxel_index >= self.max_voxels:
                            break

            if voxel_index >= self.max_voxels:
                break

        self.num_active_voxels = voxel_index
        print(f"Updated {voxel_index} voxels using CPU")

    def _map_1d_array(self, array: np.ndarray, voxels: np.ndarray) -> None:
        """Map a 1D array to voxels.

        Args:
            array: 1D array data
            voxels: Output voxel grid
        """
        x_dim = min(len(array), voxels.shape[0])
        for i in range(x_dim):
            voxels[i, 0, 0] = array[i]

    def _map_2d_array(self, array: np.ndarray, voxels: np.ndarray) -> None:
        """Map a 2D array to voxels.

        Args:
            array: 2D array data
            voxels: Output voxel grid
        """
        x_dim = min(array.shape[0], voxels.shape[0])
        y_dim = min(array.shape[1], voxels.shape[1])

        for i in range(x_dim):
            for j in range(y_dim):
                voxels[i, j, 0] = array[i, j]

    def _map_3d_array(self, array: np.ndarray, voxels: np.ndarray) -> None:
        """Map a 3D array to voxels.

        Args:
            array: 3D array data
            voxels: Output voxel grid
        """
        x_dim = min(array.shape[0], voxels.shape[0])
        y_dim = min(array.shape[1], voxels.shape[1])
        z_dim = min(array.shape[2], voxels.shape[2])

        for i in range(x_dim):
            for j in range(y_dim):
                for k in range(z_dim):
                    voxels[i, j, k] = array[i, j, k]

    def _map_1d_tensor_cuda(self, tensor: torch.Tensor, dimensions: Tuple[int, int, int]) -> torch.Tensor:
        """Map a 1D tensor to voxels using CUDA.

        Args:
            tensor: 1D tensor data
            dimensions: Dimensions of the voxel grid (x, y, z)

        Returns:
            3D tensor of voxel values
        """
        # Create empty voxel tensor on GPU
        voxels = torch.zeros(dimensions, dtype=torch.float32, device=tensor.device)

        # Map 1D data to a line along the x-axis
        x_dim = min(len(tensor), dimensions[0])
        voxels[:x_dim, 0, 0] = tensor[:x_dim]

        return voxels

    def _map_2d_tensor_cuda(self, tensor: torch.Tensor, dimensions: Tuple[int, int, int]) -> torch.Tensor:
        """Map a 2D tensor to voxels using CUDA.

        Args:
            tensor: 2D tensor data
            dimensions: Dimensions of the voxel grid (x, y, z)

        Returns:
            3D tensor of voxel values
        """
        # Create empty voxel tensor on GPU
        voxels = torch.zeros(dimensions, dtype=torch.float32, device=tensor.device)

        # Map 2D data to a plane in the x-y plane
        x_dim = min(tensor.shape[0], dimensions[0])
        y_dim = min(tensor.shape[1], dimensions[1])

        voxels[:x_dim, :y_dim, 0] = tensor[:x_dim, :y_dim]

        return voxels

    def _map_3d_tensor_cuda(self, tensor: torch.Tensor, dimensions: Tuple[int, int, int]) -> torch.Tensor:
        """Map a 3D tensor to voxels using CUDA.

        Args:
            tensor: 3D tensor data
            dimensions: Dimensions of the voxel grid (x, y, z)

        Returns:
            3D tensor of voxel values
        """
        # Create empty voxel tensor on GPU
        voxels = torch.zeros(dimensions, dtype=torch.float32, device=tensor.device)

        # Map 3D data directly to voxels
        x_dim = min(tensor.shape[0], dimensions[0])
        y_dim = min(tensor.shape[1], dimensions[1])
        z_dim = min(tensor.shape[2], dimensions[2])

        voxels[:x_dim, :y_dim, :z_dim] = tensor[:x_dim, :y_dim, :z_dim]

        return voxels

    def _map_nd_tensor_cuda(self, tensor: torch.Tensor, dimensions: Tuple[int, int, int]) -> torch.Tensor:
        """Map a higher-dimensional tensor to voxels using CUDA.

        Args:
            tensor: N-dimensional tensor data
            dimensions: Dimensions of the voxel grid (x, y, z)

        Returns:
            3D tensor of voxel values
        """
        # Flatten the tensor to 3D
        shape = tensor.shape
        if len(shape) < 3:
            raise ValueError("Tensor must have at least 3 dimensions")

        # Reshape to 3D by flattening all but the first 3 dimensions
        flattened_shape = (shape[0], shape[1], torch.prod(torch.tensor(shape[2:])).item())
        flattened_tensor = tensor.reshape(flattened_shape)

        # Map the flattened 3D tensor
        return self._map_3d_tensor_cuda(flattened_tensor, dimensions)

    def _create_instance_data_cuda(self, voxel_tensor: torch.Tensor,
                                  dimensions: Tuple[int, int, int],
                                  colormap: str) -> Any:
        """Create instance data for voxels on the GPU.

        Args:
            voxel_tensor: 3D tensor of voxel values
            dimensions: Dimensions of the voxel grid (x, y, z)
            colormap: Name of the colormap to use

        Returns:
            CUDA device allocation containing instance data
        """
        # Create a CUDA kernel to generate instance data
        # This is a placeholder - in a real implementation, you would use a CUDA kernel
        # to efficiently generate instance data directly on the GPU

        # For now, we'll use a CPU implementation and transfer the data to GPU
        # Create instance data on CPU
        instance_data = np.zeros(self.max_voxels, dtype=[
            ('position', np.float32, 3),
            ('scale', np.float32, 3),
            ('color', np.float32, 4),
            ('value', np.float32)
        ])

        # Convert voxel tensor to CPU for processing
        voxels_cpu = voxel_tensor.detach().cpu().numpy()

        # Create instance data for non-zero voxels
        voxel_index = 0
        for x in range(dimensions[0]):
            for y in range(dimensions[1]):
                for z in range(dimensions[2]):
                    if voxels_cpu[x, y, z] > 0.0:
                        # Calculate position
                        position = np.array([
                            (x / dimensions[0] - 0.5) * 2.0,
                            (y / dimensions[1] - 0.5) * 2.0,
                            (z / dimensions[2] - 0.5) * 2.0
                        ], dtype=np.float32)

                        # Calculate color based on colormap
                        value = voxels_cpu[x, y, z]
                        color = self._get_color_from_colormap(value, colormap)

                        # Set instance data
                        instance_data[voxel_index]['position'] = position
                        instance_data[voxel_index]['scale'] = np.array([0.05, 0.05, 0.05], dtype=np.float32)
                        instance_data[voxel_index]['color'] = color
                        instance_data[voxel_index]['value'] = value

                        voxel_index += 1
                        if voxel_index >= self.max_voxels:
                            break

            if voxel_index >= self.max_voxels:
                break

        # Transfer instance data to GPU
        instance_data_gpu = cuda.mem_alloc(instance_data.nbytes)
        cuda.memcpy_htod(instance_data_gpu, instance_data)

        return instance_data_gpu

    def _get_color_from_colormap(self, value: float, colormap: str) -> np.ndarray:
        """Get color from a colormap.

        Args:
            value: Value to map to color (0-1)
            colormap: Name of the colormap

        Returns:
            RGBA color
        """
        if colormap == 'viridis':
            # Viridis colormap (default)
            if value < 0.25:
                # Dark purple to teal
                t = value * 4.0
                color = np.array([0.2, 0.0 + t * 0.3, 0.3 + t * 0.3, 1.0])
            elif value < 0.5:
                # Teal to green
                t = (value - 0.25) * 4.0
                color = np.array([0.2 + t * 0.2, 0.3 + t * 0.3, 0.6 - t * 0.2, 1.0])
            elif value < 0.75:
                # Green to yellow
                t = (value - 0.5) * 4.0
                color = np.array([0.4 + t * 0.6, 0.6 + t * 0.4, 0.4 - t * 0.4, 1.0])
            else:
                # Yellow to yellow
                t = (value - 0.75) * 4.0
                color = np.array([1.0, 1.0, 0.0 + t, 1.0])
        elif colormap == 'plasma':
            # Plasma colormap (for memory)
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
        elif colormap == 'inferno':
            # Inferno colormap (for attention)
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
                color = np.array([1.0, 0.5 + t * 0.5, t, 1.0])
        else:
            # Default to grayscale
            color = np.array([value, value, value, 1.0])

        return color

    def cleanup(self) -> None:
        """Clean up OpenGL and CUDA resources."""
        # Clean up CUDA resources
        if self.cuda_available:
            if self.cuda_buffer_mapping is not None:
                try:
                    self.cuda_buffer_mapping.unmap()
                except:
                    pass

            if self.cuda_gl_buffer is not None:
                try:
                    self.cuda_gl_buffer.unregister()
                except:
                    pass

        # Call parent cleanup
        super().cleanup()
