"""
Voxel renderer for the TTM visualization engine.

This module provides functionality for rendering voxels using instanced rendering.
"""

import os
import pyglet
from pyglet.gl import *
import numpy as np
from typing import Dict, List, Tuple, Optional
import ctypes

from .shader_manager import ShaderManager


class VoxelRenderer:
    """Renders voxels using instanced rendering."""

    def __init__(self, max_voxels: int = 100000):
        """Initialize the voxel renderer.

        Args:
            max_voxels: Maximum number of voxels to render
        """
        self.max_voxels = max_voxels
        self.shader_manager = ShaderManager()

        # Initialize OpenGL resources
        self.vao = GLuint()
        self.vbo_cube = GLuint()
        self.vbo_instances = GLuint()
        self.ebo = GLuint()

        # Initialize instance data
        self.instance_data = np.zeros(max_voxels, dtype=[
            ('position', np.float32, 3),
            ('scale', np.float32, 3),
            ('color', np.float32, 4),
            ('value', np.float32)
        ])

        # Initialize tracking of modified voxels
        self.modified_voxels = set()
        self.num_active_voxels = 0

        # Initialize cube data
        self.init_cube()

        # Initialize shader program
        self.init_shaders()

        # Initialize OpenGL buffers
        self.init_buffers()

    def init_cube(self) -> None:
        """Initialize cube vertex data."""
        # Cube vertices (8 corners)
        self.cube_vertices = np.array([
            # Front face
            [-0.5, -0.5,  0.5],  # Bottom-left
            [ 0.5, -0.5,  0.5],  # Bottom-right
            [ 0.5,  0.5,  0.5],  # Top-right
            [-0.5,  0.5,  0.5],  # Top-left

            # Back face
            [-0.5, -0.5, -0.5],  # Bottom-left
            [ 0.5, -0.5, -0.5],  # Bottom-right
            [ 0.5,  0.5, -0.5],  # Top-right
            [-0.5,  0.5, -0.5]   # Top-left
        ], dtype=np.float32)

        # Cube normals (for each vertex)
        self.cube_normals = np.array([
            # Front face
            [ 0.0,  0.0,  1.0],
            [ 0.0,  0.0,  1.0],
            [ 0.0,  0.0,  1.0],
            [ 0.0,  0.0,  1.0],

            # Back face
            [ 0.0,  0.0, -1.0],
            [ 0.0,  0.0, -1.0],
            [ 0.0,  0.0, -1.0],
            [ 0.0,  0.0, -1.0]
        ], dtype=np.float32)

        # Cube texture coordinates
        self.cube_tex_coords = np.array([
            # Front face
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],

            # Back face
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0]
        ], dtype=np.float32)

        # Cube indices (6 faces, 2 triangles per face, 3 vertices per triangle)
        self.cube_indices = np.array([
            # Front face
            0, 1, 2,
            2, 3, 0,

            # Back face
            4, 5, 6,
            6, 7, 4,

            # Left face
            0, 3, 7,
            7, 4, 0,

            # Right face
            1, 5, 6,
            6, 2, 1,

            # Bottom face
            0, 4, 5,
            5, 1, 0,

            # Top face
            3, 2, 6,
            6, 7, 3
        ], dtype=np.uint32)

    def init_shaders(self) -> None:
        """Initialize shader programs."""
        # Create voxel shader program
        self.shader_manager.create_program(
            'voxel',
            'voxel',
            'voxel'
        )

    def init_buffers(self) -> None:
        """Initialize OpenGL buffers."""
        # Create VAO
        glGenVertexArrays(1, ctypes.byref(self.vao))
        glBindVertexArray(self.vao)

        # Create VBO for cube vertices
        glGenBuffers(1, ctypes.byref(self.vbo_cube))
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_cube)

        # Interleave vertex data
        vertex_data = np.zeros(8, dtype=[
            ('position', np.float32, 3),
            ('normal', np.float32, 3),
            ('texcoord', np.float32, 2)
        ])
        vertex_data['position'] = self.cube_vertices
        vertex_data['normal'] = self.cube_normals
        vertex_data['texcoord'] = self.cube_tex_coords

        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)

        # Set up vertex attributes
        stride = vertex_data.itemsize

        # Position attribute
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))

        # Normal attribute
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride,
                             ctypes.c_void_p(vertex_data.dtype['position'].itemsize))

        # Texture coordinate attribute
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride,
                             ctypes.c_void_p(vertex_data.dtype['position'].itemsize +
                                           vertex_data.dtype['normal'].itemsize))

        # Create VBO for instance data
        glGenBuffers(1, ctypes.byref(self.vbo_instances))
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_instances)
        glBufferData(GL_ARRAY_BUFFER, self.instance_data.nbytes, None, GL_DYNAMIC_DRAW)

        # Set up instance attributes
        stride = self.instance_data.itemsize

        # Instance position attribute
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glVertexAttribDivisor(3, 1)  # Advance once per instance

        # Instance scale attribute
        glEnableVertexAttribArray(4)
        glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, stride,
                             ctypes.c_void_p(self.instance_data.dtype['position'].itemsize))
        glVertexAttribDivisor(4, 1)  # Advance once per instance

        # Instance color attribute
        glEnableVertexAttribArray(5)
        glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, stride,
                             ctypes.c_void_p(self.instance_data.dtype['position'].itemsize +
                                           self.instance_data.dtype['scale'].itemsize))
        glVertexAttribDivisor(5, 1)  # Advance once per instance

        # Instance value attribute
        glEnableVertexAttribArray(6)
        glVertexAttribPointer(6, 1, GL_FLOAT, GL_FALSE, stride,
                             ctypes.c_void_p(self.instance_data.dtype['position'].itemsize +
                                           self.instance_data.dtype['scale'].itemsize +
                                           self.instance_data.dtype['color'].itemsize))
        glVertexAttribDivisor(6, 1)  # Advance once per instance

        # Create EBO
        glGenBuffers(1, ctypes.byref(self.ebo))
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.cube_indices.nbytes, self.cube_indices, GL_STATIC_DRAW)

        # Unbind VAO
        glBindVertexArray(0)

    def set_voxel(self, index: int, position: np.ndarray, scale: np.ndarray,
                 color: np.ndarray, value: float) -> None:
        """Set voxel data for a specific index.

        Args:
            index: Voxel index
            position: Position (x, y, z)
            scale: Scale (x, y, z)
            color: Color (r, g, b, a)
            value: Value for color mapping
        """
        if index >= self.max_voxels:
            raise ValueError(f"Voxel index out of range: {index} >= {self.max_voxels}")

        self.instance_data[index]['position'] = position
        self.instance_data[index]['scale'] = scale
        self.instance_data[index]['color'] = color
        self.instance_data[index]['value'] = value

        self.modified_voxels.add(index)
        self.num_active_voxels = max(self.num_active_voxels, index + 1)

    def get_voxel_data(self, index: int) -> Optional[Dict[str, Any]]:
        """Get voxel data for a specific index.

        Args:
            index: Voxel index

        Returns:
            Dictionary containing voxel data, or None if the index is out of range
        """
        if index >= self.max_voxels or index >= self.num_active_voxels:
            return None

        return {
            'position': self.instance_data[index]['position'],
            'scale': self.instance_data[index]['scale'],
            'color': self.instance_data[index]['color'],
            'value': self.instance_data[index]['value']
        }

    def update_buffers(self) -> None:
        """Update instance buffers with modified voxel data."""
        if not self.modified_voxels:
            return

        # Bind instance VBO
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_instances)

        # Update only modified voxels
        for index in self.modified_voxels:
            offset = index * self.instance_data.itemsize
            size = self.instance_data.itemsize
            glBufferSubData(GL_ARRAY_BUFFER, offset, size,
                          self.instance_data[index].tobytes())

        print(f"Updated {len(self.modified_voxels)} voxels")

        # Clear modified voxels
        self.modified_voxels.clear()

        # Unbind VBO
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def render(self, model_matrix: np.ndarray, view_matrix: np.ndarray,
              projection_matrix: np.ndarray) -> None:
        """Render voxels.

        Args:
            model_matrix: Model matrix
            view_matrix: View matrix
            projection_matrix: Projection matrix
        """
        if self.num_active_voxels == 0:
            return

        # Update buffers
        self.update_buffers()

        # Use shader program
        self.shader_manager.use_program('voxel')

        # Set uniforms
        self.shader_manager.set_uniform_matrix4fv('voxel', 'model', model_matrix)
        self.shader_manager.set_uniform_matrix4fv('voxel', 'view', view_matrix)
        self.shader_manager.set_uniform_matrix4fv('voxel', 'projection', projection_matrix)

        # Set lighting uniforms
        self.shader_manager.set_uniform_1i('voxel', 'enableLighting', 1)
        self.shader_manager.set_uniform_3f('voxel', 'lightPos', 10.0, 10.0, 10.0)
        self.shader_manager.set_uniform_3f('voxel', 'viewPos', 0.0, 0.0, 5.0)

        # Set color map uniforms
        self.shader_manager.set_uniform_1i('voxel', 'useColorMap', 0)

        # Bind VAO
        glBindVertexArray(self.vao)

        # Draw instances
        glDrawElementsInstanced(GL_TRIANGLES, len(self.cube_indices), GL_UNSIGNED_INT, None, self.num_active_voxels)

        # Unbind VAO
        glBindVertexArray(0)

    def cleanup(self) -> None:
        """Clean up OpenGL resources."""
        # Delete buffers
        glDeleteBuffers(1, ctypes.byref(self.vbo_cube))
        glDeleteBuffers(1, ctypes.byref(self.vbo_instances))
        glDeleteBuffers(1, ctypes.byref(self.ebo))

        # Delete VAO
        glDeleteVertexArrays(1, ctypes.byref(self.vao))

        # Clean up shaders
        self.shader_manager.cleanup()
