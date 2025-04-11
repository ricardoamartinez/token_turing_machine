"""
Graph Renderer for visualizing computational graphs.

This module provides functionality for rendering computational graphs using OpenGL.
"""

import os
import numpy as np
import pyglet
from pyglet.gl import *
import ctypes
from typing import Dict, List, Tuple, Optional, Any, Union

from .shader_manager import ShaderManager


class GraphRenderer:
    """Renders computational graphs using OpenGL."""

    def __init__(self, max_nodes: int = 1000, max_edges: int = 2000):
        """Initialize the graph renderer.

        Args:
            max_nodes: Maximum number of nodes to render
            max_edges: Maximum number of edges to render
        """
        self.max_nodes = max_nodes
        self.max_edges = max_edges

        # Initialize OpenGL resources
        self._init_shaders()
        self._init_sphere_geometry()
        self._init_edge_geometry()
        self._init_buffers()

        # Initialize node and edge data
        self.node_data = np.zeros(self.max_nodes, dtype=[
            ('position', np.float32, 3),
            ('scale', np.float32, 3),
            ('color', np.float32, 4),
            ('type', np.float32)
        ])

        self.edge_data = np.zeros(self.max_edges, dtype=[
            ('start', np.float32, 3),
            ('end', np.float32, 3),
            ('color', np.float32, 4),
            ('type', np.float32)
        ])

        self.num_active_nodes = 0
        self.num_active_edges = 0

        # Camera and light settings
        self.light_position = np.array([5.0, 5.0, 5.0], dtype=np.float32)
        self.view_position = np.array([0.0, 0.0, 5.0], dtype=np.float32)

        # Animation time
        self.time = 0.0

    def _init_shaders(self) -> None:
        """Initialize shaders for graph rendering."""
        self.shader_manager = ShaderManager()

        # Create node shader program
        self.node_program = self.shader_manager.create_program(
            "graph_node",
            "graph_vertex",
            "graph_fragment"
        )

        # Create edge shader program
        self.edge_program = self.shader_manager.create_program(
            "graph_edge",
            "graph_edge_vertex",
            "graph_edge_fragment"
        )

    def _init_sphere_geometry(self) -> None:
        """Initialize sphere geometry for nodes."""
        try:
            # Create a sphere mesh for nodes
            radius = 0.5
            sectors = 16
            stacks = 16

            # Generate sphere vertices, normals, and UVs
            vertices = []
            normals = []
            uvs = []

            sector_step = 2 * np.pi / sectors
            stack_step = np.pi / stacks

            for i in range(stacks + 1):
                stack_angle = np.pi / 2 - i * stack_step
                xy = radius * np.cos(stack_angle)
                z = radius * np.sin(stack_angle)

                for j in range(sectors + 1):
                    sector_angle = j * sector_step

                    # Vertex position
                    x = xy * np.cos(sector_angle)
                    y = xy * np.sin(sector_angle)
                    vertices.extend([x, y, z])

                    # Vertex normal
                    nx = x / radius
                    ny = y / radius
                    nz = z / radius
                    normals.extend([nx, ny, nz])

                    # Vertex UV
                    u = j / sectors
                    v = i / stacks
                    uvs.extend([u, v])

            # Generate sphere indices
            indices = []
            for i in range(stacks):
                k1 = i * (sectors + 1)
                k2 = k1 + sectors + 1

                for j in range(sectors):
                    if i != 0:
                        indices.extend([k1, k2, k1 + 1])

                    if i != (stacks - 1):
                        indices.extend([k1 + 1, k2, k2 + 1])

                    k1 += 1
                    k2 += 1

            # Convert to numpy arrays
            self.sphere_vertices = np.array(vertices, dtype=np.float32)
            self.sphere_normals = np.array(normals, dtype=np.float32)
            self.sphere_uvs = np.array(uvs, dtype=np.float32)
            self.sphere_indices = np.array(indices, dtype=np.uint32)

            # Store counts
            self.sphere_vertex_count = len(vertices) // 3
            self.sphere_index_count = len(indices)
        except Exception as e:
            print(f"Error generating sphere geometry: {e}")
            # Create simple fallback geometry (triangle)
            self.sphere_vertices = np.array([-0.5, -0.5, 0.0, 0.5, -0.5, 0.0, 0.0, 0.5, 0.0], dtype=np.float32)
            self.sphere_normals = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32)
            self.sphere_uvs = np.array([0.0, 0.0, 1.0, 0.0, 0.5, 1.0], dtype=np.float32)
            self.sphere_indices = np.array([0, 1, 2], dtype=np.uint32)

            # Store counts for fallback
            self.sphere_vertex_count = 3
            self.sphere_index_count = 3

    def _init_edge_geometry(self) -> None:
        """Initialize geometry for edges."""
        # Create a simple line segment for edges
        # We'll use a quad strip to give the line some thickness
        vertices = [
            # Quad strip vertices
            -0.5, -0.5, 0.0,  # Bottom left at start
            0.5, -0.5, 0.0,   # Bottom right at start
            -0.5, 0.5, 0.0,   # Top left at start
            0.5, 0.5, 0.0,    # Top right at start
            -0.5, -0.5, 1.0,  # Bottom left at end
            0.5, -0.5, 1.0,   # Bottom right at end
            -0.5, 0.5, 1.0,   # Top left at end
            0.5, 0.5, 1.0     # Top right at end
        ]

        indices = [
            0, 1, 2, 3,  # Start face
            2, 3, 6, 7,  # Top face
            0, 1, 4, 5,  # Bottom face
            0, 2, 4, 6,  # Left face
            1, 3, 5, 7,  # Right face
            4, 5, 6, 7   # End face
        ]

        # Convert to numpy arrays
        self.edge_vertices = np.array(vertices, dtype=np.float32)
        self.edge_indices = np.array(indices, dtype=np.uint32)

        # Store counts
        self.edge_vertex_count = len(vertices) // 3
        self.edge_index_count = len(indices)

    def _init_buffers(self) -> None:
        """Initialize OpenGL buffers."""
        # Create VAO for nodes
        self.node_vao = GLuint()
        glGenVertexArrays(1, ctypes.byref(self.node_vao))
        glBindVertexArray(self.node_vao)

        # Create VBO for sphere vertices
        self.vbo_sphere_vertices = GLuint()
        glGenBuffers(1, ctypes.byref(self.vbo_sphere_vertices))
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_sphere_vertices)
        glBufferData(GL_ARRAY_BUFFER, self.sphere_vertices.nbytes,
                    self.sphere_vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        # Create VBO for sphere normals
        self.vbo_sphere_normals = GLuint()
        glGenBuffers(1, ctypes.byref(self.vbo_sphere_normals))
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_sphere_normals)
        glBufferData(GL_ARRAY_BUFFER, self.sphere_normals.nbytes,
                    self.sphere_normals, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)

        # Create VBO for sphere UVs
        self.vbo_sphere_uvs = GLuint()
        glGenBuffers(1, ctypes.byref(self.vbo_sphere_uvs))
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_sphere_uvs)
        glBufferData(GL_ARRAY_BUFFER, self.sphere_uvs.nbytes,
                    self.sphere_uvs, GL_STATIC_DRAW)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(2)

        # Create VBO for node instances
        self.vbo_node_instances = GLuint()
        glGenBuffers(1, ctypes.byref(self.vbo_node_instances))
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_node_instances)
        glBufferData(GL_ARRAY_BUFFER, self.node_data.nbytes, None, GL_DYNAMIC_DRAW)

        # Set up instance attributes
        stride = self.node_data.strides[0]

        # Position
        offset = self.node_data.dtype.fields['position'][1]
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset))
        glEnableVertexAttribArray(3)
        glVertexAttribDivisor(3, 1)

        # Scale
        offset = self.node_data.dtype.fields['scale'][1]
        glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset))
        glEnableVertexAttribArray(4)
        glVertexAttribDivisor(4, 1)

        # Color
        offset = self.node_data.dtype.fields['color'][1]
        glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset))
        glEnableVertexAttribArray(5)
        glVertexAttribDivisor(5, 1)

        # Type
        offset = self.node_data.dtype.fields['type'][1]
        glVertexAttribPointer(6, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset))
        glEnableVertexAttribArray(6)
        glVertexAttribDivisor(6, 1)

        # Create EBO for sphere indices
        self.ebo_sphere = GLuint()
        glGenBuffers(1, ctypes.byref(self.ebo_sphere))
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo_sphere)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.sphere_indices.nbytes,
                    self.sphere_indices, GL_STATIC_DRAW)

        # Create VAO for edges
        self.edge_vao = GLuint()
        glGenVertexArrays(1, ctypes.byref(self.edge_vao))
        glBindVertexArray(self.edge_vao)

        # Create VBO for edge vertices
        self.vbo_edge_vertices = GLuint()
        glGenBuffers(1, ctypes.byref(self.vbo_edge_vertices))
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_edge_vertices)
        glBufferData(GL_ARRAY_BUFFER, self.edge_vertices.nbytes,
                    self.edge_vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        # Create VBO for edge instances
        self.vbo_edge_instances = GLuint()
        glGenBuffers(1, ctypes.byref(self.vbo_edge_instances))
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_edge_instances)
        glBufferData(GL_ARRAY_BUFFER, self.edge_data.nbytes, None, GL_DYNAMIC_DRAW)

        # Set up instance attributes
        stride = self.edge_data.strides[0]

        # Start position
        offset = self.edge_data.dtype.fields['start'][1]
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset))
        glEnableVertexAttribArray(1)
        glVertexAttribDivisor(1, 1)

        # End position
        offset = self.edge_data.dtype.fields['end'][1]
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset))
        glEnableVertexAttribArray(2)
        glVertexAttribDivisor(2, 1)

        # Color
        offset = self.edge_data.dtype.fields['color'][1]
        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset))
        glEnableVertexAttribArray(3)
        glVertexAttribDivisor(3, 1)

        # Type
        offset = self.edge_data.dtype.fields['type'][1]
        glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset))
        glEnableVertexAttribArray(4)
        glVertexAttribDivisor(4, 1)

        # Create EBO for edge indices
        self.ebo_edge = GLuint()
        glGenBuffers(1, ctypes.byref(self.ebo_edge))
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo_edge)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.edge_indices.nbytes,
                    self.edge_indices, GL_STATIC_DRAW)

        # Unbind VAO
        glBindVertexArray(0)

    def update_graph(self, graph_data: Dict[str, Any]) -> None:
        """Update the graph data.

        Args:
            graph_data: Dictionary containing graph data
        """
        # Extract nodes and edges
        nodes = graph_data.get('nodes', [])
        edges = graph_data.get('edges', [])

        # Update node data
        self.num_active_nodes = min(len(nodes), self.max_nodes)
        for i in range(self.num_active_nodes):
            node = nodes[i]
            self.node_data[i]['position'] = node['position']
            self.node_data[i]['scale'] = (node['size'], node['size'], node['size'])
            self.node_data[i]['color'] = node['color']
            self.node_data[i]['type'] = hash(node['type']) % 100 / 100.0  # Convert type to float for shader

        # Update edge data
        self.num_active_edges = min(len(edges), self.max_edges)
        for i in range(self.num_active_edges):
            edge = edges[i]
            # Find source and target nodes
            source_id = edge['source']
            target_id = edge['target']
            source_pos = None
            target_pos = None

            for node in nodes:
                if node['id'] == source_id:
                    source_pos = node['position']
                if node['id'] == target_id:
                    target_pos = node['position']

                if source_pos is not None and target_pos is not None:
                    break

            if source_pos is not None and target_pos is not None:
                self.edge_data[i]['start'] = source_pos
                self.edge_data[i]['end'] = target_pos
                self.edge_data[i]['color'] = edge['color']
                # Convert edge type to float for shader
                edge_type_value = 0.0
                if edge['type'] == 'data_flow':
                    edge_type_value = 1.0
                elif edge['type'] == 'parent_child':
                    edge_type_value = 2.0
                self.edge_data[i]['type'] = edge_type_value

        # Update GPU buffers
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_node_instances)
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.node_data.nbytes, self.node_data)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_edge_instances)
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.edge_data.nbytes, self.edge_data)

        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def render(self, model_matrix: np.ndarray, view_matrix: np.ndarray,
              projection_matrix: np.ndarray, delta_time: float) -> None:
        """Render the graph.

        Args:
            model_matrix: Model matrix
            view_matrix: View matrix
            projection_matrix: Projection matrix
            delta_time: Time since last frame
        """
        # Update animation time
        self.time += delta_time

        # Enable depth testing and blending
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Render edges first (behind nodes)
        self._render_edges(model_matrix, view_matrix, projection_matrix)

        # Render nodes
        self._render_nodes(model_matrix, view_matrix, projection_matrix)

        # Disable depth testing and blending
        glDisable(GL_BLEND)
        glDisable(GL_DEPTH_TEST)

    def _render_nodes(self, model_matrix: np.ndarray, view_matrix: np.ndarray,
                     projection_matrix: np.ndarray) -> None:
        """Render the graph nodes.

        Args:
            model_matrix: Model matrix
            view_matrix: View matrix
            projection_matrix: Projection matrix
        """
        if self.num_active_nodes == 0:
            return

        # Use node shader program
        self.shader_manager.use_program("graph_node")

        # Set uniforms
        self.shader_manager.set_uniform_matrix4fv("graph_node", "model", model_matrix)
        self.shader_manager.set_uniform_matrix4fv("graph_node", "view", view_matrix)
        self.shader_manager.set_uniform_matrix4fv("graph_node", "projection", projection_matrix)
        self.shader_manager.set_uniform_3f("graph_node", "lightPosition",
                                         self.light_position[0],
                                         self.light_position[1],
                                         self.light_position[2])
        self.shader_manager.set_uniform_3f("graph_node", "viewPosition",
                                         self.view_position[0],
                                         self.view_position[1],
                                         self.view_position[2])
        self.shader_manager.set_uniform_1f("graph_node", "time", self.time)

        # Bind node VAO
        glBindVertexArray(self.node_vao)

        # Draw nodes
        glDrawElementsInstanced(GL_TRIANGLES, self.sphere_index_count,
                               GL_UNSIGNED_INT, None, self.num_active_nodes)

        # Unbind VAO
        glBindVertexArray(0)

    def _render_edges(self, model_matrix: np.ndarray, view_matrix: np.ndarray,
                     projection_matrix: np.ndarray) -> None:
        """Render the graph edges.

        Args:
            model_matrix: Model matrix
            view_matrix: View matrix
            projection_matrix: Projection matrix
        """
        if self.num_active_edges == 0:
            return

        # Use edge shader program
        self.shader_manager.use_program("graph_edge")

        # Set uniforms
        self.shader_manager.set_uniform_matrix4fv("graph_edge", "model", model_matrix)
        self.shader_manager.set_uniform_matrix4fv("graph_edge", "view", view_matrix)
        self.shader_manager.set_uniform_matrix4fv("graph_edge", "projection", projection_matrix)
        self.shader_manager.set_uniform_1f("graph_edge", "time", self.time)

        # Bind edge VAO
        glBindVertexArray(self.edge_vao)

        # Draw edges
        glDrawElementsInstanced(GL_TRIANGLE_STRIP, self.edge_index_count,
                               GL_UNSIGNED_INT, None, self.num_active_edges)

        # Unbind VAO
        glBindVertexArray(0)

    def cleanup(self) -> None:
        """Clean up OpenGL resources."""
        # Delete buffers
        glDeleteBuffers(1, ctypes.byref(self.vbo_sphere_vertices))
        glDeleteBuffers(1, ctypes.byref(self.vbo_sphere_normals))
        glDeleteBuffers(1, ctypes.byref(self.vbo_sphere_uvs))
        glDeleteBuffers(1, ctypes.byref(self.vbo_node_instances))
        glDeleteBuffers(1, ctypes.byref(self.ebo_sphere))

        glDeleteBuffers(1, ctypes.byref(self.vbo_edge_vertices))
        glDeleteBuffers(1, ctypes.byref(self.vbo_edge_instances))
        glDeleteBuffers(1, ctypes.byref(self.ebo_edge))

        # Delete VAOs
        glDeleteVertexArrays(1, ctypes.byref(self.node_vao))
        glDeleteVertexArrays(1, ctypes.byref(self.edge_vao))
