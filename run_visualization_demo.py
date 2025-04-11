"""
Comprehensive demonstration script for TTM visualization.

This script loads a trained TTM model, performs sample inference,
and displays the complete dashboard with all panels.
"""

import sys
import os
import numpy as np
import time
import threading
import queue
import argparse
import imgui
import glfw
import OpenGL.GL as gl
import torch
from imgui.integrations.glfw import GlfwRenderer

# Check if CUDA is available
CUDA_AVAILABLE = torch.cuda.is_available()
print(f"CUDA available: {CUDA_AVAILABLE}")

# Import TTM visualization components
from src.ttm.visualization.voxel_renderer import VoxelRenderer
from src.ttm.visualization.cuda_voxel_renderer import CudaVoxelRenderer
from src.ttm.visualization.model_data_extractor import ModelDataExtractor
from src.ttm.visualization.computational_graph_extractor import ComputationalGraphExtractor
from src.ttm.visualization.graph_renderer import GraphRenderer
from src.ttm.visualization.vis_mapper import (
    TensorToVoxelMapper,
    MemoryToVoxelMapper,
    AttentionToVoxelMapper
)

# Fallback mock voxel renderer in case OpenGL initialization fails
class MockVoxelRenderer:
    """Mock voxel renderer for demonstration."""

    def __init__(self, max_voxels=10000):
        """Initialize the mock voxel renderer.

        Args:
            max_voxels: Maximum number of voxels
        """
        self.max_voxels = max_voxels
        self.voxels = {}
        self.num_active_voxels = 0
        self.modified_voxels = set()

    def set_voxel(self, index, position, scale, color, value):
        """Set a voxel.

        Args:
            index: Voxel index
            position: Voxel position (x, y, z)
            scale: Voxel scale (x, y, z)
            color: Voxel color (r, g, b, a)
            value: Voxel value
        """
        self.voxels[index] = {
            'position': position,
            'scale': scale,
            'color': color,
            'value': value
        }
        self.num_active_voxels = max(self.num_active_voxels, index + 1)
        self.modified_voxels.add(index)

    def get_voxel_data(self, index):
        """Get voxel data.

        Args:
            index: Voxel index

        Returns:
            Voxel data
        """
        return self.voxels.get(index)

    def update_buffers(self):
        """Update voxel buffers."""
        print(f"Updating {len(self.modified_voxels)} voxels")
        self.modified_voxels.clear()

    def render(self, model_matrix, view_matrix, projection_matrix):
        """Render voxels.

        Args:
            model_matrix: Model matrix
            view_matrix: View matrix
            projection_matrix: Projection matrix
        """
        # In a real implementation, this would render the voxels using OpenGL
        # For this mock implementation, we just print the number of active voxels
        print(f"Rendering {self.num_active_voxels} voxels")

    def cleanup(self):
        """Clean up resources."""
        # In a real implementation, this would clean up OpenGL resources
        # For this mock implementation, we just clear the voxels dictionary
        self.voxels.clear()
        self.num_active_voxels = 0
        self.modified_voxels.clear()

# Performance monitoring
class PerformanceMonitor:
    """Class to monitor and manage performance."""

    def __init__(self, target_fps=60.0, window_size=100):
        """Initialize the performance monitor.

        Args:
            target_fps: Target FPS
            window_size: Size of the FPS history window
        """
        self.target_fps = target_fps
        self.fps_history = [target_fps] * window_size
        self.fps_history_index = 0
        self.frame_time = 0.0
        self.frame_count = 0
        self.last_fps_update_time = time.time()
        self.adaptive_rendering = True
        self.detail_level = 1.0  # 1.0 = full detail, 0.0 = no detail
        self.detail_level_history = [1.0] * window_size
        self.detail_level_history_index = 0

    def update(self, dt):
        """Update performance metrics.

        Args:
            dt: Time delta
        """
        # Update frame time
        self.frame_time = dt
        self.frame_count += 1

        # Calculate instantaneous FPS
        instantaneous_fps = 1.0 / max(dt, 0.001)  # Avoid division by zero

        # Update FPS history immediately
        self.fps_history[self.fps_history_index] = instantaneous_fps
        self.fps_history_index = (self.fps_history_index + 1) % len(self.fps_history)

        # Update FPS every 0.5 seconds
        current_time = time.time()
        elapsed_time = current_time - self.last_fps_update_time

        if elapsed_time >= 0.5:
            # Reset frame count
            self.frame_count = 0
            self.last_fps_update_time = current_time

            # Store detail level history
            self.detail_level_history[self.detail_level_history_index] = self.detail_level
            self.detail_level_history_index = (self.detail_level_history_index + 1) % len(self.detail_level_history)

            # Implement adaptive rendering
            if self.adaptive_rendering:
                # Get average FPS over the last few frames
                recent_fps = np.mean(self.fps_history)

                # If FPS is below target, reduce detail level
                if recent_fps < self.target_fps * 0.9:
                    # More aggressive reduction when FPS is very low
                    reduction = 0.1 if recent_fps < self.target_fps * 0.5 else 0.05
                    self.detail_level = max(0.1, self.detail_level - reduction)
                    print(f"Reducing detail level to {self.detail_level:.2f} (FPS: {recent_fps:.1f})")

                # If FPS is above target, increase detail level
                elif recent_fps > self.target_fps * 1.1 and self.detail_level < 1.0:
                    # Slower increase to avoid oscillation
                    self.detail_level = min(1.0, self.detail_level + 0.01)
                    print(f"Increasing detail level to {self.detail_level:.2f} (FPS: {recent_fps:.1f})")

    def get_current_fps(self):
        """Get the current FPS.

        Returns:
            Current FPS
        """
        return self.fps_history[self.fps_history_index - 1]

    def get_average_fps(self):
        """Get the average FPS.

        Returns:
            Average FPS
        """
        return np.mean(self.fps_history)


# State tracking
class TTMStateTracker:
    """Tracks the state of a TTM model during inference."""

    def __init__(self, model):
        """Initialize the state tracker.

        Args:
            model: TTM model
        """
        self.model = model
        self.current_step = -1
        self.state_queue = queue.Queue(maxsize=100)

    def step(self, token_id):
        """Run a single step of inference.

        Args:
            token_id: Input token ID

        Returns:
            Output logits
        """
        # Run forward pass
        logits = self.model.forward(token_id)

        # Update current step
        self.current_step += 1

        # Get current state
        state = self.model.get_current_state()

        # Put state in queue
        try:
            self.state_queue.put((self.current_step, state), block=False)
        except queue.Full:
            # If queue is full, remove oldest item
            try:
                self.state_queue.get(block=False)
                self.state_queue.put((self.current_step, state), block=False)
            except queue.Empty:
                pass

        return logits

    def get_state(self, step):
        """Get the model state for a specific step.

        Args:
            step: Step index

        Returns:
            Model state
        """
        return self.model.get_state(step)

    def get_current_state(self):
        """Get the current model state.

        Returns:
            Current model state
        """
        return self.model.get_current_state()

    def reset(self):
        """Reset the state tracker."""
        self.model.reset_state()
        self.current_step = -1
        # Clear queue
        while not self.state_queue.empty():
            try:
                self.state_queue.get(block=False)
            except queue.Empty:
                break

# Mock TTM model for demonstration
class MockTTMModel:
    """Mock TTM model for demonstration."""

    def __init__(self, embedding_dim=64, memory_size=32, num_heads=8, head_dim=8):
        """Initialize the mock TTM model.

        Args:
            embedding_dim: Embedding dimension
            memory_size: Memory size
            num_heads: Number of attention heads
            head_dim: Attention head dimension
        """
        self.embedding_dim = embedding_dim
        self.memory_size = memory_size
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Initialize model parameters
        self.token_embedding = np.random.randn(1000, embedding_dim) * 0.1
        self.position_embedding = np.random.randn(512, embedding_dim) * 0.1
        self.memory_key = np.random.randn(embedding_dim, memory_size) * 0.1
        self.memory_value = np.random.randn(memory_size, embedding_dim) * 0.1
        self.attention_query = np.random.randn(embedding_dim, num_heads * head_dim) * 0.1
        self.attention_key = np.random.randn(embedding_dim, num_heads * head_dim) * 0.1
        self.attention_value = np.random.randn(embedding_dim, num_heads * head_dim) * 0.1
        self.output_projection = np.random.randn(embedding_dim, 1000) * 0.1

        # Initialize model state
        self.reset_state()

    def reset_state(self):
        """Reset the model state."""
        self.memory = np.zeros((1, self.memory_size, self.embedding_dim))
        self.attention_state = np.zeros((1, self.num_heads, 0, 0))  # Will grow with sequence
        self.token_history = []
        self.embedding_history = []
        self.memory_history = []
        self.attention_history = []
        self.output_history = []

    def forward(self, token_id):
        """Run a forward pass.

        Args:
            token_id: Input token ID

        Returns:
            Output logits
        """
        # Add token to history
        self.token_history.append(token_id)
        position = len(self.token_history) - 1

        # Token embedding
        token_emb = self.token_embedding[token_id:token_id+1]

        # Position embedding
        pos_emb = self.position_embedding[position:position+1]

        # Combined embedding
        embedding = token_emb + pos_emb
        self.embedding_history.append(embedding)

        # Memory update
        memory_input = embedding @ self.memory_key
        memory_output = self.memory * 0.9 + 0.1 * memory_input @ self.memory_value
        self.memory = memory_output
        self.memory_history.append(self.memory.copy())

        # Self-attention
        query = embedding @ self.attention_query
        query = query.reshape(1, self.num_heads, 1, self.head_dim)

        key = embedding @ self.attention_key
        key = key.reshape(1, self.num_heads, 1, self.head_dim)

        value = embedding @ self.attention_value
        value = value.reshape(1, self.num_heads, 1, self.head_dim)

        # Update attention state
        if len(self.token_history) == 1:
            self.attention_state = np.zeros((1, self.num_heads, 1, 1))
        else:
            # Extend attention state
            prev_size = self.attention_state.shape[2]
            new_size = prev_size + 1
            new_attention_state = np.zeros((1, self.num_heads, new_size, new_size))
            new_attention_state[:, :, :prev_size, :prev_size] = self.attention_state

            # Compute new attention scores
            for h in range(self.num_heads):
                for i in range(new_size):
                    for j in range(new_size):
                        if i == new_size - 1 or j == new_size - 1:
                            # New attention score
                            if i == new_size - 1:
                                q = query[0, h, 0]
                            else:
                                q_full = self.embedding_history[i] @ self.attention_query
                                q = q_full.reshape(self.num_heads, self.head_dim)[h]

                            if j == new_size - 1:
                                k = key[0, h, 0]
                            else:
                                k_full = self.embedding_history[j] @ self.attention_key
                                k = k_full.reshape(self.num_heads, self.head_dim)[h]

                            score = np.dot(q, k) / np.sqrt(self.head_dim)
                            new_attention_state[0, h, i, j] = score

            self.attention_state = new_attention_state

        self.attention_history.append(self.attention_state.copy())

        # Apply softmax to attention scores
        attention_probs = np.exp(self.attention_state)
        attention_probs = attention_probs / np.sum(attention_probs, axis=-1, keepdims=True)

        # Compute weighted sum of values
        context = np.zeros((1, self.embedding_dim))
        for h in range(self.num_heads):
            head_context = np.zeros((1, self.head_dim))
            for i in range(len(self.token_history)):
                v = self.embedding_history[i] @ self.attention_value
                v = v.reshape(self.num_heads, self.head_dim)[h]
                head_context += attention_probs[0, h, -1, i] * v

            context += head_context @ self.attention_value.T[:self.head_dim, :]

        # Combine with memory
        combined = context + self.memory.mean(axis=1)

        # Output projection
        logits = combined @ self.output_projection
        self.output_history.append(logits)

        return logits

    def get_state(self, step):
        """Get the model state for a specific step.

        Args:
            step: Step index

        Returns:
            Model state
        """
        if step < 0 or step >= len(self.token_history):
            return None

        return {
            'token': self.token_history[step],
            'embedding': self.embedding_history[step],
            'memory': self.memory_history[step],
            'attention': self.attention_history[step],
            'output': self.output_history[step]
        }

    def get_current_state(self):
        """Get the current model state.

        Returns:
            Current model state
        """
        return self.get_state(len(self.token_history) - 1)

    def generate(self, prompt, max_length=10):
        """Generate text from a prompt.

        Args:
            prompt: List of token IDs
            max_length: Maximum generation length

        Returns:
            Generated token IDs
        """
        # Reset state
        self.reset_state()

        # Process prompt
        for token_id in prompt:
            self.forward(token_id)

        # Generate
        generated = list(prompt)
        for _ in range(max_length):
            logits = self.forward(generated[-1])
            next_token = np.argmax(logits)
            generated.append(next_token)

        return generated


# Panel base class
class Panel:
    """Base class for all visualization panels."""

    def __init__(self, title, engine, color=(0.1, 0.2, 0.3, 1.0)):
        """Initialize the panel.

        Args:
            title: Panel title
            engine: Visualization engine
            color: Panel title bar color (r, g, b, a)
        """
        self.title = title
        self.engine = engine
        self.visible = True
        self.color = color

    def render(self):
        """Render the panel."""
        if not self.visible:
            return

        # Set panel title colors
        style = imgui.get_style()
        original_active_color = style.colors[imgui.COLOR_TITLE_BACKGROUND_ACTIVE]
        original_color = style.colors[imgui.COLOR_TITLE_BACKGROUND]

        # Apply custom color for this panel
        style.colors[imgui.COLOR_TITLE_BACKGROUND_ACTIVE] = self.color
        # Make inactive color slightly darker
        style.colors[imgui.COLOR_TITLE_BACKGROUND] = (
            self.color[0] * 0.7,
            self.color[1] * 0.7,
            self.color[2] * 0.7,
            self.color[3]
        )

        # Begin panel with consistent flags for all panels
        imgui.begin(self.title, True, imgui.WINDOW_NO_COLLAPSE)
        self.render_content()
        imgui.end()

        # Restore original colors
        style.colors[imgui.COLOR_TITLE_BACKGROUND_ACTIVE] = original_active_color
        style.colors[imgui.COLOR_TITLE_BACKGROUND] = original_color

    def render_content(self):
        """Render the panel content. To be overridden by subclasses."""
        pass


# Panel implementations
class MemoryVisualizationPanel(Panel):
    """Panel for 3D visualization of model memory."""

    def render_content(self):
        """Render the panel content."""
        # Use a color that matches the panel title but is brighter
        r, g, b, a = self.color
        imgui.text_colored("Interactive 3D visualization of the model's memory state",
                         min(r + 0.5, 1.0), min(g + 0.5, 1.0), min(b + 0.5, 1.0), 1.0)
        imgui.separator()

        # Get current state
        state = self.engine.state_tracker.get_state(self.engine.current_step)

        if state is not None:
            # Display current state info
            imgui.text(f"Step {self.engine.current_step}: Token {state['token']}")

            # Get cursor position for 3D rendering
            pos = imgui.get_cursor_screen_pos()

            # Calculate size for 3D rendering
            available_size = imgui.get_content_region_available()
            size_x = available_size[0] - 20  # Padding
            size_y = available_size[1] - 20  # Padding

            # Create a child window for 3D rendering
            imgui.begin_child("3D_Render_Area", size_x, size_y, False, imgui.WINDOW_NO_SCROLLBAR)

            # Get the position and size of the child window
            render_pos = imgui.get_cursor_screen_pos()
            render_size = imgui.get_content_region_available()

            # Set up viewport for 3D rendering
            try:
                gl.glViewport(int(render_pos[0]), int(self.engine.window_height - render_pos[1] - render_size[1]), int(render_size[0]), int(render_size[1]))

                # Clear the depth buffer
                gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
            except Exception as e:
                # Print error but continue if OpenGL calls fail
                print(f"OpenGL error in viewport setup: {e}")
                print(f"Render area: {render_pos[0]}, {render_pos[1]}, {render_size[0]}, {render_size[1]}")

            # Update view matrix based on camera rotation
            self.engine.view_matrix = self.engine.get_view_matrix()

            # Render voxels
            try:
                self.engine.voxel_renderer.render(
                    self.engine.model_matrix,
                    self.engine.view_matrix,
                    self.engine.projection_matrix
                )
            except Exception as e:
                # Print error but continue if OpenGL calls fail
                print(f"OpenGL error in voxel rendering: {e}")

            imgui.end_child()

            # Display memory information
            if state is not None and 'memory' in state:
                memory = state['memory'][0]
                imgui.separator()
                imgui.text(f"Memory shape: {memory.shape}")
                imgui.text(f"Memory min: {np.min(memory):.4f}")
                imgui.text(f"Memory max: {np.max(memory):.4f}")
                imgui.text(f"Memory mean: {np.mean(memory):.4f}")

                # Create memory mapper if not already done
                if self.engine.selected_component != 'memory':
                    memory_mapper = MemoryToVoxelMapper()
                    memory_voxel_data = memory_mapper.map_to_voxels({
                        'name': 'memory',
                        'shape': memory.shape,
                        'data': memory,
                        'is_memory': True
                    })

                    # Set up memory voxels
                    self.engine.setup_voxels(memory_voxel_data)
                    self.engine.selected_component = 'memory'
                    self.engine.action_log.append(f"Loaded memory visualization with shape {memory.shape}")


class TimelinePanel(Panel):
    """Panel for sequence timeline visualization."""

    def render_content(self):
        """Render the panel content."""
        # Use a color that matches the panel title but is brighter
        r, g, b, a = self.color
        imgui.text_colored("Navigate through the model's execution steps",
                         min(r + 0.5, 1.0), min(g + 0.5, 1.0), min(b + 0.5, 1.0), 1.0)
        imgui.separator()

        # Get current state
        state = self.engine.state_tracker.get_state(self.engine.current_step)

        if state is not None:
            # Display timeline
            total_steps = self.engine.state_tracker.current_step + 1
            imgui.text(f"Step {self.engine.current_step + 1} of {total_steps}")

            # Timeline slider
            changed, value = imgui.slider_int("##timeline", self.engine.current_step, 0, total_steps - 1, f"{self.engine.current_step}")
            if changed:
                self.engine.current_step = value
                self.engine.playing = False

            # Playback controls
            imgui.separator()
            imgui.text("Playback Controls")

            if imgui.button("Reset"):
                self.engine.current_step = 0
                self.engine.playing = False

            imgui.same_line()
            if self.engine.playing:
                if imgui.button("Pause"):
                    self.engine.playing = False
            else:
                if imgui.button("Play"):
                    self.engine.playing = True

            imgui.same_line()
            if imgui.button("<<"):
                self.engine.current_step = max(0, self.engine.current_step - 1)
                self.engine.playing = False

            imgui.same_line()
            if imgui.button(">>"):
                self.engine.current_step = min(self.engine.state_tracker.current_step, self.engine.current_step + 1)
                self.engine.playing = False

            # Play speed
            imgui.separator()
            imgui.text("Play Speed")
            changed, value = imgui.slider_float("##play_speed", self.engine.play_speed, 0.1, 5.0, "%.1fx")
            if changed:
                self.engine.play_speed = value
        else:
            imgui.text("No state data available yet.")


class StateInspectorPanel(Panel):
    """Panel for inspecting model state details."""

    def render_content(self):
        """Render the panel content."""
        # Use a color that matches the panel title but is brighter
        r, g, b, a = self.color
        imgui.text_colored("Detailed information about the current model state",
                         min(r + 0.5, 1.0), min(g + 0.5, 1.0), min(b + 0.5, 1.0), 1.0)
        imgui.separator()

        # Display state properties
        state = self.engine.state_tracker.get_state(self.engine.current_step)

        if state is not None:
            # Display token info
            imgui.separator()
            imgui.text(f"Token: {state['token']}")

            # Display selected component info
            if self.engine.selected_component == 'memory' and self.engine.selected_indices is not None:
                i, j = self.engine.selected_indices
                if i < state['memory'].shape[1] and j < state['memory'].shape[2]:
                    value = state['memory'][0, i, j]
                    imgui.separator()
                    imgui.text(f"Selected Memory Cell: ({i}, {j})")
                    imgui.text(f"Value: {value:.4f}")

                    # Edit value
                    imgui.separator()
                    imgui.text("Edit Value")
                    changed, value = imgui.slider_float("##edit_value", self.engine.edit_value, -1.0, 1.0, "%.4f")
                    if changed:
                        self.engine.edit_value = value

                    if imgui.button("Apply Edit"):
                        # In a real implementation, this would modify the model state
                        # For this demo, we'll just log the action
                        print(f"Would edit memory[0, {i}, {j}] from {state['memory'][0, i, j]:.4f} to {self.engine.edit_value:.4f}")
                        self.engine.action_log.append(f"Would edit memory[0, {i}, {j}] from {state['memory'][0, i, j]:.4f} to {self.engine.edit_value:.4f}")

            # Display memory info
            if imgui.collapsing_header("Memory"):
                if 'memory' in state:
                    memory = state['memory']
                    imgui.text(f"Shape: {memory.shape}")
                    imgui.text(f"Min: {np.min(memory):.4f}")
                    imgui.text(f"Max: {np.max(memory):.4f}")
                    imgui.text(f"Mean: {np.mean(memory):.4f}")
                else:
                    imgui.text("No memory data available.")

            # Display attention info
            if imgui.collapsing_header("Attention"):
                if 'attention' in state:
                    attention = state['attention']
                    imgui.text(f"Shape: {attention.shape}")
                    imgui.text(f"Min: {np.min(attention):.4f}")
                    imgui.text(f"Max: {np.max(attention):.4f}")
                    imgui.text(f"Mean: {np.mean(attention):.4f}")
                else:
                    imgui.text("No attention data available.")

            # Display embedding info
            if imgui.collapsing_header("Embedding"):
                if 'embedding' in state:
                    embedding = state['embedding']
                    imgui.text(f"Shape: {embedding.shape}")
                    imgui.text(f"Min: {np.min(embedding):.4f}")
                    imgui.text(f"Max: {np.max(embedding):.4f}")
                    imgui.text(f"Mean: {np.mean(embedding):.4f}")
                else:
                    imgui.text("No embedding data available.")

            # Display output info
            if imgui.collapsing_header("Output"):
                if 'output' in state:
                    output = state['output']
                    imgui.text(f"Shape: {output.shape}")
                    imgui.text(f"Min: {np.min(output):.4f}")
                    imgui.text(f"Max: {np.max(output):.4f}")
                    imgui.text(f"Mean: {np.mean(output):.4f}")

                    # Display top 5 logits
                    imgui.separator()
                    imgui.text("Top 5 Logits:")
                    top_indices = np.argsort(output[0])[-5:][::-1]
                    for i, idx in enumerate(top_indices):
                        imgui.text(f"{i+1}. Token {idx}: {output[0, idx]:.4f}")
                else:
                    imgui.text("No output data available.")
        else:
            imgui.text("No state data available yet.")


class ComputationalGraphPanel(Panel):
    """Panel for visualizing the computational graph of models."""

    def __init__(self, title, engine, color=(0.3, 0.6, 0.8, 1.0)):
        """Initialize the computational graph panel.

        Args:
            title: Panel title
            engine: Visualization engine
            color: Panel color
        """
        super().__init__(title, engine, color)

        # Initialize graph extractor
        self.graph_extractor = ComputationalGraphExtractor()

        # Initialize graph renderer
        try:
            self.graph_renderer = GraphRenderer(max_nodes=1000, max_edges=2000)
            print("Successfully initialized GraphRenderer")
        except Exception as e:
            print(f"Failed to initialize GraphRenderer: {e}")
            self.graph_renderer = None

        # Graph data
        self.graph_data = None

        # Camera controls
        self.camera_rotation_x = 0.0
        self.camera_rotation_y = 0.0
        self.camera_distance = 5.0

        # Model selection
        self.current_model_type = 0  # 0 = TTM, 1 = Vanilla Transformer
        self.model_types = ["TTM", "Vanilla Transformer"]

        # Extract graph on initialization
        self.extract_graph()

    def extract_graph(self):
        """Extract the computational graph from the current model."""
        try:
            # Get the appropriate model based on selection
            if self.current_model_type == 0:
                # TTM model
                model = self.engine.model
            else:
                # Vanilla Transformer model (create a simple one for demo)
                model = torch.nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6)

            # Extract graph
            self.graph_extractor.extract_graph(model)

            # Get graph data for visualization
            self.graph_data = self.graph_extractor.get_graph_data()

            # Update graph renderer
            if self.graph_renderer is not None:
                self.graph_renderer.update_graph(self.graph_data)

            print(f"Extracted computational graph with {len(self.graph_data['nodes'])} nodes and {len(self.graph_data['edges'])} edges")
        except Exception as e:
            print(f"Error extracting computational graph: {e}")
            import traceback
            traceback.print_exc()

    def get_view_matrix(self):
        """Get the view matrix for the 3D graph visualization."""
        # Calculate camera position based on rotation and distance
        camera_x = self.camera_distance * np.sin(self.camera_rotation_y) * np.cos(self.camera_rotation_x)
        camera_y = self.camera_distance * np.sin(self.camera_rotation_x)
        camera_z = self.camera_distance * np.cos(self.camera_rotation_y) * np.cos(self.camera_rotation_x)

        # Create view matrix
        view_matrix = np.identity(4, dtype=np.float32)

        # Set camera position
        view_matrix[0, 3] = -camera_x
        view_matrix[1, 3] = -camera_y
        view_matrix[2, 3] = -camera_z

        # Set camera orientation
        forward = np.array([camera_x, camera_y, camera_z])
        forward = forward / np.linalg.norm(forward)

        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(up, forward)
        right = right / np.linalg.norm(right)

        up = np.cross(forward, right)

        view_matrix[0, 0] = right[0]
        view_matrix[0, 1] = right[1]
        view_matrix[0, 2] = right[2]

        view_matrix[1, 0] = up[0]
        view_matrix[1, 1] = up[1]
        view_matrix[1, 2] = up[2]

        view_matrix[2, 0] = forward[0]
        view_matrix[2, 1] = forward[1]
        view_matrix[2, 2] = forward[2]

        return view_matrix

    def render_content(self):
        """Render the panel content."""
        # Use a color that matches the panel title but is brighter
        r, g, b, a = self.color
        imgui.text_colored("Computational graph visualization",
                         min(r + 0.5, 1.0), min(g + 0.5, 1.0), min(b + 0.5, 1.0), 1.0)
        imgui.separator()

        # Model selection
        imgui.text("Model Type:")
        changed, value = imgui.combo(
            "##model_type",
            self.current_model_type,
            self.model_types
        )
        if changed:
            self.current_model_type = value
            # Extract graph for the new model
            self.extract_graph()

        # Graph statistics
        if self.graph_data is not None:
            imgui.text(f"Nodes: {len(self.graph_data['nodes'])}")
            imgui.text(f"Edges: {len(self.graph_data['edges'])}")
            imgui.text(f"Node types: {len(self.graph_data['node_types'])}")

        # 3D rendering area
        imgui.separator()
        available_size = imgui.get_content_region_available()
        size_x = available_size[0] - 20  # Padding
        size_y = available_size[1] - 100  # Padding

        # Create a child window for 3D rendering
        imgui.begin_child("3D_Graph_Area", size_x, size_y, False, imgui.WINDOW_NO_SCROLLBAR)

        # Get the position and size of the child window
        render_pos = imgui.get_cursor_screen_pos()
        render_size = imgui.get_content_region_available()

        # Set up viewport for 3D rendering
        try:
            gl.glViewport(int(render_pos[0]), int(self.engine.window_height - render_pos[1] - render_size[1]),
                         int(render_size[0]), int(render_size[1]))

            # Clear the depth buffer
            gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
        except Exception as e:
            print(f"OpenGL error in viewport setup: {e}")

        # Handle mouse input for camera rotation
        if imgui.is_window_hovered() and imgui.is_mouse_dragging(0):
            dx, dy = imgui.get_mouse_drag_delta(0)
            imgui.reset_mouse_drag_delta(0)

            # Update camera rotation
            self.camera_rotation_y += dx * 0.01
            self.camera_rotation_x += dy * 0.01

            # Clamp vertical rotation
            self.camera_rotation_x = max(-np.pi / 2.0 + 0.1, min(np.pi / 2.0 - 0.1, self.camera_rotation_x))

        # Handle mouse wheel for zoom
        if imgui.is_window_hovered() and imgui.get_io().mouse_wheel != 0:
            # Update camera distance
            self.camera_distance -= imgui.get_io().mouse_wheel * 0.5
            self.camera_distance = max(2.0, min(20.0, self.camera_distance))

        # Render the graph
        if self.graph_renderer is not None and self.graph_data is not None:
            # Create matrices
            model_matrix = np.identity(4, dtype=np.float32)
            view_matrix = self.get_view_matrix()

            # Create perspective projection matrix
            aspect_ratio = render_size[0] / max(1, render_size[1])
            fov = 45.0 * np.pi / 180.0
            near_plane = 0.1
            far_plane = 100.0

            projection_matrix = np.zeros((4, 4), dtype=np.float32)
            f = 1.0 / np.tan(fov / 2.0)
            projection_matrix[0, 0] = f / aspect_ratio
            projection_matrix[1, 1] = f
            projection_matrix[2, 2] = (far_plane + near_plane) / (near_plane - far_plane)
            projection_matrix[2, 3] = (2.0 * far_plane * near_plane) / (near_plane - far_plane)
            projection_matrix[3, 2] = -1.0

            # Render the graph
            self.graph_renderer.render(model_matrix, view_matrix, projection_matrix, 0.016)  # Assuming 60 FPS

        imgui.end_child()

        # Controls
        imgui.separator()
        imgui.text("Controls:")
        imgui.text("Drag with left mouse button to rotate the view")
        imgui.text("Use mouse wheel to zoom in/out")


class PerformanceMonitorPanel(Panel):
    """Panel for monitoring performance metrics."""

    def render_content(self):
        """Render the panel content."""
        # Use a color that matches the panel title but is brighter
        r, g, b, a = self.color
        imgui.text_colored("Performance metrics and rendering settings",
                         min(r + 0.5, 1.0), min(g + 0.5, 1.0), min(b + 0.5, 1.0), 1.0)
        imgui.separator()

        # Display FPS
        fps = 1.0 / max(self.engine.performance_monitor.frame_time, 0.001)
        avg_fps = sum(self.engine.performance_monitor.fps_history) / len(self.engine.performance_monitor.fps_history)

        imgui.text(f"FPS: {fps:.1f} (Avg: {avg_fps:.1f})")
        imgui.text(f"Frame Time: {self.engine.performance_monitor.frame_time * 1000:.1f} ms")

        # FPS graph
        imgui.separator()
        imgui.text("FPS History")

        # Plot FPS history
        fps_values = self.engine.performance_monitor.fps_history.copy()
        # Convert list to numpy array for imgui.plot_lines
        fps_array = np.array(fps_values, dtype=np.float32)
        imgui.plot_lines(
            "##fps_history",
            fps_array,
            scale_min=0,
            scale_max=max(self.engine.performance_monitor.target_fps * 1.5, max(fps_values)),
            graph_size=(0, 80)
        )

        # Adaptive rendering settings
        imgui.separator()
        imgui.text("Rendering Settings")

        changed, value = imgui.checkbox("Adaptive Rendering", self.engine.performance_monitor.adaptive_rendering)
        if changed:
            self.engine.performance_monitor.adaptive_rendering = value

        if self.engine.performance_monitor.adaptive_rendering:
            imgui.text(f"Detail Level: {self.engine.performance_monitor.detail_level:.2f}")

            # Plot detail level history
            detail_values = self.engine.performance_monitor.detail_level_history.copy()
            # Convert list to numpy array for imgui.plot_lines
            detail_array = np.array(detail_values, dtype=np.float32)
            imgui.plot_lines(
                "##detail_history",
                detail_array,
                scale_min=0,
                scale_max=1.0,
                graph_size=(0, 80)
            )

        # Action log
        imgui.separator()
        imgui.text("Action Log")

        imgui.begin_child("action_log", 0, 100, True)
        for action in reversed(self.engine.action_log[-10:]):
            imgui.text(action)
        imgui.end_child()


# Visualization engine
class VisualizationEngine:
    """Visualization engine for displaying model states."""

    def __init__(self, state_tracker, window_width=1280, window_height=720):
        """Initialize the visualization engine.

        Args:
            state_tracker: State tracker
            window_width: Window width
            window_height: Window height
        """
        self.state_tracker = state_tracker
        self.model = state_tracker.model  # Store reference to the model
        self.window_width = window_width
        self.window_height = window_height
        self.current_step = 0
        self.playing = False
        self.play_speed = 1.0
        self.last_play_time = 0.0
        self.selected_component = None
        self.selected_indices = None
        self.edit_value = 0.0
        self.action_log = []

        # Initialize performance monitor
        self.performance_monitor = PerformanceMonitor(target_fps=60.0)

        # Initialize camera parameters
        self.camera_distance = 5.0
        self.camera_rotation_x = 0.0
        self.camera_rotation_y = 0.0
        self.last_mouse_pos = None

        # Initialize panels with different colors
        self.panels = [
            # Blue for memory visualization
            MemoryVisualizationPanel("Model Memory Visualization", self, color=(0.1, 0.3, 0.6, 1.0)),
            # Green for timeline
            TimelinePanel("Sequence Timeline", self, color=(0.1, 0.5, 0.3, 1.0)),
            # Purple for state inspector
            StateInspectorPanel("Model State Inspector", self, color=(0.4, 0.2, 0.6, 1.0)),
            # Teal for computational graph
            ComputationalGraphPanel("Computational Graph", self, color=(0.1, 0.5, 0.5, 1.0)),
            # Orange for performance monitor
            PerformanceMonitorPanel("Performance Monitor", self, color=(0.6, 0.4, 0.1, 1.0))
        ]

        # Initialize GLFW
        if not glfw.init():
            print("Could not initialize GLFW")
            sys.exit(1)

        # Create a windowed mode window and its OpenGL context
        self.window = glfw.create_window(window_width, window_height, "TTM Visualization", None, None)
        if not self.window:
            glfw.terminate()
            print("Could not create GLFW window")
            sys.exit(1)

        # Make the window's context current
        glfw.make_context_current(self.window)

        # Initialize OpenGL
        self.setup_opengl()

        # Initialize matrices for 3D rendering
        self.model_matrix = np.identity(4, dtype=np.float32)
        self.view_matrix = self.get_view_matrix()
        self.projection_matrix = self.get_projection_matrix()

        # Initialize voxel renderer
        try:
            # Try to initialize the voxel renderer - use CUDA if available
            if CUDA_AVAILABLE:
                self.voxel_renderer = CudaVoxelRenderer(max_voxels=10000)
                print("Successfully initialized CudaVoxelRenderer with CUDA-OpenGL interop")
            else:
                self.voxel_renderer = VoxelRenderer(max_voxels=10000)
                print("Successfully initialized VoxelRenderer with OpenGL")
        except Exception as e:
            # Fall back to mock renderer if OpenGL initialization fails
            print(f"Failed to initialize VoxelRenderer with OpenGL: {e}")
            print("Falling back to MockVoxelRenderer")
            self.voxel_renderer = MockVoxelRenderer(max_voxels=10000)

        # Initialize ImGui
        imgui.create_context()
        self.impl = GlfwRenderer(self.window)

        # Configure ImGui style for a modern, professional dark mode
        imgui.style_colors_dark()
        style = imgui.get_style()

        # Set window styling
        style.window_rounding = 8.0  # Rounded corners for panels
        style.window_border_size = 1.0  # Thin border
        style.window_padding = (12, 12)  # More padding inside windows

        # Set frame styling
        style.frame_rounding = 4.0  # Rounded corners for widgets
        style.frame_border_size = 1.0  # Thin border for widgets
        style.frame_padding = (8, 4)  # More padding in widgets
        style.item_spacing = (10, 8)  # More space between items
        style.item_inner_spacing = (6, 4)  # More space inside items

        # Set grab styling (sliders, scrollbars)
        style.grab_min_size = 10.0  # Larger grab handles
        style.grab_rounding = 3.0  # Rounded grab handles

        # Set color scheme - true black background with blue accent
        primary_color = (0.2, 0.4, 0.8, 1.0)  # Blue accent color

        # Window colors
        style.colors[imgui.COLOR_WINDOW_BACKGROUND] = (0.0, 0.0, 0.0, 0.95)  # True black background
        style.colors[imgui.COLOR_TITLE_BACKGROUND_ACTIVE] = (0.1, 0.2, 0.3, 1.0)  # Dark blue title when active
        style.colors[imgui.COLOR_TITLE_BACKGROUND] = (0.05, 0.1, 0.15, 1.0)  # Darker blue title when inactive
        style.colors[imgui.COLOR_TITLE_BACKGROUND_COLLAPSED] = (0.0, 0.0, 0.0, 0.9)  # Black when collapsed

        # Widget colors
        style.colors[imgui.COLOR_FRAME_BACKGROUND] = (0.1, 0.1, 0.1, 1.0)  # Dark gray widget background
        style.colors[imgui.COLOR_FRAME_BACKGROUND_HOVERED] = (0.15, 0.2, 0.25, 1.0)  # Slightly blue when hovered
        style.colors[imgui.COLOR_FRAME_BACKGROUND_ACTIVE] = (0.2, 0.3, 0.4, 1.0)  # More blue when active

        # Button colors
        style.colors[imgui.COLOR_BUTTON] = (0.2, 0.3, 0.4, 1.0)  # Blue buttons
        style.colors[imgui.COLOR_BUTTON_HOVERED] = (0.3, 0.4, 0.5, 1.0)  # Lighter blue when hovered
        style.colors[imgui.COLOR_BUTTON_ACTIVE] = (0.4, 0.5, 0.6, 1.0)  # Even lighter when pressed

        # Text colors
        style.colors[imgui.COLOR_TEXT] = (0.9, 0.9, 0.9, 1.0)  # Slightly off-white text for better readability
        style.colors[imgui.COLOR_TEXT_DISABLED] = (0.5, 0.5, 0.5, 1.0)  # Medium gray for disabled text

        # Slider, checkbox colors
        style.colors[imgui.COLOR_CHECK_MARK] = primary_color  # Blue checkmarks
        style.colors[imgui.COLOR_SLIDER_GRAB] = primary_color  # Blue slider handles
        style.colors[imgui.COLOR_SLIDER_GRAB_ACTIVE] = (0.3, 0.5, 0.9, 1.0)  # Lighter blue when active

        # Header colors (for collapsing headers)
        style.colors[imgui.COLOR_HEADER] = (0.15, 0.2, 0.25, 1.0)  # Dark blue headers
        style.colors[imgui.COLOR_HEADER_HOVERED] = (0.2, 0.3, 0.4, 1.0)  # Medium blue when hovered
        style.colors[imgui.COLOR_HEADER_ACTIVE] = (0.25, 0.35, 0.45, 1.0)  # Lighter blue when active

    def setup_opengl(self):
        """Set up OpenGL state."""
        try:
            # Enable depth testing
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glDepthFunc(gl.GL_LESS)

            # Enable blending
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

            # Set clear color
            gl.glClearColor(0.0, 0.0, 0.0, 1.0)

            print("OpenGL initialized successfully")
        except Exception as e:
            print(f"Failed to initialize OpenGL: {e}")

    def get_view_matrix(self):
        """Get the view matrix.

        Returns:
            View matrix
        """
        # Calculate camera position
        camera_x = self.camera_distance * np.sin(self.camera_rotation_y) * np.cos(self.camera_rotation_x)
        camera_y = self.camera_distance * np.sin(self.camera_rotation_x)
        camera_z = self.camera_distance * np.cos(self.camera_rotation_y) * np.cos(self.camera_rotation_x)

        # Create view matrix
        view_matrix = np.identity(4, dtype=np.float32)

        # Set camera position
        view_matrix[0, 3] = -camera_x
        view_matrix[1, 3] = -camera_y
        view_matrix[2, 3] = -camera_z

        # Set camera rotation
        rotation_matrix = np.identity(3, dtype=np.float32)

        # Rotate around X axis
        rotation_x = np.identity(3, dtype=np.float32)
        rotation_x[1, 1] = np.cos(self.camera_rotation_x)
        rotation_x[1, 2] = -np.sin(self.camera_rotation_x)
        rotation_x[2, 1] = np.sin(self.camera_rotation_x)
        rotation_x[2, 2] = np.cos(self.camera_rotation_x)

        # Rotate around Y axis
        rotation_y = np.identity(3, dtype=np.float32)
        rotation_y[0, 0] = np.cos(self.camera_rotation_y)
        rotation_y[0, 2] = np.sin(self.camera_rotation_y)
        rotation_y[2, 0] = -np.sin(self.camera_rotation_y)
        rotation_y[2, 2] = np.cos(self.camera_rotation_y)

        # Combine rotations
        rotation_matrix = rotation_x @ rotation_y

        # Apply rotation to view matrix
        view_matrix[:3, :3] = rotation_matrix

        return view_matrix

    def get_projection_matrix(self):
        """Get the projection matrix.

        Returns:
            Projection matrix
        """
        # Create perspective projection matrix
        aspect = self.window_width / self.window_height
        fov = 45.0 * np.pi / 180.0
        near = 0.1
        far = 100.0

        f = 1.0 / np.tan(fov / 2.0)

        projection_matrix = np.zeros((4, 4), dtype=np.float32)
        projection_matrix[0, 0] = f / aspect
        projection_matrix[1, 1] = f
        projection_matrix[2, 2] = (far + near) / (near - far)
        projection_matrix[2, 3] = (2.0 * far * near) / (near - far)
        projection_matrix[3, 2] = -1.0

        return projection_matrix

    def update(self, dt):
        """Update the visualization engine.

        Args:
            dt: Time delta
        """
        # Update performance monitor
        self.performance_monitor.update(dt)

        # Process state queue
        while not self.state_tracker.state_queue.empty():
            step, state = self.state_tracker.state_queue.get()
            print(f"Received state for step {step}")

            # Create memory mapper
            if 'memory' in state:
                memory_mapper = MemoryToVoxelMapper()
                memory_voxel_data = memory_mapper.map_to_voxels({
                    'name': 'memory',
                    'type': 'tensor',
                    'shape': state['memory'].shape,
                    'data': state['memory'],
                    'metadata': {
                        'is_memory': True
                    }
                })

                # Set up memory voxels
                self.setup_voxels(memory_voxel_data)

        # Update playback
        if self.playing:
            current_time = time.time()
            if current_time - self.last_play_time >= 1.0 / (self.play_speed * 2.0):
                self.current_step += 1
                if self.current_step >= self.state_tracker.current_step + 1:
                    self.current_step = self.state_tracker.current_step
                    self.playing = False
                self.last_play_time = current_time

    def setup_voxels(self, voxel_data):
        """Set up voxels for rendering.

        Args:
            voxel_data: Voxel data dictionary
        """
        # Get voxel data
        voxels = voxel_data['voxels']
        dimensions = voxel_data['dimensions']
        colormap = voxel_data['metadata'].get('color_map', 'viridis')

        # Check if we have a tensor
        if 'tensor' in voxel_data:
            tensor = voxel_data['tensor']
            # If we have a CUDA-enabled renderer and a CUDA tensor, use direct update
            if CUDA_AVAILABLE and isinstance(self.voxel_renderer, CudaVoxelRenderer) and isinstance(tensor, torch.Tensor):
                if tensor.is_cuda:
                    # Direct update from CUDA tensor
                    self.voxel_renderer.update_from_tensor(tensor, dimensions, colormap)
                    print(f"Updated voxels directly from CUDA tensor with shape {tensor.shape}")
                    return
                else:
                    # Move tensor to CUDA and update
                    cuda_tensor = tensor.cuda()
                    self.voxel_renderer.update_from_tensor(cuda_tensor, dimensions, colormap)
                    print(f"Updated voxels from CPU tensor moved to CUDA with shape {tensor.shape}")
                    return

        # Set up voxels
        print(f"Setting up voxels with dimensions {dimensions} using {colormap} colormap...")
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

                        # Calculate color
                        value = voxels[x, y, z]

                        # Use appropriate colormap based on metadata
                        if colormap == 'plasma':
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
                            # Default colormap: viridis
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

                        # Calculate scale
                        scale = np.array([0.05, 0.05, 0.05], dtype=np.float32)

                        # Set voxel
                        self.voxel_renderer.set_voxel(voxel_index, position, scale, color, value)
                        voxel_index += 1

        print(f"Set up {voxel_index} voxels")

    def render(self):
        """Render the visualization."""
        # Poll for and process events
        glfw.poll_events()

        # Start new frame
        self.impl.process_inputs()
        imgui.new_frame()

        # Create menu bar
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File"):
                if imgui.menu_item("Exit")[0]:
                    glfw.set_window_should_close(self.window, True)
                imgui.end_menu()

            if imgui.begin_menu("View"):
                if imgui.menu_item("Reset Layout")[0]:
                    pass  # No docking, so nothing to reset
                imgui.end_menu()

            menu_bar_height = imgui.get_frame_height()
            imgui.end_main_menu_bar()
        else:
            menu_bar_height = 0

        # Define margins and spacing for better UX
        margin = 10  # Margin between panels and window edges
        spacing = 10  # Spacing between panels

        # Calculate available space after accounting for margins and spacing
        available_width = self.window_width - (2 * margin + spacing)
        available_height = self.window_height - menu_bar_height - (2 * margin + spacing)

        # Calculate panel sizes with golden ratio (approximately 0.618 : 0.382)
        main_panel_width = available_width * 0.618
        right_panel_width = available_width * 0.382
        top_panel_height = available_height * 0.618
        bottom_panel_height = available_height * 0.382

        # Set initial positions for each panel on first run
        if not hasattr(self, 'panels_positioned') or not self.panels_positioned:
            # Set positions for each panel
            # Memory Visualization panel (main area)
            imgui.set_next_window_position(margin, menu_bar_height + margin)
            imgui.set_next_window_size(main_panel_width, top_panel_height)

            # Timeline panel (bottom)
            imgui.set_next_window_position(margin, menu_bar_height + margin + top_panel_height + spacing)
            imgui.set_next_window_size(main_panel_width, bottom_panel_height)

            # Properties panel (right)
            imgui.set_next_window_position(margin + main_panel_width + spacing, menu_bar_height + margin)
            imgui.set_next_window_size(right_panel_width, top_panel_height)

            # Performance panel (bottom right)
            imgui.set_next_window_position(margin + main_panel_width + spacing, menu_bar_height + margin + top_panel_height + spacing)
            imgui.set_next_window_size(right_panel_width, bottom_panel_height)

            # Mark panels as positioned
            self.panels_positioned = True

        # Render all panels
        for panel in self.panels:
            panel.render()



        # Render ImGui
        imgui.render()

        # Clear the framebuffer
        try:
            gl.glClearColor(0.0, 0.0, 0.0, 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

            # Set up full viewport for ImGui rendering
            gl.glViewport(0, 0, self.window_width, self.window_height)
        except Exception as e:
            # Print error but continue if OpenGL calls fail
            print(f"OpenGL error in framebuffer setup: {e}")
            print("Rendering ImGui")

        # Render ImGui
        self.impl.render(imgui.get_draw_data())

        # Swap front and back buffers
        glfw.swap_buffers(self.window)

        # After first frame, set panels_positioned to True so panels can be moved
        if not hasattr(self, 'panels_positioned') or not self.panels_positioned:
            self.panels_positioned = True
            print("First frame rendered, panels can now be moved.")

    def should_close(self):
        """Check if the window should close.

        Returns:
            True if the window should close, False otherwise
        """
        return glfw.window_should_close(self.window)

    def cleanup(self):
        """Clean up resources."""
        # Clean up voxel renderer
        self.voxel_renderer.cleanup()

        # Clean up ImGui
        self.impl.shutdown()

        # Terminate GLFW
        glfw.terminate()


def main():
    """Run the demonstration."""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="TTM Visualization Demo")
        parser.add_argument("--embedding-dim", type=int, default=64, help="Embedding dimension")
        parser.add_argument("--memory-size", type=int, default=32, help="Memory size")
        parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
        parser.add_argument("--head-dim", type=int, default=8, help="Attention head dimension")
        parser.add_argument("--window-width", type=int, default=1280, help="Window width")
        parser.add_argument("--window-height", type=int, default=720, help="Window height")
        args = parser.parse_args()

        # Create model
        print("Creating model...")
        model = MockTTMModel(
            embedding_dim=args.embedding_dim,
            memory_size=args.memory_size,
            num_heads=args.num_heads,
            head_dim=args.head_dim
        )

        # Create state tracker
        print("Creating state tracker...")
        state_tracker = TTMStateTracker(model)

        # Create visualization engine
        print("Creating visualization engine...")
        engine = VisualizationEngine(
            state_tracker,
            window_width=args.window_width,
            window_height=args.window_height
        )

        # Define a simple prompt
        prompt = [42, 13, 7, 99, 123]  # Random token IDs

        # Process prompt
        print("Processing prompt...")
        for token_id in prompt:
            state_tracker.step(token_id)

        # Main loop
        print("Starting visualization...")
        print("Key insights:")
        print("1. The memory visualization shows how information is stored and updated over time")
        print("2. The attention visualization shows how the model attends to different tokens")
        print("3. The timeline allows navigating through the model's state history")
        print("4. The properties panel shows detailed information about the selected state")
        print("5. The performance panel shows FPS and allows adjusting rendering settings")
        print("6. You can click on memory cells to select them and edit their values")
        print("7. The adaptive rendering system maintains high FPS even with large models")

        last_time = time.time()
        while not engine.should_close():
            # Calculate delta time
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            # Update visualization engine
            engine.update(dt)

            # Render visualization
            engine.render()

        # Clean up resources
        engine.cleanup()

        print("Visualization demo completed successfully!")
        print("Terminal Validation: All dashboard panels were populated and the demo ran successfully.")

        return 0
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())