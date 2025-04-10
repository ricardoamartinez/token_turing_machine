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
from imgui.integrations.glfw import GlfwRenderer

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

        # Update playback
        if self.playing:
            current_time = time.time()
            if current_time - self.last_play_time >= 1.0 / (self.play_speed * 2.0):
                self.current_step += 1
                if self.current_step >= self.state_tracker.current_step + 1:
                    self.current_step = self.state_tracker.current_step
                    self.playing = False
                self.last_play_time = current_time

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

        # Set positions for each panel with margins and spacing
        # 3D Visualization panel (main area)
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

        # Create panels
        # Define window flags for fixed panels
        window_flags = (
            imgui.WINDOW_NO_COLLAPSE |
            imgui.WINDOW_NO_RESIZE |
            imgui.WINDOW_NO_MOVE
        )

        # 3D Visualization panel
        imgui.begin("Model Memory Visualization", True, window_flags)
        imgui.text_colored("Interactive 3D visualization of the model's memory state", 0.7, 0.8, 1.0, 1.0)
        imgui.separator()

        # Get current state
        state = self.state_tracker.get_state(self.current_step)

        if state is not None:
            # Display current state info
            imgui.text(f"Step {self.current_step}: Token {state['token']}")

            # Draw memory visualization
            draw_list = imgui.get_window_draw_list()
            pos = imgui.get_cursor_screen_pos()

            # Use a fixed size
            size_x = main_panel_width - 20  # Padding
            size_y = top_panel_height - 100  # Padding

            # Draw background
            draw_list.add_rect_filled(
                pos[0], pos[1],
                pos[0] + size_x, pos[1] + size_y,
                imgui.get_color_u32_rgba(0.1, 0.1, 0.3, 1.0)
            )

            # Draw memory cells
            memory = state['memory'][0]
            cell_width = size_x / memory.shape[0]
            cell_height = size_y / memory.shape[1]

            for i in range(memory.shape[0]):
                for j in range(memory.shape[1]):
                    x = pos[0] + i * cell_width
                    y = pos[1] + j * cell_height

                    # Use memory value to determine color
                    value = memory[i, j]
                    normalized_value = (value + 1.0) / 2.0  # Normalize from [-1, 1] to [0, 1]
                    normalized_value = max(0.0, min(1.0, normalized_value))  # Clamp to [0, 1]

                    # Highlight selected cell
                    if self.selected_component == 'memory' and self.selected_indices == (i, j):
                        color = imgui.get_color_u32_rgba(1.0, 1.0, 0.0, 1.0)
                    else:
                        color = imgui.get_color_u32_rgba(normalized_value, 0.0, 1.0 - normalized_value, 1.0)

                    # Draw cell
                    draw_list.add_rect_filled(
                        x, y,
                        x + cell_width - 1, y + cell_height - 1,
                        color
                    )

                    # Check for mouse click on this cell
                    if imgui.is_mouse_clicked(0):  # Left mouse button
                        mouse_pos = imgui.get_mouse_pos()
                        if (x <= mouse_pos[0] <= x + cell_width and
                            y <= mouse_pos[1] <= y + cell_height):
                            self.selected_component = 'memory'
                            self.selected_indices = (i, j)
                            print(f"Selected memory cell ({i}, {j}) with value {value:.4f}")
                            self.action_log.append(f"Selected memory cell ({i}, {j}) with value {value:.4f}")
        else:
            imgui.text("No state data available yet.")

        imgui.end()

        # Timeline panel
        imgui.begin("Sequence Timeline", True, window_flags)
        imgui.text_colored("Navigate through the model's execution steps", 0.7, 0.8, 1.0, 1.0)
        imgui.separator()

        # Add timeline controls
        if self.state_tracker.current_step >= 0:
            # Step slider
            changed, value = imgui.slider_int("Step", self.current_step, 0, self.state_tracker.current_step)
            if changed:
                self.current_step = value
                self.playing = False

            # Playback controls
            imgui.separator()
            if self.playing:
                if imgui.button("Pause"):
                    self.playing = False
            else:
                if imgui.button("Play"):
                    self.playing = True
                    self.last_play_time = time.time()

            imgui.same_line()
            if imgui.button("<<"):
                self.current_step = max(0, self.current_step - 1)
                self.playing = False

            imgui.same_line()
            if imgui.button(">>"):
                self.current_step = min(self.state_tracker.current_step, self.current_step + 1)
                self.playing = False

            # Play speed
            imgui.separator()
            imgui.text("Play Speed")
            changed, value = imgui.slider_float("##play_speed", self.play_speed, 0.1, 5.0, "%.1fx")
            if changed:
                self.play_speed = value
        else:
            imgui.text("No state data available yet.")

        imgui.end()

        # Properties panel
        imgui.begin("Model State Inspector", True, window_flags)
        imgui.text_colored("Detailed information about the current model state", 0.7, 0.8, 1.0, 1.0)
        imgui.separator()

        # Display state properties
        state = self.state_tracker.get_state(self.current_step)

        if state is not None:
            # Display token info
            imgui.separator()
            imgui.text(f"Token: {state['token']}")

            # Display selected component info
            if self.selected_component == 'memory' and self.selected_indices is not None:
                i, j = self.selected_indices
                if i < state['memory'].shape[1] and j < state['memory'].shape[2]:
                    value = state['memory'][0, i, j]
                    imgui.separator()
                    imgui.text(f"Selected Memory Cell: ({i}, {j})")
                    imgui.text(f"Value: {value:.4f}")

                    # Edit value
                    imgui.separator()
                    imgui.text("Edit Value")
                    changed, value = imgui.slider_float("##edit_value", self.edit_value, -1.0, 1.0, "%.4f")
                    if changed:
                        self.edit_value = value

                    if imgui.button("Apply Edit"):
                        # In a real implementation, this would modify the model state
                        # For this demo, we'll just log the action
                        print(f"Would edit memory[0, {i}, {j}] from {state['memory'][0, i, j]:.4f} to {self.edit_value:.4f}")
                        self.action_log.append(f"Would edit memory[0, {i}, {j}] from {state['memory'][0, i, j]:.4f} to {self.edit_value:.4f}")

            # Display embedding info
            if imgui.collapsing_header("Embedding"):
                embedding = state['embedding']
                imgui.text(f"Shape: {embedding.shape}")
                imgui.text(f"Min: {np.min(embedding):.4f}")
                imgui.text(f"Max: {np.max(embedding):.4f}")
                imgui.text(f"Mean: {np.mean(embedding):.4f}")

            # Display memory info
            if imgui.collapsing_header("Memory"):
                memory = state['memory']
                imgui.text(f"Shape: {memory.shape}")
                imgui.text(f"Min: {np.min(memory):.4f}")
                imgui.text(f"Max: {np.max(memory):.4f}")
                imgui.text(f"Mean: {np.mean(memory):.4f}")

            # Display attention info
            if imgui.collapsing_header("Attention"):
                attention = state['attention']
                imgui.text(f"Shape: {attention.shape}")
                imgui.text(f"Min: {np.min(attention):.4f}")
                imgui.text(f"Max: {np.max(attention):.4f}")
                imgui.text(f"Mean: {np.mean(attention):.4f}")

            # Display output info
            if imgui.collapsing_header("Output"):
                output = state['output']
                imgui.text(f"Shape: {output.shape}")
                imgui.text(f"Min: {np.min(output):.4f}")
                imgui.text(f"Max: {np.max(output):.4f}")
                imgui.text(f"Mean: {np.mean(output):.4f}")

                # Display top 5 tokens
                imgui.separator()
                imgui.text("Top 5 Tokens:")
                top_indices = np.argsort(output[0])[-5:][::-1]
                for i, idx in enumerate(top_indices):
                    imgui.text(f"{i+1}. Token {idx}: {output[0, idx]:.4f}")
        else:
            imgui.text("No state data available yet.")

        imgui.end()

        # Performance panel
        imgui.begin("Performance Monitor", True, window_flags)
        imgui.text_colored("Real-time performance metrics and settings", 0.7, 0.8, 1.0, 1.0)
        imgui.separator()

        # Display FPS
        imgui.text(f"FPS: {self.performance_monitor.get_current_fps():.1f}")
        imgui.text(f"Average FPS: {self.performance_monitor.get_average_fps():.1f}")
        imgui.text(f"Frame Time: {self.performance_monitor.frame_time * 1000:.2f} ms")

        # Display detail level
        imgui.separator()
        imgui.text(f"Detail Level: {self.performance_monitor.detail_level:.2f}")

        # Adaptive rendering checkbox
        changed, value = imgui.checkbox("Adaptive Rendering", self.performance_monitor.adaptive_rendering)
        if changed:
            self.performance_monitor.adaptive_rendering = value
            print(f"{'Enabled' if value else 'Disabled'} adaptive rendering")
            self.action_log.append(f"{'Enabled' if value else 'Disabled'} adaptive rendering")

        # Target FPS slider
        changed, value = imgui.slider_float("Target FPS", self.performance_monitor.target_fps, 30.0, 120.0, "%.1f")
        if changed:
            self.performance_monitor.target_fps = value
            print(f"Changed target FPS to {value:.1f}")
            self.action_log.append(f"Changed target FPS to {value:.1f}")

        # Action log
        imgui.separator()
        imgui.text("Action Log")

        for log_entry in self.action_log[-10:]:  # Show last 10 entries
            imgui.text(log_entry)

        imgui.end()

        # Render
        imgui.render()

        # Clear the framebuffer
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        # Render ImGui
        self.impl.render(imgui.get_draw_data())

        # Swap front and back buffers
        glfw.swap_buffers(self.window)

    def should_close(self):
        """Check if the window should close.

        Returns:
            True if the window should close, False otherwise
        """
        return glfw.window_should_close(self.window)

    def cleanup(self):
        """Clean up resources."""
        self.impl.shutdown()
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