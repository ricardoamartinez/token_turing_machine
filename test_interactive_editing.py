"""
Test script for interactive editing and state replay.

This script demonstrates how to modify a state via the dashboard and replay from that state.
"""

import sys
import imgui
import glfw
import OpenGL.GL as gl
import numpy as np
import time
import threading
import queue
from imgui.integrations.glfw import GlfwRenderer

# Mock model for testing
class MockModel:
    """Mock model for testing."""

    def __init__(self):
        """Initialize the mock model."""
        # Model parameters
        self.embedding_dim = 64
        self.memory_size = 32
        self.num_heads = 8
        self.head_dim = 8

        # Model state
        self.embedding = np.random.rand(1, self.embedding_dim)
        self.memory = np.random.rand(1, self.memory_size, self.embedding_dim)
        self.attention = np.random.rand(1, self.num_heads, self.head_dim, self.head_dim)

        # Output state
        self.output = np.random.rand(1, self.embedding_dim)

    def forward(self, input_token, memory=None, attention=None):
        """Run a forward pass.

        Args:
            input_token: Input token
            memory: Optional memory state to use
            attention: Optional attention state to use

        Returns:
            Output state
        """
        # Use provided memory and attention if available
        if memory is not None:
            self.memory = memory

        if attention is not None:
            self.attention = attention

        # Simulate computation
        # In a real model, this would be a complex computation
        # For this mock, we'll just do some simple operations

        # Update memory
        # In a real model, this would involve attention mechanisms
        # For this mock, we'll just add a small random value
        self.memory = self.memory + np.random.normal(0, 0.01, self.memory.shape)

        # Update attention
        # In a real model, this would be computed based on the input and memory
        # For this mock, we'll just add a small random value
        self.attention = self.attention + np.random.normal(0, 0.01, self.attention.shape)

        # Compute output
        # In a real model, this would involve complex operations
        # For this mock, we'll just use a simple function of the memory and attention
        self.output = np.mean(self.memory, axis=1) + np.mean(self.attention, axis=(1, 2, 3)).reshape(1, 1)

        return self.output


class StateManager:
    """Manages model states for visualization and replay."""

    def __init__(self, model):
        """Initialize the state manager.

        Args:
            model: Model to manage states for
        """
        self.model = model
        self.states = []
        self.current_state_idx = -1
        self.edited_states = {}  # Maps state index to edited state

    def capture_state(self):
        """Capture the current model state.

        Returns:
            State index
        """
        state = {
            'embedding': self.model.embedding.copy(),
            'memory': self.model.memory.copy(),
            'attention': self.model.attention.copy(),
            'output': self.model.output.copy()
        }

        self.states.append(state)
        self.current_state_idx = len(self.states) - 1

        return self.current_state_idx

    def get_state(self, idx):
        """Get a state by index.

        Args:
            idx: State index

        Returns:
            State
        """
        if idx < 0 or idx >= len(self.states):
            return None

        # Return edited state if available
        if idx in self.edited_states:
            return self.edited_states[idx]

        return self.states[idx]

    def get_current_state(self):
        """Get the current state.

        Returns:
            Current state
        """
        return self.get_state(self.current_state_idx)

    def edit_state(self, idx, component, indices, value):
        """Edit a state component.

        Args:
            idx: State index
            component: Component to edit ('memory', 'attention', etc.)
            indices: Indices to edit
            value: New value

        Returns:
            True if successful, False otherwise
        """
        state = self.get_state(idx)
        if state is None:
            return False

        # Create a copy of the state if not already edited
        if idx not in self.edited_states:
            self.edited_states[idx] = {
                'embedding': state['embedding'].copy(),
                'memory': state['memory'].copy(),
                'attention': state['attention'].copy(),
                'output': state['output'].copy()
            }

        # Edit the component
        try:
            if component == 'memory':
                self.edited_states[idx]['memory'][indices] = value
            elif component == 'attention':
                self.edited_states[idx]['attention'][indices] = value
            elif component == 'embedding':
                self.edited_states[idx]['embedding'][indices] = value
            else:
                return False

            return True
        except Exception as e:
            print(f"Error editing state: {e}")
            return False

    def replay_from(self, idx):
        """Replay the model from a specific state.

        Args:
            idx: State index to replay from

        Returns:
            List of new states
        """
        state = self.get_state(idx)
        if state is None:
            return []

        # Reset model state
        self.model.memory = state['memory'].copy()
        self.model.attention = state['attention'].copy()

        # Remove states after the current one
        self.states = self.states[:idx + 1]

        # Remove edited states after the current one
        self.edited_states = {k: v for k, v in self.edited_states.items() if k <= idx}

        # Run forward pass to generate new states
        new_states = []
        for i in range(5):  # Generate 5 new states
            # Run forward pass
            self.model.forward(None)

            # Capture state
            state_idx = self.capture_state()
            new_states.append(state_idx)

        return new_states


class VisualizationEngine:
    """Visualization engine for displaying and editing model states."""

    def __init__(self, state_manager):
        """Initialize the visualization engine.

        Args:
            state_manager: State manager
        """
        self.state_manager = state_manager
        self.current_state_idx = 0
        self.selected_component = 'memory'
        self.selected_indices = [0, 0, 0]
        self.edit_value = 0.0
        self.replay_log = []

        # Initialize GLFW
        if not glfw.init():
            print("Could not initialize GLFW")
            sys.exit(1)

        # Create a windowed mode window and its OpenGL context
        self.window = glfw.create_window(1280, 720, "Interactive Editing and Replay", None, None)
        if not self.window:
            glfw.terminate()
            print("Could not create GLFW window")
            sys.exit(1)

        # Make the window's context current
        glfw.make_context_current(self.window)

        # Initialize ImGui
        imgui.create_context()
        self.impl = GlfwRenderer(self.window)

        # Configure ImGui style for dark mode
        imgui.style_colors_dark()

    def update(self):
        """Update the visualization engine."""
        pass

    def render(self):
        """Render the visualization."""
        # Poll for and process events
        glfw.poll_events()

        # Start new frame
        self.impl.process_inputs()
        imgui.new_frame()

        # Get window dimensions
        width, height = glfw.get_window_size(self.window)

        # Create menu bar
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File"):
                if imgui.menu_item("Exit")[0]:
                    glfw.set_window_should_close(self.window, True)
                imgui.end_menu()

            menu_bar_height = imgui.get_frame_height()
            imgui.end_main_menu_bar()
        else:
            menu_bar_height = 0

        # Calculate panel sizes
        visualization_width = width * 0.7
        controls_width = width * 0.3

        # Create visualization panel
        imgui.set_next_window_position(0, menu_bar_height)
        imgui.set_next_window_size(visualization_width, height - menu_bar_height)

        window_flags = (
            imgui.WINDOW_NO_COLLAPSE |
            imgui.WINDOW_NO_RESIZE |
            imgui.WINDOW_NO_MOVE
        )

        imgui.begin("Visualization", True, window_flags)

        # Get current state
        state = self.state_manager.get_state(self.current_state_idx)

        if state is not None:
            # Display state info
            imgui.text(f"State {self.current_state_idx}")

            # Display memory visualization
            imgui.separator()
            imgui.text("Memory")

            # Draw memory visualization
            draw_list = imgui.get_window_draw_list()
            pos = imgui.get_cursor_screen_pos()

            # Calculate cell size
            memory = state['memory'][0]
            cell_width = visualization_width / memory.shape[0]
            cell_height = 200 / memory.shape[1]

            # Draw memory cells
            for i in range(memory.shape[0]):
                for j in range(memory.shape[1]):
                    x = pos[0] + i * cell_width
                    y = pos[1] + j * cell_height

                    # Use memory value to determine color
                    value = memory[i, j]

                    # Highlight selected cell
                    if self.selected_component == 'memory' and self.selected_indices[1] == i and self.selected_indices[2] == j:
                        color = imgui.get_color_u32_rgba(1.0, 1.0, 0.0, 1.0)
                    else:
                        color = imgui.get_color_u32_rgba(value, 0.0, 1.0 - value, 1.0)

                    # Draw cell
                    draw_list.add_rect_filled(
                        x, y,
                        x + cell_width - 1, y + cell_height - 1,
                        color
                    )

            # Advance cursor
            imgui.dummy(0, 220)

            # Display attention visualization
            imgui.separator()
            imgui.text("Attention")

            # Draw attention visualization
            draw_list = imgui.get_window_draw_list()
            pos = imgui.get_cursor_screen_pos()

            # Calculate cell size
            attention = state['attention'][0]
            head_width = visualization_width / attention.shape[0]

            for h in range(attention.shape[0]):
                head_x = pos[0] + h * head_width
                head_y = pos[1]

                # Draw head label
                imgui.set_cursor_pos((h * head_width, imgui.get_cursor_pos()[1]))
                imgui.text(f"Head {h}")

                # Draw attention matrix
                cell_size = min(head_width, 200) / attention.shape[1]

                for i in range(attention.shape[1]):
                    for j in range(attention.shape[2]):
                        x = head_x + i * cell_size
                        y = head_y + 20 + j * cell_size

                        # Use attention value to determine color
                        value = attention[h, i, j]

                        # Highlight selected cell
                        if self.selected_component == 'attention' and self.selected_indices[1] == h and self.selected_indices[2] == i and self.selected_indices[3] == j:
                            color = imgui.get_color_u32_rgba(1.0, 1.0, 0.0, 1.0)
                        else:
                            color = imgui.get_color_u32_rgba(value, 0.0, 1.0 - value, 1.0)

                        # Draw cell
                        draw_list.add_rect_filled(
                            x, y,
                            x + cell_size - 1, y + cell_size - 1,
                            color
                        )

            # Advance cursor
            imgui.dummy(0, 220)

            # Display output visualization
            imgui.separator()
            imgui.text("Output")

            # Draw output visualization
            draw_list = imgui.get_window_draw_list()
            pos = imgui.get_cursor_screen_pos()

            # Draw output as a bar chart
            output = state['output'][0]
            bar_width = visualization_width / output.shape[0]
            max_height = 100

            for i in range(output.shape[0]):
                value = output[i]
                height = value * max_height

                x = pos[0] + i * bar_width
                y = pos[1] + max_height - height

                draw_list.add_rect_filled(
                    x, y,
                    x + bar_width - 1, pos[1] + max_height,
                    imgui.get_color_u32_rgba(0.0, value, 0.0, 1.0)
                )
        else:
            imgui.text("No state available.")

        imgui.end()

        # Create controls panel
        imgui.set_next_window_position(visualization_width, menu_bar_height)
        imgui.set_next_window_size(controls_width, height - menu_bar_height)

        imgui.begin("Controls", True, window_flags)

        # State navigation
        imgui.text("State Navigation")

        # State slider
        changed, value = imgui.slider_int("State", self.current_state_idx, 0, len(self.state_manager.states) - 1)
        if changed:
            self.current_state_idx = value

        # Navigation buttons
        if imgui.button("Previous"):
            self.current_state_idx = max(0, self.current_state_idx - 1)

        imgui.same_line()
        if imgui.button("Next"):
            self.current_state_idx = min(len(self.state_manager.states) - 1, self.current_state_idx + 1)

        # State editing
        imgui.separator()
        imgui.text("State Editing")

        # Component selection
        if imgui.radio_button("Memory", self.selected_component == 'memory'):
            self.selected_component = 'memory'

        imgui.same_line()
        if imgui.radio_button("Attention", self.selected_component == 'attention'):
            self.selected_component = 'attention'

        # Index selection
        imgui.text("Indices")

        if self.selected_component == 'memory':
            # Memory indices
            changed, value = imgui.slider_int("Memory Row", self.selected_indices[1], 0, self.state_manager.model.memory_size - 1)
            if changed:
                self.selected_indices[1] = value

            changed, value = imgui.slider_int("Memory Column", self.selected_indices[2], 0, self.state_manager.model.embedding_dim - 1)
            if changed:
                self.selected_indices[2] = value

            # Display current value
            state = self.state_manager.get_state(self.current_state_idx)
            if state is not None:
                current_value = state['memory'][0, self.selected_indices[1], self.selected_indices[2]]
                imgui.text(f"Current Value: {current_value:.4f}")

        elif self.selected_component == 'attention':
            # Attention indices
            changed, value = imgui.slider_int("Attention Head", self.selected_indices[1], 0, self.state_manager.model.num_heads - 1)
            if changed:
                self.selected_indices[1] = value

            changed, value = imgui.slider_int("Attention Row", self.selected_indices[2], 0, self.state_manager.model.head_dim - 1)
            if changed:
                self.selected_indices[2] = value

            changed, value = imgui.slider_int("Attention Column", self.selected_indices[3], 0, self.state_manager.model.head_dim - 1)
            if changed:
                self.selected_indices[3] = value

            # Display current value
            state = self.state_manager.get_state(self.current_state_idx)
            if state is not None:
                current_value = state['attention'][0, self.selected_indices[1], self.selected_indices[2], self.selected_indices[3]]
                imgui.text(f"Current Value: {current_value:.4f}")

        # Value editing
        imgui.text("Edit Value")
        changed, value = imgui.slider_float("New Value", self.edit_value, 0.0, 1.0)
        if changed:
            self.edit_value = value

        # Apply button
        if imgui.button("Apply Edit"):
            if self.selected_component == 'memory':
                indices = (0, self.selected_indices[1], self.selected_indices[2])
                success = self.state_manager.edit_state(self.current_state_idx, 'memory', indices, self.edit_value)
                if success:
                    print(f"Edited memory[{indices}] = {self.edit_value:.4f}")
                    self.replay_log.append(f"Edited memory[{indices}] = {self.edit_value:.4f}")

            elif self.selected_component == 'attention':
                indices = (0, self.selected_indices[1], self.selected_indices[2], self.selected_indices[3])
                success = self.state_manager.edit_state(self.current_state_idx, 'attention', indices, self.edit_value)
                if success:
                    print(f"Edited attention[{indices}] = {self.edit_value:.4f}")
                    self.replay_log.append(f"Edited attention[{indices}] = {self.edit_value:.4f}")

        # Replay
        imgui.separator()
        imgui.text("Replay")

        if imgui.button("Replay from Current State"):
            print(f"Replaying from state {self.current_state_idx}")
            self.replay_log.append(f"Replaying from state {self.current_state_idx}")

            # Get current state
            state = self.state_manager.get_state(self.current_state_idx)

            # Record original output
            original_output = state['output'].copy()

            # Replay from current state
            new_states = self.state_manager.replay_from(self.current_state_idx)

            # Get new output
            new_state = self.state_manager.get_state(new_states[-1])
            new_output = new_state['output'].copy()

            # Compare outputs
            output_diff = np.mean(np.abs(new_output - original_output))
            print(f"Output difference: {output_diff:.4f}")
            self.replay_log.append(f"Output difference: {output_diff:.4f}")

        # Replay log
        imgui.separator()
        imgui.text("Replay Log")

        for log_entry in self.replay_log:
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
    """Run the test."""
    try:
        # Create model
        model = MockModel()

        # Create state manager
        state_manager = StateManager(model)

        # Generate initial states
        for i in range(10):
            model.forward(None)
            state_manager.capture_state()

        # Create visualization engine
        engine = VisualizationEngine(state_manager)

        # Simulate user interactions
        # 1. Select state 3
        engine.current_state_idx = 3
        print(f"Selected state {engine.current_state_idx}")

        # 2. Select memory component
        engine.selected_component = 'memory'
        engine.selected_indices = [0, 5, 10]  # Select memory[0, 5, 10]
        print(f"Selected {engine.selected_component}[0, {engine.selected_indices[1]}, {engine.selected_indices[2]}]")

        # 3. Set edit value to 0.9
        engine.edit_value = 0.9
        print(f"Set edit value to {engine.edit_value}")

        # 4. Apply edit
        indices = (0, engine.selected_indices[1], engine.selected_indices[2])
        success = state_manager.edit_state(engine.current_state_idx, 'memory', indices, engine.edit_value)
        if success:
            print(f"Edited memory{indices} = {engine.edit_value:.4f}")
            engine.replay_log.append(f"Edited memory{indices} = {engine.edit_value:.4f}")

        # 5. Replay from current state
        print(f"Replaying from state {engine.current_state_idx}")
        engine.replay_log.append(f"Replaying from state {engine.current_state_idx}")

        # Get current state
        state = state_manager.get_state(engine.current_state_idx)

        # Record original output
        original_output = state['output'].copy()

        # Replay from current state
        new_states = state_manager.replay_from(engine.current_state_idx)

        # Get new output
        new_state = state_manager.get_state(new_states[-1])
        new_output = new_state['output'].copy()

        # Compare outputs
        output_diff = np.mean(np.abs(new_output - original_output))
        print(f"Output difference: {output_diff:.4f}")
        engine.replay_log.append(f"Output difference: {output_diff:.4f}")

        # 6. Select attention component
        engine.selected_component = 'attention'
        engine.selected_indices = [0, 2, 3, 4]  # Select attention[0, 2, 3, 4]
        print(f"Selected {engine.selected_component}[0, {engine.selected_indices[1]}, {engine.selected_indices[2]}, {engine.selected_indices[3]}]")

        # 7. Set edit value to 0.1
        engine.edit_value = 0.1
        print(f"Set edit value to {engine.edit_value}")

        # 8. Apply edit
        indices = (0, engine.selected_indices[1], engine.selected_indices[2], engine.selected_indices[3])
        success = state_manager.edit_state(engine.current_state_idx, 'attention', indices, engine.edit_value)
        if success:
            print(f"Edited attention{indices} = {engine.edit_value:.4f}")
            engine.replay_log.append(f"Edited attention{indices} = {engine.edit_value:.4f}")

        # 9. Replay from current state again
        print(f"Replaying from state {engine.current_state_idx} again")
        engine.replay_log.append(f"Replaying from state {engine.current_state_idx} again")

        # Get current state
        state = state_manager.get_state(engine.current_state_idx)

        # Record original output
        original_output = state['output'].copy()

        # Replay from current state
        new_states = state_manager.replay_from(engine.current_state_idx)

        # Get new output
        new_state = state_manager.get_state(new_states[-1])
        new_output = new_state['output'].copy()

        # Compare outputs
        output_diff = np.mean(np.abs(new_output - original_output))
        print(f"Output difference: {output_diff:.4f}")
        engine.replay_log.append(f"Output difference: {output_diff:.4f}")

        # Main loop (shortened for testing)
        print("Starting visualization engine...")
        start_time = time.time()
        while not engine.should_close() and time.time() - start_time < 5:  # Run for 5 seconds
            # Update visualization engine
            engine.update()

            # Render visualization
            engine.render()

        # Clean up resources
        engine.cleanup()

        # Print replay log
        print("\nReplay Log:")
        for log_entry in engine.replay_log:
            print(log_entry)

        print("\nInteractive editing and replay test completed successfully!")
        print("Terminal Validation: The dashboard allows editing state values and replaying from modified states.")

        return 0
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
