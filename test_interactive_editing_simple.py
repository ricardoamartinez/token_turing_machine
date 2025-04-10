"""
Simplified test script for interactive editing and state replay.

This script demonstrates how to modify a state via the dashboard and replay from that state.
"""

import sys
import imgui
import glfw
import OpenGL.GL as gl
import numpy as np
import time
from imgui.integrations.glfw import GlfwRenderer

# Mock model state
class ModelState:
    """Simple class to represent the model state."""
    
    def __init__(self):
        """Initialize the model state."""
        # Create a simple 5x5 memory matrix
        self.memory = np.random.rand(5, 5)
        
        # Create a simple 3x3 attention matrix
        self.attention = np.random.rand(3, 3)
        
        # Create a simple output vector
        self.output = np.zeros(5)
        self.update_output()
    
    def update_output(self):
        """Update the output based on memory and attention."""
        # Simple function to compute output from memory and attention
        self.output = np.mean(self.memory, axis=0) + np.mean(self.attention) * np.ones(5)
        return self.output
    
    def copy(self):
        """Create a copy of the state."""
        new_state = ModelState()
        new_state.memory = self.memory.copy()
        new_state.attention = self.attention.copy()
        new_state.output = self.output.copy()
        return new_state


def main():
    """Run the test."""
    try:
        # Initialize GLFW
        if not glfw.init():
            print("Could not initialize GLFW")
            return 1
        
        # Create a windowed mode window and its OpenGL context
        window = glfw.create_window(1280, 720, "Interactive Editing Demo", None, None)
        if not window:
            glfw.terminate()
            print("Could not create GLFW window")
            return 1
        
        # Make the window's context current
        glfw.make_context_current(window)
        
        # Initialize ImGui
        imgui.create_context()
        impl = GlfwRenderer(window)
        
        # Configure ImGui style for dark mode
        imgui.style_colors_dark()
        
        # Create initial state
        original_state = ModelState()
        current_state = original_state.copy()
        
        # Create a list to store states for replay
        states = [original_state.copy()]
        current_state_idx = 0
        
        # Create a log of actions
        action_log = []
        
        # Main loop
        print("Starting interactive editing demo...")
        print("Instructions:")
        print("1. Use the 'Memory' and 'Attention' sections to view and edit values")
        print("2. Click on cells to select them")
        print("3. Use the 'Edit Value' slider to change the selected value")
        print("4. Click 'Apply Edit' to apply the change")
        print("5. Click 'Replay' to see how the edit affects the output")
        
        # Variables for tracking selected cells
        selected_memory_cell = None
        selected_attention_cell = None
        edit_value = 0.5
        
        while not glfw.window_should_close(window):
            # Poll for and process events
            glfw.poll_events()
            
            # Start new frame
            impl.process_inputs()
            imgui.new_frame()
            
            # Create main window
            imgui.set_next_window_position(0, 0)
            imgui.set_next_window_size(1280, 720)
            
            window_flags = (
                imgui.WINDOW_NO_COLLAPSE |
                imgui.WINDOW_NO_RESIZE |
                imgui.WINDOW_NO_MOVE |
                imgui.WINDOW_NO_TITLE_BAR
            )
            
            imgui.begin("Interactive Editing Demo", True, window_flags)
            
            # Create a 2x2 grid layout
            cell_width = 1280 / 2
            cell_height = 720 / 2
            
            # Memory panel (top-left)
            imgui.set_cursor_pos((0, 0))
            imgui.begin_child("Memory", cell_width - 10, cell_height - 10)
            imgui.text("Memory Matrix")
            imgui.text("Click on a cell to select it")
            
            # Draw memory matrix
            draw_list = imgui.get_window_draw_list()
            pos = imgui.get_cursor_screen_pos()
            
            # Calculate cell size
            memory_cell_size = min(
                (cell_width - 50) / current_state.memory.shape[0],
                (cell_height - 100) / current_state.memory.shape[1]
            )
            
            # Draw memory cells
            for i in range(current_state.memory.shape[0]):
                for j in range(current_state.memory.shape[1]):
                    x = pos[0] + i * memory_cell_size
                    y = pos[1] + j * memory_cell_size
                    
                    # Get cell value
                    value = current_state.memory[i, j]
                    
                    # Determine cell color
                    if selected_memory_cell == (i, j):
                        # Selected cell is yellow
                        color = imgui.get_color_u32_rgba(1.0, 1.0, 0.0, 1.0)
                    else:
                        # Normal cell color based on value
                        color = imgui.get_color_u32_rgba(value, 0.0, 1.0 - value, 1.0)
                    
                    # Draw cell
                    draw_list.add_rect_filled(
                        x, y,
                        x + memory_cell_size - 1, y + memory_cell_size - 1,
                        color
                    )
                    
                    # Draw cell border
                    draw_list.add_rect(
                        x, y,
                        x + memory_cell_size - 1, y + memory_cell_size - 1,
                        imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 0.5),
                        0.0, 0, 1.0
                    )
                    
                    # Draw cell value
                    value_text = f"{value:.2f}"
                    text_size = imgui.calc_text_size(value_text)
                    draw_list.add_text(
                        x + (memory_cell_size - text_size[0]) / 2,
                        y + (memory_cell_size - text_size[1]) / 2,
                        imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0),
                        value_text
                    )
                    
                    # Check for mouse click on this cell
                    if imgui.is_mouse_clicked(0):  # Left mouse button
                        mouse_pos = imgui.get_mouse_pos()
                        if (x <= mouse_pos[0] <= x + memory_cell_size and
                            y <= mouse_pos[1] <= y + memory_cell_size):
                            selected_memory_cell = (i, j)
                            selected_attention_cell = None
                            print(f"Selected memory cell ({i}, {j}) with value {value:.2f}")
            
            imgui.end_child()
            
            # Attention panel (top-right)
            imgui.set_cursor_pos((cell_width, 0))
            imgui.begin_child("Attention", cell_width - 10, cell_height - 10)
            imgui.text("Attention Matrix")
            imgui.text("Click on a cell to select it")
            
            # Draw attention matrix
            draw_list = imgui.get_window_draw_list()
            pos = imgui.get_cursor_screen_pos()
            
            # Calculate cell size
            attention_cell_size = min(
                (cell_width - 50) / current_state.attention.shape[0],
                (cell_height - 100) / current_state.attention.shape[1]
            )
            
            # Draw attention cells
            for i in range(current_state.attention.shape[0]):
                for j in range(current_state.attention.shape[1]):
                    x = pos[0] + i * attention_cell_size
                    y = pos[1] + j * attention_cell_size
                    
                    # Get cell value
                    value = current_state.attention[i, j]
                    
                    # Determine cell color
                    if selected_attention_cell == (i, j):
                        # Selected cell is yellow
                        color = imgui.get_color_u32_rgba(1.0, 1.0, 0.0, 1.0)
                    else:
                        # Normal cell color based on value
                        color = imgui.get_color_u32_rgba(value, 0.0, 1.0 - value, 1.0)
                    
                    # Draw cell
                    draw_list.add_rect_filled(
                        x, y,
                        x + attention_cell_size - 1, y + attention_cell_size - 1,
                        color
                    )
                    
                    # Draw cell border
                    draw_list.add_rect(
                        x, y,
                        x + attention_cell_size - 1, y + attention_cell_size - 1,
                        imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 0.5),
                        0.0, 0, 1.0
                    )
                    
                    # Draw cell value
                    value_text = f"{value:.2f}"
                    text_size = imgui.calc_text_size(value_text)
                    draw_list.add_text(
                        x + (attention_cell_size - text_size[0]) / 2,
                        y + (attention_cell_size - text_size[1]) / 2,
                        imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0),
                        value_text
                    )
                    
                    # Check for mouse click on this cell
                    if imgui.is_mouse_clicked(0):  # Left mouse button
                        mouse_pos = imgui.get_mouse_pos()
                        if (x <= mouse_pos[0] <= x + attention_cell_size and
                            y <= mouse_pos[1] <= y + attention_cell_size):
                            selected_attention_cell = (i, j)
                            selected_memory_cell = None
                            print(f"Selected attention cell ({i}, {j}) with value {value:.2f}")
            
            imgui.end_child()
            
            # Output panel (bottom-left)
            imgui.set_cursor_pos((0, cell_height))
            imgui.begin_child("Output", cell_width - 10, cell_height - 10)
            imgui.text("Output Vector")
            
            # Draw output vector
            draw_list = imgui.get_window_draw_list()
            pos = imgui.get_cursor_screen_pos()
            
            # Draw output as a bar chart
            bar_width = (cell_width - 50) / current_state.output.shape[0]
            max_height = cell_height - 100
            
            for i in range(current_state.output.shape[0]):
                value = current_state.output[i]
                height = value * max_height
                
                x = pos[0] + i * bar_width
                y = pos[1] + max_height - height
                
                # Draw bar
                draw_list.add_rect_filled(
                    x, y,
                    x + bar_width - 1, pos[1] + max_height,
                    imgui.get_color_u32_rgba(0.0, 0.8, 0.2, 1.0)
                )
                
                # Draw value
                value_text = f"{value:.2f}"
                text_size = imgui.calc_text_size(value_text)
                draw_list.add_text(
                    x + (bar_width - text_size[0]) / 2,
                    y - text_size[1] - 5,
                    imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0),
                    value_text
                )
            
            imgui.end_child()
            
            # Controls panel (bottom-right)
            imgui.set_cursor_pos((cell_width, cell_height))
            imgui.begin_child("Controls", cell_width - 10, cell_height - 10)
            imgui.text("Controls")
            
            # Display selected cell info
            if selected_memory_cell is not None:
                i, j = selected_memory_cell
                imgui.text(f"Selected Memory Cell: ({i}, {j})")
                imgui.text(f"Current Value: {current_state.memory[i, j]:.4f}")
            elif selected_attention_cell is not None:
                i, j = selected_attention_cell
                imgui.text(f"Selected Attention Cell: ({i}, {j})")
                imgui.text(f"Current Value: {current_state.attention[i, j]:.4f}")
            else:
                imgui.text("No cell selected")
            
            # Edit value slider
            imgui.separator()
            imgui.text("Edit Value")
            changed, value = imgui.slider_float("##edit_value", edit_value, 0.0, 1.0, "%.2f")
            if changed:
                edit_value = value
            
            # Apply edit button
            if imgui.button("Apply Edit"):
                if selected_memory_cell is not None:
                    i, j = selected_memory_cell
                    old_value = current_state.memory[i, j]
                    current_state.memory[i, j] = edit_value
                    print(f"Edited memory[{i}, {j}] from {old_value:.2f} to {edit_value:.2f}")
                    action_log.append(f"Edited memory[{i}, {j}] from {old_value:.2f} to {edit_value:.2f}")
                    
                    # Update output
                    current_state.update_output()
                    
                    # Add new state to history
                    states.append(current_state.copy())
                    current_state_idx = len(states) - 1
                
                elif selected_attention_cell is not None:
                    i, j = selected_attention_cell
                    old_value = current_state.attention[i, j]
                    current_state.attention[i, j] = edit_value
                    print(f"Edited attention[{i}, {j}] from {old_value:.2f} to {edit_value:.2f}")
                    action_log.append(f"Edited attention[{i}, {j}] from {old_value:.2f} to {edit_value:.2f}")
                    
                    # Update output
                    current_state.update_output()
                    
                    # Add new state to history
                    states.append(current_state.copy())
                    current_state_idx = len(states) - 1
            
            # Replay button
            imgui.separator()
            imgui.text("Replay")
            
            if imgui.button("Replay from Original State"):
                # Reset to original state
                current_state = original_state.copy()
                current_state.update_output()
                
                # Record original output
                original_output = original_state.output.copy()
                
                # Add new state to history
                states.append(current_state.copy())
                current_state_idx = len(states) - 1
                
                # Compare outputs
                output_diff = np.mean(np.abs(current_state.output - original_output))
                print(f"Replayed from original state. Output difference: {output_diff:.4f}")
                action_log.append(f"Replayed from original state. Output difference: {output_diff:.4f}")
            
            # State navigation
            imgui.separator()
            imgui.text("State History")
            
            # State slider
            changed, value = imgui.slider_int("State", current_state_idx, 0, len(states) - 1)
            if changed:
                current_state_idx = value
                current_state = states[current_state_idx].copy()
            
            # Action log
            imgui.separator()
            imgui.text("Action Log")
            
            for log_entry in action_log[-10:]:  # Show last 10 entries
                imgui.text(log_entry)
            
            imgui.end_child()
            
            imgui.end()
            
            # Render
            imgui.render()
            
            # Clear the framebuffer
            gl.glClearColor(0.0, 0.0, 0.0, 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            
            # Render ImGui
            impl.render(imgui.get_draw_data())
            
            # Swap front and back buffers
            glfw.swap_buffers(window)
        
        # Cleanup
        impl.shutdown()
        glfw.terminate()
        
        # Print action log
        print("\nAction Log:")
        for log_entry in action_log:
            print(log_entry)
        
        print("\nInteractive editing demo completed successfully!")
        
        return 0
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
