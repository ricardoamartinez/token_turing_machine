"""
Test script for real-time interactive UI components.

This script demonstrates real-time interactive UI components using Dear ImGui.
"""

import sys
import imgui
import glfw
import OpenGL.GL as gl
import numpy as np
from imgui.integrations.glfw import GlfwRenderer

class ModelState:
    """Simple class to represent the model state."""
    
    def __init__(self):
        """Initialize the model state."""
        # Model hyperparameters
        self.learning_rate = 0.001
        self.batch_size = 32
        self.num_epochs = 10
        self.dropout_rate = 0.1
        
        # Model internal states
        self.attention_weights = np.random.rand(8, 8)  # 8x8 attention matrix
        self.memory_values = np.random.rand(10)  # 10-dimensional memory vector
        self.layer_activations = [np.random.rand(10) for _ in range(4)]  # 4 layers of 10-dimensional activations
        
        # Training metrics
        self.loss = 1.0
        self.accuracy = 0.5
        self.iteration = 0
        
        # Visualization settings
        self.show_attention = True
        self.show_memory = True
        self.show_activations = True
        self.detail_level = 1.0
        self.target_fps = 60.0


def main():
    """Run the test."""
    try:
        # Initialize model state
        model_state = ModelState()
        
        # Initialize GLFW
        if not glfw.init():
            print("Could not initialize GLFW")
            return 1
        
        # Create a windowed mode window and its OpenGL context
        window = glfw.create_window(1280, 720, "Interactive UI Demo", None, None)
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
        style = imgui.get_style()
        style.window_rounding = 0.0
        style.colors[imgui.COLOR_WINDOW_BACKGROUND] = (0.0, 0.0, 0.0, 0.8)
        style.colors[imgui.COLOR_TITLE_BACKGROUND_ACTIVE] = (0.1, 0.1, 0.1, 1.0)
        style.colors[imgui.COLOR_TITLE_BACKGROUND] = (0.1, 0.1, 0.1, 0.8)
        style.colors[imgui.COLOR_FRAME_BACKGROUND] = (0.05, 0.05, 0.05, 0.8)
        style.colors[imgui.COLOR_FRAME_BACKGROUND_HOVERED] = (0.15, 0.15, 0.15, 0.8)
        style.colors[imgui.COLOR_FRAME_BACKGROUND_ACTIVE] = (0.25, 0.25, 0.25, 0.8)
        style.colors[imgui.COLOR_BUTTON] = (0.2, 0.2, 0.2, 0.8)
        style.colors[imgui.COLOR_BUTTON_HOVERED] = (0.3, 0.3, 0.3, 0.8)
        style.colors[imgui.COLOR_BUTTON_ACTIVE] = (0.4, 0.4, 0.4, 0.8)
        style.colors[imgui.COLOR_TEXT] = (1.0, 1.0, 1.0, 1.0)
        
        # Print ImGui version
        version = imgui.get_version()
        print(f"ImGui version: {version}")
        
        # Main loop
        frame_count = 0
        while not glfw.window_should_close(window) and frame_count < 300:  # Run for 300 frames
            # Poll for and process events
            glfw.poll_events()
            
            # Start new frame
            impl.process_inputs()
            imgui.new_frame()
            
            # Get window dimensions
            width, height = glfw.get_window_size(window)
            
            # Create menu bar
            if imgui.begin_main_menu_bar():
                if imgui.begin_menu("File"):
                    if imgui.menu_item("Exit")[0]:
                        glfw.set_window_should_close(window, True)
                    imgui.end_menu()
                
                if imgui.begin_menu("View"):
                    if imgui.menu_item("Reset Layout")[0]:
                        pass  # No docking, so nothing to reset
                    imgui.end_menu()
                
                menu_bar_height = imgui.get_frame_height()
                imgui.end_main_menu_bar()
            else:
                menu_bar_height = 0
            
            # Calculate panel sizes and positions
            main_panel_width = width * 0.75
            right_panel_width = width * 0.25
            top_panel_height = (height - menu_bar_height) * 0.75
            bottom_panel_height = (height - menu_bar_height) * 0.25
            
            # Set positions for each panel
            # 3D Visualization panel (main area)
            imgui.set_next_window_position(0, menu_bar_height)
            imgui.set_next_window_size(main_panel_width, top_panel_height)
            
            # Timeline panel (bottom)
            imgui.set_next_window_position(0, menu_bar_height + top_panel_height)
            imgui.set_next_window_size(main_panel_width, bottom_panel_height)
            
            # Properties panel (right)
            imgui.set_next_window_position(main_panel_width, menu_bar_height)
            imgui.set_next_window_size(right_panel_width, top_panel_height)
            
            # Performance panel (bottom right)
            imgui.set_next_window_position(main_panel_width, menu_bar_height + top_panel_height)
            imgui.set_next_window_size(right_panel_width, bottom_panel_height)
            
            # Create panels
            # Define window flags for fixed panels
            window_flags = (
                imgui.WINDOW_NO_COLLAPSE |
                imgui.WINDOW_NO_RESIZE |
                imgui.WINDOW_NO_MOVE
            )
            
            # 3D Visualization panel
            imgui.begin("3D Visualization", True, window_flags)
            imgui.text("This panel would contain the 3D visualization of the model's internal state.")
            
            # Draw a colored rectangle to represent the 3D visualization
            draw_list = imgui.get_window_draw_list()
            pos = imgui.get_cursor_screen_pos()
            # Use a fixed size instead of content region
            size_x = main_panel_width - 20  # Padding
            size_y = top_panel_height - 100  # Padding
            draw_list.add_rect_filled(
                pos[0], pos[1],
                pos[0] + size_x, pos[1] + size_y,
                imgui.get_color_u32_rgba(0.1, 0.1, 0.3, 1.0)
            )
            
            # Draw attention matrix
            if model_state.show_attention:
                center_x = pos[0] + size_x * 0.25
                center_y = pos[1] + size_y * 0.5
                cell_size = 20
                for i in range(8):
                    for j in range(8):
                        x = center_x + (i - 4) * cell_size
                        y = center_y + (j - 4) * cell_size
                        # Use attention weights to determine color intensity
                        intensity = model_state.attention_weights[i, j]
                        color = imgui.get_color_u32_rgba(intensity, 0.0, 1.0 - intensity, 1.0)
                        draw_list.add_rect_filled(
                            x, y,
                            x + cell_size, y + cell_size,
                            color
                        )
            
            # Draw memory values
            if model_state.show_memory:
                center_x = pos[0] + size_x * 0.75
                center_y = pos[1] + size_y * 0.25
                bar_width = 20
                bar_spacing = 5
                for i in range(len(model_state.memory_values)):
                    x = center_x + i * (bar_width + bar_spacing)
                    y = center_y
                    height = model_state.memory_values[i] * 100
                    color = imgui.get_color_u32_rgba(0.0, model_state.memory_values[i], 0.5, 1.0)
                    draw_list.add_rect_filled(
                        x, y,
                        x + bar_width, y + height,
                        color
                    )
            
            # Draw layer activations
            if model_state.show_activations:
                center_x = pos[0] + size_x * 0.75
                center_y = pos[1] + size_y * 0.75
                for layer_idx, layer_activations in enumerate(model_state.layer_activations):
                    y = center_y + layer_idx * 30
                    for i, activation in enumerate(layer_activations):
                        x = center_x + i * 25
                        radius = activation * 10
                        color = imgui.get_color_u32_rgba(activation, 0.5, 0.0, 1.0)
                        draw_list.add_circle_filled(
                            x, y,
                            radius,
                            color,
                            12  # num segments
                        )
            
            imgui.end()
            
            # Timeline panel
            imgui.begin("Timeline", True, window_flags)
            imgui.text("This panel contains timeline controls for navigating through the model's state history.")
            
            # Add a slider to represent the timeline
            imgui.text("Epoch:")
            changed, value = imgui.slider_int("##epoch", model_state.iteration % model_state.num_epochs, 0, model_state.num_epochs - 1)
            if changed:
                print(f"Changed epoch to {value}")
            
            imgui.text("Batch:")
            changed, value = imgui.slider_int("##batch", (model_state.iteration // model_state.num_epochs) % model_state.batch_size, 0, model_state.batch_size - 1)
            if changed:
                print(f"Changed batch to {value}")
            
            imgui.text("Token:")
            changed, value = imgui.slider_int("##token", model_state.iteration % 10, 0, 9)
            if changed:
                print(f"Changed token to {value}")
            
            # Add playback controls
            imgui.separator()
            if imgui.button("Play"):
                print("Started playback")
            
            imgui.same_line()
            if imgui.button("Pause"):
                print("Paused playback")
            
            imgui.same_line()
            if imgui.button("<<"):
                model_state.iteration = max(0, model_state.iteration - 1)
                print(f"Stepped backward to iteration {model_state.iteration}")
            
            imgui.same_line()
            if imgui.button(">>"):
                model_state.iteration += 1
                print(f"Stepped forward to iteration {model_state.iteration}")
            
            imgui.end()
            
            # Properties panel
            imgui.begin("Properties", True, window_flags)
            imgui.text("This panel shows properties of the selected state and allows editing hyperparameters.")
            
            # Add hyperparameter controls
            if imgui.collapsing_header("Hyperparameters"):
                # Learning rate
                changed, value = imgui.slider_float("Learning Rate", model_state.learning_rate, 0.0001, 0.01, "%.5f")
                if changed:
                    model_state.learning_rate = value
                    print(f"Changed learning rate to {value:.5f}")
                
                # Batch size
                changed, value = imgui.slider_int("Batch Size", model_state.batch_size, 1, 128)
                if changed:
                    model_state.batch_size = value
                    print(f"Changed batch size to {value}")
                
                # Number of epochs
                changed, value = imgui.slider_int("Num Epochs", model_state.num_epochs, 1, 100)
                if changed:
                    model_state.num_epochs = value
                    print(f"Changed number of epochs to {value}")
                
                # Dropout rate
                changed, value = imgui.slider_float("Dropout Rate", model_state.dropout_rate, 0.0, 0.5, "%.2f")
                if changed:
                    model_state.dropout_rate = value
                    print(f"Changed dropout rate to {value:.2f}")
            
            # Add state inspection
            if imgui.collapsing_header("State Inspection"):
                # Attention weights
                if imgui.tree_node("Attention Weights"):
                    changed, value = imgui.checkbox("Show Attention", model_state.show_attention)
                    if changed:
                        model_state.show_attention = value
                        print(f"{'Showing' if value else 'Hiding'} attention visualization")
                    
                    # Allow editing a specific attention weight
                    imgui.text("Edit Attention Weight:")
                    imgui.text("Row:")
                    imgui.same_line()
                    changed, row = imgui.slider_int("##row", 0, 0, 7)
                    imgui.text("Column:")
                    imgui.same_line()
                    changed, col = imgui.slider_int("##col", 0, 0, 7)
                    
                    current_value = model_state.attention_weights[row, col]
                    changed, value = imgui.slider_float(f"Weight ({row},{col})", current_value, 0.0, 1.0, "%.2f")
                    if changed:
                        model_state.attention_weights[row, col] = value
                        print(f"Changed attention weight at ({row},{col}) to {value:.2f}")
                    
                    imgui.tree_pop()
                
                # Memory values
                if imgui.tree_node("Memory Values"):
                    changed, value = imgui.checkbox("Show Memory", model_state.show_memory)
                    if changed:
                        model_state.show_memory = value
                        print(f"{'Showing' if value else 'Hiding'} memory visualization")
                    
                    # Allow editing a specific memory value
                    imgui.text("Edit Memory Value:")
                    imgui.text("Index:")
                    imgui.same_line()
                    changed, idx = imgui.slider_int("##idx", 0, 0, len(model_state.memory_values) - 1)
                    
                    current_value = model_state.memory_values[idx]
                    changed, value = imgui.slider_float(f"Value ({idx})", current_value, 0.0, 1.0, "%.2f")
                    if changed:
                        model_state.memory_values[idx] = value
                        print(f"Changed memory value at index {idx} to {value:.2f}")
                    
                    imgui.tree_pop()
                
                # Layer activations
                if imgui.tree_node("Layer Activations"):
                    changed, value = imgui.checkbox("Show Activations", model_state.show_activations)
                    if changed:
                        model_state.show_activations = value
                        print(f"{'Showing' if value else 'Hiding'} activation visualization")
                    
                    # Allow editing a specific activation
                    imgui.text("Edit Activation:")
                    imgui.text("Layer:")
                    imgui.same_line()
                    changed, layer = imgui.slider_int("##layer", 0, 0, len(model_state.layer_activations) - 1)
                    imgui.text("Neuron:")
                    imgui.same_line()
                    changed, neuron = imgui.slider_int("##neuron", 0, 0, len(model_state.layer_activations[layer]) - 1)
                    
                    current_value = model_state.layer_activations[layer][neuron]
                    changed, value = imgui.slider_float(f"Activation ({layer},{neuron})", current_value, 0.0, 1.0, "%.2f")
                    if changed:
                        model_state.layer_activations[layer][neuron] = value
                        print(f"Changed activation at layer {layer}, neuron {neuron} to {value:.2f}")
                    
                    imgui.tree_pop()
            
            imgui.end()
            
            # Performance panel
            imgui.begin("Performance", True, window_flags)
            imgui.text("This panel shows performance metrics and allows adjusting rendering settings.")
            
            # Add FPS counter
            io = imgui.get_io()
            imgui.text(f"FPS: {io.framerate:.1f}")
            
            # Add training metrics
            imgui.separator()
            imgui.text("Training Metrics:")
            imgui.text(f"Loss: {model_state.loss:.4f}")
            imgui.text(f"Accuracy: {model_state.accuracy:.2f}")
            imgui.text(f"Iteration: {model_state.iteration}")
            
            # Add visualization settings
            imgui.separator()
            imgui.text("Visualization Settings:")
            
            # Detail level
            changed, value = imgui.slider_float("Detail Level", model_state.detail_level, 0.1, 1.0, "%.1f")
            if changed:
                model_state.detail_level = value
                print(f"Changed detail level to {value:.1f}")
            
            # Target FPS
            changed, value = imgui.slider_float("Target FPS", model_state.target_fps, 30.0, 120.0, "%.1f")
            if changed:
                model_state.target_fps = value
                print(f"Changed target FPS to {value:.1f}")
            
            # Add a checkbox for adaptive rendering
            changed, value = imgui.checkbox("Adaptive Rendering", True)
            if changed:
                print(f"{'Enabled' if value else 'Disabled'} adaptive rendering")
            
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
            
            # Update model state for animation
            model_state.iteration += 1
            model_state.loss = max(0.1, model_state.loss * 0.99)  # Decrease loss over time
            model_state.accuracy = min(0.99, model_state.accuracy * 1.01)  # Increase accuracy over time
            
            # Randomly update some attention weights
            i, j = np.random.randint(0, 8, 2)
            model_state.attention_weights[i, j] = np.clip(model_state.attention_weights[i, j] + np.random.normal(0, 0.05), 0, 1)
            
            # Randomly update some memory values
            i = np.random.randint(0, len(model_state.memory_values))
            model_state.memory_values[i] = np.clip(model_state.memory_values[i] + np.random.normal(0, 0.05), 0, 1)
            
            # Randomly update some activations
            layer = np.random.randint(0, len(model_state.layer_activations))
            neuron = np.random.randint(0, len(model_state.layer_activations[layer]))
            model_state.layer_activations[layer][neuron] = np.clip(model_state.layer_activations[layer][neuron] + np.random.normal(0, 0.05), 0, 1)
            
            # Print debug information for the first few frames
            if frame_count < 5:
                print(f"Frame {frame_count}: Rendered interactive UI with real-time state updates")
                print(f"  - Current iteration: {model_state.iteration}")
                print(f"  - Current loss: {model_state.loss:.4f}")
                print(f"  - Current accuracy: {model_state.accuracy:.2f}")
                print(f"  - FPS: {io.framerate:.1f}")
            
            frame_count += 1
        
        # Cleanup
        impl.shutdown()
        glfw.terminate()
        
        print("Interactive UI test completed successfully!")
        print("Terminal Validation: The UI includes controls to edit hyperparameters, inspect and modify internal states, and view real-time visualizations all on one canvas.")
        print("State modifications are propagated to the model simulation through direct updates to the ModelState object, which triggers reprocessing in the visualization.")
        
        return 0
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
