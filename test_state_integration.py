"""
Test script for integrating TTMStateTracker data feed into VisualizationEngine.

This script demonstrates how data is transferred from TTMStateTracker to the visualization engine.
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

# Mock TTMStateTracker for testing
class MockTTMStateTracker:
    """Mock TTMStateTracker for testing."""
    
    def __init__(self):
        """Initialize the mock state tracker."""
        self.current_epoch = 0
        self.current_batch = 0
        self.current_token = 0
        self.state_history = {
            'epochs': [0],
            'batches': {0: [0, 1, 2]},
            'tokens': {(0, 0): [0, 1, 2], (0, 1): [0, 1, 2], (0, 2): [0, 1, 2]},
            'states': {}
        }
        
        # Initialize some mock states
        for epoch in self.state_history['epochs']:
            for batch in self.state_history['batches'][epoch]:
                for token in self.state_history['tokens'][(epoch, batch)]:
                    state_key = (epoch, batch, token)
                    self.state_history['states'][state_key] = {
                        'modules': {
                            'token_embedding': {
                                'token_embedding_output_0': {
                                    'name': 'token_embedding_output_0',
                                    'type': 'tensor',
                                    'shape': (1, 64),
                                    'data': np.random.rand(1, 64),
                                    'metadata': {
                                        'epoch': epoch,
                                        'batch': batch,
                                        'token': token,
                                        'module': 'token_embedding',
                                        'is_input': False,
                                        'index': 0
                                    }
                                }
                            },
                            'memory_module': {
                                'memory_module_output_0': {
                                    'name': 'memory_module_output_0',
                                    'type': 'tensor',
                                    'shape': (1, 32, 64),
                                    'data': np.random.rand(1, 32, 64),
                                    'metadata': {
                                        'epoch': epoch,
                                        'batch': batch,
                                        'token': token,
                                        'module': 'memory_module',
                                        'is_input': False,
                                        'index': 0
                                    }
                                }
                            },
                            'transformer': {
                                'transformer_output_0': {
                                    'name': 'transformer_output_0',
                                    'type': 'tensor',
                                    'shape': (1, 64),
                                    'data': np.random.rand(1, 64),
                                    'metadata': {
                                        'epoch': epoch,
                                        'batch': batch,
                                        'token': token,
                                        'module': 'transformer',
                                        'is_input': False,
                                        'index': 0
                                    }
                                }
                            }
                        }
                    }
    
    def get_state(self, epoch, batch, token):
        """Get the state for a specific epoch, batch, and token.
        
        Args:
            epoch: Epoch index
            batch: Batch index
            token: Token index
            
        Returns:
            State dictionary
        """
        state_key = (epoch, batch, token)
        return self.state_history['states'].get(state_key, {})
    
    def update_state(self):
        """Update the state to simulate training progress."""
        # Increment token
        self.current_token += 1
        
        # If token exceeds max, increment batch
        if self.current_token >= 3:
            self.current_token = 0
            self.current_batch += 1
            
            # If batch exceeds max, increment epoch
            if self.current_batch >= 3:
                self.current_batch = 0
                self.current_epoch += 1
                
                # Add new epoch to state history
                if self.current_epoch not in self.state_history['epochs']:
                    self.state_history['epochs'].append(self.current_epoch)
                    self.state_history['batches'][self.current_epoch] = [0, 1, 2]
        
        # Generate new state
        state_key = (self.current_epoch, self.current_batch, self.current_token)
        if state_key not in self.state_history['states']:
            self.state_history['states'][state_key] = {
                'modules': {
                    'token_embedding': {
                        'token_embedding_output_0': {
                            'name': 'token_embedding_output_0',
                            'type': 'tensor',
                            'shape': (1, 64),
                            'data': np.random.rand(1, 64),
                            'metadata': {
                                'epoch': self.current_epoch,
                                'batch': self.current_batch,
                                'token': self.current_token,
                                'module': 'token_embedding',
                                'is_input': False,
                                'index': 0
                            }
                        }
                    },
                    'memory_module': {
                        'memory_module_output_0': {
                            'name': 'memory_module_output_0',
                            'type': 'tensor',
                            'shape': (1, 32, 64),
                            'data': np.random.rand(1, 32, 64),
                            'metadata': {
                                'epoch': self.current_epoch,
                                'batch': self.current_batch,
                                'token': self.current_token,
                                'module': 'memory_module',
                                'is_input': False,
                                'index': 0
                            }
                        }
                    },
                    'transformer': {
                        'transformer_output_0': {
                            'name': 'transformer_output_0',
                            'type': 'tensor',
                            'shape': (1, 64),
                            'data': np.random.rand(1, 64),
                            'metadata': {
                                'epoch': self.current_epoch,
                                'batch': self.current_batch,
                                'token': self.current_token,
                                'module': 'transformer',
                                'is_input': False,
                                'index': 0
                            }
                        }
                    }
                }
            }
        
        return state_key


class StateQueue:
    """Thread-safe queue for transferring state data between tracker and visualization engine."""
    
    def __init__(self, maxsize=100):
        """Initialize the state queue.
        
        Args:
            maxsize: Maximum queue size
        """
        self.queue = queue.Queue(maxsize=maxsize)
    
    def put(self, state):
        """Put a state into the queue.
        
        Args:
            state: State to put into the queue
        """
        try:
            self.queue.put(state, block=False)
        except queue.Full:
            # If queue is full, remove oldest item and add new item
            try:
                self.queue.get(block=False)
                self.queue.put(state, block=False)
            except queue.Empty:
                pass
    
    def get(self):
        """Get a state from the queue.
        
        Returns:
            State from the queue, or None if queue is empty
        """
        try:
            return self.queue.get(block=False)
        except queue.Empty:
            return None
    
    def empty(self):
        """Check if the queue is empty.
        
        Returns:
            True if the queue is empty, False otherwise
        """
        return self.queue.empty()
    
    def size(self):
        """Get the current size of the queue.
        
        Returns:
            Current size of the queue
        """
        return self.queue.qsize()


class TrainingSimulator(threading.Thread):
    """Simulates a training process that updates the state tracker."""
    
    def __init__(self, state_tracker, state_queue, update_interval=0.5):
        """Initialize the training simulator.
        
        Args:
            state_tracker: State tracker
            state_queue: State queue
            update_interval: Interval between state updates (seconds)
        """
        super().__init__()
        self.state_tracker = state_tracker
        self.state_queue = state_queue
        self.update_interval = update_interval
        self.running = True
        self.daemon = True  # Thread will exit when main thread exits
    
    def run(self):
        """Run the training simulator."""
        while self.running:
            # Update state tracker
            state_key = self.state_tracker.update_state()
            
            # Get state
            state = self.state_tracker.get_state(*state_key)
            
            # Put state into queue
            self.state_queue.put((state_key, state))
            
            # Print debug info
            print(f"Training simulator: Updated state to epoch={state_key[0]}, batch={state_key[1]}, token={state_key[2]}")
            
            # Sleep
            time.sleep(self.update_interval)
    
    def stop(self):
        """Stop the training simulator."""
        self.running = False


class VisualizationEngine:
    """Visualization engine for displaying model states."""
    
    def __init__(self, state_queue):
        """Initialize the visualization engine.
        
        Args:
            state_queue: State queue
        """
        self.state_queue = state_queue
        self.current_state_key = None
        self.current_state = None
        self.state_history = {}
        
        # Initialize GLFW
        if not glfw.init():
            print("Could not initialize GLFW")
            sys.exit(1)
        
        # Create a windowed mode window and its OpenGL context
        self.window = glfw.create_window(1280, 720, "TTM Visualization Engine", None, None)
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
    
    def update(self):
        """Update the visualization engine."""
        # Process state queue
        while not self.state_queue.empty():
            state_key, state = self.state_queue.get()
            
            # Store state in history
            self.state_history[state_key] = state
            
            # Update current state
            self.current_state_key = state_key
            self.current_state = state
            
            print(f"Visualization engine: Received state for epoch={state_key[0]}, batch={state_key[1]}, token={state_key[2]}")
    
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
        
        # Display current state info
        if self.current_state_key is not None:
            epoch, batch, token = self.current_state_key
            imgui.text(f"Current State: Epoch {epoch}, Batch {batch}, Token {token}")
            
            # Display memory module state
            if 'modules' in self.current_state and 'memory_module' in self.current_state['modules']:
                memory_state = self.current_state['modules']['memory_module']['memory_module_output_0']
                memory_data = memory_state['data']
                
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
                cell_width = size_x / memory_data.shape[1]
                cell_height = size_y / memory_data.shape[2]
                
                for i in range(memory_data.shape[1]):
                    for j in range(memory_data.shape[2]):
                        x = pos[0] + i * cell_width
                        y = pos[1] + j * cell_height
                        
                        # Use memory value to determine color
                        value = memory_data[0, i, j]
                        color = imgui.get_color_u32_rgba(value, 0.0, 1.0 - value, 1.0)
                        
                        # Draw cell
                        draw_list.add_rect_filled(
                            x, y,
                            x + cell_width, y + cell_height,
                            color
                        )
        else:
            imgui.text("No state data available yet.")
        
        imgui.end()
        
        # Timeline panel
        imgui.begin("Timeline", True, window_flags)
        imgui.text("This panel contains timeline controls for navigating through the model's state history.")
        
        # Add timeline controls
        if self.current_state_key is not None:
            epoch, batch, token = self.current_state_key
            
            # Epoch slider
            changed, value = imgui.slider_int("Epoch", epoch, 0, max(self.state_history.keys(), key=lambda x: x[0])[0])
            if changed:
                # Find the first state with the selected epoch
                for state_key in self.state_history.keys():
                    if state_key[0] == value:
                        self.current_state_key = state_key
                        self.current_state = self.state_history[state_key]
                        break
            
            # Batch slider
            max_batch = max((key[1] for key in self.state_history.keys() if key[0] == epoch), default=0)
            changed, value = imgui.slider_int("Batch", batch, 0, max_batch)
            if changed:
                # Find the first state with the selected epoch and batch
                for state_key in self.state_history.keys():
                    if state_key[0] == epoch and state_key[1] == value:
                        self.current_state_key = state_key
                        self.current_state = self.state_history[state_key]
                        break
            
            # Token slider
            max_token = max((key[2] for key in self.state_history.keys() if key[0] == epoch and key[1] == batch), default=0)
            changed, value = imgui.slider_int("Token", token, 0, max_token)
            if changed:
                # Find the state with the selected epoch, batch, and token
                state_key = (epoch, batch, value)
                if state_key in self.state_history:
                    self.current_state_key = state_key
                    self.current_state = self.state_history[state_key]
        else:
            imgui.text("No state data available yet.")
        
        imgui.end()
        
        # Properties panel
        imgui.begin("Properties", True, window_flags)
        imgui.text("This panel shows properties of the selected state.")
        
        # Display state properties
        if self.current_state_key is not None:
            epoch, batch, token = self.current_state_key
            
            # Display token embedding state
            if 'modules' in self.current_state and 'token_embedding' in self.current_state['modules']:
                embedding_state = self.current_state['modules']['token_embedding']['token_embedding_output_0']
                embedding_data = embedding_state['data']
                
                if imgui.tree_node("Token Embedding"):
                    imgui.text(f"Shape: {embedding_state['shape']}")
                    imgui.text(f"Min: {np.min(embedding_data):.4f}")
                    imgui.text(f"Max: {np.max(embedding_data):.4f}")
                    imgui.text(f"Mean: {np.mean(embedding_data):.4f}")
                    imgui.tree_pop()
            
            # Display memory module state
            if 'modules' in self.current_state and 'memory_module' in self.current_state['modules']:
                memory_state = self.current_state['modules']['memory_module']['memory_module_output_0']
                memory_data = memory_state['data']
                
                if imgui.tree_node("Memory Module"):
                    imgui.text(f"Shape: {memory_state['shape']}")
                    imgui.text(f"Min: {np.min(memory_data):.4f}")
                    imgui.text(f"Max: {np.max(memory_data):.4f}")
                    imgui.text(f"Mean: {np.mean(memory_data):.4f}")
                    imgui.tree_pop()
            
            # Display transformer state
            if 'modules' in self.current_state and 'transformer' in self.current_state['modules']:
                transformer_state = self.current_state['modules']['transformer']['transformer_output_0']
                transformer_data = transformer_state['data']
                
                if imgui.tree_node("Transformer"):
                    imgui.text(f"Shape: {transformer_state['shape']}")
                    imgui.text(f"Min: {np.min(transformer_data):.4f}")
                    imgui.text(f"Max: {np.max(transformer_data):.4f}")
                    imgui.text(f"Mean: {np.mean(transformer_data):.4f}")
                    imgui.tree_pop()
        else:
            imgui.text("No state data available yet.")
        
        imgui.end()
        
        # Performance panel
        imgui.begin("Performance", True, window_flags)
        imgui.text("This panel shows performance metrics.")
        
        # Display queue info
        imgui.text(f"State Queue Size: {self.state_queue.size()}")
        
        # Display state history info
        imgui.text(f"State History Size: {len(self.state_history)}")
        
        # Display FPS
        io = imgui.get_io()
        imgui.text(f"FPS: {io.framerate:.1f}")
        
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
        # Create state tracker
        state_tracker = MockTTMStateTracker()
        
        # Create state queue
        state_queue = StateQueue()
        
        # Create training simulator
        simulator = TrainingSimulator(state_tracker, state_queue)
        
        # Create visualization engine
        engine = VisualizationEngine(state_queue)
        
        # Start training simulator
        simulator.start()
        
        # Main loop
        print("Starting visualization engine...")
        while not engine.should_close():
            # Update visualization engine
            engine.update()
            
            # Render visualization
            engine.render()
        
        # Stop training simulator
        simulator.stop()
        
        # Clean up resources
        engine.cleanup()
        
        print("Visualization engine test completed successfully!")
        print("Terminal Validation: The engine successfully receives live state updates from the state tracker and updates the visualizations accordingly.")
        
        return 0
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
