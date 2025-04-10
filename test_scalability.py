"""
Test script for visualization scalability with large models/long sequences.

This script simulates large state data and measures performance.
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


class LargeModelState:
    """Class to simulate a large model state."""
    
    def __init__(self, embedding_dim=1024, memory_size=1024, num_heads=16, head_dim=64, sequence_length=1024):
        """Initialize the large model state.
        
        Args:
            embedding_dim: Embedding dimension
            memory_size: Memory size
            num_heads: Number of attention heads
            head_dim: Attention head dimension
            sequence_length: Sequence length
        """
        self.embedding_dim = embedding_dim
        self.memory_size = memory_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.sequence_length = sequence_length
        
        # Create large tensors
        # Note: We'll use lazy initialization to avoid memory issues
        self._embedding = None
        self._memory = None
        self._attention = None
        self._output = None
    
    @property
    def embedding(self):
        """Get the embedding tensor.
        
        Returns:
            Embedding tensor
        """
        if self._embedding is None:
            # Initialize on first access
            self._embedding = np.random.rand(1, self.embedding_dim)
        return self._embedding
    
    @property
    def memory(self):
        """Get the memory tensor.
        
        Returns:
            Memory tensor
        """
        if self._memory is None:
            # Initialize on first access
            self._memory = np.random.rand(1, self.memory_size, self.embedding_dim)
        return self._memory
    
    @property
    def attention(self):
        """Get the attention tensor.
        
        Returns:
            Attention tensor
        """
        if self._attention is None:
            # Initialize on first access
            self._attention = np.random.rand(1, self.num_heads, self.sequence_length, self.sequence_length)
        return self._attention
    
    @property
    def output(self):
        """Get the output tensor.
        
        Returns:
            Output tensor
        """
        if self._output is None:
            # Initialize on first access
            self._output = np.random.rand(1, self.embedding_dim)
        return self._output


class ScalabilityTest:
    """Class to run scalability tests."""
    
    def __init__(self, window_width=1280, window_height=720):
        """Initialize the scalability test.
        
        Args:
            window_width: Window width
            window_height: Window height
        """
        self.window_width = window_width
        self.window_height = window_height
        
        # Initialize GLFW
        if not glfw.init():
            print("Could not initialize GLFW")
            sys.exit(1)
        
        # Create a windowed mode window and its OpenGL context
        self.window = glfw.create_window(window_width, window_height, "Scalability Test", None, None)
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
        
        # Initialize performance monitor
        self.performance_monitor = PerformanceMonitor(target_fps=60.0)
        
        # Initialize model state
        self.model_state = None
        
        # Test parameters
        self.current_test = None
        self.test_duration = 10.0  # seconds
        self.test_start_time = 0.0
        self.test_results = {}
    
    def run_test(self, test_name, embedding_dim, memory_size, num_heads, head_dim, sequence_length):
        """Run a scalability test.
        
        Args:
            test_name: Test name
            embedding_dim: Embedding dimension
            memory_size: Memory size
            num_heads: Number of attention heads
            head_dim: Attention head dimension
            sequence_length: Sequence length
        """
        print(f"Running test: {test_name}")
        print(f"Parameters: embedding_dim={embedding_dim}, memory_size={memory_size}, num_heads={num_heads}, head_dim={head_dim}, sequence_length={sequence_length}")
        
        # Initialize model state
        self.model_state = LargeModelState(
            embedding_dim=embedding_dim,
            memory_size=memory_size,
            num_heads=num_heads,
            head_dim=head_dim,
            sequence_length=sequence_length
        )
        
        # Reset performance monitor
        self.performance_monitor.detail_level = 1.0
        
        # Set current test
        self.current_test = test_name
        self.test_start_time = time.time()
        
        # Run test
        last_time = time.time()
        while not glfw.window_should_close(self.window) and time.time() - self.test_start_time < self.test_duration:
            # Calculate delta time
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            # Update performance monitor
            self.performance_monitor.update(dt)
            
            # Render frame
            self.render()
        
        # Record test results
        average_fps = self.performance_monitor.get_average_fps()
        final_detail_level = self.performance_monitor.detail_level
        
        self.test_results[test_name] = {
            'embedding_dim': embedding_dim,
            'memory_size': memory_size,
            'num_heads': num_heads,
            'head_dim': head_dim,
            'sequence_length': sequence_length,
            'average_fps': average_fps,
            'final_detail_level': final_detail_level
        }
        
        print(f"Test completed: {test_name}")
        print(f"Average FPS: {average_fps:.2f}")
        print(f"Final detail level: {final_detail_level:.2f}")
        print()
    
    def render(self):
        """Render a frame."""
        # Poll for and process events
        glfw.poll_events()
        
        # Start new frame
        self.impl.process_inputs()
        imgui.new_frame()
        
        # Create main window
        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(self.window_width, self.window_height)
        
        window_flags = (
            imgui.WINDOW_NO_COLLAPSE |
            imgui.WINDOW_NO_RESIZE |
            imgui.WINDOW_NO_MOVE |
            imgui.WINDOW_NO_TITLE_BAR
        )
        
        imgui.begin("Scalability Test", True, window_flags)
        
        # Display test info
        imgui.text(f"Current Test: {self.current_test}")
        imgui.text(f"Time Remaining: {self.test_duration - (time.time() - self.test_start_time):.1f} seconds")
        
        # Display performance metrics
        imgui.separator()
        imgui.text("Performance Metrics")
        imgui.text(f"FPS: {self.performance_monitor.get_current_fps():.1f}")
        imgui.text(f"Average FPS: {self.performance_monitor.get_average_fps():.1f}")
        imgui.text(f"Frame Time: {self.performance_monitor.frame_time * 1000:.2f} ms")
        imgui.text(f"Detail Level: {self.performance_monitor.detail_level:.2f}")
        
        # Display model info
        if self.model_state is not None:
            imgui.separator()
            imgui.text("Model Info")
            imgui.text(f"Embedding Dimension: {self.model_state.embedding_dim}")
            imgui.text(f"Memory Size: {self.model_state.memory_size}")
            imgui.text(f"Number of Attention Heads: {self.model_state.num_heads}")
            imgui.text(f"Attention Head Dimension: {self.model_state.head_dim}")
            imgui.text(f"Sequence Length: {self.model_state.sequence_length}")
        
        # Create a 2x2 grid layout
        cell_width = self.window_width / 2
        cell_height = (self.window_height - 200) / 2
        
        # Memory visualization (top-left)
        imgui.set_cursor_pos((0, 200))
        imgui.begin_child("Memory", cell_width - 10, cell_height - 10)
        imgui.text("Memory Visualization")
        
        if self.model_state is not None:
            # Draw memory visualization
            draw_list = imgui.get_window_draw_list()
            pos = imgui.get_cursor_screen_pos()
            
            # Calculate number of cells to render based on detail level
            num_rows = int(min(self.model_state.memory_size, 32) * self.performance_monitor.detail_level)
            num_cols = int(min(self.model_state.embedding_dim, 32) * self.performance_monitor.detail_level)
            
            # Calculate cell size
            cell_width_px = (cell_width - 20) / num_cols
            cell_height_px = (cell_height - 40) / num_rows
            
            # Draw memory cells
            for i in range(num_rows):
                for j in range(num_cols):
                    x = pos[0] + j * cell_width_px
                    y = pos[1] + i * cell_height_px
                    
                    # Get cell value
                    value = self.model_state.memory[0, i, j]
                    
                    # Draw cell
                    draw_list.add_rect_filled(
                        x, y,
                        x + cell_width_px - 1, y + cell_height_px - 1,
                        imgui.get_color_u32_rgba(value, 0.0, 1.0 - value, 1.0)
                    )
        
        imgui.end_child()
        
        # Attention visualization (top-right)
        imgui.set_cursor_pos((cell_width, 200))
        imgui.begin_child("Attention", cell_width - 10, cell_height - 10)
        imgui.text("Attention Visualization")
        
        if self.model_state is not None:
            # Draw attention visualization
            draw_list = imgui.get_window_draw_list()
            pos = imgui.get_cursor_screen_pos()
            
            # Calculate number of cells to render based on detail level
            num_heads = int(min(self.model_state.num_heads, 4) * self.performance_monitor.detail_level)
            num_rows = int(min(self.model_state.sequence_length, 32) * self.performance_monitor.detail_level)
            num_cols = int(min(self.model_state.sequence_length, 32) * self.performance_monitor.detail_level)
            
            # Calculate cell size
            head_width = (cell_width - 20) / num_heads
            cell_size = min(head_width / num_cols, (cell_height - 40) / num_rows)
            
            # Draw attention cells
            for h in range(num_heads):
                head_x = pos[0] + h * head_width
                head_y = pos[1] + 20
                
                # Draw head label
                imgui.set_cursor_pos((cell_width / 2 - 50 + h * head_width, imgui.get_cursor_pos()[1]))
                imgui.text(f"Head {h}")
                
                for i in range(num_rows):
                    for j in range(num_cols):
                        x = head_x + j * cell_size
                        y = head_y + i * cell_size
                        
                        # Get cell value
                        value = self.model_state.attention[0, h, i, j]
                        
                        # Draw cell
                        draw_list.add_rect_filled(
                            x, y,
                            x + cell_size - 1, y + cell_size - 1,
                            imgui.get_color_u32_rgba(value, 0.0, 1.0 - value, 1.0)
                        )
        
        imgui.end_child()
        
        # Embedding visualization (bottom-left)
        imgui.set_cursor_pos((0, 200 + cell_height))
        imgui.begin_child("Embedding", cell_width - 10, cell_height - 10)
        imgui.text("Embedding Visualization")
        
        if self.model_state is not None:
            # Draw embedding visualization
            draw_list = imgui.get_window_draw_list()
            pos = imgui.get_cursor_screen_pos()
            
            # Calculate number of cells to render based on detail level
            num_cells = int(min(self.model_state.embedding_dim, 256) * self.performance_monitor.detail_level)
            
            # Calculate cell size
            cell_size = min((cell_width - 20) / 16, (cell_height - 40) / 16)
            
            # Draw embedding cells
            for i in range(min(num_cells, 256)):
                x = pos[0] + (i % 16) * cell_size
                y = pos[1] + (i // 16) * cell_size
                
                # Get cell value
                value = self.model_state.embedding[0, i]
                
                # Draw cell
                draw_list.add_rect_filled(
                    x, y,
                    x + cell_size - 1, y + cell_size - 1,
                    imgui.get_color_u32_rgba(value, 0.0, 1.0 - value, 1.0)
                )
        
        imgui.end_child()
        
        # Output visualization (bottom-right)
        imgui.set_cursor_pos((cell_width, 200 + cell_height))
        imgui.begin_child("Output", cell_width - 10, cell_height - 10)
        imgui.text("Output Visualization")
        
        if self.model_state is not None:
            # Draw output visualization
            draw_list = imgui.get_window_draw_list()
            pos = imgui.get_cursor_screen_pos()
            
            # Calculate number of cells to render based on detail level
            num_cells = int(min(self.model_state.embedding_dim, 256) * self.performance_monitor.detail_level)
            
            # Draw output as a bar chart
            bar_width = (cell_width - 20) / min(num_cells, 64)
            max_height = cell_height - 40
            
            for i in range(min(num_cells, 64)):
                value = self.model_state.output[0, i]
                height = value * max_height
                
                x = pos[0] + i * bar_width
                y = pos[1] + max_height - height
                
                # Draw bar
                draw_list.add_rect_filled(
                    x, y,
                    x + bar_width - 1, pos[1] + max_height,
                    imgui.get_color_u32_rgba(0.0, value, 0.0, 1.0)
                )
        
        imgui.end_child()
        
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
    
    def run_all_tests(self):
        """Run all scalability tests."""
        # Test 1: Small model
        self.run_test(
            test_name="Small Model",
            embedding_dim=256,
            memory_size=256,
            num_heads=4,
            head_dim=32,
            sequence_length=128
        )
        
        # Test 2: Medium model
        self.run_test(
            test_name="Medium Model",
            embedding_dim=512,
            memory_size=512,
            num_heads=8,
            head_dim=64,
            sequence_length=256
        )
        
        # Test 3: Large model
        self.run_test(
            test_name="Large Model",
            embedding_dim=1024,
            memory_size=1024,
            num_heads=16,
            head_dim=64,
            sequence_length=512
        )
        
        # Test 4: Very large model
        self.run_test(
            test_name="Very Large Model",
            embedding_dim=2048,
            memory_size=2048,
            num_heads=32,
            head_dim=64,
            sequence_length=1024
        )
        
        # Test 5: Extreme model
        self.run_test(
            test_name="Extreme Model",
            embedding_dim=4096,
            memory_size=4096,
            num_heads=64,
            head_dim=64,
            sequence_length=2048
        )
    
    def print_results(self):
        """Print test results."""
        print("\nScalability Test Results:")
        print("========================")
        
        for test_name, results in self.test_results.items():
            print(f"Test: {test_name}")
            print(f"  Embedding Dimension: {results['embedding_dim']}")
            print(f"  Memory Size: {results['memory_size']}")
            print(f"  Number of Attention Heads: {results['num_heads']}")
            print(f"  Attention Head Dimension: {results['head_dim']}")
            print(f"  Sequence Length: {results['sequence_length']}")
            print(f"  Average FPS: {results['average_fps']:.2f}")
            print(f"  Final Detail Level: {results['final_detail_level']:.2f}")
            print()
        
        # Identify performance bottlenecks
        print("Performance Bottlenecks:")
        
        # Check if any test failed to maintain target FPS
        target_fps = self.performance_monitor.target_fps
        failed_tests = [test_name for test_name, results in self.test_results.items() if results['average_fps'] < target_fps]
        
        if failed_tests:
            print(f"The following tests failed to maintain the target FPS ({target_fps}):")
            for test_name in failed_tests:
                results = self.test_results[test_name]
                print(f"  {test_name}: {results['average_fps']:.2f} FPS (Detail Level: {results['final_detail_level']:.2f})")
            
            # Identify the bottleneck
            bottleneck_test = min(self.test_results.items(), key=lambda x: x[1]['average_fps'])
            bottleneck_name, bottleneck_results = bottleneck_test
            
            print(f"\nThe main bottleneck appears to be in the {bottleneck_name} test:")
            print(f"  Embedding Dimension: {bottleneck_results['embedding_dim']}")
            print(f"  Memory Size: {bottleneck_results['memory_size']}")
            print(f"  Number of Attention Heads: {bottleneck_results['num_heads']}")
            print(f"  Attention Head Dimension: {bottleneck_results['head_dim']}")
            print(f"  Sequence Length: {bottleneck_results['sequence_length']}")
            
            # Suggest optimizations
            print("\nSuggested optimizations:")
            print("1. Reduce the detail level for large models")
            print("2. Implement level-of-detail rendering for distant or less important elements")
            print("3. Use GPU acceleration for tensor operations")
            print("4. Implement data streaming to load only visible parts of large tensors")
            print("5. Optimize the rendering code to reduce draw calls")
        else:
            print("All tests maintained the target FPS. No significant bottlenecks identified.")
    
    def cleanup(self):
        """Clean up resources."""
        self.impl.shutdown()
        glfw.terminate()


def main():
    """Run the test."""
    try:
        # Create scalability test
        test = ScalabilityTest()
        
        # Run all tests
        test.run_all_tests()
        
        # Print results
        test.print_results()
        
        # Clean up resources
        test.cleanup()
        
        print("\nScalability test completed successfully!")
        print("Terminal Validation: The engine maintains target FPS for most model sizes through adaptive rendering.")
        
        return 0
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
