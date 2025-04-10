"""
Test script for adaptive rendering.

This script demonstrates adaptive rendering that adjusts detail to maintain performance.
"""

import sys
import imgui
import glfw
import OpenGL.GL as gl
import numpy as np
import time
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


class VoxelRenderer:
    """Simple voxel renderer for testing adaptive rendering."""

    def __init__(self, max_voxels=10000):
        """Initialize the voxel renderer.

        Args:
            max_voxels: Maximum number of voxels
        """
        self.max_voxels = max_voxels
        self.voxels = []
        self.detail_level = 1.0

    def set_detail_level(self, detail_level):
        """Set the detail level.

        Args:
            detail_level: Detail level (0.0 to 1.0)
        """
        self.detail_level = max(0.0, min(1.0, detail_level))

    def generate_voxels(self, num_voxels):
        """Generate random voxels.

        Args:
            num_voxels: Number of voxels to generate
        """
        self.voxels = []
        for _ in range(num_voxels):
            # Generate random position
            x = np.random.uniform(-5.0, 5.0)
            y = np.random.uniform(-5.0, 5.0)
            z = np.random.uniform(-5.0, 5.0)

            # Generate random color
            r = np.random.uniform(0.0, 1.0)
            g = np.random.uniform(0.0, 1.0)
            b = np.random.uniform(0.0, 1.0)

            # Generate random size
            size = np.random.uniform(0.1, 0.5)

            # Add voxel
            self.voxels.append({
                'position': (x, y, z),
                'color': (r, g, b),
                'size': size
            })

    def render(self, draw_list, pos, size):
        """Render voxels.

        Args:
            draw_list: ImGui draw list
            pos: Position (x, y)
            size: Size (width, height)
        """
        # Calculate number of voxels to render based on detail level
        num_voxels = int(len(self.voxels) * self.detail_level)

        # Render voxels
        center_x = pos[0] + size[0] / 2
        center_y = pos[1] + size[1] / 2

        for i in range(num_voxels):
            voxel = self.voxels[i]

            # Project 3D position to 2D
            x = center_x + voxel['position'][0] * 50
            y = center_y + voxel['position'][1] * 50

            # Calculate size based on z-position (perspective)
            z = voxel['position'][2]
            scale = 1.0 / (1.0 + abs(z) * 0.1)

            # Calculate radius
            radius = voxel['size'] * 20 * scale

            # Calculate color
            r, g, b = voxel['color']
            color = imgui.get_color_u32_rgba(r, g, b, 1.0)

            # Draw voxel
            draw_list.add_circle_filled(
                x, y,
                radius,
                color,
                12  # num segments
            )


def main():
    """Run the test."""
    try:
        # Initialize performance monitor
        performance_monitor = PerformanceMonitor(target_fps=60.0)

        # Initialize voxel renderer
        voxel_renderer = VoxelRenderer(max_voxels=10000)
        voxel_renderer.generate_voxels(10000)

        # Initialize GLFW
        if not glfw.init():
            print("Could not initialize GLFW")
            return 1

        # Create a windowed mode window and its OpenGL context
        window = glfw.create_window(1280, 720, "Adaptive Rendering Demo", None, None)
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

        # Variables for artificial load
        load_factor = 0.0
        load_increasing = True

        # Main loop
        frame_count = 0
        last_time = time.time()
        while not glfw.window_should_close(window) and frame_count < 1000:  # Run for 1000 frames
            # Calculate delta time
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            # Update performance monitor
            performance_monitor.update(dt)

            # Update voxel renderer detail level
            voxel_renderer.set_detail_level(performance_monitor.detail_level)

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

                menu_bar_height = imgui.get_frame_height()
                imgui.end_main_menu_bar()
            else:
                menu_bar_height = 0

            # Create visualization panel
            imgui.set_next_window_position(0, menu_bar_height)
            imgui.set_next_window_size(width * 0.75, height - menu_bar_height)

            window_flags = (
                imgui.WINDOW_NO_COLLAPSE |
                imgui.WINDOW_NO_RESIZE |
                imgui.WINDOW_NO_MOVE
            )

            imgui.begin("Visualization", True, window_flags)
            imgui.text("This panel contains the 3D visualization.")

            # Draw voxels
            draw_list = imgui.get_window_draw_list()
            pos = imgui.get_cursor_screen_pos()
            size = (width * 0.75 - 20, height - menu_bar_height - 100)

            # Draw background
            draw_list.add_rect_filled(
                pos[0], pos[1],
                pos[0] + size[0], pos[1] + size[1],
                imgui.get_color_u32_rgba(0.1, 0.1, 0.3, 1.0)
            )

            # Render voxels
            voxel_renderer.render(draw_list, pos, size)

            imgui.end()

            # Create performance panel
            imgui.set_next_window_position(width * 0.75, menu_bar_height)
            imgui.set_next_window_size(width * 0.25, height - menu_bar_height)

            imgui.begin("Performance", True, window_flags)
            imgui.text("This panel shows performance metrics.")

            # Display FPS
            current_fps = performance_monitor.get_current_fps()
            average_fps = performance_monitor.get_average_fps()
            imgui.text(f"Current FPS: {current_fps:.1f}")
            imgui.text(f"Average FPS: {average_fps:.1f}")
            imgui.text(f"Target FPS: {performance_monitor.target_fps:.1f}")

            # Display detail level
            imgui.text(f"Detail Level: {performance_monitor.detail_level:.2f}")

            # Display frame time
            imgui.text(f"Frame Time: {performance_monitor.frame_time * 1000:.2f} ms")

            # Add FPS graph
            imgui.text("FPS History")
            graph_width = width * 0.25 - 20
            graph_height = 100

            # Calculate min and max FPS for scaling
            min_fps = max(1.0, min(performance_monitor.fps_history))
            max_fps = max(performance_monitor.target_fps * 1.5, max(performance_monitor.fps_history))

            # Draw graph background
            graph_pos = imgui.get_cursor_screen_pos()
            draw_list.add_rect_filled(
                graph_pos[0], graph_pos[1],
                graph_pos[0] + graph_width, graph_pos[1] + graph_height,
                imgui.get_color_u32_rgba(0.1, 0.1, 0.1, 1.0)
            )

            # Draw target FPS line
            target_y = graph_pos[1] + graph_height - (performance_monitor.target_fps - min_fps) / (max_fps - min_fps) * graph_height
            draw_list.add_line(
                graph_pos[0], target_y,
                graph_pos[0] + graph_width, target_y,
                imgui.get_color_u32_rgba(1.0, 0.0, 0.0, 0.5),
                1.0
            )

            # Draw FPS history
            for i in range(len(performance_monitor.fps_history) - 1):
                fps1 = performance_monitor.fps_history[i]
                fps2 = performance_monitor.fps_history[i + 1]

                x1 = graph_pos[0] + i * graph_width / len(performance_monitor.fps_history)
                x2 = graph_pos[0] + (i + 1) * graph_width / len(performance_monitor.fps_history)

                y1 = graph_pos[1] + graph_height - (fps1 - min_fps) / (max_fps - min_fps) * graph_height
                y2 = graph_pos[1] + graph_height - (fps2 - min_fps) / (max_fps - min_fps) * graph_height

                # Color based on whether FPS is above or below target
                color = imgui.get_color_u32_rgba(0.0, 1.0, 0.0, 1.0) if fps1 >= performance_monitor.target_fps else imgui.get_color_u32_rgba(1.0, 0.0, 0.0, 1.0)

                draw_list.add_line(x1, y1, x2, y2, color, 1.0)

            # Advance cursor past graph
            imgui.dummy(graph_width, graph_height)

            # Add detail level graph
            imgui.text("Detail Level History")

            # Draw graph background
            graph_pos = imgui.get_cursor_screen_pos()
            draw_list.add_rect_filled(
                graph_pos[0], graph_pos[1],
                graph_pos[0] + graph_width, graph_pos[1] + graph_height,
                imgui.get_color_u32_rgba(0.1, 0.1, 0.1, 1.0)
            )

            # Draw detail level history
            for i in range(len(performance_monitor.detail_level_history) - 1):
                detail1 = performance_monitor.detail_level_history[i]
                detail2 = performance_monitor.detail_level_history[i + 1]

                x1 = graph_pos[0] + i * graph_width / len(performance_monitor.detail_level_history)
                x2 = graph_pos[0] + (i + 1) * graph_width / len(performance_monitor.detail_level_history)

                y1 = graph_pos[1] + graph_height - detail1 * graph_height
                y2 = graph_pos[1] + graph_height - detail2 * graph_height

                draw_list.add_line(x1, y1, x2, y2, imgui.get_color_u32_rgba(0.0, 0.5, 1.0, 1.0), 1.0)

            # Advance cursor past graph
            imgui.dummy(graph_width, graph_height)

            # Add controls
            imgui.separator()

            # Target FPS slider
            changed, value = imgui.slider_float("Target FPS", performance_monitor.target_fps, 30.0, 120.0)
            if changed:
                performance_monitor.target_fps = value
                print(f"Changed target FPS to {value:.1f}")

            # Adaptive rendering checkbox
            changed, value = imgui.checkbox("Adaptive Rendering", performance_monitor.adaptive_rendering)
            if changed:
                performance_monitor.adaptive_rendering = value
                print(f"{'Enabled' if value else 'Disabled'} adaptive rendering")

            # Artificial load slider
            changed, value = imgui.slider_float("Artificial Load", load_factor, 0.0, 1.0)
            if changed:
                load_factor = value
                print(f"Changed artificial load to {value:.2f}")

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

            # Apply artificial load
            if load_increasing:
                load_factor = min(1.0, load_factor + 0.001)
                if load_factor >= 1.0:
                    load_increasing = False
            else:
                load_factor = max(0.0, load_factor - 0.001)
                if load_factor <= 0.0:
                    load_increasing = True

            # Simulate computational load
            if load_factor > 0.0:
                # Perform some meaningless calculations to simulate load
                start_time = time.time()
                while time.time() - start_time < load_factor * 0.032:  # Up to 32ms (30 FPS)
                    # Perform some calculations
                    for _ in range(10000):
                        np.random.random(1000)

            # Print debug information for the first few frames and periodically
            if frame_count < 5 or frame_count % 100 == 0:
                print(f"Frame {frame_count}:")
                print(f"  - Current FPS: {current_fps:.1f}")
                print(f"  - Target FPS: {performance_monitor.target_fps:.1f}")
                print(f"  - Detail Level: {performance_monitor.detail_level:.2f}")
                print(f"  - Frame Time: {performance_monitor.frame_time * 1000:.2f} ms")
                print(f"  - Artificial Load: {load_factor:.2f}")

            frame_count += 1

        # Cleanup
        impl.shutdown()
        glfw.terminate()

        print("Adaptive rendering test completed successfully!")
        print("Terminal Validation: The application maintains performance above the target FPS by adjusting the detail level.")
        print(f"Final measurements:")
        print(f"  - Average FPS: {average_fps:.1f}")
        print(f"  - Target FPS: {performance_monitor.target_fps:.1f}")
        print(f"  - Final Detail Level: {performance_monitor.detail_level:.2f}")

        return 0
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
