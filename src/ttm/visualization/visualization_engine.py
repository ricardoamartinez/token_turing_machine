import pyglet
from pyglet.gl import *
import numpy as np
import math
import time
import imgui
from .imgui_renderer import CustomPygletRenderer
from typing import Dict, List, Tuple, Optional, Any

from .voxel_renderer import VoxelRenderer
from .vis_mapper import create_mapper_for_state
from .visualization_manager import VisualizationManager

class VisualizationEngine(pyglet.window.Window):
    """
    High-performance Pyglet/OpenGL visualization engine for TTM.
    Renders voxel-based visualizations of model states using instanced rendering.
    """
    def __init__(self, width=1280, height=720, caption='TTM Visualization Engine', resizable=True):
        super().__init__(width, height, caption=caption, resizable=resizable)
        self.setup_opengl()
        print(f"OpenGL Version: {self.context.get_info().get_version()}")

        # Get GLSL version
        try:
            glsl_version = glGetString(GL_SHADING_LANGUAGE_VERSION)
            if glsl_version:
                if hasattr(glsl_version, 'decode'):
                    print(f"GLSL Version: {glsl_version.decode('utf-8')}")
                else:
                    print(f"GLSL Version: {glsl_version}")
            else:
                print("GLSL Version: Unknown")
        except Exception as e:
            print(f"Error getting GLSL version: {e}")

        # Initialize camera parameters
        self.camera_position = np.array([0.0, 0.0, 5.0], dtype=np.float32)
        self.camera_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.camera_fov = 45.0
        self.camera_near = 0.1
        self.camera_far = 100.0

        # Initialize picking parameters
        self.selected_voxel = None
        self.hovered_voxel = None
        self.tooltip_visible = False
        self.tooltip_text = ""
        self.tooltip_position = (0, 0)

        # Initialize ImGui
        try:
            self.init_imgui()
            self.imgui_enabled = True
        except Exception as e:
            print(f"Error initializing ImGui: {e}")
            self.imgui_enabled = False

        # Initialize state editing parameters
        self.editing_voxel = None
        self.editing_value = 0.0
        self.editing_window_open = False

        # Initialize timeline parameters
        self.timeline_window_open = True
        self.current_epoch = 0
        self.current_batch = 0
        self.current_token = 0
        self.max_epoch = 0
        self.max_batch = 0
        self.max_token = 0
        self.playing = False
        self.playback_speed = 1.0
        self.last_playback_time = 0.0

        # Initialize performance monitoring parameters
        self.performance_window_open = True
        self.fps_history = [60.0] * 100  # Store last 100 FPS values
        self.fps_history_index = 0
        self.frame_time = 0.0
        self.frame_count = 0
        self.last_fps_update_time = time.time()
        self.target_fps = 60.0
        self.adaptive_rendering = True
        self.detail_level = 1.0  # 1.0 = full detail, 0.0 = no detail

        # Initialize matrices
        self.model_matrix = np.identity(4, dtype=np.float32)
        self.view_matrix = self._get_view_matrix()
        self.projection_matrix = self._get_projection_matrix()

        # Initialize voxel renderer
        self.voxel_renderer = VoxelRenderer(max_voxels=100000)

        # Initialize visualization manager
        self.visualization_manager = VisualizationManager(self.voxel_renderer)

        # Initialize demo voxels
        self._init_demo_voxels()

        # Set up keyboard and mouse handlers
        self.keys = pyglet.window.key.KeyStateHandler()
        self.push_handlers(self.keys)

        # Set up clock for animation
        pyglet.clock.schedule_interval(self.update, 1/60.0)

    def setup_opengl(self):
        """Sets up basic OpenGL states."""
        # Set the background color to true black (RGBA)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        # Enable depth testing for 3D rendering
        glEnable(GL_DEPTH_TEST)
        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        # Enable backface culling
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)

    def init_imgui(self):
        """Initialize Dear ImGui."""
        # Create ImGui context
        imgui.create_context()

        # Configure ImGui style
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

        # Create custom Pyglet renderer
        self.imgui_renderer = CustomPygletRenderer(self)

        # Set up ImGui IO
        io = imgui.get_io()
        io.display_size = self.width, self.height

    def _init_demo_voxels(self):
        """Initialize demo voxels for testing."""
        # Create a grid of voxels
        grid_size = 10
        for x in range(grid_size):
            for y in range(grid_size):
                for z in range(grid_size):
                    # Calculate position
                    position = np.array([
                        (x - grid_size/2 + 0.5) * 0.2,
                        (y - grid_size/2 + 0.5) * 0.2,
                        (z - grid_size/2 + 0.5) * 0.2
                    ], dtype=np.float32)

                    # Calculate color based on position
                    color = np.array([
                        x / grid_size,
                        y / grid_size,
                        z / grid_size,
                        0.8  # Alpha
                    ], dtype=np.float32)

                    # Calculate scale
                    scale = np.array([0.1, 0.1, 0.1], dtype=np.float32)

                    # Calculate value
                    value = (x + y + z) / (3 * grid_size)

                    # Set voxel
                    index = x * grid_size * grid_size + y * grid_size + z
                    self.voxel_renderer.set_voxel(index, position, scale, color, value)

    def _get_view_matrix(self) -> np.ndarray:
        """Calculate view matrix based on camera parameters."""
        # Calculate camera axes
        forward = self.camera_target - self.camera_position
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, self.camera_up)
        right = right / np.linalg.norm(right)

        up = np.cross(right, forward)

        # Create view matrix
        view_matrix = np.identity(4, dtype=np.float32)

        view_matrix[0, 0] = right[0]
        view_matrix[0, 1] = right[1]
        view_matrix[0, 2] = right[2]

        view_matrix[1, 0] = up[0]
        view_matrix[1, 1] = up[1]
        view_matrix[1, 2] = up[2]

        view_matrix[2, 0] = -forward[0]
        view_matrix[2, 1] = -forward[1]
        view_matrix[2, 2] = -forward[2]

        view_matrix[0, 3] = -np.dot(right, self.camera_position)
        view_matrix[1, 3] = -np.dot(up, self.camera_position)
        view_matrix[2, 3] = np.dot(forward, self.camera_position)

        return view_matrix

    def _get_projection_matrix(self) -> np.ndarray:
        """Calculate projection matrix based on camera parameters."""
        aspect = self.width / self.height
        f = 1.0 / math.tan(math.radians(self.camera_fov) / 2.0)

        projection_matrix = np.zeros((4, 4), dtype=np.float32)

        projection_matrix[0, 0] = f / aspect
        projection_matrix[1, 1] = f
        projection_matrix[2, 2] = (self.camera_far + self.camera_near) / (self.camera_near - self.camera_far)
        projection_matrix[2, 3] = (2.0 * self.camera_far * self.camera_near) / (self.camera_near - self.camera_far)
        projection_matrix[3, 2] = -1.0

        return projection_matrix

    def on_draw(self):
        """Called by pyglet to draw the window contents."""
        self.clear()

        # Render voxels
        self.voxel_renderer.render(
            self.model_matrix,
            self.view_matrix,
            self.projection_matrix
        )

        # Draw tooltip if visible and not editing
        if self.tooltip_visible and self.tooltip_text and not self.editing_window_open:
            # Create a label for the tooltip
            x, y = self.tooltip_position
            label = pyglet.text.Label(
                self.tooltip_text,
                font_name='Arial',
                font_size=10,
                x=x + 15,  # Offset from cursor
                y=y - 15,
                anchor_x='left',
                anchor_y='top',
                color=(255, 255, 255, 255),
                multiline=True,
                width=300
            )

            # Draw a background for the tooltip
            background_width = label.content_width + 10
            background_height = label.content_height + 10
            pyglet.graphics.draw(4, GL_QUADS,
                ('v2f', (
                    x + 10, y - 10,
                    x + 10 + background_width, y - 10,
                    x + 10 + background_width, y - 10 - background_height,
                    x + 10, y - 10 - background_height
                )),
                ('c4B', (0, 0, 0, 200) * 4)
            )

            # Draw the label
            label.draw()

        # Render ImGui UI
        imgui.new_frame()
        self._render_imgui()
        imgui.render()
        self.imgui_renderer.render(imgui.get_draw_data())

    def _render_imgui(self):
        """Render ImGui UI."""
        # Render state editing window if open
        if self.editing_window_open and self.editing_voxel is not None:
            self._render_state_editing_window()

        # Render timeline window if open
        if self.timeline_window_open:
            self._render_timeline_window()

        # Render performance monitoring window if open
        if self.performance_window_open:
            self._render_performance_window()

    def _render_performance_window(self):
        """Render the performance monitoring window."""
        # Set window position and size
        window_width = 300
        window_height = 200
        window_x = self.width - window_width - 10
        window_y = self.height - window_height - 120  # Position below timeline window
        imgui.set_next_window_position(window_x, window_y)
        imgui.set_next_window_size(window_width, window_height)

        # Begin window
        expanded, open = imgui.begin("Performance", True, imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE)
        if not open:
            self.performance_window_open = False
            imgui.end()
            return

        # Display current FPS
        current_fps = self.fps_history[self.fps_history_index]
        imgui.text(f"FPS: {current_fps:.1f}")

        # Display target FPS
        imgui.text(f"Target FPS: {self.target_fps:.1f}")

        # Display detail level
        imgui.text(f"Detail Level: {self.detail_level:.2f}")

        # Display frame time
        imgui.text(f"Frame Time: {self.frame_time * 1000:.2f} ms")

        # FPS history graph
        imgui.text("FPS History")
        graph_width = window_width - 20
        graph_height = 80

        # Calculate min and max FPS for scaling
        min_fps = min(self.fps_history)
        max_fps = max(self.fps_history)
        if max_fps == min_fps:
            max_fps = min_fps + 1.0

        # Draw graph background
        draw_list = imgui.get_window_draw_list()
        pos = imgui.get_cursor_screen_pos()
        draw_list.add_rect_filled(
            pos[0], pos[1],
            pos[0] + graph_width, pos[1] + graph_height,
            imgui.get_color_u32_rgba(0.1, 0.1, 0.1, 1.0)
        )

        # Draw target FPS line
        target_y = pos[1] + graph_height - (self.target_fps - min_fps) / (max_fps - min_fps) * graph_height
        draw_list.add_line(
            pos[0], target_y,
            pos[0] + graph_width, target_y,
            imgui.get_color_u32_rgba(1.0, 0.0, 0.0, 0.5),
            1.0
        )

        # Draw FPS history
        for i in range(len(self.fps_history) - 1):
            fps1 = self.fps_history[i]
            fps2 = self.fps_history[i + 1]
            x1 = pos[0] + i * graph_width / len(self.fps_history)
            x2 = pos[0] + (i + 1) * graph_width / len(self.fps_history)
            y1 = pos[1] + graph_height - (fps1 - min_fps) / (max_fps - min_fps) * graph_height
            y2 = pos[1] + graph_height - (fps2 - min_fps) / (max_fps - min_fps) * graph_height

            # Color based on whether FPS is above or below target
            color = imgui.get_color_u32_rgba(0.0, 1.0, 0.0, 1.0) if fps1 >= self.target_fps else imgui.get_color_u32_rgba(1.0, 0.0, 0.0, 1.0)

            draw_list.add_line(x1, y1, x2, y2, color, 1.0)

        # Advance cursor past graph
        imgui.dummy(graph_width, graph_height)

        # Adaptive rendering toggle
        changed, value = imgui.checkbox("Adaptive Rendering", self.adaptive_rendering)
        if changed:
            self.adaptive_rendering = value

        # Target FPS slider
        changed, value = imgui.slider_float("Target FPS", self.target_fps, 30.0, 120.0)
        if changed:
            self.target_fps = value

        # End window
        imgui.end()

    def _render_timeline_window(self):
        """Render the timeline window."""
        # Set window position and size
        window_width = self.width - 20
        window_height = 100
        window_x = 10
        window_y = 10
        imgui.set_next_window_position(window_x, window_y)
        imgui.set_next_window_size(window_width, window_height)

        # Begin window
        expanded, open = imgui.begin("Timeline", True, imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE)
        if not open:
            self.timeline_window_open = False
            imgui.end()
            return

        # Get available epochs, batches, and tokens from visualization manager
        epochs = self.visualization_manager.get_available_epochs()
        batches = self.visualization_manager.get_available_batches(self.current_epoch)
        tokens = self.visualization_manager.get_available_tokens(self.current_epoch, self.current_batch)

        # Update max values
        if epochs:
            self.max_epoch = max(epochs)
        if batches:
            self.max_batch = max(batches)
        if tokens:
            self.max_token = max(tokens)

        # Display current state
        imgui.text(f"Epoch: {self.current_epoch}, Batch: {self.current_batch}, Token: {self.current_token}")

        # Epoch slider
        if epochs:
            changed, value = imgui.slider_int("Epoch", self.current_epoch, min(epochs), max(epochs))
            if changed:
                self.current_epoch = value
                # Reset batch and token when epoch changes
                batches = self.visualization_manager.get_available_batches(self.current_epoch)
                if batches:
                    self.current_batch = min(batches)
                    tokens = self.visualization_manager.get_available_tokens(self.current_epoch, self.current_batch)
                    if tokens:
                        self.current_token = min(tokens)
                # Load state for new epoch/batch/token
                self.visualization_manager.load_state(self.current_epoch, self.current_batch, self.current_token)

        # Batch slider
        if batches:
            changed, value = imgui.slider_int("Batch", self.current_batch, min(batches), max(batches))
            if changed:
                self.current_batch = value
                # Reset token when batch changes
                tokens = self.visualization_manager.get_available_tokens(self.current_epoch, self.current_batch)
                if tokens:
                    self.current_token = min(tokens)
                # Load state for new batch/token
                self.visualization_manager.load_state(self.current_epoch, self.current_batch, self.current_token)

        # Token slider
        if tokens:
            changed, value = imgui.slider_int("Token", self.current_token, min(tokens), max(tokens))
            if changed:
                self.current_token = value
                # Load state for new token
                self.visualization_manager.load_state(self.current_epoch, self.current_batch, self.current_token)

        # Playback controls
        imgui.separator()

        # Play/pause button
        if self.playing:
            if imgui.button("Pause", 80, 30):
                self.playing = False
        else:
            if imgui.button("Play", 80, 30):
                self.playing = True
                self.last_playback_time = time.time()

        imgui.same_line()

        # Step backward button
        if imgui.button("<<", 40, 30):
            self._step_backward()

        imgui.same_line()

        # Step forward button
        if imgui.button(">>", 40, 30):
            self._step_forward()

        imgui.same_line()

        # Playback speed slider
        changed, value = imgui.slider_float("Speed", self.playback_speed, 0.1, 5.0)
        if changed:
            self.playback_speed = value

        # End window
        imgui.end()

    def _render_state_editing_window(self):
        """Render the state editing window."""
        # Get voxel data
        voxel_data = self.voxel_renderer.get_voxel_data(self.editing_voxel)
        if voxel_data is None:
            self.editing_window_open = False
            return

        # Get state name
        state_name = self.visualization_manager.get_state_name_for_voxel(self.editing_voxel)
        if state_name is None:
            state_name = "Unknown"

        # Set window position and size
        window_width = 300
        window_height = 200
        window_x = self.width - window_width - 10
        window_y = self.height - window_height - 10
        imgui.set_next_window_position(window_x, window_y)
        imgui.set_next_window_size(window_width, window_height)

        # Begin window
        expanded, open = imgui.begin(f"Edit State: {state_name}", True)
        if not open:
            self.editing_window_open = False
            imgui.end()
            return

        # Display voxel information
        imgui.text(f"Voxel Index: {self.editing_voxel}")
        imgui.text(f"Position: {voxel_data['position']}")
        imgui.text(f"Scale: {voxel_data['scale']}")
        imgui.text(f"Color: {voxel_data['color']}")

        # Edit value
        changed, value = imgui.slider_float("Value", voxel_data['value'], 0.0, 1.0)
        if changed:
            # Update voxel value
            self.editing_value = value

            # Update voxel in renderer
            self.voxel_renderer.set_voxel(
                self.editing_voxel,
                voxel_data['position'],
                voxel_data['scale'],
                voxel_data['color'],
                value
            )

            # Update state in visualization manager
            self.visualization_manager.update_state_value(state_name, value)

            # Print debug information
            print(f"Updated voxel {self.editing_voxel} value to {value}")

        # Add apply and cancel buttons
        if imgui.button("Apply", 100, 30):
            # Apply changes to the state
            self.visualization_manager.apply_state_changes(state_name)
            self.editing_window_open = False

        imgui.same_line()

        if imgui.button("Cancel", 100, 30):
            # Revert changes
            self.visualization_manager.revert_state_changes(state_name)
            self.editing_window_open = False

        # End window
        imgui.end()

    def on_resize(self, width, height):
        """Called when the window is resized."""
        super().on_resize(width, height)

        # Update viewport
        glViewport(0, 0, width, height)

        # Update projection matrix
        self.projection_matrix = self._get_projection_matrix()

        # Update ImGui display size
        io = imgui.get_io()
        io.display_size = width, height

    def on_mouse_motion(self, x, y, dx, dy):
        """Called when the mouse is moved."""
        # Skip if ImGui is capturing mouse
        io = imgui.get_io()
        if io.want_capture_mouse:
            return

        # Update tooltip position
        self.tooltip_position = (x, y)

        # Skip voxel picking if editing window is open
        if self.editing_window_open:
            return

        # Perform ray casting for voxel picking
        voxel_index = self._pick_voxel(x, y)

        # Update hovered voxel
        if voxel_index != self.hovered_voxel:
            self.hovered_voxel = voxel_index

            # Update tooltip
            if voxel_index is not None:
                # Get voxel data
                voxel_data = self.voxel_renderer.get_voxel_data(voxel_index)

                # Get state name from visualization manager
                state_name = self.visualization_manager.get_state_name_for_voxel(voxel_index)

                # Create tooltip text
                self.tooltip_text = f"Voxel: {voxel_index}\n"
                self.tooltip_text += f"State: {state_name}\n"
                self.tooltip_text += f"Position: {voxel_data['position']}\n"
                self.tooltip_text += f"Color: {voxel_data['color']}\n"
                self.tooltip_text += f"Value: {voxel_data['value']:.4f}"
                self.tooltip_text += f"\n\nClick to edit"

                # Show tooltip
                self.tooltip_visible = True
            else:
                # Hide tooltip
                self.tooltip_visible = False

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        """Called when the mouse is dragged."""
        # Skip if ImGui is capturing mouse
        io = imgui.get_io()
        if io.want_capture_mouse:
            return

        # Update tooltip position
        self.tooltip_position = (x, y)

        # Rotate camera around target
        if buttons & pyglet.window.mouse.LEFT:
            # Convert pixel movement to rotation angles
            yaw = dx * 0.01
            pitch = dy * 0.01

            # Rotate camera position around target
            # First, translate to origin
            position = self.camera_position - self.camera_target

            # Rotate around y-axis (yaw)
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            position_yaw = np.array([
                position[0] * cos_yaw + position[2] * sin_yaw,
                position[1],
                -position[0] * sin_yaw + position[2] * cos_yaw
            ])

            # Rotate around x-axis (pitch)
            cos_pitch = math.cos(pitch)
            sin_pitch = math.sin(pitch)
            position_pitch = np.array([
                position_yaw[0],
                position_yaw[1] * cos_pitch - position_yaw[2] * sin_pitch,
                position_yaw[1] * sin_pitch + position_yaw[2] * cos_pitch
            ])

            # Translate back
            self.camera_position = position_pitch + self.camera_target

            # Update view matrix
            self.view_matrix = self._get_view_matrix()

            # Hide tooltip while dragging
            self.tooltip_visible = False

    def on_mouse_press(self, x, y, button, modifiers):
        """Called when a mouse button is pressed."""
        # Skip if ImGui is capturing mouse
        io = imgui.get_io()
        if io.want_capture_mouse:
            return

        if button == pyglet.window.mouse.LEFT:
            # Perform ray casting for voxel picking
            voxel_index = self._pick_voxel(x, y)

            # Update selected voxel
            if voxel_index != self.selected_voxel:
                self.selected_voxel = voxel_index

                # Highlight selected voxel
                if voxel_index is not None:
                    # Get voxel data
                    voxel_data = self.voxel_renderer.get_voxel_data(voxel_index)

                    # Highlight by changing color
                    highlight_color = np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32)  # Yellow
                    self.voxel_renderer.set_voxel(
                        voxel_index,
                        voxel_data['position'],
                        voxel_data['scale'] * 1.2,  # Increase size
                        highlight_color,
                        voxel_data['value']
                    )

                    # Open editing window
                    self.editing_voxel = voxel_index
                    self.editing_value = voxel_data['value']
                    self.editing_window_open = True

                    # Hide tooltip while editing
                    self.tooltip_visible = False

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        """Called when the mouse wheel is scrolled."""
        # Skip if ImGui is capturing mouse
        io = imgui.get_io()
        if io.want_capture_mouse:
            return

        # Zoom camera
        zoom_factor = 1.1 ** scroll_y

        # Calculate new position
        direction = self.camera_position - self.camera_target
        new_direction = direction * zoom_factor

        # Update camera position
        self.camera_position = self.camera_target + new_direction

        # Update view matrix
        self.view_matrix = self._get_view_matrix()

    def _step_forward(self) -> None:
        """Step forward in the timeline."""
        # Get available epochs, batches, and tokens
        epochs = self.visualization_manager.get_available_epochs()
        batches = self.visualization_manager.get_available_batches(self.current_epoch)
        tokens = self.visualization_manager.get_available_tokens(self.current_epoch, self.current_batch)

        if not epochs or not batches or not tokens:
            return

        # Try to step forward in tokens
        if self.current_token < max(tokens):
            self.current_token += 1
        # If at the end of tokens, try to step forward in batches
        elif self.current_batch < max(batches):
            self.current_batch += 1
            # Reset token to the beginning of the new batch
            tokens = self.visualization_manager.get_available_tokens(self.current_epoch, self.current_batch)
            if tokens:
                self.current_token = min(tokens)
        # If at the end of batches, try to step forward in epochs
        elif self.current_epoch < max(epochs):
            self.current_epoch += 1
            # Reset batch and token to the beginning of the new epoch
            batches = self.visualization_manager.get_available_batches(self.current_epoch)
            if batches:
                self.current_batch = min(batches)
                tokens = self.visualization_manager.get_available_tokens(self.current_epoch, self.current_batch)
                if tokens:
                    self.current_token = min(tokens)

        # Load state for new epoch/batch/token
        self.visualization_manager.load_state(self.current_epoch, self.current_batch, self.current_token)

    def _step_backward(self) -> None:
        """Step backward in the timeline."""
        # Get available epochs, batches, and tokens
        epochs = self.visualization_manager.get_available_epochs()
        batches = self.visualization_manager.get_available_batches(self.current_epoch)
        tokens = self.visualization_manager.get_available_tokens(self.current_epoch, self.current_batch)

        if not epochs or not batches or not tokens:
            return

        # Try to step backward in tokens
        if self.current_token > min(tokens):
            self.current_token -= 1
        # If at the beginning of tokens, try to step backward in batches
        elif self.current_batch > min(batches):
            self.current_batch -= 1
            # Reset token to the end of the previous batch
            tokens = self.visualization_manager.get_available_tokens(self.current_epoch, self.current_batch)
            if tokens:
                self.current_token = max(tokens)
        # If at the beginning of batches, try to step backward in epochs
        elif self.current_epoch > min(epochs):
            self.current_epoch -= 1
            # Reset batch and token to the end of the previous epoch
            batches = self.visualization_manager.get_available_batches(self.current_epoch)
            if batches:
                self.current_batch = max(batches)
                tokens = self.visualization_manager.get_available_tokens(self.current_epoch, self.current_batch)
                if tokens:
                    self.current_token = max(tokens)

        # Load state for new epoch/batch/token
        self.visualization_manager.load_state(self.current_epoch, self.current_batch, self.current_token)

    def update(self, dt):
        """Update the scene."""
        # Update frame time and FPS
        self.frame_time = dt
        self.frame_count += 1

        current_time = time.time()
        elapsed_time = current_time - self.last_fps_update_time

        # Update FPS every 0.5 seconds
        if elapsed_time >= 0.5:
            fps = self.frame_count / elapsed_time
            self.fps_history[self.fps_history_index] = fps
            self.fps_history_index = (self.fps_history_index + 1) % len(self.fps_history)
            self.frame_count = 0
            self.last_fps_update_time = current_time

            # Implement adaptive rendering
            if self.adaptive_rendering:
                current_fps = self.fps_history[self.fps_history_index]

                # If FPS is below target, reduce detail level
                if current_fps < self.target_fps and self.detail_level > 0.1:
                    self.detail_level = max(0.1, self.detail_level - 0.05)
                    print(f"Reducing detail level to {self.detail_level:.2f} (FPS: {current_fps:.1f})")

                    # Apply detail level to visualization manager
                    self.visualization_manager.set_detail_level(self.detail_level)

                # If FPS is above target, increase detail level
                elif current_fps > self.target_fps * 1.2 and self.detail_level < 1.0:
                    self.detail_level = min(1.0, self.detail_level + 0.05)
                    print(f"Increasing detail level to {self.detail_level:.2f} (FPS: {current_fps:.1f})")

                    # Apply detail level to visualization manager
                    self.visualization_manager.set_detail_level(self.detail_level)

        # Handle keyboard input
        self._handle_keyboard_input(dt)

        # Update visualization manager
        self.visualization_manager.update()

        # Update playback
        if self.playing:
            current_time = time.time()
            elapsed_time = current_time - self.last_playback_time

            # Step forward based on playback speed
            if elapsed_time >= 1.0 / self.playback_speed:
                self._step_forward()
                self.last_playback_time = current_time

        # Rotate model
        angle = 15.0 * dt  # 15 degrees per second
        rotation_matrix = np.identity(4, dtype=np.float32)

        # Rotate around y-axis
        cos_angle = math.cos(math.radians(angle))
        sin_angle = math.sin(math.radians(angle))

        rotation_matrix[0, 0] = cos_angle
        rotation_matrix[0, 2] = sin_angle
        rotation_matrix[2, 0] = -sin_angle
        rotation_matrix[2, 2] = cos_angle

        # Apply rotation
        self.model_matrix = np.matmul(rotation_matrix, self.model_matrix)

    def _pick_voxel(self, x: int, y: int) -> Optional[int]:
        """Perform ray casting for voxel picking.

        Args:
            x: Mouse x coordinate
            y: Mouse y coordinate

        Returns:
            Voxel index if a voxel is picked, None otherwise
        """
        # Convert screen coordinates to normalized device coordinates
        # Note: Pyglet's y-coordinate is from bottom to top
        ndcX = 2.0 * x / self.width - 1.0
        ndcY = 2.0 * y / self.height - 1.0

        # Create a ray in clip space
        ray_clip = np.array([ndcX, ndcY, -1.0, 1.0], dtype=np.float32)

        # Convert to eye space
        inv_projection = np.linalg.inv(self.projection_matrix)
        ray_eye = np.dot(inv_projection, ray_clip)
        ray_eye = np.array([ray_eye[0], ray_eye[1], -1.0, 0.0], dtype=np.float32)

        # Convert to world space
        inv_view = np.linalg.inv(self.view_matrix)
        ray_world = np.dot(inv_view, ray_eye)
        ray_world = np.array([ray_world[0], ray_world[1], ray_world[2]], dtype=np.float32)
        ray_world = ray_world / np.linalg.norm(ray_world)

        # Ray origin is the camera position
        ray_origin = self.camera_position

        # Check for intersection with voxels
        closest_voxel = None
        closest_distance = float('inf')

        # Get active voxels from the renderer
        for voxel_index in range(self.voxel_renderer.num_active_voxels):
            # Get voxel data
            voxel_data = self.voxel_renderer.get_voxel_data(voxel_index)

            # Skip if voxel is not active
            if voxel_data is None:
                continue

            # Get voxel position and scale
            voxel_position = voxel_data['position']
            voxel_scale = voxel_data['scale']

            # Apply model matrix to voxel position
            voxel_position_world = np.dot(self.model_matrix, np.append(voxel_position, 1.0))[:3]

            # Calculate voxel bounds
            voxel_min = voxel_position_world - voxel_scale * 0.5
            voxel_max = voxel_position_world + voxel_scale * 0.5

            # Check for ray-box intersection
            t_min = float('-inf')
            t_max = float('inf')

            for i in range(3):
                if abs(ray_world[i]) < 1e-6:
                    # Ray is parallel to slab, check if ray origin is inside slab
                    if ray_origin[i] < voxel_min[i] or ray_origin[i] > voxel_max[i]:
                        # Ray is outside slab, no intersection
                        break
                else:
                    # Calculate intersection with slab
                    t1 = (voxel_min[i] - ray_origin[i]) / ray_world[i]
                    t2 = (voxel_max[i] - ray_origin[i]) / ray_world[i]

                    # Ensure t1 <= t2
                    if t1 > t2:
                        t1, t2 = t2, t1

                    # Update t_min and t_max
                    t_min = max(t_min, t1)
                    t_max = min(t_max, t2)

                    if t_min > t_max:
                        # Ray misses the box
                        break
            else:
                # Ray intersects the box
                if t_min > 0 and t_min < closest_distance:
                    closest_distance = t_min
                    closest_voxel = voxel_index

        return closest_voxel

    def _handle_keyboard_input(self, dt):
        """Handle keyboard input."""
        # Camera movement speed
        speed = 2.0 * dt

        # Calculate camera axes
        forward = self.camera_target - self.camera_position
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, self.camera_up)
        right = right / np.linalg.norm(right)

        up = np.cross(right, forward)

        # Move camera
        if self.keys[pyglet.window.key.W]:
            self.camera_position += forward * speed
            self.camera_target += forward * speed

        if self.keys[pyglet.window.key.S]:
            self.camera_position -= forward * speed
            self.camera_target -= forward * speed

        if self.keys[pyglet.window.key.A]:
            self.camera_position -= right * speed
            self.camera_target -= right * speed

        if self.keys[pyglet.window.key.D]:
            self.camera_position += right * speed
            self.camera_target += right * speed

        if self.keys[pyglet.window.key.Q]:
            self.camera_position -= up * speed
            self.camera_target -= up * speed

        if self.keys[pyglet.window.key.E]:
            self.camera_position += up * speed
            self.camera_target += up * speed

        # Update view matrix if camera moved
        if any([self.keys[k] for k in [pyglet.window.key.W, pyglet.window.key.S,
                                     pyglet.window.key.A, pyglet.window.key.D,
                                     pyglet.window.key.Q, pyglet.window.key.E]]):
            self.view_matrix = self._get_view_matrix()

    def load_state_history(self, filepath: str) -> None:
        """Load state history from a file.

        Args:
            filepath: Path to the state history file
        """
        self.visualization_manager.load_state_history(filepath)

    def run(self):
        """Starts the pyglet event loop."""
        pyglet.app.run()

    def cleanup(self):
        """Clean up OpenGL resources."""
        # Clear visualization manager
        self.visualization_manager.clear()

        # Clean up voxel renderer
        self.voxel_renderer.cleanup()

        # Clean up ImGui
        if hasattr(self, 'imgui_renderer'):
            self.imgui_renderer.shutdown()

if __name__ == '__main__':
    # Example usage: Create and run the engine
    engine = VisualizationEngine()
    try:
        engine.run()
    finally:
        engine.cleanup()
