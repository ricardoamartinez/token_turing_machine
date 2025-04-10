import pyglet
from pyglet.gl import *
import numpy as np
import math
from typing import Dict, List, Tuple, Optional

from .voxel_renderer import VoxelRenderer
from .vis_mapper import create_mapper_for_state

class VisualizationEngine(pyglet.window.Window):
    """
    High-performance Pyglet/OpenGL visualization engine for TTM.
    Renders voxel-based visualizations of model states using instanced rendering.
    """
    def __init__(self, width=1280, height=720, caption='TTM Visualization Engine', resizable=True):
        super().__init__(width, height, caption=caption, resizable=resizable)
        self.setup_opengl()
        print(f"OpenGL Version: {self.context.get_info().get_version()}")
        print(f"GLSL Version: {self.context.get_info().get_shading_language_version()}")

        # Initialize camera parameters
        self.camera_position = np.array([0.0, 0.0, 5.0], dtype=np.float32)
        self.camera_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.camera_fov = 45.0
        self.camera_near = 0.1
        self.camera_far = 100.0

        # Initialize matrices
        self.model_matrix = np.identity(4, dtype=np.float32)
        self.view_matrix = self._get_view_matrix()
        self.projection_matrix = self._get_projection_matrix()

        # Initialize voxel renderer
        self.voxel_renderer = VoxelRenderer(max_voxels=100000)

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

    def on_resize(self, width, height):
        """Called when the window is resized."""
        super().on_resize(width, height)

        # Update viewport
        glViewport(0, 0, width, height)

        # Update projection matrix
        self.projection_matrix = self._get_projection_matrix()

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        """Called when the mouse is dragged."""
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

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        """Called when the mouse wheel is scrolled."""
        # Zoom camera
        zoom_factor = 1.1 ** scroll_y

        # Calculate new position
        direction = self.camera_position - self.camera_target
        new_direction = direction * zoom_factor

        # Update camera position
        self.camera_position = self.camera_target + new_direction

        # Update view matrix
        self.view_matrix = self._get_view_matrix()

    def update(self, dt):
        """Update the scene."""
        # Handle keyboard input
        self._handle_keyboard_input(dt)

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

    def run(self):
        """Starts the pyglet event loop."""
        pyglet.app.run()

    def cleanup(self):
        """Clean up OpenGL resources."""
        self.voxel_renderer.cleanup()

if __name__ == '__main__':
    # Example usage: Create and run the engine
    engine = VisualizationEngine()
    try:
        engine.run()
    finally:
        engine.cleanup()
