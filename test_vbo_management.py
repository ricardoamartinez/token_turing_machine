"""
Test script for dynamic VBO management.

This script tests that the VBO management is working correctly by updating voxel data
and measuring the number of updates.
"""

import os
import sys
import pyglet
from pyglet.gl import *
import numpy as np
import time

from src.ttm.visualization.voxel_renderer import VoxelRenderer

class TestWindow(pyglet.window.Window):
    """Test window for VBO management."""
    
    def __init__(self):
        """Initialize the test window."""
        super().__init__(width=800, height=600, caption="VBO Management Test", visible=True)
        
        # Initialize OpenGL
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_DEPTH_TEST)
        
        # Print OpenGL version
        print(f"OpenGL Version: {self.context.get_info().get_version()}")
        print(f"GLSL Version: {self.context.get_info().get_shading_language_version()}")
        
        # Create voxel renderer
        self.voxel_renderer = VoxelRenderer(max_voxels=1000)
        
        # Initialize matrices
        self.model_matrix = np.identity(4, dtype=np.float32)
        self.view_matrix = np.identity(4, dtype=np.float32)
        self.view_matrix[2, 3] = -5.0  # Move camera back
        self.projection_matrix = self._get_projection_matrix()
        
        # Initialize test data
        self._init_test_data()
        
        # Set up clock for animation
        pyglet.clock.schedule_interval(self.update, 1/60.0)
        
        # Initialize test metrics
        self.frame_count = 0
        self.update_count = 0
        self.start_time = time.time()
    
    def _get_projection_matrix(self) -> np.ndarray:
        """Calculate projection matrix."""
        aspect = self.width / self.height
        fov = 45.0
        near = 0.1
        far = 100.0
        
        f = 1.0 / np.tan(np.radians(fov) / 2.0)
        
        projection_matrix = np.zeros((4, 4), dtype=np.float32)
        
        projection_matrix[0, 0] = f / aspect
        projection_matrix[1, 1] = f
        projection_matrix[2, 2] = (far + near) / (near - far)
        projection_matrix[2, 3] = (2.0 * far * near) / (near - far)
        projection_matrix[3, 2] = -1.0
        
        return projection_matrix
    
    def _init_test_data(self):
        """Initialize test data."""
        # Create a grid of voxels
        grid_size = 10
        for x in range(grid_size):
            for y in range(grid_size):
                # Calculate position
                position = np.array([
                    (x - grid_size/2 + 0.5) * 0.2,
                    (y - grid_size/2 + 0.5) * 0.2,
                    0.0
                ], dtype=np.float32)
                
                # Calculate color
                color = np.array([
                    x / grid_size,
                    y / grid_size,
                    0.5,
                    1.0
                ], dtype=np.float32)
                
                # Calculate scale
                scale = np.array([0.1, 0.1, 0.1], dtype=np.float32)
                
                # Calculate value
                value = (x + y) / (2 * grid_size)
                
                # Set voxel
                index = x * grid_size + y
                self.voxel_renderer.set_voxel(index, position, scale, color, value)
    
    def on_draw(self):
        """Called by pyglet to draw the window contents."""
        self.clear()
        
        # Render voxels
        self.voxel_renderer.render(
            self.model_matrix,
            self.view_matrix,
            self.projection_matrix
        )
        
        # Update frame count
        self.frame_count += 1
    
    def update(self, dt):
        """Update the scene."""
        # Update test data
        self._update_test_data(dt)
        
        # Print metrics every 60 frames
        if self.frame_count % 60 == 0:
            elapsed_time = time.time() - self.start_time
            fps = self.frame_count / elapsed_time
            updates_per_frame = self.update_count / self.frame_count
            
            print(f"FPS: {fps:.2f}, Updates per frame: {updates_per_frame:.2f}")
            print(f"Total updates: {self.update_count}, Total frames: {self.frame_count}")
            
            # Reset metrics
            self.frame_count = 0
            self.update_count = 0
            self.start_time = time.time()
    
    def _update_test_data(self, dt):
        """Update test data."""
        # Update a subset of voxels each frame
        grid_size = 10
        num_updates = 10  # Update 10 voxels per frame
        
        for i in range(num_updates):
            # Choose a random voxel
            x = np.random.randint(0, grid_size)
            y = np.random.randint(0, grid_size)
            index = x * grid_size + y
            
            # Calculate new position with some animation
            position = np.array([
                (x - grid_size/2 + 0.5) * 0.2,
                (y - grid_size/2 + 0.5) * 0.2,
                np.sin(time.time() * 2.0 + x * 0.5 + y * 0.3) * 0.1
            ], dtype=np.float32)
            
            # Calculate new color with some animation
            color = np.array([
                (np.sin(time.time() + x * 0.2) + 1.0) * 0.5,
                (np.sin(time.time() + y * 0.2 + 2.0) + 1.0) * 0.5,
                (np.sin(time.time() * 0.5) + 1.0) * 0.5,
                1.0
            ], dtype=np.float32)
            
            # Calculate scale with some animation
            scale_factor = (np.sin(time.time() * 3.0 + x * 0.4 + y * 0.6) + 1.5) * 0.1
            scale = np.array([scale_factor, scale_factor, scale_factor], dtype=np.float32)
            
            # Calculate value
            value = (np.sin(time.time() + x * 0.1 + y * 0.1) + 1.0) * 0.5
            
            # Set voxel
            self.voxel_renderer.set_voxel(index, position, scale, color, value)
            
            # Update count
            self.update_count += 1
    
    def cleanup(self):
        """Clean up OpenGL resources."""
        self.voxel_renderer.cleanup()

def main():
    """Run the test."""
    # Create test window
    window = TestWindow()
    
    try:
        # Run the test
        pyglet.app.run()
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        # Clean up
        window.cleanup()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
