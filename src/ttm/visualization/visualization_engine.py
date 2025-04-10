import pyglet
from pyglet.gl import *

class VisualizationEngine(pyglet.window.Window):
    """
    Basic Pyglet/OpenGL window for TTM visualization.
    Initializes a window with a black background and OpenGL context.
    """
    def __init__(self, width=1280, height=720, caption='TTM Visualization Engine', resizable=True):
        super().__init__(width, height, caption=caption, resizable=resizable)
        self.setup_opengl()
        print(f"OpenGL Version: {self.context.get_info().get_version()}")
        print(f"GLSL Version: {self.context.get_info().get_shading_language_version()}")

    def setup_opengl(self):
        """Sets up basic OpenGL states."""
        # Set the background color to true black (RGBA)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        # Enable depth testing for 3D rendering
        glEnable(GL_DEPTH_TEST)
        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def on_draw(self):
        """Called by pyglet to draw the window contents."""
        self.clear()
        # Future drawing code will go here
        # For now, just clears to black

    def run(self):
        """Starts the pyglet event loop."""
        pyglet.app.run()

if __name__ == '__main__':
    # Example usage: Create and run the engine
    engine = VisualizationEngine()
    engine.run()
