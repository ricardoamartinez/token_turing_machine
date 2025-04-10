"""
Official ImGui demo script.

This script runs the official ImGui demo to verify that ImGui is installed and working correctly.
"""

import sys
import imgui
import glfw
import OpenGL.GL as gl
from imgui.integrations.glfw import GlfwRenderer

def main():
    """Run the test."""
    try:
        # Initialize GLFW
        if not glfw.init():
            print("Could not initialize GLFW")
            return 1
        
        # Create a windowed mode window and its OpenGL context
        window = glfw.create_window(1280, 720, "Dear ImGui GLFW+OpenGL3 Demo", None, None)
        if not window:
            glfw.terminate()
            print("Could not create GLFW window")
            return 1
        
        # Make the window's context current
        glfw.make_context_current(window)
        
        # Initialize ImGui
        imgui.create_context()
        impl = GlfwRenderer(window)
        
        # Print ImGui version
        version = imgui.get_version()
        print(f"ImGui version: {version}")
        
        # Main loop
        while not glfw.window_should_close(window):
            # Poll for and process events
            glfw.poll_events()
            
            # Start new frame
            impl.process_inputs()
            imgui.new_frame()
            
            # Show demo window
            imgui.show_demo_window()
            
            # Render
            imgui.render()
            
            # Clear the framebuffer
            gl.glClearColor(0.1, 0.1, 0.1, 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            
            # Render ImGui
            impl.render(imgui.get_draw_data())
            
            # Swap front and back buffers
            glfw.swap_buffers(window)
        
        # Cleanup
        impl.shutdown()
        glfw.terminate()
        
        print("Dear ImGui is working correctly!")
        print("Terminal Validation: ImGui is properly installed and functioning.")
        
        return 0
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
