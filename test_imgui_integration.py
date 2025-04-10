"""
Test script for Dear ImGui integration.

This script tests the integration of Dear ImGui with our OpenGL context.
"""

import os
import sys
import pyglet
from pyglet.gl import *
import imgui
import ctypes
import numpy as np

class ImGuiPygletRenderer:
    """Custom ImGui renderer for Pyglet."""
    
    def __init__(self, window):
        """Initialize the renderer.
        
        Args:
            window: Pyglet window
        """
        self.window = window
        self._font_texture = None
        self._shader_program = None
        self._vao = None
        self._vbo = None
        self._ebo = None
        self._attrib_location_tex = None
        self._attrib_location_proj_mtx = None
        self._attrib_location_position = None
        self._attrib_location_uv = None
        self._attrib_location_color = None
        
        # Initialize ImGui
        io = imgui.get_io()
        io.display_size = window.width, window.height
        io.display_fb_scale = 1.0, 1.0
        
        # Create font texture
        self._create_font_texture()
        
        # Create shaders
        self._create_shaders()
        
        # Create buffers
        self._create_buffers()
        
        # Set up event handlers
        window.push_handlers(
            on_mouse_motion=self._on_mouse_motion,
            on_mouse_press=self._on_mouse_press,
            on_mouse_release=self._on_mouse_release,
            on_mouse_scroll=self._on_mouse_scroll,
            on_key_press=self._on_key_press,
            on_key_release=self._on_key_release,
            on_text=self._on_text
        )
    
    def _create_font_texture(self):
        """Create font texture."""
        io = imgui.get_io()
        
        # Get font texture data
        width, height, pixels = io.fonts.get_tex_data_as_rgba32()
        
        # Create texture
        self._font_texture = GLuint(0)
        glGenTextures(1, ctypes.byref(self._font_texture))
        glBindTexture(GL_TEXTURE_2D, self._font_texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels)
        
        # Set texture ID in ImGui
        io.fonts.tex_id = self._font_texture.value
    
    def _create_shaders(self):
        """Create shaders."""
        # Vertex shader
        vertex_shader = """
        #version 330 core
        uniform mat4 ProjMtx;
        in vec2 Position;
        in vec2 UV;
        in vec4 Color;
        out vec2 Frag_UV;
        out vec4 Frag_Color;
        void main() {
            Frag_UV = UV;
            Frag_Color = Color;
            gl_Position = ProjMtx * vec4(Position.xy, 0.0, 1.0);
        }
        """
        
        # Fragment shader
        fragment_shader = """
        #version 330 core
        uniform sampler2D Texture;
        in vec2 Frag_UV;
        in vec4 Frag_Color;
        out vec4 Out_Color;
        void main() {
            Out_Color = Frag_Color * texture(Texture, Frag_UV.st);
        }
        """
        
        # Compile vertex shader
        vertex_shader_id = glCreateShader(GL_VERTEX_SHADER)
        
        # Convert shader source to bytes and create a pointer to it
        vertex_shader_bytes = vertex_shader.encode('utf-8')
        c_vertex_shader = ctypes.c_char_p(vertex_shader_bytes)
        c_vertex_shader_ptr = ctypes.cast(ctypes.pointer(c_vertex_shader), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)))
        
        # Set shader source
        glShaderSource(vertex_shader_id, 1, c_vertex_shader_ptr, None)
        
        # Compile shader
        glCompileShader(vertex_shader_id)
        
        # Check vertex shader compilation
        status = GLint(0)
        glGetShaderiv(vertex_shader_id, GL_COMPILE_STATUS, ctypes.byref(status))
        if status.value != GL_TRUE:
            # Get error message
            log_length = GLint(0)
            glGetShaderiv(vertex_shader_id, GL_INFO_LOG_LENGTH, ctypes.byref(log_length))
            log = ctypes.create_string_buffer(log_length.value)
            glGetShaderInfoLog(vertex_shader_id, log_length, None, log)
            print(f"Vertex shader compilation failed: {log.value.decode('utf-8')}")
            glDeleteShader(vertex_shader_id)
            return
        
        # Compile fragment shader
        fragment_shader_id = glCreateShader(GL_FRAGMENT_SHADER)
        
        # Convert shader source to bytes and create a pointer to it
        fragment_shader_bytes = fragment_shader.encode('utf-8')
        c_fragment_shader = ctypes.c_char_p(fragment_shader_bytes)
        c_fragment_shader_ptr = ctypes.cast(ctypes.pointer(c_fragment_shader), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)))
        
        # Set shader source
        glShaderSource(fragment_shader_id, 1, c_fragment_shader_ptr, None)
        
        # Compile shader
        glCompileShader(fragment_shader_id)
        
        # Check fragment shader compilation
        glGetShaderiv(fragment_shader_id, GL_COMPILE_STATUS, ctypes.byref(status))
        if status.value != GL_TRUE:
            # Get error message
            log_length = GLint(0)
            glGetShaderiv(fragment_shader_id, GL_INFO_LOG_LENGTH, ctypes.byref(log_length))
            log = ctypes.create_string_buffer(log_length.value)
            glGetShaderInfoLog(fragment_shader_id, log_length, None, log)
            print(f"Fragment shader compilation failed: {log.value.decode('utf-8')}")
            glDeleteShader(vertex_shader_id)
            glDeleteShader(fragment_shader_id)
            return
        
        # Create shader program
        self._shader_program = glCreateProgram()
        glAttachShader(self._shader_program, vertex_shader_id)
        glAttachShader(self._shader_program, fragment_shader_id)
        glLinkProgram(self._shader_program)
        
        # Check program linking
        glGetProgramiv(self._shader_program, GL_LINK_STATUS, ctypes.byref(status))
        if status.value != GL_TRUE:
            # Get error message
            log_length = GLint(0)
            glGetProgramiv(self._shader_program, GL_INFO_LOG_LENGTH, ctypes.byref(log_length))
            log = ctypes.create_string_buffer(log_length.value)
            glGetProgramInfoLog(self._shader_program, log_length, None, log)
            print(f"Shader program linking failed: {log.value.decode('utf-8')}")
            glDeleteShader(vertex_shader_id)
            glDeleteShader(fragment_shader_id)
            glDeleteProgram(self._shader_program)
            self._shader_program = None
            return
        
        # Delete shaders (they're linked to the program now)
        glDeleteShader(vertex_shader_id)
        glDeleteShader(fragment_shader_id)
        
        # Get attribute locations
        self._attrib_location_tex = glGetUniformLocation(self._shader_program, b"Texture")
        self._attrib_location_proj_mtx = glGetUniformLocation(self._shader_program, b"ProjMtx")
        self._attrib_location_position = glGetAttribLocation(self._shader_program, b"Position")
        self._attrib_location_uv = glGetAttribLocation(self._shader_program, b"UV")
        self._attrib_location_color = glGetAttribLocation(self._shader_program, b"Color")
    
    def _create_buffers(self):
        """Create buffers."""
        # Create VAO
        self._vao = GLuint(0)
        glGenVertexArrays(1, ctypes.byref(self._vao))
        glBindVertexArray(self._vao)
        
        # Create VBO
        self._vbo = GLuint(0)
        glGenBuffers(1, ctypes.byref(self._vbo))
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, imgui.VERTEX_SIZE * 65536, None, GL_DYNAMIC_DRAW)
        
        # Create EBO
        self._ebo = GLuint(0)
        glGenBuffers(1, ctypes.byref(self._ebo))
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, ctypes.sizeof(ctypes.c_ushort) * 65536, None, GL_DYNAMIC_DRAW)
        
        # Set up vertex attributes
        glEnableVertexAttribArray(self._attrib_location_position)
        glEnableVertexAttribArray(self._attrib_location_uv)
        glEnableVertexAttribArray(self._attrib_location_color)
        
        glVertexAttribPointer(self._attrib_location_position, 2, GL_FLOAT, GL_FALSE, imgui.VERTEX_SIZE, ctypes.c_void_p(0))
        glVertexAttribPointer(self._attrib_location_uv, 2, GL_FLOAT, GL_FALSE, imgui.VERTEX_SIZE, ctypes.c_void_p(8))
        glVertexAttribPointer(self._attrib_location_color, 4, GL_UNSIGNED_BYTE, GL_TRUE, imgui.VERTEX_SIZE, ctypes.c_void_p(16))
        
        # Unbind VAO
        glBindVertexArray(0)
    
    def render(self, draw_data):
        """Render ImGui draw data.
        
        Args:
            draw_data: ImGui draw data
        """
        # Check if shader program is valid
        if self._shader_program is None:
            return
        
        # Get display size
        display_width, display_height = self.window.width, self.window.height
        
        # Set up render state
        glEnable(GL_BLEND)
        glBlendEquation(GL_FUNC_ADD)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_CULL_FACE)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_SCISSOR_TEST)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        # Set up viewport
        glViewport(0, 0, int(display_width), int(display_height))
        
        # Set up projection matrix
        ortho_projection = np.array([
            [2.0 / display_width, 0.0, 0.0, -1.0],
            [0.0, 2.0 / -display_height, 0.0, 1.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        # Set up shader program
        glUseProgram(self._shader_program)
        glUniform1i(self._attrib_location_tex, 0)
        glUniformMatrix4fv(self._attrib_location_proj_mtx, 1, GL_FALSE, ortho_projection.flatten())
        
        # Bind VAO
        glBindVertexArray(self._vao)
        
        # Render command lists
        for commands in draw_data.commands_lists:
            # Get vertex and index buffer data
            vertex_buffer_data = commands.vtx_buffer_data
            index_buffer_data = commands.idx_buffer_data
            
            # Upload vertex buffer
            glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
            glBufferData(GL_ARRAY_BUFFER, vertex_buffer_data, GL_STREAM_DRAW)
            
            # Upload index buffer
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._ebo)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_buffer_data, GL_STREAM_DRAW)
            
            # Render command lists
            idx_pos = 0
            for command in commands.commands:
                # Set up scissor box
                x, y, z, w = command.clip_rect
                glScissor(int(x), int(display_height - w), int(z - x), int(w - y))
                
                # Bind texture
                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, command.texture_id)
                
                # Draw elements
                glDrawElements(GL_TRIANGLES, command.elem_count, GL_UNSIGNED_SHORT, ctypes.c_void_p(idx_pos))
                
                # Update index position
                idx_pos += command.elem_count * ctypes.sizeof(ctypes.c_ushort)
        
        # Restore state
        glDisable(GL_SCISSOR_TEST)
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        glUseProgram(0)
    
    def shutdown(self):
        """Clean up resources."""
        if self._vao:
            glDeleteVertexArrays(1, ctypes.byref(self._vao))
            self._vao = None
        
        if self._vbo:
            glDeleteBuffers(1, ctypes.byref(self._vbo))
            self._vbo = None
        
        if self._ebo:
            glDeleteBuffers(1, ctypes.byref(self._ebo))
            self._ebo = None
        
        if self._font_texture:
            glDeleteTextures(1, ctypes.byref(self._font_texture))
            self._font_texture = None
            imgui.get_io().fonts.tex_id = 0
        
        if self._shader_program:
            glDeleteProgram(self._shader_program)
            self._shader_program = None
    
    def _on_mouse_motion(self, x, y, dx, dy):
        """Handle mouse motion events.
        
        Args:
            x: Mouse x position
            y: Mouse y position
            dx: Mouse x delta
            dy: Mouse y delta
        """
        io = imgui.get_io()
        io.mouse_pos = x, y
        return True
    
    def _on_mouse_press(self, x, y, button, modifiers):
        """Handle mouse press events.
        
        Args:
            x: Mouse x position
            y: Mouse y position
            button: Mouse button
            modifiers: Keyboard modifiers
        """
        io = imgui.get_io()
        
        if button == pyglet.window.mouse.LEFT:
            io.mouse_down[0] = True
        elif button == pyglet.window.mouse.RIGHT:
            io.mouse_down[1] = True
        elif button == pyglet.window.mouse.MIDDLE:
            io.mouse_down[2] = True
        
        return io.want_capture_mouse
    
    def _on_mouse_release(self, x, y, button, modifiers):
        """Handle mouse release events.
        
        Args:
            x: Mouse x position
            y: Mouse y position
            button: Mouse button
            modifiers: Keyboard modifiers
        """
        io = imgui.get_io()
        
        if button == pyglet.window.mouse.LEFT:
            io.mouse_down[0] = False
        elif button == pyglet.window.mouse.RIGHT:
            io.mouse_down[1] = False
        elif button == pyglet.window.mouse.MIDDLE:
            io.mouse_down[2] = False
        
        return io.want_capture_mouse
    
    def _on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        """Handle mouse scroll events.
        
        Args:
            x: Mouse x position
            y: Mouse y position
            scroll_x: Horizontal scroll amount
            scroll_y: Vertical scroll amount
        """
        io = imgui.get_io()
        io.mouse_wheel = scroll_y
        io.mouse_wheel_h = scroll_x
        return io.want_capture_mouse
    
    def _on_key_press(self, symbol, modifiers):
        """Handle key press events.
        
        Args:
            symbol: Key symbol
            modifiers: Keyboard modifiers
        """
        io = imgui.get_io()
        
        # Update modifier keys
        io.key_ctrl = modifiers & pyglet.window.key.MOD_CTRL
        io.key_shift = modifiers & pyglet.window.key.MOD_SHIFT
        io.key_alt = modifiers & pyglet.window.key.MOD_ALT
        
        # Update key state
        io.keys_down[symbol] = True
        
        return io.want_capture_keyboard
    
    def _on_key_release(self, symbol, modifiers):
        """Handle key release events.
        
        Args:
            symbol: Key symbol
            modifiers: Keyboard modifiers
        """
        io = imgui.get_io()
        
        # Update modifier keys
        io.key_ctrl = modifiers & pyglet.window.key.MOD_CTRL
        io.key_shift = modifiers & pyglet.window.key.MOD_SHIFT
        io.key_alt = modifiers & pyglet.window.key.MOD_ALT
        
        # Update key state
        io.keys_down[symbol] = False
        
        return io.want_capture_keyboard
    
    def _on_text(self, text):
        """Handle text input events.
        
        Args:
            text: Input text
        """
        io = imgui.get_io()
        
        for char in text:
            io.add_input_character(ord(char))
        
        return io.want_capture_keyboard


class ImGuiTestWindow(pyglet.window.Window):
    """Test window for Dear ImGui integration."""
    
    def __init__(self, width=1280, height=720, title="Dear ImGui Test"):
        """Initialize the window.
        
        Args:
            width: Window width
            height: Window height
            title: Window title
        """
        super().__init__(width, height, title, resizable=True)
        
        # Set up OpenGL
        self.setup_opengl()
        
        # Initialize ImGui
        imgui.create_context()
        self.imgui_renderer = ImGuiPygletRenderer(self)
        
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
        
        # Set up ImGui IO
        io = imgui.get_io()
        io.display_size = width, height
        
        # Initialize test variables
        self.show_demo_window = True
        self.show_custom_window = True
        self.clear_color = (0.0, 0.0, 0.0, 1.0)
        self.f = 0.0
        self.counter = 0
        
        # Set up update function
        pyglet.clock.schedule_interval(self.update, 1/60.0)
    
    def setup_opengl(self):
        """Set up OpenGL."""
        # Set the background color to true black (RGBA)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        # Enable depth testing for 3D rendering
        glEnable(GL_DEPTH_TEST)
        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    def update(self, dt):
        """Update the window.
        
        Args:
            dt: Time delta
        """
        # Update ImGui
        io = imgui.get_io()
        io.delta_time = dt
    
    def on_draw(self):
        """Draw the window."""
        # Clear the window
        self.clear()
        
        # Start ImGui frame
        imgui.new_frame()
        
        # Show demo window
        if self.show_demo_window:
            expanded, self.show_demo_window = imgui.begin("Dear ImGui Demo", True)
            if expanded:
                imgui.text("This is a demo window for Dear ImGui.")
                imgui.text(f"ImGui version: {imgui.get_version()}")
                imgui.text(f"OpenGL version: {glGetString(GL_VERSION).decode('utf-8')}")
                imgui.text(f"GLSL version: {glGetString(GL_SHADING_LANGUAGE_VERSION).decode('utf-8')}")
                
                # Add a slider
                changed, self.f = imgui.slider_float("Float", self.f, 0.0, 1.0)
                
                # Add a color picker
                changed, self.clear_color = imgui.color_edit3("Clear Color", *self.clear_color[:3])
                
                # Add a button
                if imgui.button("Button"):
                    self.counter += 1
                
                imgui.same_line()
                imgui.text(f"Counter: {self.counter}")
                
                # Add a checkbox
                changed, self.show_custom_window = imgui.checkbox("Show Custom Window", self.show_custom_window)
            
            imgui.end()
        
        # Show custom window
        if self.show_custom_window:
            expanded, self.show_custom_window = imgui.begin("Custom Window", True)
            if expanded:
                imgui.text("This is a custom window.")
                
                # Add a progress bar
                imgui.progress_bar(self.f, (0, 0), "Progress")
                
                # Add a collapsing header
                if imgui.collapsing_header("Collapsing Header"):
                    imgui.text("This is inside the collapsing header.")
                    
                    # Add a tree node
                    if imgui.tree_node("Tree Node"):
                        imgui.text("This is inside the tree node.")
                        imgui.tree_pop()
            
            imgui.end()
        
        # Render ImGui
        imgui.render()
        self.imgui_renderer.render(imgui.get_draw_data())
    
    def on_resize(self, width, height):
        """Handle window resize events.
        
        Args:
            width: New window width
            height: New window height
        """
        # Update viewport
        glViewport(0, 0, width, height)
        
        # Update ImGui display size
        io = imgui.get_io()
        io.display_size = width, height
    
    def on_close(self):
        """Handle window close events."""
        # Clean up ImGui
        self.imgui_renderer.shutdown()
        
        # Close the window
        super().on_close()


def main():
    """Run the test."""
    try:
        # Print ImGui version
        print(f"ImGui version: {imgui.get_version()}")
        
        # Create the window
        window = ImGuiTestWindow()
        
        # Run the application
        pyglet.app.run()
        
        return 0
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
