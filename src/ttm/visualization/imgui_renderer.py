"""
Custom ImGui renderer for Pyglet.

This module provides a custom ImGui renderer for Pyglet that works with modern OpenGL.
"""

import imgui
import pyglet
from pyglet.gl import *
import numpy as np
import ctypes

class CustomPygletRenderer:
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
        glShaderSource(vertex_shader_id, 1, ctypes.cast(ctypes.pointer(ctypes.create_string_buffer(vertex_shader.encode('utf-8'))), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))), None)
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
        glShaderSource(fragment_shader_id, 1, ctypes.cast(ctypes.pointer(ctypes.create_string_buffer(fragment_shader.encode('utf-8'))), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))), None)
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
