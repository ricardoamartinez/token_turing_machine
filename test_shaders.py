"""
Test script for shader compilation.

This script tests that the GLSL shaders compile correctly.
"""

import os
import sys
import pyglet
from pyglet.gl import *
import numpy as np

from src.ttm.visualization.shader_manager import ShaderManager

def main():
    """Test shader compilation."""
    # Create a window to initialize OpenGL context
    window = pyglet.window.Window(width=1, height=1, visible=False)

    # Print OpenGL version
    print(f"OpenGL Version: {window.context.get_info().get_version()}")

    # Get GLSL version
    glsl_version = glGetString(GL_SHADING_LANGUAGE_VERSION)
    if glsl_version:
        print(f"GLSL Version: {glsl_version.decode('utf-8')}")
    else:
        print("GLSL Version: Unknown")

    # Create shader manager
    shader_dir = os.path.join(os.path.dirname(__file__), 'src/ttm/visualization/shaders')
    shader_manager = ShaderManager(shader_dir)

    try:
        # Load vertex shader
        print("\nCompiling vertex shader...")
        vertex_shader = shader_manager.load_shader('voxel', GL_VERTEX_SHADER)
        print("Vertex shader compiled successfully.")

        # Load fragment shader
        print("\nCompiling fragment shader...")
        fragment_shader = shader_manager.load_shader('voxel', GL_FRAGMENT_SHADER)
        print("Fragment shader compiled successfully.")

        # Create program
        print("\nLinking shader program...")
        program = shader_manager.create_program('voxel', 'voxel', 'voxel')
        print("Shader program linked successfully.")

        # Print active attributes
        print("\nActive attributes:")
        num_attributes = GLint()
        glGetProgramiv(program, GL_ACTIVE_ATTRIBUTES, num_attributes)

        for i in range(num_attributes.value):
            name_length = GLint()
            attribute_size = GLint()
            attribute_type = GLenum()
            name_buffer = (GLchar * 100)()

            glGetActiveAttrib(program, i, 100, name_length, attribute_size, attribute_type, name_buffer)
            name = name_buffer.value.decode('utf-8')
            location = glGetAttribLocation(program, name.encode('utf-8'))

            print(f"  {name} (location {location})")

        # Print active uniforms
        print("\nActive uniforms:")
        num_uniforms = GLint()
        glGetProgramiv(program, GL_ACTIVE_UNIFORMS, num_uniforms)

        for i in range(num_uniforms.value):
            name_length = GLint()
            uniform_size = GLint()
            uniform_type = GLenum()
            name_buffer = (GLchar * 100)()

            glGetActiveUniform(program, i, 100, name_length, uniform_size, uniform_type, name_buffer)
            name = name_buffer.value.decode('utf-8')
            location = glGetUniformLocation(program, name.encode('utf-8'))

            print(f"  {name} (location {location})")

        print("\nShader compilation test passed!")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    finally:
        # Clean up
        shader_manager.cleanup()
        window.close()

    return 0

if __name__ == "__main__":
    sys.exit(main())
