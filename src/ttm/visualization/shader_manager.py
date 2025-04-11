"""
Shader manager for the TTM visualization engine.

This module provides functionality for loading, compiling, and managing OpenGL shaders.
"""

import os
import ctypes
from ctypes import c_char_p, c_int
import pyglet
from pyglet.gl import *
import numpy as np
from typing import Dict, List, Tuple, Optional


class ShaderManager:
    """Manages OpenGL shaders for the visualization engine."""

    def __init__(self, shader_dir: str = None):
        """Initialize the shader manager.

        Args:
            shader_dir: Directory containing shader files
        """
        if shader_dir is None:
            # Default to the 'shaders' directory in the same directory as this file
            shader_dir = os.path.join(os.path.dirname(__file__), 'shaders')

        self.shader_dir = shader_dir
        self.programs: Dict[str, int] = {}  # Maps program names to program IDs
        self.shaders: Dict[str, int] = {}   # Maps shader names to shader IDs

    def load_shader(self, name: str, shader_type: int) -> int:
        """Load and compile a shader from a file.

        Args:
            name: Name of the shader file (without extension)
            shader_type: Type of shader (GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, etc.)

        Returns:
            Shader ID
        """
        print(f"Loading shader: {name} (type: {shader_type})")
        # Determine file extension based on shader type
        if shader_type == GL_VERTEX_SHADER:
            ext = '.vert'
        elif shader_type == GL_FRAGMENT_SHADER:
            ext = '.frag'
        elif shader_type == GL_GEOMETRY_SHADER:
            ext = '.geom'
        elif shader_type == GL_COMPUTE_SHADER:
            ext = '.comp'
        else:
            raise ValueError(f"Unsupported shader type: {shader_type}")

        # Load shader source
        shader_path = os.path.join(self.shader_dir, name + ext)
        print(f"Loading shader from: {shader_path}")
        try:
            with open(shader_path, 'r') as f:
                shader_source = f.read()
            print(f"Shader source loaded successfully! ({len(shader_source)} bytes)")
        except Exception as e:
            print(f"Error loading shader source: {e}")
            raise

        # Create and compile shader
        print("Creating shader...")
        shader = glCreateShader(shader_type)
        print(f"Shader created with ID: {shader}")

        # Use pyglet's utility function to handle shader source
        # This handles the ctypes conversion properly
        print("Converting shader source to bytes...")
        shader_source_bytes = shader_source.encode('utf-8') if isinstance(shader_source, str) else shader_source
        print(f"Shader source converted to bytes: {len(shader_source_bytes)} bytes")

        print("Using alternative shader loading approach...")
        try:
            # Use a simpler approach with ctypes
            # Convert shader source to bytes
            shader_source_bytes = shader_source.encode('utf-8')

            # Create a ctypes string buffer
            c_source = ctypes.create_string_buffer(shader_source_bytes)

            # Create a pointer to the source
            c_source_ptr = ctypes.cast(ctypes.pointer(c_source), ctypes.POINTER(ctypes.c_char))

            # Create an array of pointers to the source
            sources = (ctypes.POINTER(ctypes.c_char) * 1)(c_source_ptr)

            # Create an array of lengths
            lengths = (ctypes.c_int * 1)(len(shader_source_bytes))

            # Set the shader source
            print("Setting shader source...")
            glShaderSource(shader, 1, sources, lengths)
            print("Shader source set successfully!")

            # Compile the shader
            print("Compiling shader...")
            glCompileShader(shader)
            print("Shader compiled successfully!")
        except Exception as e:
            print(f"Error during shader compilation: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Check for compilation errors
        status = GLint()
        glGetShaderiv(shader, GL_COMPILE_STATUS, status)
        if status.value != GL_TRUE:
            # Get error message
            log_length = GLint()
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, log_length)
            log = glGetShaderInfoLog(shader)
            raise RuntimeError(f"Shader compilation failed ({name}{ext}):\n{log.decode('utf-8')}")

        # Store shader
        self.shaders[name] = shader
        return shader

    def create_program(self, name: str, vertex_shader: str, fragment_shader: str,
                      geometry_shader: Optional[str] = None, compute_shader: Optional[str] = None) -> int:
        """Create a shader program from shaders.

        Args:
            name: Name of the program
            vertex_shader: Name of the vertex shader (None for compute-only programs)
            fragment_shader: Name of the fragment shader (None for compute-only programs)
            geometry_shader: Optional name of the geometry shader
            compute_shader: Optional name of the compute shader

        Returns:
            Program ID
        """
        print(f"Creating shader program: {name}")

        # Check if this is a compute shader program
        is_compute_program = compute_shader is not None and vertex_shader is None and fragment_shader is None

        if not is_compute_program:
            # Load regular graphics shaders if not already loaded
            print(f"Loading vertex shader: {vertex_shader}")
            if vertex_shader not in self.shaders:
                self.load_shader(vertex_shader, GL_VERTEX_SHADER)
            print(f"Vertex shader loaded with ID: {self.shaders[vertex_shader]}")

            print(f"Loading fragment shader: {fragment_shader}")
            if fragment_shader not in self.shaders:
                self.load_shader(fragment_shader, GL_FRAGMENT_SHADER)
            print(f"Fragment shader loaded with ID: {self.shaders[fragment_shader]}")

            if geometry_shader is not None:
                print(f"Loading geometry shader: {geometry_shader}")
                if geometry_shader not in self.shaders:
                    self.load_shader(geometry_shader, GL_GEOMETRY_SHADER)
                print(f"Geometry shader loaded with ID: {self.shaders[geometry_shader]}")

        # Load compute shader if specified
        if compute_shader is not None:
            print(f"Loading compute shader: {compute_shader}")
            if compute_shader not in self.shaders:
                self.load_shader(compute_shader, GL_COMPUTE_SHADER)
            print(f"Compute shader loaded with ID: {self.shaders[compute_shader]}")

        # Create program
        print("Creating program...")
        program = glCreateProgram()
        print(f"Program created with ID: {program}")

        # Attach shaders
        try:
            if not is_compute_program:
                # Attach graphics shaders
                print(f"Attaching vertex shader {self.shaders[vertex_shader]} to program {program}...")
                glAttachShader(program, self.shaders[vertex_shader])
                print("Vertex shader attached successfully!")

                print(f"Attaching fragment shader {self.shaders[fragment_shader]} to program {program}...")
                glAttachShader(program, self.shaders[fragment_shader])
                print("Fragment shader attached successfully!")

                if geometry_shader is not None:
                    print(f"Attaching geometry shader {self.shaders[geometry_shader]} to program {program}...")
                    glAttachShader(program, self.shaders[geometry_shader])
                    print("Geometry shader attached successfully!")

            # Attach compute shader if specified
            if compute_shader is not None:
                print(f"Attaching compute shader {self.shaders[compute_shader]} to program {program}...")
                glAttachShader(program, self.shaders[compute_shader])
                print("Compute shader attached successfully!")
        except Exception as e:
            print(f"Error attaching shaders: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Link program
        glLinkProgram(program)

        # Check for linking errors
        status = GLint()
        glGetProgramiv(program, GL_LINK_STATUS, status)
        if status.value != GL_TRUE:
            # Get error message
            log_length = GLint()
            glGetProgramiv(program, GL_INFO_LOG_LENGTH, log_length)
            log = glGetProgramInfoLog(program)
            raise RuntimeError(f"Program linking failed ({name}):\n{log.decode('utf-8')}")

        # Store program
        self.programs[name] = program
        return program

    def use_program(self, name: str) -> None:
        """Use a shader program.

        Args:
            name: Name of the program
        """
        if name not in self.programs:
            raise ValueError(f"Program not found: {name}")

        glUseProgram(self.programs[name])

    def set_uniform_1i(self, program_name: str, uniform_name: str, value: int) -> None:
        """Set an integer uniform value.

        Args:
            program_name: Name of the program
            uniform_name: Name of the uniform
            value: Value to set
        """
        if program_name not in self.programs:
            raise ValueError(f"Program not found: {program_name}")

        program = self.programs[program_name]
        # Convert uniform name to bytes
        uniform_name_bytes = uniform_name.encode('utf-8')
        location = glGetUniformLocation(program, uniform_name_bytes)
        if location == -1:
            print(f"Warning: Uniform '{uniform_name}' not found in program '{program_name}'")
            return

        glUseProgram(program)
        glUniform1i(location, value)

    def set_uniform_1f(self, program_name: str, uniform_name: str, value: float) -> None:
        """Set a float uniform value.

        Args:
            program_name: Name of the program
            uniform_name: Name of the uniform
            value: Value to set
        """
        if program_name not in self.programs:
            raise ValueError(f"Program not found: {program_name}")

        program = self.programs[program_name]
        # Convert uniform name to bytes
        uniform_name_bytes = uniform_name.encode('utf-8')
        location = glGetUniformLocation(program, uniform_name_bytes)
        if location == -1:
            print(f"Warning: Uniform '{uniform_name}' not found in program '{program_name}'")
            return

        glUseProgram(program)
        glUniform1f(location, value)

    def set_uniform_3f(self, program_name: str, uniform_name: str,
                      x: float, y: float, z: float) -> None:
        """Set a vec3 uniform value.

        Args:
            program_name: Name of the program
            uniform_name: Name of the uniform
            x: X component
            y: Y component
            z: Z component
        """
        if program_name not in self.programs:
            raise ValueError(f"Program not found: {program_name}")

        program = self.programs[program_name]
        # Convert uniform name to bytes
        uniform_name_bytes = uniform_name.encode('utf-8')
        location = glGetUniformLocation(program, uniform_name_bytes)
        if location == -1:
            print(f"Warning: Uniform '{uniform_name}' not found in program '{program_name}'")
            return

        glUseProgram(program)
        glUniform3f(location, x, y, z)

    def set_uniform_4f(self, program_name: str, uniform_name: str,
                      x: float, y: float, z: float, w: float) -> None:
        """Set a vec4 uniform value.

        Args:
            program_name: Name of the program
            uniform_name: Name of the uniform
            x: X component
            y: Y component
            z: Z component
            w: W component
        """
        if program_name not in self.programs:
            raise ValueError(f"Program not found: {program_name}")

        program = self.programs[program_name]
        # Convert uniform name to bytes
        uniform_name_bytes = uniform_name.encode('utf-8')
        location = glGetUniformLocation(program, uniform_name_bytes)
        if location == -1:
            print(f"Warning: Uniform '{uniform_name}' not found in program '{program_name}'")
            return

        glUseProgram(program)
        glUniform4f(location, x, y, z, w)

    def set_uniform_matrix4fv(self, program_name: str, uniform_name: str,
                             matrix: np.ndarray) -> None:
        """Set a mat4 uniform value.

        Args:
            program_name: Name of the program
            uniform_name: Name of the uniform
            matrix: 4x4 matrix
        """
        if program_name not in self.programs:
            raise ValueError(f"Program not found: {program_name}")

        program = self.programs[program_name]
        # Convert uniform name to bytes
        uniform_name_bytes = uniform_name.encode('utf-8')
        location = glGetUniformLocation(program, uniform_name_bytes)
        if location == -1:
            print(f"Warning: Uniform '{uniform_name}' not found in program '{program_name}'")
            return

        glUseProgram(program)
        # Convert numpy array to ctypes array
        matrix_flat = matrix.flatten()
        c_matrix = (ctypes.c_float * len(matrix_flat))(*matrix_flat)
        glUniformMatrix4fv(location, 1, GL_FALSE, c_matrix)

    def dispatch_compute(self, program_name: str, num_groups_x: int, num_groups_y: int, num_groups_z: int) -> None:
        """Dispatch a compute shader.

        Args:
            program_name: Name of the compute shader program
            num_groups_x: Number of work groups in X dimension
            num_groups_y: Number of work groups in Y dimension
            num_groups_z: Number of work groups in Z dimension
        """
        if program_name not in self.programs:
            raise ValueError(f"Program not found: {program_name}")

        program = self.programs[program_name]
        glUseProgram(program)
        glDispatchCompute(num_groups_x, num_groups_y, num_groups_z)

        # Wait for compute shader to finish
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_ATOMIC_COUNTER_BARRIER_BIT)

    def cleanup(self) -> None:
        """Clean up all shaders and programs."""
        # Delete programs
        for program in self.programs.values():
            glDeleteProgram(program)

        # Delete shaders
        for shader in self.shaders.values():
            glDeleteShader(shader)

        self.programs.clear()
        self.shaders.clear()
