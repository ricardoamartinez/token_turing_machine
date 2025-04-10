"""
Shader manager for the TTM visualization engine.

This module provides functionality for loading, compiling, and managing OpenGL shaders.
"""

import os
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
        with open(shader_path, 'r') as f:
            shader_source = f.read()
        
        # Create and compile shader
        shader = glCreateShader(shader_type)
        glShaderSource(shader, shader_source)
        glCompileShader(shader)
        
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
                      geometry_shader: Optional[str] = None) -> int:
        """Create a shader program from vertex and fragment shaders.
        
        Args:
            name: Name of the program
            vertex_shader: Name of the vertex shader
            fragment_shader: Name of the fragment shader
            geometry_shader: Optional name of the geometry shader
            
        Returns:
            Program ID
        """
        # Load shaders if not already loaded
        if vertex_shader not in self.shaders:
            self.load_shader(vertex_shader, GL_VERTEX_SHADER)
        
        if fragment_shader not in self.shaders:
            self.load_shader(fragment_shader, GL_FRAGMENT_SHADER)
        
        if geometry_shader is not None and geometry_shader not in self.shaders:
            self.load_shader(geometry_shader, GL_GEOMETRY_SHADER)
        
        # Create program
        program = glCreateProgram()
        
        # Attach shaders
        glAttachShader(program, self.shaders[vertex_shader])
        glAttachShader(program, self.shaders[fragment_shader])
        if geometry_shader is not None:
            glAttachShader(program, self.shaders[geometry_shader])
        
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
        location = glGetUniformLocation(program, uniform_name)
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
        location = glGetUniformLocation(program, uniform_name)
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
        location = glGetUniformLocation(program, uniform_name)
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
        location = glGetUniformLocation(program, uniform_name)
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
        location = glGetUniformLocation(program, uniform_name)
        if location == -1:
            print(f"Warning: Uniform '{uniform_name}' not found in program '{program_name}'")
            return
        
        glUseProgram(program)
        glUniformMatrix4fv(location, 1, GL_FALSE, matrix.flatten())
    
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
