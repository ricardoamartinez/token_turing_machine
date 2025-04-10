"""
Test script for VisMapper classes.

This script creates instances of the different VisMapper classes and tests
their functionality with sample data.
"""

import numpy as np
import torch
from src.ttm.visualization.vis_mapper import (
    VisMapper,
    TensorToVoxelMapper,
    MemoryToVoxelMapper,
    AttentionToVoxelMapper,
    create_mapper_for_state
)

def test_tensor_mapper():
    """Test the TensorToVoxelMapper class."""
    print("\nTesting TensorToVoxelMapper...")
    
    # Create a mapper
    mapper = TensorToVoxelMapper(name="test_tensor_mapper")
    
    # Test with 1D tensor
    print("Testing with 1D tensor...")
    tensor_1d = np.random.rand(10)
    state_1d = {
        'name': 'test_1d_tensor',
        'data': tensor_1d,
        'type': 'tensor',
        'shape': tensor_1d.shape,
        'metadata': {'test': True}
    }
    
    voxel_data_1d = mapper.map_to_voxels(state_1d)
    print(f"Voxel data name: {voxel_data_1d['name']}")
    print(f"Voxel dimensions: {voxel_data_1d['dimensions']}")
    print(f"Voxel shape: {voxel_data_1d['voxels'].shape}")
    print(f"Non-zero voxels: {np.count_nonzero(voxel_data_1d['voxels'])}")
    
    # Test with 2D tensor
    print("\nTesting with 2D tensor...")
    tensor_2d = np.random.rand(8, 8)
    state_2d = {
        'name': 'test_2d_tensor',
        'data': tensor_2d,
        'type': 'tensor',
        'shape': tensor_2d.shape,
        'metadata': {'test': True}
    }
    
    voxel_data_2d = mapper.map_to_voxels(state_2d)
    print(f"Voxel data name: {voxel_data_2d['name']}")
    print(f"Voxel dimensions: {voxel_data_2d['dimensions']}")
    print(f"Voxel shape: {voxel_data_2d['voxels'].shape}")
    print(f"Non-zero voxels: {np.count_nonzero(voxel_data_2d['voxels'])}")
    
    # Test with 3D tensor
    print("\nTesting with 3D tensor...")
    tensor_3d = np.random.rand(4, 4, 4)
    state_3d = {
        'name': 'test_3d_tensor',
        'data': tensor_3d,
        'type': 'tensor',
        'shape': tensor_3d.shape,
        'metadata': {'test': True}
    }
    
    voxel_data_3d = mapper.map_to_voxels(state_3d)
    print(f"Voxel data name: {voxel_data_3d['name']}")
    print(f"Voxel dimensions: {voxel_data_3d['dimensions']}")
    print(f"Voxel shape: {voxel_data_3d['voxels'].shape}")
    print(f"Non-zero voxels: {np.count_nonzero(voxel_data_3d['voxels'])}")
    
    # Test layout and color map
    layout = mapper.get_voxel_layout()
    color_map = mapper.get_color_map()
    
    print("\nVoxel layout:")
    print(f"  Dimensions: {layout['dimensions']}")
    print(f"  Scale: {layout['scale']}")
    print(f"  Origin: {layout['origin']}")
    
    print("\nColor map:")
    print(f"  Name: {color_map['name']}")
    print(f"  Min value: {color_map['min_value']}")
    print(f"  Max value: {color_map['max_value']}")
    print(f"  Alpha: {color_map['alpha']}")

def test_memory_mapper():
    """Test the MemoryToVoxelMapper class."""
    print("\nTesting MemoryToVoxelMapper...")
    
    # Create a mapper
    mapper = MemoryToVoxelMapper()
    
    # Test with memory matrix
    print("Testing with memory matrix...")
    memory = np.random.rand(8, 32)  # r x d
    state = {
        'name': 'test_memory',
        'data': memory,
        'type': 'tensor',
        'shape': memory.shape,
        'metadata': {'is_memory': True}
    }
    
    voxel_data = mapper.map_to_voxels(state)
    print(f"Voxel data name: {voxel_data['name']}")
    print(f"Voxel dimensions: {voxel_data['dimensions']}")
    print(f"Voxel shape: {voxel_data['voxels'].shape}")
    print(f"Non-zero voxels: {np.count_nonzero(voxel_data['voxels'])}")
    print(f"Is memory: {voxel_data['metadata']['is_memory']}")

def test_attention_mapper():
    """Test the AttentionToVoxelMapper class."""
    print("\nTesting AttentionToVoxelMapper...")
    
    # Create a mapper
    mapper = AttentionToVoxelMapper()
    
    # Test with attention matrix
    print("Testing with attention matrix...")
    attention = np.random.rand(1, 8, 8)  # batch x seq_len x seq_len
    state = {
        'name': 'test_attention',
        'data': attention,
        'type': 'tensor',
        'shape': attention.shape,
        'metadata': {'is_attention': True}
    }
    
    voxel_data = mapper.map_to_voxels(state)
    print(f"Voxel data name: {voxel_data['name']}")
    print(f"Voxel dimensions: {voxel_data['dimensions']}")
    print(f"Voxel shape: {voxel_data['voxels'].shape}")
    print(f"Non-zero voxels: {np.count_nonzero(voxel_data['voxels'])}")
    print(f"Is attention: {voxel_data['metadata']['is_attention']}")

def test_factory_function():
    """Test the create_mapper_for_state factory function."""
    print("\nTesting create_mapper_for_state factory function...")
    
    # Test with memory state
    memory_state = {
        'name': 'memory_0',
        'data': np.random.rand(8, 32),
        'type': 'tensor',
        'shape': (8, 32),
        'metadata': {}
    }
    
    memory_mapper = create_mapper_for_state(memory_state)
    print(f"Mapper for memory state: {memory_mapper.name}")
    
    # Test with attention state
    attention_state = {
        'name': 'attention_weights',
        'data': np.random.rand(1, 8, 8),
        'type': 'tensor',
        'shape': (1, 8, 8),
        'metadata': {}
    }
    
    attention_mapper = create_mapper_for_state(attention_state)
    print(f"Mapper for attention state: {attention_mapper.name}")
    
    # Test with generic tensor state
    tensor_state = {
        'name': 'embedding',
        'data': np.random.rand(10, 64),
        'type': 'tensor',
        'shape': (10, 64),
        'metadata': {}
    }
    
    tensor_mapper = create_mapper_for_state(tensor_state)
    print(f"Mapper for tensor state: {tensor_mapper.name}")

def main():
    """Run all tests."""
    test_tensor_mapper()
    test_memory_mapper()
    test_attention_mapper()
    test_factory_function()

if __name__ == "__main__":
    main()
