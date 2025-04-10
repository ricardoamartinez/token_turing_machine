"""
Test script for VisMapper integration with the rendering engine.

This script tests that the VisMapper output is correctly integrated with the rendering engine.
"""

import os
import sys
import pyglet
from pyglet.gl import *
import numpy as np
import time
import pickle

from src.ttm.visualization.visualization_engine import VisualizationEngine
from src.ttm.visualization.vis_mapper import TensorToVoxelMapper


def create_test_state_history(filepath: str) -> None:
    """Create a test state history file.

    Args:
        filepath: Path to save the state history file
    """
    # Create a state history dictionary
    state_history = {
        'epochs': [0],
        'batches': {0: [0]},
        'tokens': {(0, 0): [0]},
        'states': {}
    }

    # Create a state key
    state_key = (0, 0, 0)  # (epoch, batch, token)

    # Create a state dictionary
    state_dict = {
        'modules': {},
        'gradients': {}
    }

    # Create a module dictionary
    module_dict = {}

    # Create a tensor state
    tensor_state = {
        'name': 'test_tensor',
        'type': 'tensor',
        'shape': (10, 10),
        'data': np.random.rand(10, 10),
        'metadata': {
            'epoch': 0,
            'batch': 0,
            'token': 0,
            'module': 'test_module',
            'is_input': True
        }
    }

    # Add the tensor state to the module dictionary
    module_dict['inputs'] = tensor_state

    # Create another tensor state
    tensor_state2 = {
        'name': 'test_tensor2',
        'type': 'tensor',
        'shape': (5, 5),
        'data': np.random.rand(5, 5),
        'metadata': {
            'epoch': 0,
            'batch': 0,
            'token': 0,
            'module': 'test_module',
            'is_input': False
        }
    }

    # Add the tensor state to the module dictionary
    module_dict['outputs'] = tensor_state2

    # Add the module dictionary to the state dictionary
    state_dict['modules']['test_module'] = module_dict

    # Create a gradient state
    gradient_state = {
        'name': 'test_gradient',
        'type': 'tensor',
        'shape': (10, 10),
        'data': np.random.rand(10, 10),
        'metadata': {
            'epoch': 0,
            'batch': 0,
            'token': 0,
            'parameter_name': 'test_parameter',
            'is_gradient': True
        }
    }

    # Add the gradient state to the state dictionary
    state_dict['gradients']['test_parameter'] = gradient_state

    # Add the state dictionary to the state history
    state_history['states'][state_key] = state_dict

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Save the state history
    with open(filepath, 'wb') as f:
        pickle.dump(state_history, f)

    print(f"Created test state history file: {filepath}")


def main():
    """Run the test."""
    try:
        # Create a test state history file
        test_filepath = './visualization_data/test_state_history.pkl'
        create_test_state_history(test_filepath)

        print("\nCreating visualization engine...")
        # Create the visualization engine
        engine = VisualizationEngine(width=1280, height=720, caption="VisMapper Integration Test")

        print("\nLoading state history...")
        # Load the state history
        engine.load_state_history(test_filepath)

        # Print a summary of the visualization
        print("\nVisualization summary:")
        print(f"Number of states: {len(engine.visualization_manager.states)}")
        print(f"Active states: {engine.visualization_manager.active_states}")
        print(f"Number of mappers: {len(engine.visualization_manager.mappers)}")
        print(f"Number of voxel mappings: {len(engine.visualization_manager.voxel_mapping)}")
        print(f"Next voxel index: {engine.visualization_manager.next_voxel_index}")

        # Print a summary of the voxel renderer
        print("\nVoxel renderer summary:")
        print(f"Number of active voxels: {engine.voxel_renderer.num_active_voxels}")
        print(f"Number of modified voxels: {len(engine.voxel_renderer.modified_voxels)}")

        print("\nRunning visualization engine...")
        # Run the engine
        try:
            engine.run()
        except KeyboardInterrupt:
            print("\nTest interrupted by user.")
        finally:
            # Clean up
            engine.cleanup()

        return 0
    except Exception as e:
        import traceback
        print(f"\nError: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
