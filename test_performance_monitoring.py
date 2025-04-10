"""
Test script for performance monitoring and adaptive rendering.

This script tests the performance monitoring and adaptive rendering features by creating a visualization engine,
loading a test state history with a large number of voxels, and monitoring the FPS and detail level.
"""

import os
import sys
import numpy as np
import pickle
import random
import time

from src.ttm.visualization.visualization_engine import VisualizationEngine


def create_test_state_history(filepath: str) -> None:
    """Create a test state history file with a large number of voxels.
    
    Args:
        filepath: Path to save the state history file
    """
    # Create a state history dictionary
    state_history = {
        'epochs': [],
        'batches': {},
        'tokens': {},
        'states': {}
    }
    
    # Create a single state with a large tensor
    epoch = 0
    batch = 0
    token = 0
    
    # Create state key
    state_key = (epoch, batch, token)
    
    # Create a state dictionary
    state_dict = {
        'modules': {},
        'gradients': {}
    }
    
    # Create a module dictionary
    module_dict = {}
    
    # Create a tensor state with a large tensor
    tensor_size = 100  # 100x100x100 = 1,000,000 voxels
    tensor_state = {
        'name': 'large_tensor',
        'type': 'tensor',
        'shape': (tensor_size, tensor_size, tensor_size),
        'data': np.random.rand(tensor_size, tensor_size, tensor_size),
        'metadata': {
            'epoch': epoch,
            'batch': batch,
            'token': token,
            'module': 'test_module',
            'is_input': True
        }
    }
    
    # Add the tensor state to the module dictionary
    module_dict['inputs'] = tensor_state
    
    # Add the module dictionary to the state dictionary
    state_dict['modules']['test_module'] = module_dict
    
    # Add the state dictionary to the state history
    state_history['states'][state_key] = state_dict
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save the state history
    with open(filepath, 'wb') as f:
        pickle.dump(state_history, f)
    
    print(f"Created test state history file: {filepath}")
    print(f"Number of states: {len(state_history['states'])}")
    print(f"Tensor size: {tensor_size}x{tensor_size}x{tensor_size} = {tensor_size**3} voxels")


def main():
    """Run the test."""
    try:
        # Create a test state history file
        test_filepath = './visualization_data/test_performance.pkl'
        create_test_state_history(test_filepath)
        
        print("\nCreating visualization engine...")
        # Create the visualization engine
        engine = VisualizationEngine(width=1280, height=720, caption="Performance Monitoring Test")
        
        print("\nLoading state history...")
        # Load the state history
        engine.load_state_history(test_filepath)
        
        print("\nRunning visualization engine...")
        print("Instructions:")
        print("1. Observe the FPS and detail level in the performance monitoring window")
        print("2. Toggle adaptive rendering on/off to see the effect on FPS")
        print("3. Adjust the target FPS to see the effect on detail level")
        
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
