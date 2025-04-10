"""
Test script for interactive state editing.

This script tests the interactive state editing interface by creating a visualization engine,
loading a test state history, and allowing the user to edit state values.
"""

import os
import sys
import numpy as np
import pickle

from src.ttm.visualization.visualization_engine import VisualizationEngine


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
        engine = VisualizationEngine(width=1280, height=720, caption="Interactive State Editing Test")
        
        print("\nLoading state history...")
        # Load the state history
        engine.load_state_history(test_filepath)
        
        print("\nRunning visualization engine...")
        print("Instructions:")
        print("1. Hover over a voxel to see its information")
        print("2. Click on a voxel to open the editing interface")
        print("3. Use the slider to change the value")
        print("4. Click 'Apply' to apply the changes or 'Cancel' to revert")
        
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
