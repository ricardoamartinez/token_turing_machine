"""
Test script for state timeline/playback controls.

This script tests the state timeline/playback controls by creating a visualization engine,
loading a test state history, and allowing the user to navigate through the timeline.
"""

import os
import sys
import numpy as np
import pickle
import random

from src.ttm.visualization.visualization_engine import VisualizationEngine


def create_test_state_history(filepath: str) -> None:
    """Create a test state history file with multiple epochs, batches, and tokens.
    
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
    
    # Create multiple epochs, batches, and tokens
    num_epochs = 3
    num_batches = 5
    num_tokens = 10
    
    for epoch in range(num_epochs):
        for batch in range(num_batches):
            for token in range(num_tokens):
                # Create state key
                state_key = (epoch, batch, token)
                
                # Create a state dictionary
                state_dict = {
                    'modules': {},
                    'gradients': {}
                }
                
                # Create a module dictionary
                module_dict = {}
                
                # Create a tensor state
                tensor_state = {
                    'name': f'tensor_{epoch}_{batch}_{token}',
                    'type': 'tensor',
                    'shape': (10, 10),
                    'data': np.random.rand(10, 10),
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
                
                # Create another tensor state
                tensor_state2 = {
                    'name': f'tensor2_{epoch}_{batch}_{token}',
                    'type': 'tensor',
                    'shape': (5, 5),
                    'data': np.random.rand(5, 5),
                    'metadata': {
                        'epoch': epoch,
                        'batch': batch,
                        'token': token,
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
                    'name': f'gradient_{epoch}_{batch}_{token}',
                    'type': 'tensor',
                    'shape': (10, 10),
                    'data': np.random.rand(10, 10),
                    'metadata': {
                        'epoch': epoch,
                        'batch': batch,
                        'token': token,
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
    print(f"Number of states: {len(state_history['states'])}")


def main():
    """Run the test."""
    try:
        # Create a test state history file
        test_filepath = './visualization_data/test_timeline.pkl'
        create_test_state_history(test_filepath)
        
        print("\nCreating visualization engine...")
        # Create the visualization engine
        engine = VisualizationEngine(width=1280, height=720, caption="Timeline Playback Test")
        
        print("\nLoading state history...")
        # Load the state history
        engine.load_state_history(test_filepath)
        
        print("\nRunning visualization engine...")
        print("Instructions:")
        print("1. Use the timeline sliders to navigate through epochs, batches, and tokens")
        print("2. Use the play/pause button to start/stop automatic playback")
        print("3. Use the step forward/backward buttons to navigate one step at a time")
        print("4. Use the speed slider to adjust playback speed")
        
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
