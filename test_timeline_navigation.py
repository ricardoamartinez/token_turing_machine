"""
Test script for timeline navigation.

This script tests the timeline navigation methods in the VisualizationEngine class.
"""

import os
import sys
import numpy as np
import pickle
import random

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


def main():
    """Run the test."""
    try:
        # Create a test state history file
        test_filepath = './visualization_data/test_navigation.pkl'
        create_test_state_history(test_filepath)
        
        # Create a mock visualization manager
        class MockVisualizationManager:
            def __init__(self):
                self.state_history = {
                    'epochs': [],
                    'batches': {},
                    'tokens': {},
                    'states': {}
                }
                self.current_state_key = None
                self.current_epoch = 0
                self.current_batch = 0
                self.current_token = 0
            
            def load_state_history(self, filepath):
                with open(filepath, 'rb') as f:
                    self.state_history = pickle.load(f)
                
                # Extract available epochs, batches, and tokens
                self.state_history['epochs'] = sorted(list({key[0] for key in self.state_history['states'].keys()}))
                
                # Extract available batches for each epoch
                for epoch in self.state_history['epochs']:
                    self.state_history['batches'][epoch] = sorted(list({key[1] for key in self.state_history['states'].keys() if key[0] == epoch}))
                
                # Extract available tokens for each (epoch, batch) pair
                for epoch in self.state_history['epochs']:
                    for batch in self.state_history['batches'][epoch]:
                        key = (epoch, batch)
                        self.state_history['tokens'][key] = sorted(list({key[2] for key in self.state_history['states'].keys() if key[0] == epoch and key[1] == batch}))
                
                # Get the first state key
                if self.state_history['states']:
                    first_key = next(iter(self.state_history['states'].keys()))
                    self.current_state_key = first_key
                    self.current_epoch, self.current_batch, self.current_token = first_key
            
            def load_state(self, epoch, batch, token):
                self.current_epoch = epoch
                self.current_batch = batch
                self.current_token = token
                self.current_state_key = (epoch, batch, token)
                print(f"Loaded state for epoch {epoch}, batch {batch}, token {token}")
            
            def get_available_epochs(self):
                return self.state_history['epochs']
            
            def get_available_batches(self, epoch):
                if epoch in self.state_history['batches']:
                    return self.state_history['batches'][epoch]
                return []
            
            def get_available_tokens(self, epoch, batch):
                key = (epoch, batch)
                if key in self.state_history['tokens']:
                    return self.state_history['tokens'][key]
                return []
        
        # Create a mock visualization engine
        class MockVisualizationEngine:
            def __init__(self):
                self.visualization_manager = MockVisualizationManager()
                self.current_epoch = 0
                self.current_batch = 0
                self.current_token = 0
            
            def load_state_history(self, filepath):
                self.visualization_manager.load_state_history(filepath)
                self.current_epoch = self.visualization_manager.current_epoch
                self.current_batch = self.visualization_manager.current_batch
                self.current_token = self.visualization_manager.current_token
            
            def _step_forward(self):
                # Get available epochs, batches, and tokens
                epochs = self.visualization_manager.get_available_epochs()
                batches = self.visualization_manager.get_available_batches(self.current_epoch)
                tokens = self.visualization_manager.get_available_tokens(self.current_epoch, self.current_batch)
                
                if not epochs or not batches or not tokens:
                    return
                
                # Try to step forward in tokens
                if self.current_token < max(tokens):
                    self.current_token += 1
                # If at the end of tokens, try to step forward in batches
                elif self.current_batch < max(batches):
                    self.current_batch += 1
                    # Reset token to the beginning of the new batch
                    tokens = self.visualization_manager.get_available_tokens(self.current_epoch, self.current_batch)
                    if tokens:
                        self.current_token = min(tokens)
                # If at the end of batches, try to step forward in epochs
                elif self.current_epoch < max(epochs):
                    self.current_epoch += 1
                    # Reset batch and token to the beginning of the new epoch
                    batches = self.visualization_manager.get_available_batches(self.current_epoch)
                    if batches:
                        self.current_batch = min(batches)
                        tokens = self.visualization_manager.get_available_tokens(self.current_epoch, self.current_batch)
                        if tokens:
                            self.current_token = min(tokens)
                
                # Load state for new epoch/batch/token
                self.visualization_manager.load_state(self.current_epoch, self.current_batch, self.current_token)
            
            def _step_backward(self):
                # Get available epochs, batches, and tokens
                epochs = self.visualization_manager.get_available_epochs()
                batches = self.visualization_manager.get_available_batches(self.current_epoch)
                tokens = self.visualization_manager.get_available_tokens(self.current_epoch, self.current_batch)
                
                if not epochs or not batches or not tokens:
                    return
                
                # Try to step backward in tokens
                if self.current_token > min(tokens):
                    self.current_token -= 1
                # If at the beginning of tokens, try to step backward in batches
                elif self.current_batch > min(batches):
                    self.current_batch -= 1
                    # Reset token to the end of the previous batch
                    tokens = self.visualization_manager.get_available_tokens(self.current_epoch, self.current_batch)
                    if tokens:
                        self.current_token = max(tokens)
                # If at the beginning of batches, try to step backward in epochs
                elif self.current_epoch > min(epochs):
                    self.current_epoch -= 1
                    # Reset batch and token to the end of the previous epoch
                    batches = self.visualization_manager.get_available_batches(self.current_epoch)
                    if batches:
                        self.current_batch = max(batches)
                        tokens = self.visualization_manager.get_available_tokens(self.current_epoch, self.current_batch)
                        if tokens:
                            self.current_token = max(tokens)
                
                # Load state for new epoch/batch/token
                self.visualization_manager.load_state(self.current_epoch, self.current_batch, self.current_token)
        
        # Create a visualization engine
        print("\nCreating visualization engine...")
        engine = MockVisualizationEngine()
        
        # Load the state history
        print("\nLoading state history...")
        engine.load_state_history(test_filepath)
        
        # Test stepping forward
        print("\nTesting stepping forward...")
        print(f"Initial state: epoch {engine.current_epoch}, batch {engine.current_batch}, token {engine.current_token}")
        
        # Step forward 10 times
        for i in range(10):
            print(f"Step forward {i+1}...")
            engine._step_forward()
            print(f"Current state: epoch {engine.current_epoch}, batch {engine.current_batch}, token {engine.current_token}")
        
        # Test stepping backward
        print("\nTesting stepping backward...")
        
        # Step backward 10 times
        for i in range(10):
            print(f"Step backward {i+1}...")
            engine._step_backward()
            print(f"Current state: epoch {engine.current_epoch}, batch {engine.current_batch}, token {engine.current_token}")
        
        # Test stepping forward to the end
        print("\nTesting stepping forward to the end...")
        
        # Step forward until we reach the end
        max_steps = 1000  # Limit to avoid infinite loop
        step_count = 0
        last_state = (engine.current_epoch, engine.current_batch, engine.current_token)
        
        while step_count < max_steps:
            engine._step_forward()
            current_state = (engine.current_epoch, engine.current_batch, engine.current_token)
            
            if current_state == last_state:
                print(f"Reached the end at epoch {engine.current_epoch}, batch {engine.current_batch}, token {engine.current_token}")
                break
            
            last_state = current_state
            step_count += 1
        
        # Test stepping backward to the beginning
        print("\nTesting stepping backward to the beginning...")
        
        # Step backward until we reach the beginning
        max_steps = 1000  # Limit to avoid infinite loop
        step_count = 0
        last_state = (engine.current_epoch, engine.current_batch, engine.current_token)
        
        while step_count < max_steps:
            engine._step_backward()
            current_state = (engine.current_epoch, engine.current_batch, engine.current_token)
            
            if current_state == last_state:
                print(f"Reached the beginning at epoch {engine.current_epoch}, batch {engine.current_batch}, token {engine.current_token}")
                break
            
            last_state = current_state
            step_count += 1
        
        print("\nTest completed successfully!")
        return 0
    except Exception as e:
        import traceback
        print(f"\nError: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
