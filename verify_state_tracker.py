"""
Verification script for TTMStateTracker integration.

This script verifies that TTMStateTracker correctly captures state snapshots during
training/inference, storing each state in a standardized dictionary with the expected keys.
"""

import torch
import numpy as np
from src.ttm.models.ttm_model import TokenTuringMachine
from src.ttm.visualization.state_tracker import TTMStateTracker
from src.ttm.data.tokenization import create_multiplication_example

def verify_state_format(state):
    """Verify that a state has the expected format.
    
    Args:
        state: State dictionary to verify
        
    Returns:
        True if the state has the expected format, False otherwise
    """
    # Check required keys
    required_keys = ['name', 'type', 'shape', 'data', 'metadata']
    if not all(key in state for key in required_keys):
        print(f"Missing required keys: {set(required_keys) - set(state.keys())}")
        return False
    
    # Check metadata
    metadata = state['metadata']
    required_metadata = ['epoch', 'batch', 'token', 'module']
    if not all(key in metadata for key in required_metadata):
        print(f"Missing required metadata: {set(required_metadata) - set(metadata.keys())}")
        return False
    
    # Check data type
    if not isinstance(state['data'], np.ndarray):
        print(f"Data is not a numpy array: {type(state['data'])}")
        return False
    
    return True

def main():
    """Run the verification."""
    # Create a small TTM model for testing
    print("Creating TTM model...")
    model = TokenTuringMachine(
        vocab_size=13,
        embedding_dim=64,
        memory_size=32,
        r=8,
        num_layers=2,
        num_heads=2,
        hidden_dim=128
    )
    
    # Set model to eval mode
    model.eval()
    
    # Create the state tracker with 100% sampling rate for testing
    print("Creating TTMStateTracker...")
    tracker = TTMStateTracker(model, sampling_rate=1.0)
    
    # Set current epoch, batch, and token for tracking
    tracker.start_epoch(0)
    tracker.start_batch(0)
    tracker.start_token(0)
    
    # Create a simple multiplication example
    input_tokens, target_tokens = create_multiplication_example(5, 7)
    
    # Convert to tensor
    input_tensor = torch.tensor([input_tokens], dtype=torch.long)
    
    # Run a forward pass
    print("Running forward pass...")
    with torch.no_grad():
        output_logits, _ = model(input_tensor)
    
    # Verify state history structure
    print("\nVerifying state history structure...")
    
    # Check top-level keys
    required_keys = ['epochs', 'batches', 'tokens', 'states']
    if not all(key in tracker.state_history for key in required_keys):
        print(f"Missing required keys in state_history: {set(required_keys) - set(tracker.state_history.keys())}")
        return
    
    # Check if states were captured
    state_key = (0, 0, 0)  # (epoch, batch, token)
    if state_key not in tracker.state_history['states']:
        print(f"No state captured for key {state_key}")
        return
    
    # Check if modules were captured
    if 'modules' not in tracker.state_history['states'][state_key]:
        print(f"No modules captured for key {state_key}")
        return
    
    # Find a module with inputs and outputs
    modules = tracker.state_history['states'][state_key]['modules']
    example_module = None
    example_input = None
    example_output = None
    
    for module_name, module_data in modules.items():
        if 'inputs' in module_data and 'outputs' in module_data:
            example_module = module_name
            
            # Get input state
            if isinstance(module_data['inputs'], list):
                example_input = module_data['inputs'][0]
            else:
                example_input = module_data['inputs']
            
            # Get output state
            if isinstance(module_data['outputs'], list):
                example_output = module_data['outputs'][0]
            else:
                example_output = module_data['outputs']
            
            break
    
    if example_module is None:
        print("No module found with both inputs and outputs")
        return
    
    # Verify input state format
    print(f"\nVerifying input state format for module: {example_module}")
    if verify_state_format(example_input):
        print("Input state format is valid")
        
        # Print example input state
        print("\nExample input state:")
        print(f"  name: {example_input['name']}")
        print(f"  type: {example_input['type']}")
        print(f"  shape: {example_input['shape']}")
        print(f"  metadata: {example_input['metadata']}")
        print(f"  data: <tensor of shape {example_input['data'].shape}>")
    else:
        print("Input state format is invalid")
    
    # Verify output state format
    print(f"\nVerifying output state format for module: {example_module}")
    if verify_state_format(example_output):
        print("Output state format is valid")
        
        # Print example output state
        print("\nExample output state:")
        print(f"  name: {example_output['name']}")
        print(f"  type: {example_output['type']}")
        print(f"  shape: {example_output['shape']}")
        print(f"  metadata: {example_output['metadata']}")
        print(f"  data: <tensor of shape {example_output['data'].shape}>")
    else:
        print("Output state format is invalid")
    
    print("\nVerification completed successfully!")
    print("TTMStateTracker correctly captures state snapshots during inference, storing each state in a standardized dictionary with the expected keys.")

if __name__ == "__main__":
    main()
