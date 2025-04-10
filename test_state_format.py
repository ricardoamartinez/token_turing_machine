"""
Test script for TTMStateTracker standardized state format.

This script runs a sample forward pass through the TTM model and verifies that
the state tracker correctly captures and stores the model states in the standardized format.
"""

import torch
import pickle
from src.ttm.models.ttm_model import TokenTuringMachine
from src.ttm.visualization.state_tracker import TTMStateTracker
from src.ttm.data.tokenization import create_multiplication_example

def main():
    # Create a small TTM model for testing
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
    print("\nRunning forward pass...")
    with torch.no_grad():
        output_logits, _ = model(input_tensor)
    
    # Check if states were captured in standardized format
    print("\nChecking standardized state format...")
    state_key = (0, 0, 0)  # (epoch, batch, token)
    
    if state_key in tracker.state_history['states']:
        print(f"State captured for key {state_key}")
        
        # Check if modules were captured
        if 'modules' in tracker.state_history['states'][state_key]:
            modules = tracker.state_history['states'][state_key]['modules']
            
            # Find a module with inputs and outputs
            for module_name, module_data in modules.items():
                if 'inputs' in module_data and 'outputs' in module_data:
                    print(f"\nExample of standardized state format for module: {module_name}")
                    
                    # Print input state format
                    if isinstance(module_data['inputs'], list):
                        input_state = module_data['inputs'][0]
                    else:
                        input_state = module_data['inputs']
                    
                    print("\nInput state format:")
                    print(f"  name: {input_state['name']}")
                    print(f"  type: {input_state['type']}")
                    print(f"  shape: {input_state['shape']}")
                    print(f"  metadata: {input_state['metadata']}")
                    print(f"  data: <tensor of shape {input_state['data'].shape}>")
                    
                    # Print output state format
                    if isinstance(module_data['outputs'], list):
                        output_state = module_data['outputs'][0]
                    else:
                        output_state = module_data['outputs']
                    
                    print("\nOutput state format:")
                    print(f"  name: {output_state['name']}")
                    print(f"  type: {output_state['type']}")
                    print(f"  shape: {output_state['shape']}")
                    print(f"  metadata: {output_state['metadata']}")
                    print(f"  data: <tensor of shape {output_state['data'].shape}>")
                    
                    break
            
            # Check if gradients were captured
            if 'gradients' in tracker.state_history['states'][state_key]:
                gradients = tracker.state_history['states'][state_key]['gradients']
                if gradients:
                    # Print a sample gradient state
                    grad_name = list(gradients.keys())[0]
                    grad_state = gradients[grad_name]
                    
                    print("\nGradient state format:")
                    print(f"  name: {grad_state['name']}")
                    print(f"  type: {grad_state['type']}")
                    print(f"  shape: {grad_state['shape']}")
                    print(f"  metadata: {grad_state['metadata']}")
                    print(f"  data: <tensor of shape {grad_state['data'].shape}>")
    
    # Save the state history to a file
    tracker.save_state_history("test_standardized_state.pkl")
    print("\nState history saved to test_standardized_state.pkl")
    
    # Load the state history from the file to verify it can be serialized and deserialized
    print("\nLoading state history from file...")
    with open("./visualization_data/test_standardized_state.pkl", 'rb') as f:
        loaded_state = pickle.load(f)
    
    print("State history loaded successfully.")
    print(f"Number of states: {len(loaded_state['states'])}")

if __name__ == "__main__":
    main()
