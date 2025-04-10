"""
Test script for TTMStateTracker state recording.

This script runs a sample forward pass through the TTM model and verifies that
the state tracker correctly captures and stores the model states.
"""

import torch
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
    
    # Check if states were captured
    print("\nChecking captured states...")
    state_key = (0, 0, 0)  # (epoch, batch, token)
    
    if state_key in tracker.state_history['states']:
        print(f"State captured for key {state_key}")
        
        # Check if modules were captured
        if 'modules' in tracker.state_history['states'][state_key]:
            modules = tracker.state_history['states'][state_key]['modules']
            print(f"Number of modules captured: {len(modules)}")
            
            # Print a sample of captured modules
            print("\nSample of captured modules:")
            for i, (module_name, module_data) in enumerate(modules.items()):
                print(f"  Module: {module_name}")
                
                # Print input shape if available
                if 'inputs' in module_data:
                    inputs = module_data['inputs']
                    if isinstance(inputs, list):
                        for j, inp in enumerate(inputs):
                            if hasattr(inp, 'shape'):
                                print(f"    Input {j} shape: {inp.shape}")
                    elif hasattr(inputs, 'shape'):
                        print(f"    Input shape: {inputs.shape}")
                
                # Print output shape if available
                if 'outputs' in module_data:
                    outputs = module_data['outputs']
                    if isinstance(outputs, list):
                        for j, out in enumerate(outputs):
                            if hasattr(out, 'shape'):
                                print(f"    Output {j} shape: {out.shape}")
                    elif hasattr(outputs, 'shape'):
                        print(f"    Output shape: {outputs.shape}")
                
                # Only print a few modules to avoid overwhelming output
                if i >= 2:
                    print("  ...")
                    break
            
            # Print metadata stored with the state
            print("\nMetadata stored with the state:")
            print(f"  Epoch: {tracker.current_epoch}")
            print(f"  Batch: {tracker.current_batch}")
            print(f"  Token: {tracker.current_token}")
        else:
            print("No modules captured")
    else:
        print(f"No state captured for key {state_key}")
    
    # Save the state history to a file
    tracker.save_state_history("test_state_history.pkl")
    print("\nState history saved to test_state_history.pkl")

if __name__ == "__main__":
    main()
