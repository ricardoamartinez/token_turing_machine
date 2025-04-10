"""
Test script for TTMStateTracker hook registration.

This script instantiates a TTMStateTracker and prints the list of target module names
where hooks are registered.
"""

import torch
from src.ttm.models.ttm_model import TokenTuringMachine
from src.ttm.visualization.state_tracker import TTMStateTracker

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
    
    # Create the state tracker
    tracker = TTMStateTracker(model)
    
    # Print the target modules
    print("\nTarget modules for hook registration:")
    for module_type, module_names in tracker.target_modules.items():
        print(f"  {module_type}: {module_names}")
    
    # Print the number of hooks registered
    print(f"\nTotal hooks registered: {len(tracker.hooks)}")
    
    # Print the model structure to see available modules
    print("\nModel structure:")
    for name, module in model.named_modules():
        print(f"  {name}: {type(module).__name__}")

if __name__ == "__main__":
    main()
