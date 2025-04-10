"""
Minimal test script for TTMStateTracker integration with TTMTrainer.
"""

import os
import sys
import traceback

def main():
    print("Starting minimal test script")

    try:
        print("Importing TTM modules...")
        import torch
        from torch.utils.data import DataLoader
        from src.ttm.models.ttm_model import TokenTuringMachine
        from src.ttm.training.trainer import TTMTrainer
        from src.ttm.data.multiplication_dataset import MultiplicationDataset
        from src.ttm.visualization.state_tracker import TTMStateTracker

        print("All modules imported successfully")

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
        print("TTM model created successfully")

        # Create a state tracker directly
        print("Creating state tracker...")
        tracker = TTMStateTracker(model, sampling_rate=1.0)
        print("State tracker created successfully")

        # Test state tracker methods
        print("Testing state tracker methods...")
        tracker.start_epoch(0)
        tracker.start_batch(0)
        tracker.start_token(0)

        # Create a dummy input
        print("Creating dummy input...")
        dummy_input = torch.randint(0, 13, (1, 5))
        print(f"Dummy input shape: {dummy_input.shape}")

        # Run a forward pass
        print("Running forward pass...")
        with torch.no_grad():
            output, _ = model(dummy_input)
        print("Forward pass completed successfully")

        # Check if states were captured
        print("Checking captured states...")
        state_key = (0, 0, 0)
        if state_key in tracker.state_history['states']:
            print(f"State captured for key {state_key}")
            print(f"Number of states: {len(tracker.state_history['states'])}")
        else:
            print(f"No state captured for key {state_key}")

        # Save state history
        print("Saving state history...")
        vis_dir = './visualization_data'
        os.makedirs(vis_dir, exist_ok=True)
        vis_file = os.path.join(vis_dir, 'test_minimal.pkl')
        tracker.save_state_history(vis_file)
        print(f"State history saved to {vis_file}")

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
