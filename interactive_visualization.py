"""
Interactive Visualization for TTM model.

This script runs the interactive visualization dashboard for the TTM model.
"""

import argparse
import os
import torch
import numpy as np
import pickle
from datetime import datetime

from src.ttm.visualization.interactive_dashboard import TTMDashboard
from src.ttm.visualization.training_callbacks import register_training_callbacks
from src.ttm.visualization.inference_callbacks import register_inference_callbacks


def main():
    """Run the interactive visualization dashboard."""
    parser = argparse.ArgumentParser(description='TTM Interactive Visualization')
    parser.add_argument('--state_history', type=str, default=None,
                        help='Path to state history file')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--port', type=int, default=8050,
                        help='Port to run the dashboard on')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Create dashboard
    dashboard = TTMDashboard(
        state_history_path=args.state_history,
        model_path=args.model,
        port=args.port
    )
    
    # Register callbacks
    register_training_callbacks(dashboard.app, dashboard)
    register_inference_callbacks(dashboard.app, dashboard)
    
    # Run server
    print(f"Starting dashboard on http://localhost:{args.port}")
    dashboard.run_server(debug=args.debug)


if __name__ == '__main__':
    main()
