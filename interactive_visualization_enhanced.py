"""
Enhanced Interactive Visualization for TTM model.

This script runs the enhanced interactive visualization dashboard for the TTM model,
providing a comprehensive interface for visualizing and manipulating model states.
"""

import argparse
import os
import torch
import numpy as np
import pickle
from datetime import datetime
import dash
from dash import html, dcc, callback_context
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px

from src.ttm.visualization.theme import COLORS, STYLES, create_figure_layout
from src.ttm.visualization.components import (
    card, sidebar, main_content, button, input_field, dropdown, slider, tabs,
    create_memory_heatmap, create_attention_heatmap, create_parameter_histogram,
    create_timeline_plot, create_state_transition_graph
)
from src.ttm.visualization.dashboard import ModernTTMDashboard
from src.ttm.visualization.training_callbacks import register_training_callbacks
from src.ttm.visualization.inference_callbacks import register_inference_callbacks
from src.ttm.visualization.experiment_callbacks import register_experiment_callbacks


def main():
    """Run the enhanced interactive visualization dashboard."""
    parser = argparse.ArgumentParser(description='Enhanced TTM Interactive Visualization')
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
    dashboard = ModernTTMDashboard(
        state_history_path=args.state_history,
        model_path=args.model,
        port=args.port
    )
    
    # Register callbacks
    register_training_callbacks(dashboard.app, dashboard)
    register_inference_callbacks(dashboard.app, dashboard)
    register_experiment_callbacks(dashboard.app, dashboard)
    
    # Run server
    print(f"Starting enhanced dashboard on http://localhost:{args.port}")
    dashboard.run_server(debug=args.debug)


if __name__ == '__main__':
    main()
