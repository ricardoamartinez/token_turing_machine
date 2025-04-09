"""
Interactive Dashboard for TTM model.

This module provides an interactive dashboard for visualizing and manipulating TTM model states.
"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import torch
import os
import json
from typing import Dict, List, Tuple, Any, Optional, Union
import pickle
from datetime import datetime

from src.ttm.visualization.visualization_utils import (
    create_state_transition_graph,
    create_memory_heatmap,
    create_attention_heatmap,
    create_parameter_histogram,
    create_timeline_plot,
    tokens_to_labels
)


class TTMDashboard:
    """
    Interactive dashboard for visualizing and manipulating TTM model states.

    This class provides a web-based dashboard for exploring model states,
    running inference, and experimenting with state modifications.
    """

    def __init__(self,
                 state_history_path: Optional[str] = None,
                 model_path: Optional[str] = None,
                 port: int = 8050):
        """
        Initialize the dashboard.

        Args:
            state_history_path: Optional path to state history file
            model_path: Optional path to model checkpoint
            port: Port to run the dashboard on
        """
        self.app = dash.Dash(__name__,
                            external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'],
                            suppress_callback_exceptions=True)
        self.port = port

        # State management
        self.state_history = {}
        self.current_state = {}
        self.modified_states = {}
        self.inference_state = None
        self.model = None

        # Load state history if provided
        if state_history_path and os.path.exists(state_history_path):
            self._load_state_history(state_history_path)

        # Load model if provided
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)

        # Setup layout and callbacks
        self._setup_layout()
        self._setup_callbacks()

    def _load_state_history(self, filepath: str):
        """
        Load state history from file.

        Args:
            filepath: Path to state history file
        """
        try:
            with open(filepath, 'rb') as f:
                self.state_history = pickle.load(f)
            print(f"State history loaded from {filepath}")
        except Exception as e:
            print(f"Error loading state history: {e}")
            self.state_history = {
                'epochs': [],
                'batches': {},
                'tokens': {},
                'states': {}
            }

    def _load_model(self, filepath: str):
        """
        Load model from checkpoint.

        Args:
            filepath: Path to model checkpoint
        """
        try:
            # In a real implementation, we would need to initialize the model first
            # and then load the state dict
            from src.ttm.model.ttm import TTM

            # Create a dummy model for demonstration purposes
            self.model = TTM(
                vocab_size=20,  # Placeholder value
                d_model=128,    # Placeholder value
                memory_size=16, # Placeholder value
                n_layers=4      # Placeholder value
            )

            # Load the state dict
            checkpoint = torch.load(filepath)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Try loading directly if it's just the state dict
                self.model.load_state_dict(checkpoint)

            print(f"Model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def _setup_layout(self):
        """Set up the dashboard layout."""
        self.app.layout = html.Div([
            # Header with mode selection
            html.Div([
                html.H1("TTM Interactive Dashboard", style={"color": "white", "marginBottom": "20px"}),
                html.Div([
                    dcc.Tabs(id='mode-tabs', value='training', children=[
                        dcc.Tab(label='Training History', value='training'),
                        dcc.Tab(label='Inference Testing', value='inference'),
                        dcc.Tab(label='State Experimentation', value='experiment')
                    ], style={'color': 'white'})
                ], style={'marginBottom': '20px'})
            ], style={'backgroundColor': '#000000', 'padding': '20px', 'borderBottom': '1px solid #333'}),

            # Dynamic content area
            html.Div(id='mode-content'),

            # Hidden divs for storing state
            html.Div(id='current-state-storage', style={'display': 'none'}),
            html.Div(id='history-storage', style={'display': 'none'}),
            html.Div(id='modified-state-storage', style={'display': 'none'})
        ], style={'backgroundColor': '#000000', 'color': 'white', 'minHeight': '100vh'})

    def _setup_callbacks(self):
        """Set up the dashboard callbacks."""
        # Mode selection callback
        @self.app.callback(
            Output('mode-content', 'children'),
            [Input('mode-tabs', 'value')]
        )
        def update_mode_content(mode):
            if mode == 'training':
                return self._generate_training_mode_layout()
            elif mode == 'inference':
                return self._generate_inference_mode_layout()
            elif mode == 'experiment':
                return self._generate_experiment_mode_layout()
            else:
                return html.Div("Invalid mode selected")

    def _generate_training_mode_layout(self):
        """Generate layout for training history mode."""
        # Check if we have training history data
        if not self.state_history or not self.state_history.get('epochs', []):
            return html.Div([
                html.H2("No Training History Available", style={'textAlign': 'center'}),
                html.P("Please load a state history file to view training data.",
                      style={'textAlign': 'center'})
            ])

        # Get available epochs, batches, and tokens
        epochs = self.state_history.get('epochs', [])
        max_epoch = max(epochs) if epochs else 0

        # Create layout
        return html.Div([
            # Training history navigation
            html.Div([
                html.Div([
                    html.H3("Training Navigation"),
                    html.Div([
                        html.Label("Epoch:"),
                        dcc.Slider(
                            id='epoch-slider',
                            min=0,
                            max=max_epoch,
                            value=0,
                            marks={i: str(i) for i in range(0, max_epoch+1, max(1, max_epoch//10))},
                            step=1
                        )
                    ], style={'marginBottom': '20px'}),

                    html.Div([
                        html.Label("Batch:"),
                        dcc.Slider(
                            id='batch-slider',
                            min=0,
                            max=0,  # Will be updated by callback
                            value=0,
                            step=1
                        )
                    ], style={'marginBottom': '20px'}),

                    html.Div([
                        html.Label("Token:"),
                        dcc.Slider(
                            id='token-slider',
                            min=0,
                            max=0,  # Will be updated by callback
                            value=0,
                            step=1
                        )
                    ], style={'marginBottom': '20px'}),

                    html.Div([
                        html.Button('Play/Pause', id='play-button', n_clicks=0,
                                   style={'marginRight': '10px'}),
                        html.Button('Step Forward', id='step-forward', n_clicks=0,
                                   style={'marginRight': '10px'}),
                        html.Button('Step Backward', id='step-backward', n_clicks=0)
                    ])
                ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top',
                          'padding': '20px', 'backgroundColor': '#111111', 'borderRadius': '5px'}),

                html.Div([
                    html.H3("Training Metrics"),
                    dcc.Graph(id='training-metrics-graph')
                ], style={'width': '70%', 'display': 'inline-block', 'verticalAlign': 'top',
                          'padding': '20px', 'marginLeft': '20px', 'backgroundColor': '#111111',
                          'borderRadius': '5px'})
            ], style={'marginBottom': '20px'}),

            # State visualization
            html.Div([
                html.H2("State Visualization", style={'textAlign': 'center'}),

                # State transition graph
                html.Div([
                    html.H3("Process Flow"),
                    dcc.Graph(id='state-transition-graph', style={'height': '300px'})
                ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top',
                          'padding': '20px', 'backgroundColor': '#111111', 'borderRadius': '5px'}),

                # Current state details
                html.Div([
                    html.H3("Current State Details"),
                    html.Div(id='current-state-details')
                ], style={'width': '65%', 'display': 'inline-block', 'verticalAlign': 'top',
                          'padding': '20px', 'marginLeft': '20px', 'backgroundColor': '#111111',
                          'borderRadius': '5px'})
            ], style={'marginBottom': '20px'}),

            # Detailed visualizations
            html.Div([
                dcc.Tabs([
                    dcc.Tab(label='Memory', children=[
                        html.Div([
                            html.Div([
                                html.H4("Memory Controls"),
                                html.Label("Memory Slot:"),
                                dcc.Slider(
                                    id='memory-slot-slider',
                                    min=0,
                                    max=15,  # Assuming 16 memory slots
                                    value=0,
                                    step=1
                                )
                            ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                            html.Div([
                                dcc.Graph(id='memory-heatmap')
                            ], style={'width': '70%', 'display': 'inline-block', 'verticalAlign': 'top'})
                        ]),
                        dcc.Graph(id='memory-timeline')
                    ]),

                    dcc.Tab(label='Attention', children=[
                        html.Div([
                            html.Div([
                                html.H4("Attention Controls"),
                                html.Label("Layer:"),
                                dcc.Slider(
                                    id='attention-layer-slider',
                                    min=0,
                                    max=3,  # Assuming 4 layers
                                    value=0,
                                    step=1
                                ),
                                html.Label("Head:"),
                                dcc.Slider(
                                    id='attention-head-slider',
                                    min=0,
                                    max=3,  # Assuming 4 heads
                                    value=0,
                                    step=1
                                )
                            ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                            html.Div([
                                dcc.Graph(id='attention-heatmap')
                            ], style={'width': '70%', 'display': 'inline-block', 'verticalAlign': 'top'})
                        ]),
                        dcc.Graph(id='attention-timeline')
                    ]),

                    dcc.Tab(label='Parameters', children=[
                        html.Div([
                            html.Div([
                                html.H4("Parameter Controls"),
                                html.Label("Layer:"),
                                dcc.Dropdown(
                                    id='parameter-layer-dropdown',
                                    options=[
                                        {'label': 'Embedding', 'value': 'embedding'},
                                        {'label': 'Memory Module', 'value': 'memory_module'},
                                        {'label': 'Transformer Layer 1', 'value': 'transformer_1'},
                                        {'label': 'Transformer Layer 2', 'value': 'transformer_2'},
                                        {'label': 'Transformer Layer 3', 'value': 'transformer_3'},
                                        {'label': 'Transformer Layer 4', 'value': 'transformer_4'},
                                        {'label': 'Output Projection', 'value': 'output_proj'}
                                    ],
                                    value='embedding'
                                )
                            ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                            html.Div([
                                dcc.Graph(id='parameter-histogram')
                            ], style={'width': '70%', 'display': 'inline-block', 'verticalAlign': 'top'})
                        ]),
                        dcc.Graph(id='gradient-histogram')
                    ])
                ], style={'color': 'white', 'backgroundColor': '#111111'})
            ])
        ])

    def _generate_inference_mode_layout(self):
        """Generate layout for inference testing mode."""
        # Get available checkpoints
        checkpoints = []
        if os.path.exists('./checkpoints'):
            for file in os.listdir('./checkpoints'):
                if file.endswith('.pt'):
                    checkpoints.append(os.path.join('./checkpoints', file))

        return html.Div([
            # Checkpoint selection and input
            html.Div([
                html.Div([
                    html.H3("Model Selection"),
                    html.Label("Checkpoint:"),
                    dcc.Dropdown(
                        id='checkpoint-dropdown',
                        options=[{'label': os.path.basename(cp), 'value': cp} for cp in checkpoints],
                        value=checkpoints[0] if checkpoints else None,
                        style={'color': 'black'}
                    ),
                    html.Button('Load Model', id='load-model-button', n_clicks=0,
                               style={'marginTop': '10px'})
                ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top',
                          'padding': '20px', 'backgroundColor': '#111111', 'borderRadius': '5px'}),

                html.Div([
                    html.H3("Test Input"),
                    html.Div([
                        html.Label("First Number:"),
                        dcc.Input(id='num1-input', type='number', value=5,
                                 style={'marginRight': '10px', 'width': '100px'}),
                        html.Label("Second Number:"),
                        dcc.Input(id='num2-input', type='number', value=7,
                                 style={'width': '100px'})
                    ], style={'marginBottom': '20px'}),
                    html.Button('Run Inference', id='run-inference-button', n_clicks=0)
                ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top',
                          'padding': '20px', 'marginLeft': '20px', 'backgroundColor': '#111111',
                          'borderRadius': '5px'}),

                html.Div([
                    html.H3("Results"),
                    html.Div(id='inference-results')
                ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top',
                          'padding': '20px', 'marginLeft': '20px', 'backgroundColor': '#111111',
                          'borderRadius': '5px'})
            ], style={'marginBottom': '20px'}),

            # Token navigation for inference
            html.Div([
                html.H3("Inference Step Navigation"),
                html.Div([
                    html.Label("Token:"),
                    dcc.Slider(
                        id='inference-token-slider',
                        min=0,
                        max=20,  # Default max tokens
                        value=0,
                        step=1
                    )
                ], style={'marginBottom': '20px'}),

                html.Div([
                    html.Button('Step Forward', id='inference-step-forward', n_clicks=0,
                               style={'marginRight': '10px'}),
                    html.Button('Step Backward', id='inference-step-backward', n_clicks=0)
                ])
            ], style={'marginBottom': '20px', 'padding': '20px', 'backgroundColor': '#111111',
                      'borderRadius': '5px'}),

            # State visualization
            html.Div([
                html.H2("State Visualization", style={'textAlign': 'center'}),
                dcc.Tabs([
                    dcc.Tab(label='Memory', children=[
                        html.Div([
                            html.Div([
                                html.H4("Memory Controls"),
                                html.Label("Memory Slot:"),
                                dcc.Slider(
                                    id='inference-memory-slot-slider',
                                    min=0,
                                    max=15,  # Assuming 16 memory slots
                                    value=0,
                                    step=1
                                )
                            ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                            html.Div([
                                dcc.Graph(id='inference-memory-heatmap')
                            ], style={'width': '70%', 'display': 'inline-block', 'verticalAlign': 'top'})
                        ])
                    ]),

                    dcc.Tab(label='Attention', children=[
                        html.Div([
                            html.Div([
                                html.H4("Attention Controls"),
                                html.Label("Layer:"),
                                dcc.Slider(
                                    id='inference-attention-layer-slider',
                                    min=0,
                                    max=3,  # Assuming 4 layers
                                    value=0,
                                    step=1
                                ),
                                html.Label("Head:"),
                                dcc.Slider(
                                    id='inference-attention-head-slider',
                                    min=0,
                                    max=3,  # Assuming 4 heads
                                    value=0,
                                    step=1
                                )
                            ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                            html.Div([
                                dcc.Graph(id='inference-attention-heatmap')
                            ], style={'width': '70%', 'display': 'inline-block', 'verticalAlign': 'top'})
                        ])
                    ])
                ], style={'color': 'white', 'backgroundColor': '#111111'})
            ])
        ])

    def _generate_experiment_mode_layout(self):
        """Generate layout for state experimentation mode."""
        return html.Div([
            # Source state selection
            html.Div([
                html.Div([
                    html.H3("Source State Selection"),
                    html.Label("Source:"),
                    dcc.RadioItems(
                        id='source-selection',
                        options=[
                            {'label': 'Training History', 'value': 'training'},
                            {'label': 'Inference Result', 'value': 'inference'}
                        ],
                        value='training'
                    )
                ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top',
                          'padding': '20px', 'backgroundColor': '#111111', 'borderRadius': '5px'}),

                html.Div([
                    html.H3("State Selection"),
                    html.Label("Component:"),
                    dcc.Dropdown(
                        id='component-dropdown',
                        options=[
                            {'label': 'Memory', 'value': 'memory'},
                            {'label': 'Attention Weights', 'value': 'attention'}
                        ],
                        value='memory',
                        style={'color': 'black'}
                    )
                ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top',
                          'padding': '20px', 'marginLeft': '20px', 'backgroundColor': '#111111',
                          'borderRadius': '5px'}),

                html.Div([
                    html.H3("Experiment Controls"),
                    html.Button('Load State', id='load-state-button', n_clicks=0,
                               style={'marginBottom': '10px', 'width': '100%'}),
                    html.Button('Apply Changes', id='apply-changes-button', n_clicks=0,
                               style={'marginBottom': '10px', 'width': '100%'}),
                    html.Button('Run Forward Pass', id='run-forward-button', n_clicks=0,
                               style={'marginBottom': '10px', 'width': '100%'}),
                    html.Button('Reset Experiment', id='reset-experiment-button', n_clicks=0,
                               style={'width': '100%'})
                ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top',
                          'padding': '20px', 'marginLeft': '20px', 'backgroundColor': '#111111',
                          'borderRadius': '5px'})
            ], style={'marginBottom': '20px'}),

            # State editor and visualization
            html.Div([
                html.Div([
                    html.H3("State Editor"),
                    html.Div(id='state-editor', children=[
                        html.P("Select a state and click 'Load State' to begin editing.")
                    ])
                ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top',
                          'padding': '20px', 'backgroundColor': '#111111', 'borderRadius': '5px'}),

                html.Div([
                    html.H3("State Visualization"),
                    html.Div(id='experiment-visualization', children=[
                        html.P("No state loaded for visualization.")
                    ])
                ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top',
                          'padding': '20px', 'marginLeft': '20px', 'backgroundColor': '#111111',
                          'borderRadius': '5px'})
            ], style={'marginBottom': '20px'}),

            # Results comparison
            html.Div([
                html.H3("Results Comparison"),
                html.Div([
                    html.Div([
                        html.H4("Original Output"),
                        html.Div(id='original-output', children=[
                            html.P("No original output available.")
                        ])
                    ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                    html.Div([
                        html.H4("Modified Output"),
                        html.Div(id='modified-output', children=[
                            html.P("No modified output available.")
                        ])
                    ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top',
                              'marginLeft': '20px'})
                ])
            ], style={'padding': '20px', 'backgroundColor': '#111111', 'borderRadius': '5px'})
        ])

    def run_server(self, debug: bool = True):
        """
        Run the dashboard server.

        Args:
            debug: Whether to run in debug mode
        """
        self.app.run(debug=debug, port=self.port)
