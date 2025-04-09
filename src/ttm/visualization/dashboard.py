"""
Modern TTM Interactive Dashboard.

This module provides a modern, modular dashboard for visualizing and manipulating TTM model states.
"""

import dash
from dash import html, dcc
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

from src.ttm.visualization.theme import COLORS, STYLES
from src.ttm.visualization.components import (
    card, sidebar, main_content, button, input_field, dropdown, slider, tabs,
    create_memory_heatmap, create_attention_heatmap, create_parameter_histogram,
    create_timeline_plot, create_state_transition_graph
)


class ModernTTMDashboard:
    """
    Modern interactive dashboard for visualizing and manipulating TTM model states.
    
    This class provides a web-based dashboard with a modern UI for exploring model states,
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
        self.app = dash.Dash(
            __name__, 
            external_stylesheets=[
                'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap'
            ],
            suppress_callback_exceptions=True
        )
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
        self.app.layout = html.Div(
            style=STYLES['page'],
            children=[
                # Left sidebar
                sidebar(
                    title="TTM Dashboard",
                    id="sidebar",
                    content=[
                        html.Div([
                            html.H3("Navigation", style=STYLES['heading']),
                            dcc.RadioItems(
                                id='mode-selector',
                                options=[
                                    {'label': 'Training History', 'value': 'training'},
                                    {'label': 'Inference Testing', 'value': 'inference'},
                                    {'label': 'State Experimentation', 'value': 'experiment'}
                                ],
                                value='training',
                                style={'marginBottom': '20px', 'color': COLORS['text']}
                            ),
                            
                            # Dynamic controls based on selected mode
                            html.Div(id='sidebar-controls')
                        ])
                    ]
                ),
                
                # Main content area
                main_content(
                    id="main-content",
                    content=[
                        # Header
                        html.Div([
                            html.H1("TTM Interactive Visualization", style={
                                'color': COLORS['text'],
                                'marginBottom': '20px',
                                'borderBottom': f'1px solid {COLORS["border"]}',
                                'paddingBottom': '10px'
                            }),
                            
                            # Status bar
                            html.Div(
                                id='status-bar',
                                style={
                                    'backgroundColor': COLORS['card'],
                                    'padding': '10px',
                                    'borderRadius': '4px',
                                    'marginBottom': '20px',
                                    'display': 'flex',
                                    'justifyContent': 'space-between',
                                    'alignItems': 'center'
                                },
                                children=[
                                    html.Div(id='status-message', children="Ready"),
                                    html.Div(id='status-actions')
                                ]
                            )
                        ]),
                        
                        # Dynamic content based on selected mode
                        html.Div(id='mode-content')
                    ]
                ),
                
                # Hidden divs for storing state
                html.Div(id='current-state-storage', style={'display': 'none'}),
                html.Div(id='history-storage', style={'display': 'none'}),
                html.Div(id='modified-state-storage', style={'display': 'none'})
            ]
        )
    
    def _setup_callbacks(self):
        """Set up the dashboard callbacks."""
        # Mode selection callback
        @self.app.callback(
            [Output('sidebar-controls', 'children'),
             Output('mode-content', 'children'),
             Output('status-message', 'children'),
             Output('status-actions', 'children')],
            [Input('mode-selector', 'value')]
        )
        def update_mode(mode):
            if mode == 'training':
                return (
                    self._generate_training_sidebar_controls(),
                    self._generate_training_mode_layout(),
                    "Training History Mode",
                    html.Div([
                        button("Load History", id="load-history-button", style={'marginRight': '10px'}),
                        button("Export Visualization", id="export-viz-button")
                    ])
                )
            elif mode == 'inference':
                return (
                    self._generate_inference_sidebar_controls(),
                    self._generate_inference_mode_layout(),
                    "Inference Testing Mode",
                    html.Div([
                        button("Load Model", id="load-model-button", style={'marginRight': '10px'}),
                        button("Run Inference", id="run-inference-button")
                    ])
                )
            elif mode == 'experiment':
                return (
                    self._generate_experiment_sidebar_controls(),
                    self._generate_experiment_mode_layout(),
                    "State Experimentation Mode",
                    html.Div([
                        button("Load State", id="load-state-button", style={'marginRight': '10px'}),
                        button("Apply Changes", id="apply-changes-button", style={'marginRight': '10px'}),
                        button("Reset", id="reset-experiment-button")
                    ])
                )
            else:
                return html.Div("Invalid mode selected"), html.Div("Invalid mode selected"), "Error", html.Div()
    
    def _generate_training_sidebar_controls(self):
        """Generate sidebar controls for training history mode."""
        # Check if we have training history data
        if not self.state_history or not self.state_history.get('epochs', []):
            return html.Div([
                html.P("No training history available.", style={'color': COLORS['text_muted']}),
                button("Load History", id="load-history-sidebar-button")
            ])
        
        # Get available epochs, batches, and tokens
        epochs = self.state_history.get('epochs', [])
        max_epoch = max(epochs) if epochs else 0
        
        return html.Div([
            slider(
                label="Epoch",
                id='epoch-slider',
                min=0,
                max=max_epoch,
                value=0,
                marks={i: str(i) for i in range(0, max_epoch+1, max(1, max_epoch//5))}
            ),
            
            slider(
                label="Batch",
                id='batch-slider',
                min=0,
                max=0,  # Will be updated by callback
                value=0
            ),
            
            slider(
                label="Token",
                id='token-slider',
                min=0,
                max=0,  # Will be updated by callback
                value=0
            ),
            
            html.Div(
                style={'display': 'flex', 'justifyContent': 'space-between', 'marginTop': '20px'},
                children=[
                    button("◀", id='step-backward', style={'width': '30%'}),
                    button("▶/❚❚", id='play-button', style={'width': '30%'}),
                    button("▶", id='step-forward', style={'width': '30%'})
                ]
            )
        ])
    
    def _generate_inference_sidebar_controls(self):
        """Generate sidebar controls for inference testing mode."""
        # Get available checkpoints
        checkpoints = []
        if os.path.exists('./checkpoints'):
            for file in os.listdir('./checkpoints'):
                if file.endswith('.pt'):
                    checkpoints.append(os.path.join('./checkpoints', file))
        
        return html.Div([
            dropdown(
                label="Model Checkpoint",
                id='checkpoint-dropdown',
                options=[{'label': os.path.basename(cp), 'value': cp} for cp in checkpoints],
                value=checkpoints[0] if checkpoints else None
            ),
            
            input_field(
                label="First Number",
                id='num1-input',
                type="number",
                value=5
            ),
            
            input_field(
                label="Second Number",
                id='num2-input',
                type="number",
                value=7
            ),
            
            slider(
                label="Token Position",
                id='inference-token-slider',
                min=0,
                max=20,  # Default max tokens
                value=0
            ),
            
            html.Div(
                style={'display': 'flex', 'justifyContent': 'space-between', 'marginTop': '20px'},
                children=[
                    button("◀", id='inference-step-backward', style={'width': '48%'}),
                    button("▶", id='inference-step-forward', style={'width': '48%'})
                ]
            )
        ])
    
    def _generate_experiment_sidebar_controls(self):
        """Generate sidebar controls for state experimentation mode."""
        return html.Div([
            dropdown(
                label="Source",
                id='source-selection',
                options=[
                    {'label': 'Training History', 'value': 'training'},
                    {'label': 'Inference Result', 'value': 'inference'}
                ],
                value='training'
            ),
            
            dropdown(
                label="Component",
                id='component-dropdown',
                options=[
                    {'label': 'Memory', 'value': 'memory'},
                    {'label': 'Attention Weights', 'value': 'attention'},
                    {'label': 'Transformer Output', 'value': 'transformer'},
                    {'label': 'Model Parameters', 'value': 'parameters'}
                ],
                value='memory'
            ),
            
            html.Div(id='component-controls')
        ])
    
    def _generate_training_mode_layout(self):
        """Generate layout for training history mode."""
        # Check if we have training history data
        if not self.state_history or not self.state_history.get('epochs', []):
            return card(
                title="No Training History Available",
                content=html.P("Please load a state history file to view training data.")
            )
        
        # Create layout with multiple visualizations
        return html.Div([
            # Top row - Overview
            html.Div(
                style={'display': 'flex', 'marginBottom': '20px'},
                children=[
                    # Training metrics
                    html.Div(
                        style={'width': '60%', 'marginRight': '20px'},
                        children=[
                            card(
                                title="Training Metrics",
                                content=dcc.Graph(id='training-metrics-graph')
                            )
                        ]
                    ),
                    
                    # State transition graph
                    html.Div(
                        style={'width': '40%'},
                        children=[
                            card(
                                title="Process Flow",
                                content=dcc.Graph(id='state-transition-graph')
                            )
                        ]
                    )
                ]
            ),
            
            # Middle row - Memory and Attention
            html.Div(
                style={'display': 'flex', 'marginBottom': '20px'},
                children=[
                    # Memory visualization
                    html.Div(
                        style={'width': '50%', 'marginRight': '20px'},
                        children=[
                            card(
                                title="Memory Visualization",
                                content=[
                                    html.Div(
                                        style={'display': 'flex', 'marginBottom': '10px'},
                                        children=[
                                            html.Div(
                                                style={'width': '30%', 'marginRight': '10px'},
                                                children=[
                                                    html.Label("Memory Slot:"),
                                                    dcc.Slider(
                                                        id='memory-slot-slider',
                                                        min=0,
                                                        max=15,  # Assuming 16 memory slots
                                                        value=0,
                                                        step=1
                                                    )
                                                ]
                                            ),
                                            html.Div(
                                                style={'width': '70%'},
                                                children=[
                                                    dcc.Graph(id='memory-heatmap')
                                                ]
                                            )
                                        ]
                                    ),
                                    dcc.Graph(id='memory-timeline')
                                ]
                            )
                        ]
                    ),
                    
                    # Attention visualization
                    html.Div(
                        style={'width': '50%'},
                        children=[
                            card(
                                title="Attention Visualization",
                                content=[
                                    html.Div(
                                        style={'display': 'flex', 'marginBottom': '10px'},
                                        children=[
                                            html.Div(
                                                style={'width': '30%', 'marginRight': '10px'},
                                                children=[
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
                                                ]
                                            ),
                                            html.Div(
                                                style={'width': '70%'},
                                                children=[
                                                    dcc.Graph(id='attention-heatmap')
                                                ]
                                            )
                                        ]
                                    ),
                                    dcc.Graph(id='attention-timeline')
                                ]
                            )
                        ]
                    )
                ]
            ),
            
            # Bottom row - Parameters and Gradients
            html.Div(
                style={'display': 'flex'},
                children=[
                    # Parameter visualization
                    html.Div(
                        style={'width': '50%', 'marginRight': '20px'},
                        children=[
                            card(
                                title="Parameter Visualization",
                                content=[
                                    html.Div(
                                        style={'marginBottom': '10px'},
                                        children=[
                                            dropdown(
                                                label="Layer",
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
                                        ]
                                    ),
                                    dcc.Graph(id='parameter-histogram')
                                ]
                            )
                        ]
                    ),
                    
                    # Gradient visualization
                    html.Div(
                        style={'width': '50%'},
                        children=[
                            card(
                                title="Gradient Visualization",
                                content=dcc.Graph(id='gradient-histogram')
                            )
                        ]
                    )
                ]
            )
        ])
    
    def _generate_inference_mode_layout(self):
        """Generate layout for inference testing mode."""
        return html.Div([
            # Top row - Results and Process Flow
            html.Div(
                style={'display': 'flex', 'marginBottom': '20px'},
                children=[
                    # Inference results
                    html.Div(
                        style={'width': '60%', 'marginRight': '20px'},
                        children=[
                            card(
                                title="Inference Results",
                                content=html.Div(id='inference-results', children=[
                                    html.P("Load a model and run inference to see results.")
                                ])
                            )
                        ]
                    ),
                    
                    # Process flow
                    html.Div(
                        style={'width': '40%'},
                        children=[
                            card(
                                title="Process Flow",
                                content=dcc.Graph(id='inference-state-graph')
                            )
                        ]
                    )
                ]
            ),
            
            # Bottom row - Memory and Attention
            html.Div(
                style={'display': 'flex'},
                children=[
                    # Memory visualization
                    html.Div(
                        style={'width': '50%', 'marginRight': '20px'},
                        children=[
                            card(
                                title="Memory Visualization",
                                content=[
                                    html.Div(
                                        style={'display': 'flex', 'marginBottom': '10px'},
                                        children=[
                                            html.Div(
                                                style={'width': '30%', 'marginRight': '10px'},
                                                children=[
                                                    html.Label("Memory Slot:"),
                                                    dcc.Slider(
                                                        id='inference-memory-slot-slider',
                                                        min=0,
                                                        max=15,  # Assuming 16 memory slots
                                                        value=0,
                                                        step=1
                                                    )
                                                ]
                                            ),
                                            html.Div(
                                                style={'width': '70%'},
                                                children=[
                                                    dcc.Graph(id='inference-memory-heatmap')
                                                ]
                                            )
                                        ]
                                    )
                                ]
                            )
                        ]
                    ),
                    
                    # Attention visualization
                    html.Div(
                        style={'width': '50%'},
                        children=[
                            card(
                                title="Attention Visualization",
                                content=[
                                    html.Div(
                                        style={'display': 'flex', 'marginBottom': '10px'},
                                        children=[
                                            html.Div(
                                                style={'width': '30%', 'marginRight': '10px'},
                                                children=[
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
                                                ]
                                            ),
                                            html.Div(
                                                style={'width': '70%'},
                                                children=[
                                                    dcc.Graph(id='inference-attention-heatmap')
                                                ]
                                            )
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ]
            )
        ])
    
    def _generate_experiment_mode_layout(self):
        """Generate layout for state experimentation mode."""
        return html.Div([
            # Top row - State Editor and Visualization
            html.Div(
                style={'display': 'flex', 'marginBottom': '20px'},
                children=[
                    # State editor
                    html.Div(
                        style={'width': '50%', 'marginRight': '20px'},
                        children=[
                            card(
                                title="State Editor",
                                content=html.Div(id='state-editor', children=[
                                    html.P("Select a state and click 'Load State' to begin editing.")
                                ])
                            )
                        ]
                    ),
                    
                    # State visualization
                    html.Div(
                        style={'width': '50%'},
                        children=[
                            card(
                                title="State Visualization",
                                content=html.Div(id='experiment-visualization', children=[
                                    html.P("No state loaded for visualization.")
                                ])
                            )
                        ]
                    )
                ]
            ),
            
            # Bottom row - Results Comparison
            html.Div(
                children=[
                    card(
                        title="Results Comparison",
                        content=html.Div(
                            style={'display': 'flex'},
                            children=[
                                html.Div(
                                    style={'width': '50%', 'marginRight': '20px'},
                                    children=[
                                        html.H4("Original Output", style=STYLES['heading']),
                                        html.Div(id='original-output', children=[
                                            html.P("No original output available.")
                                        ])
                                    ]
                                ),
                                
                                html.Div(
                                    style={'width': '50%'},
                                    children=[
                                        html.H4("Modified Output", style=STYLES['heading']),
                                        html.Div(id='modified-output', children=[
                                            html.P("No modified output available.")
                                        ])
                                    ]
                                )
                            ]
                        )
                    )
                ]
            )
        ])
    
    def run_server(self, debug: bool = True):
        """
        Run the dashboard server.
        
        Args:
            debug: Whether to run in debug mode
        """
        self.app.run(debug=debug, port=self.port)
