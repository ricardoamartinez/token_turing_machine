"""
Comprehensive TTM Interactive Dashboard.

This module provides a comprehensive, modern dashboard for visualizing and manipulating TTM model states,
with all visualizations visible simultaneously and full state manipulation capabilities.
"""

import dash
from dash import html, dcc, callback_context
from dash.dependencies import Input, Output, State, ALL
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import torch
import os
import json
from typing import Dict, List, Tuple, Any, Optional, Union
import pickle
from datetime import datetime
import copy

from src.ttm.visualization.theme import COLORS, STYLES, create_figure_layout
from src.ttm.visualization.components import (
    card, sidebar, main_content, button, input_field, dropdown, slider, tabs,
    create_memory_heatmap, create_attention_heatmap, create_parameter_histogram,
    create_timeline_plot, create_state_transition_graph
)
from src.ttm.data.tokenization import (
    create_multiplication_example,
    tokens_to_string,
    number_to_tokens,
    tokens_to_labels
)


class ComprehensiveTTMDashboard:
    """
    Comprehensive interactive dashboard for visualizing and manipulating TTM model states.
    
    This class provides a web-based dashboard with a modern UI for exploring model states,
    running inference, and experimenting with state modifications, with all visualizations
    visible simultaneously.
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
        self.original_state = None
        self.modified_state = None
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
                # Header
                html.Div(
                    style={
                        'backgroundColor': COLORS['background'],
                        'padding': '20px',
                        'borderBottom': f'1px solid {COLORS["border"]}',
                        'display': 'flex',
                        'justifyContent': 'space-between',
                        'alignItems': 'center',
                        'position': 'fixed',
                        'top': 0,
                        'left': 0,
                        'right': 0,
                        'zIndex': 1000
                    },
                    children=[
                        html.H1("TTM Interactive Visualization", style={
                            'color': COLORS['text'],
                            'margin': 0,
                            'fontSize': '24px'
                        }),
                        
                        # Mode selector
                        html.Div(
                            style={'display': 'flex', 'gap': '10px'},
                            children=[
                                html.Button(
                                    "Training",
                                    id='training-mode-button',
                                    style={
                                        'backgroundColor': COLORS['primary'],
                                        'color': COLORS['text'],
                                        'border': 'none',
                                        'borderRadius': '4px',
                                        'padding': '8px 16px',
                                        'cursor': 'pointer'
                                    }
                                ),
                                html.Button(
                                    "Inference",
                                    id='inference-mode-button',
                                    style={
                                        'backgroundColor': COLORS['card'],
                                        'color': COLORS['text'],
                                        'border': f'1px solid {COLORS["border"]}',
                                        'borderRadius': '4px',
                                        'padding': '8px 16px',
                                        'cursor': 'pointer'
                                    }
                                ),
                                html.Button(
                                    "Experiment",
                                    id='experiment-mode-button',
                                    style={
                                        'backgroundColor': COLORS['card'],
                                        'color': COLORS['text'],
                                        'border': f'1px solid {COLORS["border"]}',
                                        'borderRadius': '4px',
                                        'padding': '8px 16px',
                                        'cursor': 'pointer'
                                    }
                                )
                            ]
                        ),
                        
                        # Status and actions
                        html.Div(
                            style={'display': 'flex', 'gap': '10px', 'alignItems': 'center'},
                            children=[
                                html.Div(id='status-message', children="Ready", style={'color': COLORS['text']}),
                                html.Div(id='status-actions')
                            ]
                        )
                    ]
                ),
                
                # Main content
                html.Div(
                    style={
                        'display': 'flex',
                        'marginTop': '64px',  # To account for fixed header
                        'height': 'calc(100vh - 64px)'  # Full height minus header
                    },
                    children=[
                        # Left sidebar
                        html.Div(
                            id='sidebar-container',
                            style={
                                'width': '280px',
                                'backgroundColor': COLORS['card'],
                                'borderRight': f'1px solid {COLORS["border"]}',
                                'padding': '20px',
                                'overflowY': 'auto',
                                'height': '100%'
                            },
                            children=[
                                html.H3("Controls", style=STYLES['heading']),
                                html.Div(id='sidebar-content')
                            ]
                        ),
                        
                        # Main visualization area
                        html.Div(
                            style={
                                'flex': 1,
                                'padding': '20px',
                                'overflowY': 'auto',
                                'height': '100%'
                            },
                            children=[
                                # Dynamic content based on mode
                                html.Div(id='main-content')
                            ]
                        )
                    ]
                ),
                
                # Hidden divs for storing state
                html.Div(id='current-mode', children='training', style={'display': 'none'}),
                html.Div(id='current-state-storage', style={'display': 'none'}),
                html.Div(id='history-storage', style={'display': 'none'}),
                html.Div(id='modified-state-storage', style={'display': 'none'})
            ]
        )
    
    def _setup_callbacks(self):
        """Set up the dashboard callbacks."""
        # Mode selection callbacks
        @self.app.callback(
            [Output('current-mode', 'children'),
             Output('training-mode-button', 'style'),
             Output('inference-mode-button', 'style'),
             Output('experiment-mode-button', 'style')],
            [Input('training-mode-button', 'n_clicks'),
             Input('inference-mode-button', 'n_clicks'),
             Input('experiment-mode-button', 'n_clicks')],
            [State('current-mode', 'children')]
        )
        def update_mode(training_clicks, inference_clicks, experiment_clicks, current_mode):
            # Default button styles
            training_style = {
                'backgroundColor': COLORS['card'],
                'color': COLORS['text'],
                'border': f'1px solid {COLORS["border"]}',
                'borderRadius': '4px',
                'padding': '8px 16px',
                'cursor': 'pointer'
            }
            
            inference_style = training_style.copy()
            experiment_style = training_style.copy()
            
            # Determine which button was clicked
            ctx = callback_context
            if not ctx.triggered:
                # No button clicked, use current mode
                mode = current_mode
            else:
                button_id = ctx.triggered[0]['prop_id'].split('.')[0]
                if button_id == 'training-mode-button':
                    mode = 'training'
                elif button_id == 'inference-mode-button':
                    mode = 'inference'
                elif button_id == 'experiment-mode-button':
                    mode = 'experiment'
                else:
                    mode = current_mode
            
            # Highlight the active button
            if mode == 'training':
                training_style['backgroundColor'] = COLORS['primary']
                training_style['border'] = 'none'
            elif mode == 'inference':
                inference_style['backgroundColor'] = COLORS['primary']
                inference_style['border'] = 'none'
            elif mode == 'experiment':
                experiment_style['backgroundColor'] = COLORS['primary']
                experiment_style['border'] = 'none'
            
            return mode, training_style, inference_style, experiment_style
        
        # Update sidebar and main content based on mode
        @self.app.callback(
            [Output('sidebar-content', 'children'),
             Output('main-content', 'children'),
             Output('status-message', 'children'),
             Output('status-actions', 'children')],
            [Input('current-mode', 'children')]
        )
        def update_content(mode):
            if mode == 'training':
                return (
                    self._generate_training_sidebar(),
                    self._generate_training_content(),
                    "Training History Mode",
                    html.Div([
                        button("Load History", id="load-history-button", style={'marginRight': '10px'}),
                        button("Export Visualization", id="export-viz-button")
                    ])
                )
            elif mode == 'inference':
                return (
                    self._generate_inference_sidebar(),
                    self._generate_inference_content(),
                    "Inference Testing Mode",
                    html.Div([
                        button("Load Model", id="load-model-button", style={'marginRight': '10px'}),
                        button("Run Inference", id="run-inference-button")
                    ])
                )
            elif mode == 'experiment':
                return (
                    self._generate_experiment_sidebar(),
                    self._generate_experiment_content(),
                    "State Experimentation Mode",
                    html.Div([
                        button("Load State", id="load-state-button", style={'marginRight': '10px'}),
                        button("Apply Changes", id="apply-changes-button", style={'marginRight': '10px'}),
                        button("Reset", id="reset-experiment-button")
                    ])
                )
            else:
                return html.Div(), html.Div(), "Unknown Mode", html.Div()
    
    def _generate_training_sidebar(self):
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
            ),
            
            html.Hr(style={'margin': '20px 0', 'borderColor': COLORS['border']}),
            
            html.H3("Visualization Controls", style=STYLES['heading']),
            
            dropdown(
                label="Memory Slot",
                id='memory-slot-dropdown',
                options=[{'label': f'Slot {i}', 'value': i} for i in range(16)],
                value=0
            ),
            
            dropdown(
                label="Attention Layer",
                id='attention-layer-dropdown',
                options=[{'label': f'Layer {i+1}', 'value': i} for i in range(4)],
                value=0
            ),
            
            dropdown(
                label="Attention Head",
                id='attention-head-dropdown',
                options=[{'label': f'Head {i+1}', 'value': i} for i in range(4)],
                value=0
            ),
            
            dropdown(
                label="Parameter Layer",
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
        ])
    
    def _generate_inference_sidebar(self):
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
            ),
            
            html.Hr(style={'margin': '20px 0', 'borderColor': COLORS['border']}),
            
            html.H3("Visualization Controls", style=STYLES['heading']),
            
            dropdown(
                label="Memory Slot",
                id='inference-memory-slot-dropdown',
                options=[{'label': f'Slot {i}', 'value': i} for i in range(16)],
                value=0
            ),
            
            dropdown(
                label="Attention Layer",
                id='inference-attention-layer-dropdown',
                options=[{'label': f'Layer {i+1}', 'value': i} for i in range(4)],
                value=0
            ),
            
            dropdown(
                label="Attention Head",
                id='inference-attention-head-dropdown',
                options=[{'label': f'Head {i+1}', 'value': i} for i in range(4)],
                value=0
            )
        ])
    
    def _generate_experiment_sidebar(self):
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
            
            html.Div(id='component-controls'),
            
            html.Hr(style={'margin': '20px 0', 'borderColor': COLORS['border']}),
            
            html.H3("Experiment Controls", style=STYLES['heading']),
            
            dropdown(
                label="Modification Type",
                id='modification-type-dropdown',
                options=[
                    {'label': 'Manual Edit', 'value': 'manual'},
                    {'label': 'Random Perturbation', 'value': 'random'},
                    {'label': 'Systematic Change', 'value': 'systematic'}
                ],
                value='manual'
            ),
            
            html.Div(id='modification-controls')
        ])
    
    def _generate_training_content(self):
        """Generate main content for training history mode."""
        # Check if we have training history data
        if not self.state_history or not self.state_history.get('epochs', []):
            return html.Div([
                card(
                    title="No Training History Available",
                    content=html.P("Please load a state history file to view training data.")
                )
            ])
        
        # Create a comprehensive layout with all visualizations
        return html.Div([
            # Top row - Overview and Process Flow
            html.Div(
                style={'display': 'flex', 'marginBottom': '20px', 'gap': '20px'},
                children=[
                    # Training metrics
                    html.Div(
                        style={'width': '60%'},
                        children=[
                            card(
                                title="Training Metrics",
                                content=dcc.Graph(id='training-metrics-graph', style={'height': '300px'})
                            )
                        ]
                    ),
                    
                    # State transition graph
                    html.Div(
                        style={'width': '40%'},
                        children=[
                            card(
                                title="Process Flow",
                                content=dcc.Graph(id='state-transition-graph', style={'height': '300px'})
                            )
                        ]
                    )
                ]
            ),
            
            # Middle row - Memory and Attention
            html.Div(
                style={'display': 'flex', 'marginBottom': '20px', 'gap': '20px'},
                children=[
                    # Memory visualization
                    html.Div(
                        style={'width': '50%'},
                        children=[
                            card(
                                title="Memory Visualization",
                                content=[
                                    dcc.Graph(id='memory-heatmap', style={'height': '300px'})
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
                                    dcc.Graph(id='attention-heatmap', style={'height': '300px'})
                                ]
                            )
                        ]
                    )
                ]
            ),
            
            # Bottom row - Timelines and Parameters
            html.Div(
                style={'display': 'flex', 'gap': '20px'},
                children=[
                    # Memory timeline
                    html.Div(
                        style={'width': '33%'},
                        children=[
                            card(
                                title="Memory Timeline",
                                content=dcc.Graph(id='memory-timeline', style={'height': '300px'})
                            )
                        ]
                    ),
                    
                    # Attention timeline
                    html.Div(
                        style={'width': '33%'},
                        children=[
                            card(
                                title="Attention Timeline",
                                content=dcc.Graph(id='attention-timeline', style={'height': '300px'})
                            )
                        ]
                    ),
                    
                    # Parameter distribution
                    html.Div(
                        style={'width': '33%'},
                        children=[
                            card(
                                title="Parameter Distribution",
                                content=dcc.Graph(id='parameter-histogram', style={'height': '300px'})
                            )
                        ]
                    )
                ]
            )
        ])
    
    def _generate_inference_content(self):
        """Generate main content for inference testing mode."""
        return html.Div([
            # Top row - Results and Process Flow
            html.Div(
                style={'display': 'flex', 'marginBottom': '20px', 'gap': '20px'},
                children=[
                    # Inference results
                    html.Div(
                        style={'width': '60%'},
                        children=[
                            card(
                                title="Inference Results",
                                content=html.Div(id='inference-results', children=[
                                    html.Div(
                                        style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'height': '300px'},
                                        children=[
                                            html.P("Load a model and run inference to see results.", style={'color': COLORS['text_muted']})
                                        ]
                                    )
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
                                content=dcc.Graph(id='inference-state-graph', style={'height': '300px'})
                            )
                        ]
                    )
                ]
            ),
            
            # Middle row - Memory and Attention
            html.Div(
                style={'display': 'flex', 'marginBottom': '20px', 'gap': '20px'},
                children=[
                    # Memory visualization
                    html.Div(
                        style={'width': '50%'},
                        children=[
                            card(
                                title="Memory Visualization",
                                content=dcc.Graph(id='inference-memory-heatmap', style={'height': '300px'})
                            )
                        ]
                    ),
                    
                    # Attention visualization
                    html.Div(
                        style={'width': '50%'},
                        children=[
                            card(
                                title="Attention Visualization",
                                content=dcc.Graph(id='inference-attention-heatmap', style={'height': '300px'})
                            )
                        ]
                    )
                ]
            ),
            
            # Bottom row - Token Predictions
            html.Div(
                children=[
                    card(
                        title="Token Predictions",
                        content=html.Div(id='token-predictions', children=[
                            html.Div(
                                style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'height': '300px'},
                                children=[
                                    html.P("Run inference to see token predictions.", style={'color': COLORS['text_muted']})
                                ]
                            )
                        ])
                    )
                ]
            )
        ])
    
    def _generate_experiment_content(self):
        """Generate main content for state experimentation mode."""
        return html.Div([
            # Top row - State Editor and Visualization
            html.Div(
                style={'display': 'flex', 'marginBottom': '20px', 'gap': '20px'},
                children=[
                    # State editor
                    html.Div(
                        style={'width': '50%'},
                        children=[
                            card(
                                title="State Editor",
                                content=html.Div(id='state-editor', children=[
                                    html.Div(
                                        style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'height': '300px'},
                                        children=[
                                            html.P("Select a state and click 'Load State' to begin editing.", style={'color': COLORS['text_muted']})
                                        ]
                                    )
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
                                    html.Div(
                                        style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'height': '300px'},
                                        children=[
                                            html.P("No state loaded for visualization.", style={'color': COLORS['text_muted']})
                                        ]
                                    )
                                ])
                            )
                        ]
                    )
                ]
            ),
            
            # Bottom row - Results Comparison
            html.Div(
                style={'display': 'flex', 'gap': '20px'},
                children=[
                    # Original output
                    html.Div(
                        style={'width': '50%'},
                        children=[
                            card(
                                title="Original Output",
                                content=html.Div(id='original-output', children=[
                                    html.Div(
                                        style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'height': '300px'},
                                        children=[
                                            html.P("No original output available.", style={'color': COLORS['text_muted']})
                                        ]
                                    )
                                ])
                            )
                        ]
                    ),
                    
                    # Modified output
                    html.Div(
                        style={'width': '50%'},
                        children=[
                            card(
                                title="Modified Output",
                                content=html.Div(id='modified-output', children=[
                                    html.Div(
                                        style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'height': '300px'},
                                        children=[
                                            html.P("No modified output available.", style={'color': COLORS['text_muted']})
                                        ]
                                    )
                                ])
                            )
                        ]
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
