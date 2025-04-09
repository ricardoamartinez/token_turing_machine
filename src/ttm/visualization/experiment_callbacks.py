"""
Experiment mode callbacks for the TTM Interactive Dashboard.

This module provides callback functions for the state experimentation mode of the dashboard.
"""

from dash import html, dcc, callback_context
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import numpy as np
import json
import torch
import os
import copy
from typing import Dict, List, Tuple, Any, Optional, Union

from src.ttm.visualization.theme import COLORS, STYLES
from src.ttm.visualization.components import (
    card, button, input_field, dropdown, slider,
    create_memory_heatmap, create_attention_heatmap
)


def register_experiment_callbacks(app, dashboard):
    """
    Register callbacks for state experimentation mode.
    
    Args:
        app: Dash app
        dashboard: TTMDashboard instance
    """
    
    # Update component controls based on selected component
    @app.callback(
        Output('component-controls', 'children'),
        [Input('component-dropdown', 'value'),
         Input('source-selection', 'value')]
    )
    def update_component_controls(component, source):
        if component == 'memory':
            return html.Div([
                slider(
                    label="Memory Slot",
                    id='experiment-memory-slot-slider',
                    min=0,
                    max=15,  # Assuming 16 memory slots
                    value=0
                )
            ])
        elif component == 'attention':
            return html.Div([
                slider(
                    label="Layer",
                    id='experiment-attention-layer-slider',
                    min=0,
                    max=3,  # Assuming 4 layers
                    value=0
                ),
                slider(
                    label="Head",
                    id='experiment-attention-head-slider',
                    min=0,
                    max=3,  # Assuming 4 heads
                    value=0
                )
            ])
        elif component == 'transformer':
            return html.Div([
                slider(
                    label="Layer",
                    id='experiment-transformer-layer-slider',
                    min=0,
                    max=3,  # Assuming 4 layers
                    value=0
                )
            ])
        elif component == 'parameters':
            return html.Div([
                dropdown(
                    label="Layer",
                    id='experiment-parameter-layer-dropdown',
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
        else:
            return html.Div("Select a component to see controls")
    
    # Load state button callback
    @app.callback(
        [Output('state-editor', 'children'),
         Output('experiment-visualization', 'children'),
         Output('original-output', 'children')],
        [Input('load-state-button', 'n_clicks')],
        [State('source-selection', 'value'),
         State('component-dropdown', 'value'),
         State('epoch-slider', 'value') if hasattr(dashboard, 'state_history') and dashboard.state_history else None,
         State('batch-slider', 'value') if hasattr(dashboard, 'state_history') and dashboard.state_history else None,
         State('token-slider', 'value') if hasattr(dashboard, 'state_history') and dashboard.state_history else None]
    )
    def load_state(n_clicks, source, component, epoch=None, batch=None, token=None):
        if n_clicks is None or n_clicks == 0:
            return (
                html.P("Select a state and click 'Load State' to begin editing."),
                html.P("No state loaded for visualization."),
                html.P("No original output available.")
            )
        
        # Get state based on source
        if source == 'training' and hasattr(dashboard, 'state_history') and dashboard.state_history:
            if epoch is None or batch is None or token is None:
                return (
                    html.P("Please select epoch, batch, and token first."),
                    html.P("No state loaded for visualization."),
                    html.P("No original output available.")
                )
            
            state_key = (epoch, batch, token)
            state = dashboard.state_history['states'].get(state_key, {})
            
            if not state:
                return (
                    html.P(f"No state found for epoch {epoch}, batch {batch}, token {token}."),
                    html.P("No state loaded for visualization."),
                    html.P("No original output available.")
                )
            
            # Store the original state for comparison
            dashboard.original_state = copy.deepcopy(state)
            dashboard.modified_state = copy.deepcopy(state)
            
            # Create state editor and visualization based on component
            if component == 'memory' and 'memory' in state:
                memory_data = state['memory']
                if not isinstance(memory_data, np.ndarray):
                    memory_data = np.array(memory_data)
                
                # Create a grid editor for memory
                memory_size, embedding_dim = memory_data.shape
                
                # Create a simple grid editor (in a real implementation, this would be more sophisticated)
                rows = []
                for i in range(min(10, memory_size)):  # Show first 10 rows for simplicity
                    cells = []
                    for j in range(min(10, embedding_dim)):  # Show first 10 columns for simplicity
                        cells.append(
                            html.Td(
                                dcc.Input(
                                    id={'type': 'memory-cell-input', 'row': i, 'col': j},
                                    type='number',
                                    value=float(memory_data[i, j]),
                                    style={'width': '60px'}
                                ),
                                style={'padding': '5px'}
                            )
                        )
                    rows.append(html.Tr(cells))
                
                editor = html.Div([
                    html.H4("Memory Editor", style=STYLES['heading']),
                    html.P("Edit memory values directly:"),
                    html.Table(
                        html.Tbody(rows),
                        style={'borderCollapse': 'collapse', 'width': '100%', 'overflowX': 'auto'}
                    ),
                    html.P("Note: Only showing first 10x10 values for simplicity.")
                ])
                
                # Create memory visualization
                visualization = html.Div([
                    html.H4("Memory Visualization", style=STYLES['heading']),
                    dcc.Graph(figure=create_memory_heatmap(memory_data, "Memory Content"))
                ])
                
                # Show original output
                if 'outputs' in state:
                    outputs = state['outputs']
                    if isinstance(outputs, np.ndarray):
                        # Get the top prediction
                        if outputs.ndim > 1:
                            top_pred = np.argmax(outputs, axis=-1)
                            original_output = html.Div([
                                html.H4("Original Prediction", style=STYLES['heading']),
                                html.P(f"Top prediction indices: {top_pred}")
                            ])
                        else:
                            original_output = html.Div([
                                html.H4("Original Output", style=STYLES['heading']),
                                html.P(f"Output: {outputs}")
                            ])
                    else:
                        original_output = html.Div([
                            html.H4("Original Output", style=STYLES['heading']),
                            html.P(f"Output: {outputs}")
                        ])
                else:
                    original_output = html.P("No output data available.")
                
                return editor, visualization, original_output
                
            elif component == 'attention' and 'attention' in state:
                attention_data = state['attention']
                
                # Extract attention for the first layer and head
                if isinstance(attention_data, dict) and 'layer_0' in attention_data:
                    layer_data = attention_data['layer_0']
                    if isinstance(layer_data, dict) and 'head_0' in layer_data:
                        head_data = layer_data['head_0']
                    elif isinstance(layer_data, np.ndarray) and layer_data.ndim >= 3:
                        head_data = layer_data[0]  # First head
                    else:
                        head_data = layer_data
                elif 'combined' in attention_data:
                    head_data = attention_data['combined']
                else:
                    head_data = None
                
                if head_data is not None and isinstance(head_data, np.ndarray):
                    # Create a grid editor for attention
                    seq_len = head_data.shape[0]
                    
                    # Create a simple grid editor
                    rows = []
                    for i in range(min(10, seq_len)):  # Show first 10 rows for simplicity
                        cells = []
                        for j in range(min(10, seq_len)):  # Show first 10 columns for simplicity
                            cells.append(
                                html.Td(
                                    dcc.Input(
                                        id={'type': 'attention-cell-input', 'row': i, 'col': j},
                                        type='number',
                                        value=float(head_data[i, j]),
                                        style={'width': '60px'}
                                    ),
                                    style={'padding': '5px'}
                                )
                            )
                        rows.append(html.Tr(cells))
                    
                    editor = html.Div([
                        html.H4("Attention Editor", style=STYLES['heading']),
                        html.P("Edit attention weights directly:"),
                        html.Table(
                            html.Tbody(rows),
                            style={'borderCollapse': 'collapse', 'width': '100%', 'overflowX': 'auto'}
                        ),
                        html.P("Note: Only showing first 10x10 values for simplicity.")
                    ])
                    
                    # Create attention visualization
                    visualization = html.Div([
                        html.H4("Attention Visualization", style=STYLES['heading']),
                        dcc.Graph(figure=create_attention_heatmap(head_data, title="Attention Weights"))
                    ])
                    
                    # Show original output
                    if 'outputs' in state:
                        outputs = state['outputs']
                        if isinstance(outputs, np.ndarray):
                            # Get the top prediction
                            if outputs.ndim > 1:
                                top_pred = np.argmax(outputs, axis=-1)
                                original_output = html.Div([
                                    html.H4("Original Prediction", style=STYLES['heading']),
                                    html.P(f"Top prediction indices: {top_pred}")
                                ])
                            else:
                                original_output = html.Div([
                                    html.H4("Original Output", style=STYLES['heading']),
                                    html.P(f"Output: {outputs}")
                                ])
                        else:
                            original_output = html.Div([
                                html.H4("Original Output", style=STYLES['heading']),
                                html.P(f"Output: {outputs}")
                            ])
                    else:
                        original_output = html.P("No output data available.")
                    
                    return editor, visualization, original_output
                else:
                    return (
                        html.P("Attention data not available in the expected format."),
                        html.P("No state loaded for visualization."),
                        html.P("No original output available.")
                    )
            else:
                return (
                    html.P(f"Component '{component}' not found in the state or not yet implemented."),
                    html.P("No state loaded for visualization."),
                    html.P("No original output available.")
                )
        elif source == 'inference' and hasattr(dashboard, 'inference_state') and dashboard.inference_state:
            # Similar implementation for inference state
            return (
                html.P("Inference state editing not yet implemented."),
                html.P("No state loaded for visualization."),
                html.P("No original output available.")
            )
        else:
            return (
                html.P(f"No {source} state available."),
                html.P("No state loaded for visualization."),
                html.P("No original output available.")
            )
    
    # Apply changes button callback
    @app.callback(
        Output('modified-output', 'children'),
        [Input('apply-changes-button', 'n_clicks')],
        [State({'type': 'memory-cell-input', 'row': dash.dependencies.ALL, 'col': dash.dependencies.ALL}, 'value'),
         State({'type': 'attention-cell-input', 'row': dash.dependencies.ALL, 'col': dash.dependencies.ALL}, 'value'),
         State('component-dropdown', 'value')]
    )
    def apply_changes(n_clicks, memory_values, attention_values, component):
        if n_clicks is None or n_clicks == 0:
            return html.P("No modified output available.")
        
        if not hasattr(dashboard, 'modified_state') or not dashboard.modified_state:
            return html.P("No state loaded to modify.")
        
        # Apply changes based on component
        if component == 'memory' and memory_values:
            # Get the original memory data
            if 'memory' in dashboard.modified_state:
                memory_data = dashboard.modified_state['memory']
                if not isinstance(memory_data, np.ndarray):
                    memory_data = np.array(memory_data)
                
                # Update memory values
                for i, row_values in enumerate(memory_values):
                    for j, value in enumerate(row_values):
                        if i < memory_data.shape[0] and j < memory_data.shape[1]:
                            memory_data[i, j] = float(value)
                
                # Update the modified state
                dashboard.modified_state['memory'] = memory_data
                
                # In a real implementation, we would run a forward pass with the modified state
                # For now, just show a message
                return html.Div([
                    html.H4("Modified State Applied", style=STYLES['heading']),
                    html.P("Memory values have been updated."),
                    html.P("In a real implementation, a forward pass would be run with the modified state.")
                ])
        elif component == 'attention' and attention_values:
            # Similar implementation for attention
            return html.Div([
                html.H4("Modified State Applied", style=STYLES['heading']),
                html.P("Attention values have been updated."),
                html.P("In a real implementation, a forward pass would be run with the modified state.")
            ])
        
        return html.P("No changes applied.")
    
    # Reset experiment button callback
    @app.callback(
        [Output('state-editor', 'children', allow_duplicate=True),
         Output('experiment-visualization', 'children', allow_duplicate=True),
         Output('original-output', 'children', allow_duplicate=True),
         Output('modified-output', 'children', allow_duplicate=True)],
        [Input('reset-experiment-button', 'n_clicks')],
        prevent_initial_call=True
    )
    def reset_experiment(n_clicks):
        if n_clicks is None or n_clicks == 0:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        # Reset the modified state
        if hasattr(dashboard, 'original_state'):
            dashboard.modified_state = copy.deepcopy(dashboard.original_state)
        
        return (
            html.P("Experiment reset. Select a state and click 'Load State' to begin editing."),
            html.P("No state loaded for visualization."),
            html.P("No original output available."),
            html.P("No modified output available.")
        )
