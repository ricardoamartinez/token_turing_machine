"""
Comprehensive Interactive Visualization for TTM model.

This script runs the comprehensive interactive visualization dashboard for the TTM model,
providing a modern, feature-rich interface for visualizing and manipulating model states.
"""

import argparse
import os
import torch
import numpy as np
import pickle
from datetime import datetime
import dash
from dash import html, dcc, callback_context
from dash.dependencies import Input, Output, State, ALL
import plotly.graph_objects as go
import plotly.express as px

from src.ttm.visualization.theme import COLORS, STYLES, create_figure_layout
from src.ttm.visualization.components import (
    card, sidebar, main_content, button, input_field, dropdown, slider, tabs,
    create_memory_heatmap, create_attention_heatmap, create_parameter_histogram,
    create_timeline_plot, create_state_transition_graph
)
from src.ttm.visualization.comprehensive_dashboard import ComprehensiveTTMDashboard
from src.ttm.visualization.training_callbacks import register_training_callbacks
from src.ttm.visualization.inference_callbacks import register_inference_callbacks
from src.ttm.visualization.experiment_callbacks import register_experiment_callbacks


def register_comprehensive_callbacks(app, dashboard):
    """
    Register callbacks for the comprehensive dashboard.
    
    Args:
        app: Dash app
        dashboard: ComprehensiveTTMDashboard instance
    """
    # Update batch slider based on epoch selection
    @app.callback(
        [Output('batch-slider', 'max'),
         Output('batch-slider', 'marks'),
         Output('batch-slider', 'value')],
        [Input('epoch-slider', 'value')]
    )
    def update_batch_slider(epoch):
        if not dashboard.state_history or 'batches' not in dashboard.state_history:
            return 0, {}, 0
        
        batches = dashboard.state_history['batches'].get(epoch, [])
        if not batches:
            return 0, {}, 0
        
        max_batch = max(batches)
        marks = {i: str(i) for i in range(0, max_batch+1, max(1, max_batch//5))}
        return max_batch, marks, 0
    
    # Update token slider based on epoch and batch selection
    @app.callback(
        [Output('token-slider', 'max'),
         Output('token-slider', 'marks'),
         Output('token-slider', 'value')],
        [Input('epoch-slider', 'value'),
         Input('batch-slider', 'value')]
    )
    def update_token_slider(epoch, batch):
        if not dashboard.state_history or 'tokens' not in dashboard.state_history:
            return 0, {}, 0
        
        key = (epoch, batch)
        tokens = dashboard.state_history['tokens'].get(key, [])
        if not tokens:
            return 0, {}, 0
        
        max_token = max(tokens)
        marks = {i: str(i) for i in range(0, max_token+1, max(1, max_token//5))}
        return max_token, marks, 0
    
    # Update current state based on epoch, batch, and token selection
    @app.callback(
        Output('current-state-storage', 'children'),
        [Input('epoch-slider', 'value'),
         Input('batch-slider', 'value'),
         Input('token-slider', 'value')]
    )
    def update_current_state(epoch, batch, token):
        if not dashboard.state_history or 'states' not in dashboard.state_history:
            return json.dumps({})
        
        state_key = (epoch, batch, token)
        current_state = dashboard.state_history['states'].get(state_key, {})
        
        # Update dashboard's current state
        dashboard.current_state = {
            'epoch': epoch,
            'batch': batch,
            'token': token,
            **current_state
        }
        
        # Return serializable version for storage
        serializable_state = {
            'epoch': epoch,
            'batch': batch,
            'token': token,
            'has_memory': 'memory' in current_state,
            'has_attention': 'attention' in current_state,
            'has_inputs': 'inputs' in current_state,
            'has_outputs': 'outputs' in current_state,
            'has_gradients': 'gradients' in current_state
        }
        
        return json.dumps(serializable_state)
    
    # Update memory heatmap
    @app.callback(
        Output('memory-heatmap', 'figure'),
        [Input('current-state-storage', 'children'),
         Input('memory-slot-dropdown', 'value')]
    )
    def update_memory_heatmap(current_state_json, memory_slot):
        if not current_state_json:
            return create_memory_heatmap(None, "Memory Content")
        
        # Get current state info
        current_state_info = json.loads(current_state_json)
        epoch = current_state_info.get('epoch', 0)
        batch = current_state_info.get('batch', 0)
        token = current_state_info.get('token', 0)
        
        # Get full state from dashboard
        state_key = (epoch, batch, token)
        state = dashboard.state_history['states'].get(state_key, {})
        
        # Get memory data
        memory_data = state.get('memory')
        
        # Create title
        title = f"Memory Content (Epoch {epoch}, Batch {batch}, Token {token})"
        
        # If memory data is not available, return empty figure
        if memory_data is None:
            return create_memory_heatmap(None, title)
        
        # Convert to numpy if needed
        if not isinstance(memory_data, np.ndarray):
            memory_data = np.array(memory_data)
        
        # Highlight selected memory slot if applicable
        if memory_slot is not None and 0 <= memory_slot < memory_data.shape[0]:
            # Create a copy to avoid modifying the original
            highlighted_memory = memory_data.copy()
            
            # Scale the selected row for highlighting
            max_val = np.max(np.abs(highlighted_memory))
            highlight_factor = 0.2 * max_val
            
            # Add highlight effect
            highlighted_memory[memory_slot] = highlighted_memory[memory_slot] + highlight_factor
            
            return create_memory_heatmap(highlighted_memory, title)
        
        return create_memory_heatmap(memory_data, title)
    
    # Update attention heatmap
    @app.callback(
        Output('attention-heatmap', 'figure'),
        [Input('current-state-storage', 'children'),
         Input('attention-layer-dropdown', 'value'),
         Input('attention-head-dropdown', 'value')]
    )
    def update_attention_heatmap(current_state_json, layer, head):
        if not current_state_json:
            return create_attention_heatmap(None, title="Attention Weights")
        
        # Get current state info
        current_state_info = json.loads(current_state_json)
        epoch = current_state_info.get('epoch', 0)
        batch = current_state_info.get('batch', 0)
        token = current_state_info.get('token', 0)
        
        # Get full state from dashboard
        state_key = (epoch, batch, token)
        state = dashboard.state_history['states'].get(state_key, {})
        
        # Get attention data
        attention_data = state.get('attention')
        
        # Create title
        title = f"Attention Weights (Epoch {epoch}, Batch {batch}, Token {token}, Layer {layer+1}, Head {head+1})"
        
        # If attention data is not available, return empty figure
        if attention_data is None:
            return create_attention_heatmap(None, title=title)
        
        # Extract the specific layer and head
        layer_key = f'layer_{layer}'
        if layer_key in attention_data:
            layer_data = attention_data[layer_key]
            
            # Check if we have head-specific data
            if isinstance(layer_data, dict) and f'head_{head}' in layer_data:
                head_data = layer_data[f'head_{head}']
            elif isinstance(layer_data, np.ndarray) and layer_data.ndim >= 3:
                # If we have a tensor with dimensions [num_heads, seq_len, seq_len]
                head_data = layer_data[head]
            else:
                # Use whatever we have
                head_data = layer_data
        elif 'combined' in attention_data:
            # If we only have combined attention, use that
            head_data = attention_data['combined']
        else:
            return create_attention_heatmap(None, title=title)
        
        # Convert to numpy if needed
        if not isinstance(head_data, np.ndarray):
            head_data = np.array(head_data)
        
        # Get token labels if available
        token_labels = None
        if 'inputs' in state:
            inputs = state['inputs']
            if isinstance(inputs, np.ndarray):
                # Get the first batch item
                if inputs.ndim > 1:
                    inputs = inputs[0]
                token_labels = tokens_to_labels(inputs)
        
        return create_attention_heatmap(head_data, token_labels, title)
    
    # Update memory timeline
    @app.callback(
        Output('memory-timeline', 'figure'),
        [Input('epoch-slider', 'value'),
         Input('batch-slider', 'value'),
         Input('token-slider', 'value'),
         Input('memory-slot-dropdown', 'value')]
    )
    def update_memory_timeline(epoch, batch, current_token, memory_slot):
        # Get all tokens for this epoch and batch
        key = (epoch, batch)
        tokens = dashboard.state_history['tokens'].get(key, [])
        if not tokens:
            return create_timeline_plot([], [], "Memory Usage Timeline")
        
        # Collect memory usage data for each token
        memory_usage = []
        for token in sorted(tokens):
            state_key = (epoch, batch, token)
            state = dashboard.state_history['states'].get(state_key, {})
            memory_data = state.get('memory')
            
            if memory_data is not None:
                if not isinstance(memory_data, np.ndarray):
                    memory_data = np.array(memory_data)
                
                # If memory slot is specified, track that slot's activation
                if memory_slot is not None and 0 <= memory_slot < memory_data.shape[0]:
                    slot_data = memory_data[memory_slot]
                    usage = np.mean(np.abs(slot_data))
                else:
                    # Otherwise track overall memory usage
                    usage = np.mean(np.abs(memory_data))
                
                memory_usage.append(usage)
            else:
                memory_usage.append(0)
        
        title = f"Memory Usage Across Tokens (Epoch {epoch}, Batch {batch})"
        if memory_slot is not None:
            title += f", Slot {memory_slot}"
        
        return create_timeline_plot(
            memory_usage, 
            sorted(tokens),
            title=title,
            x_label="Token Position",
            y_label="Average Activation",
            highlight_idx=current_token
        )
    
    # Update attention timeline
    @app.callback(
        Output('attention-timeline', 'figure'),
        [Input('epoch-slider', 'value'),
         Input('batch-slider', 'value'),
         Input('token-slider', 'value'),
         Input('attention-layer-dropdown', 'value')]
    )
    def update_attention_timeline(epoch, batch, current_token, layer):
        # Get all tokens for this epoch and batch
        key = (epoch, batch)
        tokens = dashboard.state_history['tokens'].get(key, [])
        if not tokens:
            return create_timeline_plot([], [], "Attention Focus Timeline")
        
        # Collect attention focus data for each token
        attention_focus = []
        for token in sorted(tokens):
            state_key = (epoch, batch, token)
            state = dashboard.state_history['states'].get(state_key, {})
            attention_data = state.get('attention')
            
            if attention_data is not None:
                # Extract the specific layer
                layer_key = f'layer_{layer}'
                if layer_key in attention_data:
                    layer_data = attention_data[layer_key]
                    
                    # Calculate attention focus (1 - entropy)
                    if isinstance(layer_data, np.ndarray):
                        # Average over heads if needed
                        if layer_data.ndim > 2:
                            layer_data = np.mean(layer_data, axis=0)
                        
                        # Calculate entropy of attention weights
                        # Higher entropy = more distributed attention
                        # Lower entropy = more focused attention
                        epsilon = 1e-10  # To avoid log(0)
                        entropy = -np.sum(layer_data * np.log(layer_data + epsilon), axis=-1)
                        max_entropy = -np.log(1.0 / layer_data.shape[-1])  # Maximum possible entropy
                        normalized_entropy = entropy / max_entropy  # Between 0 and 1
                        
                        # 1 - normalized entropy gives us focus (1 = focused, 0 = distributed)
                        focus = 1 - np.mean(normalized_entropy)
                        attention_focus.append(focus)
                    else:
                        attention_focus.append(0)
                else:
                    attention_focus.append(0)
            else:
                attention_focus.append(0)
        
        title = f"Attention Focus Across Tokens (Epoch {epoch}, Batch {batch}, Layer {layer+1})"
        
        return create_timeline_plot(
            attention_focus, 
            sorted(tokens),
            title=title,
            x_label="Token Position",
            y_label="Attention Focus (1-Entropy)",
            highlight_idx=current_token
        )
    
    # Update parameter histogram
    @app.callback(
        Output('parameter-histogram', 'figure'),
        [Input('current-state-storage', 'children'),
         Input('parameter-layer-dropdown', 'value')]
    )
    def update_parameter_histogram(current_state_json, layer):
        if not current_state_json:
            return create_parameter_histogram(None, "Parameter Distribution")
        
        # Get current state info
        current_state_info = json.loads(current_state_json)
        epoch = current_state_info.get('epoch', 0)
        batch = current_state_info.get('batch', 0)
        token = current_state_info.get('token', 0)
        
        # In a real implementation, we would extract parameters for the specified layer
        # For now, we'll generate synthetic data
        param_data = np.random.normal(0, 0.1, 1000)
        
        title = f"Parameter Distribution (Epoch {epoch}, Batch {batch}, Token {token}, Layer {layer})"
        
        return create_parameter_histogram(param_data, title)
    
    # Update state transition graph
    @app.callback(
        Output('state-transition-graph', 'figure'),
        [Input('current-state-storage', 'children')]
    )
    def update_state_transition_graph(current_state_json):
        if not current_state_json:
            return create_state_transition_graph()
        
        current_state = json.loads(current_state_json)
        token = current_state.get('token', 0)
        
        # Map token position to state in the processing flow
        states = [
            "Token Embedding", 
            "Memory Initialization",
            "Memory Reading",
            "Token Summarization",
            "Transformer Processing",
            "Memory Writing",
            "Output Projection",
            "Loss Computation",
            "Backward Pass",
            "Parameter Update"
        ]
        
        # Simple mapping based on token position modulo number of states
        current_state_name = states[token % len(states)]
        
        return create_state_transition_graph(current_state_name)
    
    # Update training metrics graph
    @app.callback(
        Output('training-metrics-graph', 'figure'),
        [Input('epoch-slider', 'value')]
    )
    def update_training_metrics_graph(current_epoch):
        # In a real implementation, we would extract metrics from the state history
        # For now, we'll generate synthetic data
        epochs = list(range(current_epoch + 1))
        
        # Generate synthetic metrics
        np.random.seed(42)  # For reproducibility
        loss = 1.0 - 0.8 * (np.array(epochs) / max(1, current_epoch))
        loss += np.random.normal(0, 0.05, size=len(epochs))
        loss = np.clip(loss, 0.1, 1.0)
        
        accuracy = 0.8 * (np.array(epochs) / max(1, current_epoch))
        accuracy += np.random.normal(0, 0.05, size=len(epochs))
        accuracy = np.clip(accuracy, 0.0, 0.95)
        
        # Create figure
        fig = go.Figure()
        
        # Add loss trace
        fig.add_trace(go.Scatter(
            x=epochs,
            y=loss,
            mode='lines+markers',
            name='Loss',
            line=dict(color=COLORS['viz_4'], width=2),
            marker=dict(size=8)
        ))
        
        # Add accuracy trace
        fig.add_trace(go.Scatter(
            x=epochs,
            y=accuracy,
            mode='lines+markers',
            name='Accuracy',
            line=dict(color=COLORS['viz_1'], width=2),
            marker=dict(size=8)
        ))
        
        # Highlight current epoch
        fig.add_shape(
            type="line",
            x0=current_epoch, y0=0,
            x1=current_epoch, y1=1,
            line=dict(color=COLORS['viz_3'], width=2, dash="dash")
        )
        
        # Update layout
        fig.update_layout(
            create_figure_layout(
                title='Training Metrics',
                xaxis_title='Epoch',
                yaxis_title='Value',
                legend=dict(
                    bgcolor='rgba(0,0,0,0.5)',
                    bordercolor=COLORS['border'],
                    borderwidth=1
                )
            )
        )
        
        return fig


def main():
    """Run the comprehensive interactive visualization dashboard."""
    parser = argparse.ArgumentParser(description='Comprehensive TTM Interactive Visualization')
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
    dashboard = ComprehensiveTTMDashboard(
        state_history_path=args.state_history,
        model_path=args.model,
        port=args.port
    )
    
    # Register callbacks
    register_comprehensive_callbacks(dashboard.app, dashboard)
    
    # Run server
    print(f"Starting comprehensive dashboard on http://localhost:{args.port}")
    dashboard.run_server(debug=args.debug)


if __name__ == '__main__':
    main()
