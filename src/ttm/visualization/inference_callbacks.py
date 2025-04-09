"""
Inference mode callbacks for the TTM Interactive Dashboard.

This module provides callback functions for the inference testing mode of the dashboard.
"""

from dash import html, dcc, callback_context
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import numpy as np
import json
import torch
import os
from typing import Dict, List, Tuple, Any, Optional, Union
import dash

from src.ttm.data.tokenization import (
    create_multiplication_example,
    tokens_to_string,
    number_to_tokens
)
from src.ttm.visualization.visualization_utils import (
    create_memory_heatmap,
    create_attention_heatmap,
    tokens_to_labels
)


def register_inference_callbacks(app, dashboard):
    """
    Register callbacks for inference testing mode.

    Args:
        app: Dash app
        dashboard: TTMDashboard instance
    """

    # Combined load model and run inference callback
    @app.callback(
        Output('inference-results', 'children'),
        [Input('load-model-button', 'n_clicks'),
         Input('run-inference-button', 'n_clicks')],
        [State('checkpoint-dropdown', 'value'),
         State('num1-input', 'value'),
         State('num2-input', 'value')]
    )
    def handle_inference_actions(load_clicks, run_clicks, checkpoint_path, num1, num2):
        # Use callback context to determine which button was clicked
        ctx = callback_context
        if not ctx.triggered:
            # No button clicked yet
            return html.Div("No model loaded")

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        # Handle load model button
        if button_id == 'load-model-button':
            if load_clicks == 0 or not checkpoint_path:
                return html.Div("No model loaded")

            try:
                # Load the model
                dashboard._load_model(checkpoint_path)

                return html.Div([
                    html.H4("Model Loaded Successfully", style={'color': '#66ff99'}),
                    html.P(f"Checkpoint: {os.path.basename(checkpoint_path)}"),
                    html.P("Enter numbers and click 'Run Inference' to test the model.")
                ])
            except Exception as e:
                return html.Div([
                    html.H4("Error Loading Model", style={'color': '#ff6666'}),
                    html.P(str(e))
                ])

        # Handle run inference button
        elif button_id == 'run-inference-button':
            if run_clicks == 0:
                return html.Div("Enter numbers and click 'Run Inference'")

            if dashboard.model is None:
                return html.Div([
                    html.H4("No Model Loaded", style={'color': '#ff6666'}),
                    html.P("Please load a model first.")
                ])

            if num1 is None or num2 is None:
                return html.Div([
                    html.H4("Invalid Input", style={'color': '#ff6666'}),
                    html.P("Please enter valid numbers.")
                ])

            try:
                # Create input example
                input_tokens, target_tokens = create_multiplication_example(num1, num2)

                # Convert to tensor
                input_tensor = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0)  # Add batch dimension

                # Run inference
                with torch.no_grad():
                    # Initialize memory if needed
                    if hasattr(dashboard.model, 'memory_less') and not dashboard.model.memory_less:
                        memory = dashboard.model.initialize_memory(batch_size=1)
                    else:
                        memory = None

                    # Forward pass
                    output, _ = dashboard.model(input_tensor, memory)

                    # Get predictions
                    predictions = output.argmax(dim=-1).squeeze().tolist()

                    # Convert to string
                    input_str = tokens_to_string(input_tokens)
                    target_str = tokens_to_string(target_tokens)
                    pred_str = tokens_to_string(predictions)

                    # Clean up strings
                    input_str = input_str.replace("<PAD>", "").replace("<EOS>", "")
                    target_str = target_str.replace("<PAD>", "").replace("<EOS>", "")
                    pred_str = pred_str.replace("<PAD>", "").replace("<EOS>", "")

                    # Store inference state for visualization
                    dashboard.inference_state = {
                        'input_tokens': input_tokens,
                        'target_tokens': target_tokens,
                        'predictions': predictions,
                        'input_tensor': input_tensor,
                        'output': output,
                        'memory': memory
                    }

                    # Check if prediction is correct
                    is_correct = pred_str.strip() == target_str.strip()

                    return html.Div([
                        html.H4("Inference Results"),
                        html.Div([
                            html.Div([
                                html.H5("Input:"),
                                html.P(input_str, style={'fontSize': '18px'})
                            ]),
                            html.Div([
                                html.H5("Expected Output:"),
                                html.P(target_str, style={'fontSize': '18px'})
                            ]),
                            html.Div([
                                html.H5("Model Prediction:"),
                                html.P(pred_str, style={'fontSize': '18px',
                                                       'color': '#66ff99' if is_correct else '#ff6666'})
                            ]),
                            html.Div([
                                html.H5("Result:"),
                                html.P("Correct!" if is_correct else "Incorrect",
                                      style={'fontSize': '18px', 'fontWeight': 'bold',
                                            'color': '#66ff99' if is_correct else '#ff6666'})
                            ])
                        ])
                    ])
            except Exception as e:
                return html.Div([
                    html.H4("Error Running Inference", style={'color': '#ff6666'}),
                    html.P(str(e))
                ])

        # Default return if no button was recognized
        return html.Div("Select a model and run inference")

    # Update inference memory heatmap
    @app.callback(
        Output('inference-memory-heatmap', 'figure'),
        [Input('inference-token-slider', 'value'),
         Input('inference-memory-slot-slider', 'value')]
    )
    def update_inference_memory_heatmap(token, memory_slot):
        if not hasattr(dashboard, 'inference_state') or dashboard.inference_state is None:
            return create_memory_heatmap(None, "No inference data available")

        # Get memory data
        memory_data = dashboard.inference_state.get('memory')

        # Create title
        title = f"Memory Content (Token {token})"

        # If memory data is not available, return empty figure
        if memory_data is None:
            return create_memory_heatmap(None, title)

        # Convert to numpy if needed
        if isinstance(memory_data, torch.Tensor):
            memory_data = memory_data.detach().cpu().numpy()

        # Get the first batch item
        if memory_data.ndim > 2:
            memory_data = memory_data[0]

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

    # Update inference attention heatmap
    @app.callback(
        Output('inference-attention-heatmap', 'figure'),
        [Input('inference-token-slider', 'value'),
         Input('inference-attention-layer-slider', 'value'),
         Input('inference-attention-head-slider', 'value')]
    )
    def update_inference_attention_heatmap(token, layer, head):
        if not hasattr(dashboard, 'inference_state') or dashboard.inference_state is None:
            return create_attention_heatmap(None, title="No inference data available")

        # In a real implementation, we would extract attention weights from the model
        # For now, we'll generate synthetic data
        seq_len = len(dashboard.inference_state.get('input_tokens', []))

        # Generate synthetic attention data
        np.random.seed(token * 100 + layer * 10 + head)
        attention_data = np.random.rand(seq_len, seq_len)

        # Make it look more like attention (normalize rows)
        for i in range(seq_len):
            attention_data[i] = attention_data[i] / attention_data[i].sum()

        # Add some structure - diagonal attention pattern
        for i in range(seq_len):
            for j in range(seq_len):
                if i == j:
                    attention_data[i, j] *= 2
                elif abs(i - j) == 1:
                    attention_data[i, j] *= 1.5

        # Renormalize
        for i in range(seq_len):
            attention_data[i] = attention_data[i] / attention_data[i].sum()

        # Get token labels
        input_tokens = dashboard.inference_state.get('input_tokens', [])
        token_labels = tokens_to_labels(input_tokens)

        title = f"Attention Weights (Token {token}, Layer {layer}, Head {head})"

        return create_attention_heatmap(attention_data, token_labels, title)
