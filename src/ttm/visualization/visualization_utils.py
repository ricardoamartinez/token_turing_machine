"""
Visualization utilities for TTM model.

This module provides utility functions for visualizing TTM model states.
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Any, Optional, Union
import networkx as nx
import torch
from src.ttm.data.tokenization import tokens_to_string


def create_state_transition_graph(current_state: str = None) -> go.Figure:
    """
    Create a state transition graph for the TTM model.
    
    Args:
        current_state: Current state to highlight
        
    Returns:
        Plotly figure object
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Define the main states in the training loop
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
    
    # Add nodes
    for i, state in enumerate(states):
        G.add_node(i, name=state)
        
    # Add edges (transitions)
    for i in range(len(states)-1):
        G.add_edge(i, i+1)
    
    # Add edge from last to first (for next token)
    G.add_edge(len(states)-1, 0)
    
    # Highlight current state if provided
    current_state_idx = None
    if current_state is not None:
        try:
            current_state_idx = states.index(current_state)
        except ValueError:
            pass
    
    # Convert to plotly figure
    pos = nx.spring_layout(G, seed=42)  # For consistent layout
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
    node_colors = ['#FF4136' if i == current_state_idx else '#17BECF' for i in range(len(states))]
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[G.nodes[i]['name'] for i in range(len(states))],
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color=node_colors,
            size=30,
            line_width=2))
    
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='TTM Training State Transition',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor='#000000',
                        paper_bgcolor='#000000',
                        font=dict(color='white')
                    ))
    
    return fig


def create_memory_heatmap(memory_data: np.ndarray, title: str = "Memory Content") -> go.Figure:
    """
    Create a heatmap visualization of memory content.
    
    Args:
        memory_data: Memory data as numpy array [memory_size, embedding_dim]
        title: Title for the plot
        
    Returns:
        Plotly figure object
    """
    if memory_data is None:
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="Memory data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(color="white", size=20)
        )
        fig.update_layout(
            title=title,
            plot_bgcolor='#000000',
            paper_bgcolor='#000000',
            font=dict(color='white')
        )
        return fig
    
    # Create axis labels
    memory_size, embedding_dim = memory_data.shape
    x_labels = [f"Dim {i}" for i in range(embedding_dim)]
    y_labels = [f"Slot {i}" for i in range(memory_size)]
    
    # Create heatmap
    fig = px.imshow(
        memory_data,
        labels=dict(x="Embedding Dimension", y="Memory Slot", color="Value"),
        x=x_labels,
        y=y_labels,
        color_continuous_scale='plasma'
    )
    
    fig.update_layout(
        title=title,
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        font=dict(color='white')
    )
    
    return fig


def create_attention_heatmap(attention_data: np.ndarray, 
                            token_labels: Optional[List[str]] = None,
                            title: str = "Attention Weights") -> go.Figure:
    """
    Create a heatmap visualization of attention weights.
    
    Args:
        attention_data: Attention weights as numpy array [seq_len, seq_len]
        token_labels: Optional list of token labels
        title: Title for the plot
        
    Returns:
        Plotly figure object
    """
    if attention_data is None:
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="Attention data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(color="white", size=20)
        )
        fig.update_layout(
            title=title,
            plot_bgcolor='#000000',
            paper_bgcolor='#000000',
            font=dict(color='white')
        )
        return fig
    
    # Create heatmap
    fig = px.imshow(
        attention_data,
        labels=dict(x="Key Position", y="Query Position", color="Attention Weight"),
        color_continuous_scale='plasma'
    )
    
    # Add token labels if provided
    if token_labels is not None:
        fig.update_xaxes(ticktext=token_labels, tickvals=list(range(len(token_labels))))
        fig.update_yaxes(ticktext=token_labels, tickvals=list(range(len(token_labels))))
    
    fig.update_layout(
        title=title,
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        font=dict(color='white')
    )
    
    return fig


def create_parameter_histogram(parameter_data: np.ndarray, 
                              title: str = "Parameter Distribution") -> go.Figure:
    """
    Create a histogram visualization of parameter values.
    
    Args:
        parameter_data: Parameter values as flattened numpy array
        title: Title for the plot
        
    Returns:
        Plotly figure object
    """
    if parameter_data is None or parameter_data.size == 0:
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="Parameter data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(color="white", size=20)
        )
        fig.update_layout(
            title=title,
            plot_bgcolor='#000000',
            paper_bgcolor='#000000',
            font=dict(color='white')
        )
        return fig
    
    # Flatten parameter data if needed
    if parameter_data.ndim > 1:
        parameter_data = parameter_data.flatten()
    
    # Create histogram
    fig = go.Figure(data=go.Histogram(
        x=parameter_data,
        nbinsx=50,
        marker_color='#FF851B'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Parameter Value',
        yaxis_title='Count',
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        font=dict(color='white')
    )
    
    return fig


def create_timeline_plot(data: List[float], 
                        x_values: List[int],
                        title: str = "Timeline",
                        x_label: str = "Step",
                        y_label: str = "Value",
                        highlight_idx: Optional[int] = None) -> go.Figure:
    """
    Create a timeline plot of values.
    
    Args:
        data: List of values to plot
        x_values: List of x-axis values
        title: Title for the plot
        x_label: Label for x-axis
        y_label: Label for y-axis
        highlight_idx: Optional index to highlight
        
    Returns:
        Plotly figure object
    """
    if not data or len(data) == 0:
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="Timeline data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(color="white", size=20)
        )
        fig.update_layout(
            title=title,
            plot_bgcolor='#000000',
            paper_bgcolor='#000000',
            font=dict(color='white')
        )
        return fig
    
    # Create line plot
    fig = go.Figure(data=go.Scatter(
        x=x_values,
        y=data,
        mode='lines+markers',
        marker=dict(size=10, color='#36A2EB'),
        line=dict(width=2, color='#36A2EB')
    ))
    
    # Highlight specific point if requested
    if highlight_idx is not None and highlight_idx in x_values:
        idx = x_values.index(highlight_idx)
        if 0 <= idx < len(data):
            fig.add_trace(go.Scatter(
                x=[x_values[idx]],
                y=[data[idx]],
                mode='markers',
                marker=dict(size=15, color='#FF4136', symbol='circle-open', line=dict(width=3)),
                showlegend=False
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        font=dict(color='white')
    )
    
    return fig


def tokens_to_labels(token_indices: List[int]) -> List[str]:
    """
    Convert token indices to human-readable labels.
    
    Args:
        token_indices: List of token indices
        
    Returns:
        List of token labels
    """
    # Convert to numpy array if it's a tensor
    if isinstance(token_indices, torch.Tensor):
        token_indices = token_indices.cpu().numpy()
    
    # Convert to string representation
    token_str = tokens_to_string(token_indices)
    
    # Split into individual tokens
    tokens = token_str.split()
    
    # Remove special tokens for cleaner display
    tokens = [t for t in tokens if t not in ["<PAD>", "<EOS>"]]
    
    return tokens
