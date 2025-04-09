"""
Reusable UI components for the TTM Interactive Dashboard.

This module provides reusable UI components for building the dashboard.
"""

from dash import html, dcc
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

from src.ttm.visualization.theme import COLORS, STYLES, create_figure_layout


def card(title, content, id=None, className=None, style=None):
    """
    Create a card component with a title and content.
    
    Args:
        title: Card title
        content: Card content (dash components)
        id: Optional ID for the card
        className: Optional CSS class
        style: Optional additional styles
        
    Returns:
        Dash component
    """
    card_style = STYLES['card'].copy()
    if style:
        card_style.update(style)
    
    return html.Div(
        id=id,
        className=className,
        style=card_style,
        children=[
            html.H3(title, style=STYLES['heading']),
            html.Div(content)
        ]
    )


def sidebar(title, content, id=None):
    """
    Create a sidebar component.
    
    Args:
        title: Sidebar title
        content: Sidebar content (dash components)
        id: Optional ID for the sidebar
        
    Returns:
        Dash component
    """
    return html.Div(
        id=id,
        style=STYLES['sidebar'],
        children=[
            html.H2(title, style=STYLES['heading']),
            html.Div(content)
        ]
    )


def main_content(content, id=None):
    """
    Create a main content component.
    
    Args:
        content: Main content (dash components)
        id: Optional ID for the main content
        
    Returns:
        Dash component
    """
    return html.Div(
        id=id,
        style=STYLES['main_content'],
        children=content
    )


def button(label, id=None, style=None, n_clicks=0, disabled=False, color="primary"):
    """
    Create a button component.
    
    Args:
        label: Button label
        id: Optional ID for the button
        style: Optional additional styles
        n_clicks: Initial number of clicks
        disabled: Whether the button is disabled
        color: Button color (primary or secondary)
        
    Returns:
        Dash component
    """
    button_style = STYLES['button'].copy() if color == "primary" else STYLES['button_secondary'].copy()
    
    if disabled:
        button_style['opacity'] = 0.5
        button_style['cursor'] = 'not-allowed'
    
    if style:
        button_style.update(style)
    
    return html.Button(
        label,
        id=id,
        n_clicks=n_clicks,
        disabled=disabled,
        style=button_style
    )


def input_field(label, id=None, type="text", value=None, placeholder=None, style=None):
    """
    Create an input field with a label.
    
    Args:
        label: Input label
        id: Optional ID for the input
        type: Input type
        value: Initial value
        placeholder: Placeholder text
        style: Optional additional styles
        
    Returns:
        Dash component
    """
    input_style = STYLES['input'].copy()
    if style:
        input_style.update(style)
    
    return html.Div(
        style={'marginBottom': '16px'},
        children=[
            html.Label(label, style={'marginBottom': '8px', 'display': 'block'}),
            dcc.Input(
                id=id,
                type=type,
                value=value,
                placeholder=placeholder,
                style=input_style
            )
        ]
    )


def dropdown(label, id=None, options=None, value=None, multi=False, style=None):
    """
    Create a dropdown with a label.
    
    Args:
        label: Dropdown label
        id: Optional ID for the dropdown
        options: Dropdown options
        value: Initial value
        multi: Whether multiple selection is allowed
        style: Optional additional styles
        
    Returns:
        Dash component
    """
    dropdown_style = {
        'backgroundColor': COLORS['background'],
        'color': COLORS['text'],
        'border': f'1px solid {COLORS["border"]}',
        'borderRadius': '4px'
    }
    
    if style:
        dropdown_style.update(style)
    
    return html.Div(
        style={'marginBottom': '16px'},
        children=[
            html.Label(label, style={'marginBottom': '8px', 'display': 'block'}),
            dcc.Dropdown(
                id=id,
                options=options or [],
                value=value,
                multi=multi,
                style=dropdown_style
            )
        ]
    )


def slider(label, id=None, min=0, max=10, step=1, value=0, marks=None, style=None):
    """
    Create a slider with a label.
    
    Args:
        label: Slider label
        id: Optional ID for the slider
        min: Minimum value
        max: Maximum value
        step: Step size
        value: Initial value
        marks: Slider marks
        style: Optional additional styles
        
    Returns:
        Dash component
    """
    slider_style = {'marginBottom': '24px'}
    if style:
        slider_style.update(style)
    
    # Generate default marks if none provided
    if marks is None:
        marks = {i: str(i) for i in range(min, max + 1, max(1, (max - min) // 5))}
    
    return html.Div(
        style=slider_style,
        children=[
            html.Label(label, style={'marginBottom': '8px', 'display': 'block'}),
            dcc.Slider(
                id=id,
                min=min,
                max=max,
                step=step,
                value=value,
                marks=marks
            )
        ]
    )


def tabs(id=None, tabs=None, value=None, style=None):
    """
    Create a tabs component.
    
    Args:
        id: Optional ID for the tabs
        tabs: List of tab dictionaries with 'label' and 'value'
        value: Initial selected tab value
        style: Optional additional styles
        
    Returns:
        Dash component
    """
    tabs_style = STYLES['tabs'].copy()
    if style:
        tabs_style.update(style)
    
    return dcc.Tabs(
        id=id,
        value=value or (tabs[0]['value'] if tabs else None),
        children=[dcc.Tab(label=tab['label'], value=tab['value']) for tab in (tabs or [])],
        style=tabs_style
    )


def create_memory_heatmap(memory_data, title="Memory Content"):
    """
    Create a memory heatmap visualization.
    
    Args:
        memory_data: Memory data as numpy array [memory_size, embedding_dim]
        title: Title for the plot
        
    Returns:
        Plotly figure
    """
    if memory_data is None or not isinstance(memory_data, np.ndarray):
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="Memory data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(color=COLORS['text'], size=16)
        )
        fig.update_layout(create_figure_layout(title))
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
        color_continuous_scale=[[0, COLORS['viz_1']], [0.5, COLORS['viz_3']], [1, COLORS['viz_4']]]
    )
    
    fig.update_layout(create_figure_layout(title))
    
    return fig


def create_attention_heatmap(attention_data, token_labels=None, title="Attention Weights"):
    """
    Create an attention heatmap visualization.
    
    Args:
        attention_data: Attention weights as numpy array [seq_len, seq_len]
        token_labels: Optional list of token labels
        title: Title for the plot
        
    Returns:
        Plotly figure
    """
    if attention_data is None or not isinstance(attention_data, np.ndarray):
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="Attention data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(color=COLORS['text'], size=16)
        )
        fig.update_layout(create_figure_layout(title))
        return fig
    
    # Create heatmap
    fig = px.imshow(
        attention_data,
        labels=dict(x="Key Position", y="Query Position", color="Attention Weight"),
        color_continuous_scale=[[0, COLORS['viz_1']], [0.5, COLORS['viz_3']], [1, COLORS['viz_4']]]
    )
    
    # Add token labels if provided
    if token_labels is not None:
        fig.update_xaxes(ticktext=token_labels, tickvals=list(range(len(token_labels))))
        fig.update_yaxes(ticktext=token_labels, tickvals=list(range(len(token_labels))))
    
    fig.update_layout(create_figure_layout(title))
    
    return fig


def create_parameter_histogram(parameter_data, title="Parameter Distribution"):
    """
    Create a parameter histogram visualization.
    
    Args:
        parameter_data: Parameter values as flattened numpy array
        title: Title for the plot
        
    Returns:
        Plotly figure
    """
    if parameter_data is None or not isinstance(parameter_data, np.ndarray) or parameter_data.size == 0:
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="Parameter data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(color=COLORS['text'], size=16)
        )
        fig.update_layout(create_figure_layout(title))
        return fig
    
    # Flatten parameter data if needed
    if parameter_data.ndim > 1:
        parameter_data = parameter_data.flatten()
    
    # Create histogram
    fig = go.Figure(data=go.Histogram(
        x=parameter_data,
        nbinsx=50,
        marker_color=COLORS['viz_2']
    ))
    
    fig.update_layout(
        create_figure_layout(
            title=title,
            xaxis_title='Parameter Value',
            yaxis_title='Count'
        )
    )
    
    return fig


def create_timeline_plot(data, x_values, title="Timeline", x_label="Step", y_label="Value", highlight_idx=None):
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
        Plotly figure
    """
    if not data or len(data) == 0:
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="Timeline data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(color=COLORS['text'], size=16)
        )
        fig.update_layout(create_figure_layout(title))
        return fig
    
    # Create line plot
    fig = go.Figure(data=go.Scatter(
        x=x_values,
        y=data,
        mode='lines+markers',
        marker=dict(size=8, color=COLORS['viz_1']),
        line=dict(width=2, color=COLORS['viz_1'])
    ))
    
    # Highlight specific point if requested
    if highlight_idx is not None and highlight_idx in x_values:
        idx = x_values.index(highlight_idx)
        if 0 <= idx < len(data):
            fig.add_trace(go.Scatter(
                x=[x_values[idx]],
                y=[data[idx]],
                mode='markers',
                marker=dict(size=12, color=COLORS['viz_4'], symbol='circle-open', line=dict(width=2)),
                showlegend=False
            ))
    
    fig.update_layout(
        create_figure_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label
        )
    )
    
    return fig


def create_state_transition_graph(current_state=None):
    """
    Create a state transition graph for the TTM model.
    
    Args:
        current_state: Current state to highlight
        
    Returns:
        Plotly figure
    """
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
    
    # Define node positions in a circular layout
    import math
    radius = 1
    node_positions = {}
    for i, state in enumerate(states):
        angle = 2 * math.pi * i / len(states)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        node_positions[i] = (x, y)
    
    # Create edges
    edge_x = []
    edge_y = []
    for i in range(len(states)):
        x0, y0 = node_positions[i]
        x1, y1 = node_positions[(i + 1) % len(states)]
        
        # Create a curved edge
        cpx = (x0 + x1) / 2 - (y1 - y0) * 0.2  # Control point x
        cpy = (y0 + y1) / 2 + (x1 - x0) * 0.2  # Control point y
        
        # Add points for the curve
        t_values = np.linspace(0, 1, 20)
        for t in t_values:
            # Quadratic Bezier curve
            xt = (1-t)**2 * x0 + 2*(1-t)*t * cpx + t**2 * x1
            yt = (1-t)**2 * y0 + 2*(1-t)*t * cpy + t**2 * y1
            edge_x.append(xt)
            edge_y.append(yt)
        
        # Add None to create a break in the line
        edge_x.append(None)
        edge_y.append(None)
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color=COLORS['border']),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node traces
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    
    for i, state in enumerate(states):
        x, y = node_positions[i]
        node_x.append(x)
        node_y.append(y)
        node_text.append(state)
        
        # Highlight current state if provided
        if current_state == state:
            node_colors.append(COLORS['viz_4'])
        else:
            node_colors.append(COLORS['viz_1'])
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        textfont=dict(color=COLORS['text'], size=10),
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color=node_colors,
            size=15,
            line=dict(width=1, color=COLORS['border'])
        )
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace])
    
    # Update layout
    fig.update_layout(
        create_figure_layout(
            title="TTM Training State Transition",
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    
    return fig
