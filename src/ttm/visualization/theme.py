"""
Theme and styling for the TTM Interactive Dashboard.

This module provides consistent styling and theme elements for the dashboard.
"""

# Color palette
COLORS = {
    # Base colors
    'background': '#09090b',      # Slate 950
    'card': '#1c1c1f',            # Slate 900
    'card_hover': '#27272a',      # Slate 800
    'border': '#3f3f46',          # Slate 700
    'muted': '#71717a',           # Slate 500
    'text': '#f4f4f5',            # Slate 100
    'text_muted': '#a1a1aa',      # Slate 400
    
    # Accent colors
    'primary': '#0ea5e9',         # Sky 500
    'primary_hover': '#0284c7',   # Sky 600
    'secondary': '#8b5cf6',       # Violet 500
    'secondary_hover': '#7c3aed', # Violet 600
    
    # Semantic colors
    'success': '#22c55e',         # Green 500
    'warning': '#f59e0b',         # Amber 500
    'error': '#ef4444',           # Red 500
    'info': '#3b82f6',            # Blue 500
    
    # Visualization colors
    'viz_1': '#0ea5e9',           # Sky 500
    'viz_2': '#8b5cf6',           # Violet 500
    'viz_3': '#f59e0b',           # Amber 500
    'viz_4': '#ef4444',           # Red 500
    'viz_5': '#22c55e',           # Green 500
    'viz_6': '#ec4899',           # Pink 500
    'viz_7': '#14b8a6',           # Teal 500
    'viz_8': '#f97316',           # Orange 500
}

# Visualization color scales
COLOR_SCALES = {
    'heatmap': [
        [0, COLORS['viz_1']],
        [0.5, COLORS['viz_3']],
        [1, COLORS['viz_4']]
    ],
    'diverging': [
        [0, COLORS['viz_4']],
        [0.5, COLORS['text_muted']],
        [1, COLORS['viz_5']]
    ],
    'sequential': [
        [0, COLORS['card']],
        [1, COLORS['viz_1']]
    ]
}

# Typography
TYPOGRAPHY = {
    'font_family': 'Inter, system-ui, sans-serif',
    'heading': {
        'font_weight': '600',
        'line_height': '1.2',
        'letter_spacing': '-0.025em'
    },
    'body': {
        'font_weight': '400',
        'line_height': '1.5',
        'letter_spacing': '0'
    }
}

# Spacing
SPACING = {
    'xs': '4px',
    'sm': '8px',
    'md': '16px',
    'lg': '24px',
    'xl': '32px',
    '2xl': '48px',
    '3xl': '64px'
}

# Border radius
RADIUS = {
    'sm': '4px',
    'md': '6px',
    'lg': '8px',
    'xl': '12px',
    'full': '9999px'
}

# Shadows
SHADOWS = {
    'sm': '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
    'md': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
    'lg': '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
    'xl': '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)'
}

# Common styles
STYLES = {
    'page': {
        'backgroundColor': COLORS['background'],
        'color': COLORS['text'],
        'fontFamily': TYPOGRAPHY['font_family'],
        'minHeight': '100vh',
        'padding': SPACING['lg']
    },
    'sidebar': {
        'backgroundColor': COLORS['card'],
        'borderRight': f'1px solid {COLORS["border"]}',
        'padding': SPACING['lg'],
        'width': '280px',
        'height': '100vh',
        'position': 'fixed',
        'top': '0',
        'left': '0',
        'overflowY': 'auto'
    },
    'main_content': {
        'marginLeft': '280px',
        'padding': SPACING['lg']
    },
    'card': {
        'backgroundColor': COLORS['card'],
        'borderRadius': RADIUS['lg'],
        'border': f'1px solid {COLORS["border"]}',
        'padding': SPACING['lg'],
        'marginBottom': SPACING['lg']
    },
    'heading': {
        'color': COLORS['text'],
        'fontWeight': TYPOGRAPHY['heading']['font_weight'],
        'lineHeight': TYPOGRAPHY['heading']['line_height'],
        'letterSpacing': TYPOGRAPHY['heading']['letter_spacing'],
        'marginBottom': SPACING['md']
    },
    'button': {
        'backgroundColor': COLORS['primary'],
        'color': COLORS['text'],
        'border': 'none',
        'borderRadius': RADIUS['md'],
        'padding': f'{SPACING["sm"]} {SPACING["md"]}',
        'fontWeight': '500',
        'cursor': 'pointer',
        'transition': 'background-color 0.2s',
        'outline': 'none'
    },
    'button_hover': {
        'backgroundColor': COLORS['primary_hover']
    },
    'button_secondary': {
        'backgroundColor': COLORS['secondary'],
        'color': COLORS['text'],
        'border': 'none',
        'borderRadius': RADIUS['md'],
        'padding': f'{SPACING["sm"]} {SPACING["md"]}',
        'fontWeight': '500',
        'cursor': 'pointer',
        'transition': 'background-color 0.2s',
        'outline': 'none'
    },
    'button_secondary_hover': {
        'backgroundColor': COLORS['secondary_hover']
    },
    'input': {
        'backgroundColor': COLORS['background'],
        'color': COLORS['text'],
        'border': f'1px solid {COLORS["border"]}',
        'borderRadius': RADIUS['md'],
        'padding': SPACING['sm'],
        'outline': 'none',
        'width': '100%'
    },
    'input_focus': {
        'borderColor': COLORS['primary']
    },
    'select': {
        'backgroundColor': COLORS['background'],
        'color': COLORS['text'],
        'border': f'1px solid {COLORS["border"]}',
        'borderRadius': RADIUS['md'],
        'padding': SPACING['sm'],
        'outline': 'none',
        'width': '100%'
    },
    'tabs': {
        'borderBottom': f'1px solid {COLORS["border"]}',
        'marginBottom': SPACING['lg']
    },
    'tab': {
        'padding': f'{SPACING["sm"]} {SPACING["md"]}',
        'marginRight': SPACING['sm'],
        'borderBottom': '2px solid transparent',
        'cursor': 'pointer'
    },
    'tab_selected': {
        'borderBottomColor': COLORS['primary'],
        'color': COLORS['primary']
    },
    'graph': {
        'backgroundColor': COLORS['card'],
        'borderRadius': RADIUS['lg'],
        'border': f'1px solid {COLORS["border"]}',
        'height': '400px'
    }
}

# Plotly figure layout defaults
FIGURE_LAYOUT = {
    'template': 'plotly_dark',
    'paper_bgcolor': COLORS['card'],
    'plot_bgcolor': COLORS['card'],
    'font': {
        'family': TYPOGRAPHY['font_family'],
        'color': COLORS['text']
    },
    'margin': {'l': 40, 'r': 40, 't': 40, 'b': 40},
    'xaxis': {
        'gridcolor': COLORS['border'],
        'zerolinecolor': COLORS['border']
    },
    'yaxis': {
        'gridcolor': COLORS['border'],
        'zerolinecolor': COLORS['border']
    },
    'coloraxis': {
        'colorscale': COLOR_SCALES['heatmap']
    }
}

# Function to create a consistent figure layout
def create_figure_layout(title=None, **kwargs):
    """
    Create a consistent figure layout with optional overrides.
    
    Args:
        title: Optional title for the figure
        **kwargs: Additional layout parameters to override defaults
    
    Returns:
        Dictionary with figure layout parameters
    """
    layout = FIGURE_LAYOUT.copy()
    
    if title:
        layout['title'] = {
            'text': title,
            'font': {
                'family': TYPOGRAPHY['font_family'],
                'size': 16,
                'color': COLORS['text']
            },
            'x': 0.05
        }
    
    # Update with any additional parameters
    for key, value in kwargs.items():
        if key in layout and isinstance(layout[key], dict) and isinstance(value, dict):
            layout[key].update(value)
        else:
            layout[key] = value
    
    return layout
