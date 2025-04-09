"""
Run script for the TTM Interactive Dashboard.

This script provides a simple way to run the TTM Interactive Dashboard.
"""

import os
import sys
import argparse

def main():
    """Run the TTM Interactive Dashboard."""
    parser = argparse.ArgumentParser(description='Run TTM Interactive Dashboard')
    parser.add_argument('--enhanced', action='store_true',
                        help='Run the enhanced dashboard')
    parser.add_argument('--state_history', type=str, default=None,
                        help='Path to state history file')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--port', type=int, default=8050,
                        help='Port to run the dashboard on')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Build command
    cmd = []
    
    # Python executable
    cmd.append(sys.executable)
    
    # Script to run
    if args.enhanced:
        cmd.append('interactive_visualization_enhanced.py')
    else:
        cmd.append('interactive_visualization.py')
    
    # Add arguments
    if args.state_history:
        cmd.append(f'--state_history={args.state_history}')
    
    if args.model:
        cmd.append(f'--model={args.model}')
    
    if args.port != 8050:
        cmd.append(f'--port={args.port}')
    
    if args.debug:
        cmd.append('--debug')
    
    # Run command
    cmd_str = ' '.join(cmd)
    print(f"Running: {cmd_str}")
    os.system(cmd_str)

if __name__ == '__main__':
    main()
