"""
Dashboard for visualizing TTM model training in real-time.

This module provides a web-based dashboard for monitoring the training of the
Token Turing Machine (TTM) model in real-time.
"""

import os
import sys
import json
import logging
import threading
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver
import queue

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import io
import base64

from ..models.ttm_model import TokenTuringMachine


# Global variables for dashboard state
dashboard_data = {
    'metrics': {
        'loss': [],
        'position_accuracy': [],
        'sequence_accuracy': [],
        'learning_rate': [],
        'difficulty_stage': []
    },
    'examples': [],
    'parameter_stats': {},
    'gradient_stats': {},
    'memory_visualizations': [],
    'attention_visualizations': [],
    'training_status': 'idle',
    'last_update': None,
    'issues': []
}

# Queue for thread-safe updates
update_queue = queue.Queue()


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the dashboard."""

    def _set_headers(self, content_type='text/html'):
        """Set response headers.

        Args:
            content_type: Content type of the response
        """
        self.send_response(200)
        self.send_header('Content-type', content_type)
        self.end_headers()

    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/':
            # Serve main dashboard page
            self._set_headers()
            self.wfile.write(self._get_dashboard_html().encode())
        elif self.path == '/data':
            # Serve dashboard data as JSON
            self._set_headers('application/json')
            self.wfile.write(json.dumps(dashboard_data).encode())
        elif self.path.startswith('/static/'):
            # Serve static files (CSS, JS)
            file_path = self.path[8:]  # Remove '/static/' prefix
            try:
                with open(os.path.join('dashboard', 'static', file_path), 'rb') as f:
                    content_type = 'text/css' if file_path.endswith('.css') else 'application/javascript'
                    self._set_headers(content_type)
                    self.wfile.write(f.read())
            except FileNotFoundError:
                self.send_response(404)
                self.end_headers()
        else:
            # Handle 404
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        """Handle POST requests."""
        if self.path == '/control':
            # Handle control commands (pause, resume, etc.)
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            command = json.loads(post_data)

            # Process command
            if command['action'] == 'pause':
                dashboard_data['training_status'] = 'paused'
            elif command['action'] == 'resume':
                dashboard_data['training_status'] = 'training'
            elif command['action'] == 'stop':
                dashboard_data['training_status'] = 'stopped'
            elif command['action'] == 'update_params':
                # Update training parameters
                pass

            # Send response
            self._set_headers('application/json')
            self.wfile.write(json.dumps({'status': 'success'}).encode())
        else:
            # Handle 404
            self.send_response(404)
            self.end_headers()

    def _get_dashboard_html(self):
        """Get the HTML for the dashboard.

        Returns:
            HTML string
        """
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TTM Training Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            background-color: #333;
            color: white;
            padding: 10px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 {
            margin: 0;
        }
        .status {
            display: flex;
            align-items: center;
        }
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 10px;
        }
        .status-idle {
            background-color: gray;
        }
        .status-training {
            background-color: green;
        }
        .status-paused {
            background-color: orange;
        }
        .status-stopped {
            background-color: red;
        }
        .controls {
            display: flex;
            gap: 10px;
        }
        .controls button {
            padding: 5px 10px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .controls button:hover {
            opacity: 0.8;
        }
        .btn-pause {
            background-color: orange;
            color: white;
        }
        .btn-resume {
            background-color: green;
            color: white;
        }
        .btn-stop {
            background-color: red;
            color: white;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-top: 20px;
        }
        .card {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            padding: 20px;
        }
        .card h2 {
            margin-top: 0;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }
        .metric {
            background-color: #f9f9f9;
            padding: 10px;
            border-radius: 4px;
        }
        .metric h3 {
            margin: 0;
            font-size: 14px;
            color: #666;
        }
        .metric p {
            margin: 5px 0 0;
            font-size: 24px;
            font-weight: bold;
        }
        .chart-container {
            width: 100%;
            height: 300px;
            margin-top: 10px;
            position: relative;
            overflow: hidden;
        }
        .chart {
            width: 100%;
            height: 100%;
        }
        .examples {
            margin-top: 10px;
        }
        .example {
            background-color: #f9f9f9;
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 10px;
        }
        .example p {
            margin: 0;
        }
        .example .correct {
            color: green;
        }
        .example .incorrect {
            color: red;
        }
        .issues {
            margin-top: 10px;
        }
        .issue {
            background-color: #fff0f0;
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 10px;
            border-left: 4px solid red;
        }
        .visualization-container {
            width: 100%;
            height: 300px;
            margin-top: 10px;
            position: relative;
            overflow: hidden;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .visualization {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }
        .footer {
            margin-top: 20px;
            text-align: center;
            color: #666;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>TTM Training Dashboard</h1>
        <div class="status">
            <div id="status-indicator" class="status-indicator status-idle"></div>
            <span id="status-text">Idle</span>
        </div>
        <div class="controls">
            <button id="btn-pause" class="btn-pause">Pause</button>
            <button id="btn-resume" class="btn-resume">Resume</button>
            <button id="btn-stop" class="btn-stop">Stop</button>
        </div>
    </div>

    <div class="container">
        <div class="grid">
            <div class="card">
                <h2>Training Metrics</h2>
                <div class="metrics-grid">
                    <div class="metric">
                        <h3>Loss</h3>
                        <p id="metric-loss">-</p>
                    </div>
                    <div class="metric">
                        <h3>Position Accuracy</h3>
                        <p id="metric-position-accuracy">-</p>
                    </div>
                    <div class="metric">
                        <h3>Sequence Accuracy</h3>
                        <p id="metric-sequence-accuracy">-</p>
                    </div>
                    <div class="metric">
                        <h3>Difficulty Stage</h3>
                        <p id="metric-difficulty-stage">-</p>
                    </div>
                </div>
                <div class="chart-container">
                    <canvas id="chart-metrics" class="chart"></canvas>
                </div>
            </div>

            <div class="card">
                <h2>Example Predictions</h2>
                <div id="examples" class="examples">
                    <p>No examples available</p>
                </div>
            </div>

            <div class="card">
                <h2>Memory Visualization</h2>
                <div class="visualization-container">
                    <img id="memory-viz" class="visualization" src="" alt="Memory visualization not available">
                </div>
            </div>

            <div class="card">
                <h2>Attention Visualization</h2>
                <div class="visualization-container">
                    <img id="attention-viz" class="visualization" src="" alt="Attention visualization not available">
                </div>
            </div>

            <div class="card">
                <h2>Parameter Distribution</h2>
                <div class="chart-container">
                    <canvas id="chart-params" class="chart"></canvas>
                </div>
            </div>

            <div class="card">
                <h2>Training Issues</h2>
                <div id="issues" class="issues">
                    <p>No issues detected</p>
                </div>
            </div>
        </div>

        <div class="footer">
            <p>Last updated: <span id="last-update">Never</span></p>
        </div>
    </div>

    <script>
        // Dashboard update interval (ms)
        const UPDATE_INTERVAL = 1000;

        // Charts
        let metricsChart;
        let paramsChart;

        // Initialize dashboard
        function initDashboard() {
            // Initialize charts
            const metricsCtx = document.getElementById('chart-metrics').getContext('2d');
            metricsChart = new Chart(metricsCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: 'Loss',
                            data: [],
                            borderColor: 'red',
                            backgroundColor: 'rgba(255, 0, 0, 0.1)',
                            fill: true
                        },
                        {
                            label: 'Position Accuracy',
                            data: [],
                            borderColor: 'blue',
                            backgroundColor: 'rgba(0, 0, 255, 0.1)',
                            fill: true
                        },
                        {
                            label: 'Sequence Accuracy',
                            data: [],
                            borderColor: 'green',
                            backgroundColor: 'rgba(0, 255, 0, 0.1)',
                            fill: true
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Epoch'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Value'
                            }
                        }
                    },
                    animation: {
                        duration: 0 // Disable animations for better performance
                    },
                    plugins: {
                        legend: {
                            position: 'top',
                            labels: {
                                boxWidth: 10,
                                font: {
                                    size: 10
                                }
                            }
                        }
                    }
                }
            });

            const paramsCtx = document.getElementById('chart-params').getContext('2d');
            paramsChart = new Chart(paramsCtx, {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: 'Mean',
                            data: [],
                            backgroundColor: 'rgba(54, 162, 235, 0.5)'
                        },
                        {
                            label: 'Std',
                            data: [],
                            backgroundColor: 'rgba(255, 99, 132, 0.5)'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Parameter'
                            },
                            ticks: {
                                display: false // Hide x-axis labels for better display
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Value'
                            }
                        }
                    },
                    animation: {
                        duration: 0 // Disable animations for better performance
                    },
                    plugins: {
                        legend: {
                            position: 'top',
                            labels: {
                                boxWidth: 10,
                                font: {
                                    size: 10
                                }
                            }
                        }
                    }
                }
            });

            // Set up control buttons
            document.getElementById('btn-pause').addEventListener('click', () => {
                sendCommand('pause');
            });

            document.getElementById('btn-resume').addEventListener('click', () => {
                sendCommand('resume');
            });

            document.getElementById('btn-stop').addEventListener('click', () => {
                sendCommand('stop');
            });

            // Start update loop
            updateDashboard();
            setInterval(updateDashboard, UPDATE_INTERVAL);
        }

        // Update dashboard with latest data
        async function updateDashboard() {
            try {
                const response = await fetch('/data');
                const data = await response.json();

                // Update status
                updateStatus(data.training_status);

                // Update metrics
                updateMetrics(data.metrics);

                // Update examples
                updateExamples(data.examples);

                // Update visualizations
                updateVisualizations(data);

                // Update issues
                updateIssues(data.issues);

                // Update last update time
                if (data.last_update) {
                    document.getElementById('last-update').textContent = new Date(data.last_update).toLocaleString();
                }
            } catch (error) {
                console.error('Error updating dashboard:', error);
            }
        }

        // Update training status
        function updateStatus(status) {
            const indicator = document.getElementById('status-indicator');
            const text = document.getElementById('status-text');

            // Remove all status classes
            indicator.classList.remove('status-idle', 'status-training', 'status-paused', 'status-stopped');

            // Add appropriate class
            indicator.classList.add(`status-${status}`);

            // Update text
            text.textContent = status.charAt(0).toUpperCase() + status.slice(1);
        }

        // Update metrics
        function updateMetrics(metrics) {
            // Update metric values
            if (metrics.loss.length > 0) {
                document.getElementById('metric-loss').textContent = metrics.loss[metrics.loss.length - 1].toFixed(4);
            }

            if (metrics.position_accuracy.length > 0) {
                document.getElementById('metric-position-accuracy').textContent =
                    (metrics.position_accuracy[metrics.position_accuracy.length - 1] * 100).toFixed(2) + '%';
            }

            if (metrics.sequence_accuracy.length > 0) {
                document.getElementById('metric-sequence-accuracy').textContent =
                    (metrics.sequence_accuracy[metrics.sequence_accuracy.length - 1] * 100).toFixed(2) + '%';
            }

            if (metrics.difficulty_stage.length > 0) {
                document.getElementById('metric-difficulty-stage').textContent =
                    metrics.difficulty_stage[metrics.difficulty_stage.length - 1];
            }

            // Update charts
            // Limit the number of data points to prevent overcrowding
            const MAX_DATA_POINTS = 100;
            let epochs, loss_data, pos_acc_data, seq_acc_data;

            if (metrics.loss.length > MAX_DATA_POINTS) {
                // If we have too many points, sample them to reduce the number
                const step = Math.ceil(metrics.loss.length / MAX_DATA_POINTS);
                epochs = [];
                loss_data = [];
                pos_acc_data = [];
                seq_acc_data = [];

                for (let i = 0; i < metrics.loss.length; i += step) {
                    epochs.push(i + 1);
                    loss_data.push(metrics.loss[i]);
                    pos_acc_data.push(metrics.position_accuracy[i]);
                    seq_acc_data.push(metrics.sequence_accuracy[i]);
                }

                // Always include the most recent data point
                if (epochs[epochs.length - 1] !== metrics.loss.length) {
                    epochs.push(metrics.loss.length);
                    loss_data.push(metrics.loss[metrics.loss.length - 1]);
                    pos_acc_data.push(metrics.position_accuracy[metrics.position_accuracy.length - 1]);
                    seq_acc_data.push(metrics.sequence_accuracy[metrics.sequence_accuracy.length - 1]);
                }
            } else {
                epochs = Array.from({length: metrics.loss.length}, (_, i) => i + 1);
                loss_data = metrics.loss;
                pos_acc_data = metrics.position_accuracy;
                seq_acc_data = metrics.sequence_accuracy;
            }

            metricsChart.data.labels = epochs;
            metricsChart.data.datasets[0].data = loss_data;
            metricsChart.data.datasets[1].data = pos_acc_data;
            metricsChart.data.datasets[2].data = seq_acc_data;
            metricsChart.update();
        }

        // Update example predictions
        function updateExamples(examples) {
            const container = document.getElementById('examples');

            if (examples.length === 0) {
                container.innerHTML = '<p>No examples available</p>';
                return;
            }

            container.innerHTML = '';

            examples.forEach(example => {
                const div = document.createElement('div');
                div.className = 'example';

                const isCorrect = example.predicted === example.expected;
                const resultClass = isCorrect ? 'correct' : 'incorrect';

                div.innerHTML = `
                    <p>${example.num1} × ${example.num2} = ${example.expected}</p>
                    <p class="${resultClass}">Predicted: ${example.predicted}</p>
                `;

                container.appendChild(div);
            });
        }

        // Update visualizations
        function updateVisualizations(data) {
            // Update memory visualization
            if (data.memory_visualizations.length > 0) {
                document.getElementById('memory-viz').src = data.memory_visualizations[data.memory_visualizations.length - 1];
            }

            // Update attention visualization
            if (data.attention_visualizations.length > 0) {
                document.getElementById('attention-viz').src = data.attention_visualizations[data.attention_visualizations.length - 1];
            }

            // Update parameter distribution chart
            if (Object.keys(data.parameter_stats).length > 0) {
                // Limit the number of parameters to display
                const MAX_PARAMS = 20;
                let paramNames = Object.keys(data.parameter_stats);

                // If we have too many parameters, group them by layer
                if (paramNames.length > MAX_PARAMS) {
                    const layerStats = {};

                    // Group parameters by layer
                    paramNames.forEach(name => {
                        const layerName = name.split('.')[0];
                        if (!layerStats[layerName]) {
                            layerStats[layerName] = {
                                count: 0,
                                meanSum: 0,
                                stdSum: 0
                            };
                        }
                        layerStats[layerName].count++;
                        layerStats[layerName].meanSum += data.parameter_stats[name].mean;
                        layerStats[layerName].stdSum += data.parameter_stats[name].std;
                    });

                    // Calculate average stats for each layer
                    paramNames = Object.keys(layerStats);
                    const means = paramNames.map(layer => layerStats[layer].meanSum / layerStats[layer].count);
                    const stds = paramNames.map(layer => layerStats[layer].stdSum / layerStats[layer].count);

                    paramsChart.data.labels = paramNames;
                    paramsChart.data.datasets[0].data = means;
                    paramsChart.data.datasets[1].data = stds;
                } else {
                    const means = paramNames.map(name => data.parameter_stats[name].mean);
                    const stds = paramNames.map(name => data.parameter_stats[name].std);

                    paramsChart.data.labels = paramNames;
                    paramsChart.data.datasets[0].data = means;
                    paramsChart.data.datasets[1].data = stds;
                }

                paramsChart.update();
            }
        }

        // Update issues
        function updateIssues(issues) {
            const container = document.getElementById('issues');

            if (issues.length === 0) {
                container.innerHTML = '<p>No issues detected</p>';
                return;
            }

            container.innerHTML = '';

            issues.forEach(issue => {
                const div = document.createElement('div');
                div.className = 'issue';
                div.textContent = issue;
                container.appendChild(div);
            });
        }

        // Send command to server
        async function sendCommand(action) {
            try {
                const response = await fetch('/control', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ action })
                });

                const data = await response.json();
                console.log('Command response:', data);
            } catch (error) {
                console.error('Error sending command:', error);
            }
        }

        // Load Chart.js from CDN
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/chart.js';
        script.onload = initDashboard;
        document.head.appendChild(script);
    </script>
</body>
</html>
        """


class Dashboard:
    """Dashboard for visualizing TTM model training in real-time."""

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 8080,
        open_browser: bool = True
    ):
        """Initialize the dashboard.

        Args:
            host: Host to run the dashboard server on
            port: Port to run the dashboard server on
            open_browser: Whether to open the browser automatically
        """
        self.host = host
        self.port = port
        self.open_browser = open_browser
        self.server = None
        self.server_thread = None
        self.running = False

        # Set up logging
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the dashboard.

        Returns:
            Logger instance
        """
        logger = logging.getLogger('ttm_dashboard')
        logger.setLevel(logging.INFO)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(console_handler)

        return logger

    def start(self):
        """Start the dashboard server."""
        if self.running:
            self.logger.warning("Dashboard already running")
            return

        # Create and start server
        self.server = HTTPServer((self.host, self.port), DashboardHandler)
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
        self.running = True

        self.logger.info(f"Dashboard server started at http://{self.host}:{self.port}")

        # Open browser
        if self.open_browser:
            webbrowser.open(f"http://{self.host}:{self.port}")

        # Start update thread
        self.update_thread = threading.Thread(target=self._process_updates)
        self.update_thread.daemon = True
        self.update_thread.start()

    def stop(self):
        """Stop the dashboard server."""
        if not self.running:
            self.logger.warning("Dashboard not running")
            return

        # Stop server
        self.server.shutdown()
        self.server.server_close()
        self.running = False

        self.logger.info("Dashboard server stopped")

    def _process_updates(self):
        """Process updates from the update queue."""
        while self.running:
            try:
                # Get update from queue (non-blocking)
                try:
                    update = update_queue.get(block=False)
                    self._apply_update(update)
                    update_queue.task_done()
                except queue.Empty:
                    pass

                # Sleep to avoid busy waiting
                time.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Error processing updates: {e}")

    def _apply_update(self, update: Dict[str, Any]):
        """Apply an update to the dashboard data.

        Args:
            update: Update data
        """
        global dashboard_data

        # Update metrics
        if 'metrics' in update:
            for key, value in update['metrics'].items():
                if key in dashboard_data['metrics']:
                    dashboard_data['metrics'][key].append(value)

        # Update examples
        if 'examples' in update:
            dashboard_data['examples'] = update['examples']

        # Update parameter stats
        if 'parameter_stats' in update:
            dashboard_data['parameter_stats'] = update['parameter_stats']

        # Update gradient stats
        if 'gradient_stats' in update:
            dashboard_data['gradient_stats'] = update['gradient_stats']

        # Update visualizations
        if 'memory_visualization' in update:
            dashboard_data['memory_visualizations'].append(update['memory_visualization'])

        if 'attention_visualization' in update:
            dashboard_data['attention_visualizations'].append(update['attention_visualization'])

        # Update training status
        if 'training_status' in update:
            dashboard_data['training_status'] = update['training_status']

        # Update issues
        if 'issues' in update:
            dashboard_data['issues'] = update['issues']

        # Update last update time
        dashboard_data['last_update'] = datetime.now().isoformat()

    def update_metrics(
        self,
        loss: float,
        position_accuracy: float,
        sequence_accuracy: float,
        learning_rate: float,
        difficulty_stage: int
    ):
        """Update training metrics.

        Args:
            loss: Loss value
            position_accuracy: Position accuracy
            sequence_accuracy: Sequence accuracy
            learning_rate: Learning rate
            difficulty_stage: Difficulty stage
        """
        update_queue.put({
            'metrics': {
                'loss': loss,
                'position_accuracy': position_accuracy,
                'sequence_accuracy': sequence_accuracy,
                'learning_rate': learning_rate,
                'difficulty_stage': difficulty_stage
            }
        })

    def update_examples(self, examples: List[Dict[str, Any]]):
        """Update example predictions.

        Args:
            examples: List of example predictions
        """
        update_queue.put({
            'examples': examples
        })

    def update_parameter_stats(self, stats: Dict[str, Dict[str, float]]):
        """Update parameter statistics.

        Args:
            stats: Parameter statistics
        """
        update_queue.put({
            'parameter_stats': stats
        })

    def update_gradient_stats(self, stats: Dict[str, float]):
        """Update gradient statistics.

        Args:
            stats: Gradient statistics
        """
        update_queue.put({
            'gradient_stats': stats
        })

    def update_memory_visualization(self, fig: plt.Figure):
        """Update memory visualization.

        Args:
            fig: Matplotlib figure
        """
        # Convert figure to base64 image
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')

        update_queue.put({
            'memory_visualization': f"data:image/png;base64,{img_str}"
        })

    def update_attention_visualization(self, fig: plt.Figure):
        """Update attention visualization.

        Args:
            fig: Matplotlib figure
        """
        # Convert figure to base64 image
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')

        update_queue.put({
            'attention_visualization': f"data:image/png;base64,{img_str}"
        })

    def update_training_status(self, status: str):
        """Update training status.

        Args:
            status: Training status ('idle', 'training', 'paused', 'stopped')
        """
        update_queue.put({
            'training_status': status
        })

    def update_issues(self, issues: List[str]):
        """Update training issues.

        Args:
            issues: List of training issues
        """
        update_queue.put({
            'issues': issues
        })

    def get_training_status(self) -> str:
        """Get the current training status.

        Returns:
            Training status
        """
        return dashboard_data['training_status']


def create_dashboard(
    host: str = 'localhost',
    port: int = 8080,
    open_browser: bool = True
) -> Dashboard:
    """Create a dashboard instance.

    Args:
        host: Host to run the dashboard server on
        port: Port to run the dashboard server on
        open_browser: Whether to open the browser automatically

    Returns:
        Dashboard instance
    """
    return Dashboard(host, port, open_browser)
