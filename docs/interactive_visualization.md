# TTM Interactive Visualization System

This document explains how to use the interactive visualization and manipulation system for the Token Turing Machine (TTM) model.

## Overview

The TTM Interactive Visualization System provides a comprehensive interface for:

1. **Exploring training history** in detail, token by token
2. **Testing inference** with custom inputs
3. **Experimenting with state modifications** to understand model behavior

## Installation

The visualization system is built into the TTM codebase. Make sure you have all the required dependencies:

```bash
pip install dash plotly networkx
```

## Usage

### Running the Dashboard

To start the interactive dashboard, run:

```bash
python interactive_visualization.py [options]
```

Available options:
- `--state_history PATH`: Path to a saved state history file
- `--model PATH`: Path to a model checkpoint
- `--port PORT`: Port to run the dashboard on (default: 8050)
- `--debug`: Run in debug mode

### Collecting State History

To collect state history during training, use the `TTMStateTracker`:

```python
from src.ttm.visualization.state_tracker import TTMStateTracker

# Initialize the state tracker with your model
tracker = TTMStateTracker(
    model=model,
    storage_dir='./visualization_data',
    sampling_rate=0.1  # Sample 10% of batches
)

# Start tracking an epoch
tracker.start_epoch(epoch_idx)

# For each batch
tracker.start_batch(batch_idx)

# For each token
tracker.start_token(token_idx)

# Train as usual...

# Save state history
tracker.save_state_history()
```

## Dashboard Features

### Training History Mode

This mode allows you to explore the model's internal states during training:

1. **Navigation Controls**:
   - Epoch slider: Select the training epoch
   - Batch slider: Select a specific batch within the epoch
   - Token slider: Select a specific token within the batch
   - Play/Pause, Step Forward, Step Backward buttons: Navigate through tokens

2. **Training Metrics**:
   - View loss and accuracy trends over epochs
   - Current epoch is highlighted

3. **State Visualization**:
   - Process Flow: Visual representation of the training loop
   - Current State Details: Summary of available data for the current state

4. **Detailed Visualizations**:
   - Memory tab: View memory content and usage patterns
   - Attention tab: Explore attention weights and focus patterns
   - Parameters tab: Examine parameter and gradient distributions

### Inference Testing Mode

This mode allows you to test the model with custom inputs:

1. **Model Selection**:
   - Choose a checkpoint to load
   - Load the model with the "Load Model" button

2. **Test Input**:
   - Enter two numbers to multiply
   - Run inference with the "Run Inference" button

3. **Results**:
   - View input, expected output, and model prediction
   - See if the prediction is correct

4. **State Visualization**:
   - Navigate through tokens in the inference process
   - View memory and attention patterns for each token

### State Experimentation Mode

This mode allows you to modify internal states and observe the effects:

1. **Source State Selection**:
   - Choose a state from training history or inference
   - Select the specific component to modify

2. **State Editor**:
   - Directly edit values in memory, attention weights, or parameters
   - Apply changes with the "Apply Changes" button

3. **Results Comparison**:
   - Compare original and modified outputs
   - Understand how modifications affect model behavior

## Understanding Visualizations

### Memory Visualization

The memory heatmap shows:
- X-axis: Embedding dimensions
- Y-axis: Memory slots
- Color intensity: Value in each memory cell

The memory timeline shows:
- X-axis: Token positions
- Y-axis: Average memory activation
- Highlighted point: Current token position

### Attention Visualization

The attention heatmap shows:
- X-axis: Key positions (token positions being attended to)
- Y-axis: Query positions (tokens doing the attending)
- Color intensity: Attention weight

The attention timeline shows:
- X-axis: Token positions
- Y-axis: Attention focus (1 - entropy)
- Higher values indicate more focused attention

### Parameter Visualization

The parameter histogram shows:
- X-axis: Parameter values
- Y-axis: Count
- Distribution shape indicates potential issues

The gradient histogram shows:
- X-axis: Gradient values
- Y-axis: Count
- Distribution shape indicates training health

## Tips for Effective Use

1. **Start with Training History**:
   - Explore how the model learns over time
   - Identify epochs where performance changes significantly

2. **Test Specific Examples**:
   - Use inference mode to test edge cases
   - Compare easy and difficult examples

3. **Experiment Methodically**:
   - Modify one component at a time
   - Keep track of how changes affect the output

4. **Look for Patterns**:
   - How does memory usage change across tokens?
   - Which tokens receive the most attention?
   - How do parameter distributions evolve during training?

## Extending the System

The visualization system is designed to be extensible:

1. **Adding New Visualizations**:
   - Create new visualization functions in `visualization_utils.py`
   - Add new tabs or sections to the dashboard layout

2. **Tracking Additional States**:
   - Modify the `TTMStateTracker` to capture additional information
   - Update the dashboard to display the new data

3. **Custom Experiments**:
   - Implement new state manipulation functions
   - Add new experiment types to the dashboard
