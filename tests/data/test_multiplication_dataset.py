"""Tests for multiplication dataset.

This module contains tests for the multiplication dataset implementation.
"""

import jax
import jax.numpy as jnp
import pytest
import numpy as np

from src.ttm.data.multiplication_dataset import MultiplicationDataset


def test_dataset_initialization():
    """Test dataset initialization."""
    dataset = MultiplicationDataset(batch_size=32, seq_len=12)
    
    assert dataset.batch_size == 32
    assert dataset.seq_len == 12
    assert dataset.vocab_size == 13
    assert dataset.current_stage == 0


def test_generate_batch():
    """Test batch generation."""
    dataset = MultiplicationDataset(batch_size=32, seq_len=12)
    
    inputs, targets = dataset.generate_batch()
    
    assert inputs.shape == (32, 12)
    assert targets.shape == (32, 12)
    
    # Check that inputs contain multiplication symbol
    assert np.any(inputs == dataset.TIMES_TOKEN)
    
    # Check that both inputs and targets contain EOS token
    assert np.any(inputs == dataset.EOS_TOKEN)
    assert np.any(targets == dataset.EOS_TOKEN)
    
    # Check that both inputs and targets contain padding
    assert np.any(inputs == dataset.PAD_TOKEN)
    assert np.any(targets == dataset.PAD_TOKEN)


def test_increase_difficulty():
    """Test difficulty progression."""
    dataset = MultiplicationDataset(batch_size=32, seq_len=12)
    
    assert dataset.current_stage == 0
    
    # Progress through all stages
    for i in range(len(dataset.stages) - 1):
        assert dataset.increase_difficulty() == True
        assert dataset.current_stage == i + 1
    
    # Try to progress beyond max stage
    assert dataset.increase_difficulty() == False
    assert dataset.current_stage == len(dataset.stages) - 1


def test_curriculum():
    """Test curriculum progression."""
    dataset = MultiplicationDataset(batch_size=32, seq_len=12)
    
    # Test stage 0
    inputs, targets = dataset.generate_batch()
    
    # Extract numbers from inputs
    for i in range(5):  # Check a few examples
        # Find multiplication symbol
        times_pos = np.where(inputs[i] == dataset.TIMES_TOKEN)[0][0]
        
        # Extract first number
        num1 = int(''.join([str(d) for d in inputs[i, :times_pos] if d < 10]))
        
        # Find EOS token
        eos_pos = np.where(inputs[i] == dataset.EOS_TOKEN)[0][0]
        
        # Extract second number
        num2 = int(''.join([str(d) for d in inputs[i, times_pos+1:eos_pos] if d < 10]))
        
        # Check that numbers are within range for stage 0
        assert 1 <= num1 <= 2
        assert 1 <= num2 <= 2
    
    # Progress to stage 3
    dataset.current_stage = 3
    
    # Test stage 3
    inputs, targets = dataset.generate_batch()
    
    # Extract numbers from inputs
    for i in range(5):  # Check a few examples
        # Find multiplication symbol
        times_pos = np.where(inputs[i] == dataset.TIMES_TOKEN)[0][0]
        
        # Extract first number
        num1 = int(''.join([str(d) for d in inputs[i, :times_pos] if d < 10]))
        
        # Find EOS token
        eos_pos = np.where(inputs[i] == dataset.EOS_TOKEN)[0][0]
        
        # Extract second number
        num2 = int(''.join([str(d) for d in inputs[i, times_pos+1:eos_pos] if d < 10]))
        
        # Check that numbers are within range for stage 3
        assert 1 <= num1 <= 9
        assert 1 <= num2 <= 9


def test_data_augmentation():
    """Test data augmentation."""
    dataset = MultiplicationDataset(batch_size=32, seq_len=12)
    
    # Generate batch
    inputs, targets = dataset.generate_batch()
    
    # Apply augmentation
    aug_inputs, aug_targets = dataset.augment_batch(inputs, targets)
    
    assert aug_inputs.shape == inputs.shape
    assert aug_targets.shape == targets.shape
    
    # Check that augmentation changed the data
    assert not np.array_equal(aug_inputs, inputs) or not np.array_equal(aug_targets, targets)
    
    # Verify that augmented examples are still valid
    for i in range(5):  # Check a few examples
        # Find multiplication symbol
        times_pos = np.where(aug_inputs[i] == dataset.TIMES_TOKEN)[0][0]
        
        # Extract first number
        num1 = int(''.join([str(d) for d in aug_inputs[i, :times_pos] if d < 10]))
        
        # Find EOS token
        eos_pos = np.where(aug_inputs[i] == dataset.EOS_TOKEN)[0][0]
        
        # Extract second number
        num2 = int(''.join([str(d) for d in aug_inputs[i, times_pos+1:eos_pos] if d < 10]))
        
        # Extract result
        result_eos_pos = np.where(aug_targets[i] == dataset.EOS_TOKEN)[0][0]
        result = int(''.join([str(d) for d in aug_targets[i, :result_eos_pos] if d < 10]))
        
        # Check that result is correct
        assert result == num1 * num2
