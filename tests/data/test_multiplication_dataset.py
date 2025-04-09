"""Tests for the multiplication dataset module.

This module contains tests for the MultiplicationDataset class.
"""

import unittest
import torch
import numpy as np
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.ttm.data.multiplication_dataset import MultiplicationDataset
from src.ttm.data.tokenization import TIMES_TOKEN, EOS_TOKEN, PAD_TOKEN


class TestMultiplicationDataset(unittest.TestCase):
    """Test cases for the MultiplicationDataset class."""

    def setUp(self):
        """Set up test fixtures."""
        self.dataset = MultiplicationDataset(batch_size=4, max_seq_len=12, seed=42)

    def test_initialization(self):
        """Test that the dataset is initialized correctly."""
        self.assertEqual(self.dataset.batch_size, 4)
        self.assertEqual(self.dataset.max_seq_len, 12)
        self.assertEqual(self.dataset.current_stage, 0)
        self.assertEqual(len(self.dataset.stages), 7)
        self.assertEqual(self.dataset.stages[0], (1, 9))  # First stage is single-digit multiplication

    def test_get_stage_range(self):
        """Test getting the current stage range."""
        self.assertEqual(self.dataset.get_stage_range(), (1, 9))

        # Test after increasing difficulty
        self.dataset.increase_difficulty()
        self.assertEqual(self.dataset.get_stage_range(), (10, 99))

    def test_increase_difficulty(self):
        """Test increasing the difficulty."""
        self.assertEqual(self.dataset.current_stage, 0)

        # Increase difficulty
        result = self.dataset.increase_difficulty()
        self.assertTrue(result)
        self.assertEqual(self.dataset.current_stage, 1)

        # Increase to max difficulty
        for _ in range(5):
            self.dataset.increase_difficulty()

        # Try to increase beyond max
        result = self.dataset.increase_difficulty()
        self.assertFalse(result)
        self.assertEqual(self.dataset.current_stage, 6)  # Should stay at max stage

    def test_generate_batch(self):
        """Test generating a batch of examples."""
        inputs, targets = self.dataset.generate_batch()

        # Check shapes
        self.assertEqual(inputs.shape, (4, 12))
        self.assertEqual(targets.shape, (4, 12))

        # Check types
        self.assertEqual(inputs.dtype, torch.long)
        self.assertEqual(targets.dtype, torch.long)


    def test_should_increase_difficulty(self):
        """Test the difficulty progression logic."""
        # Not enough history yet
        self.assertFalse(self.dataset.should_increase_difficulty(0.8))
        self.assertFalse(self.dataset.should_increase_difficulty(0.85))
        self.assertFalse(self.dataset.should_increase_difficulty(0.9))
        self.assertFalse(self.dataset.should_increase_difficulty(0.95))

        # Now we have enough history, but average is below threshold
        self.assertFalse(self.dataset.should_increase_difficulty(0.8))

        # Reset history
        self.dataset.accuracy_history = []

        # Add high accuracy values
        self.assertFalse(self.dataset.should_increase_difficulty(0.95))
        self.assertFalse(self.dataset.should_increase_difficulty(0.95))
        self.assertFalse(self.dataset.should_increase_difficulty(0.95))
        self.assertFalse(self.dataset.should_increase_difficulty(0.95))

        # This should trigger progression
        self.assertTrue(self.dataset.should_increase_difficulty(0.95))

    def test_curriculum(self):
        """Test curriculum progression."""
        # Test stage 0 (single-digit multiplication)
        inputs, targets = self.dataset.generate_batch()

        # Extract numbers from inputs
        for i in range(2):  # Check a couple of examples
            # Find multiplication symbol
            times_pos = -1
            for j in range(self.dataset.max_seq_len):
                if inputs[i, j].item() == TIMES_TOKEN:
                    times_pos = j
                    break

            self.assertGreater(times_pos, 0, "Multiplication symbol not found")

            # Extract first number
            num1_tokens = inputs[i, :times_pos].tolist()
            num1 = int(''.join([str(d) for d in num1_tokens if d < 10]))

            # Find EOS token
            eos_pos = -1
            for j in range(times_pos + 1, self.dataset.max_seq_len):
                if inputs[i, j].item() == EOS_TOKEN:
                    eos_pos = j
                    break

            self.assertGreater(eos_pos, times_pos, "EOS token not found after multiplication symbol")

            # Extract second number
            num2_tokens = inputs[i, times_pos+1:eos_pos].tolist()
            num2 = int(''.join([str(d) for d in num2_tokens if d < 10]))

            # Check that numbers are within range for stage 0
            self.assertGreaterEqual(num1, 1)
            self.assertLessEqual(num1, 9)
            self.assertGreaterEqual(num2, 1)
            self.assertLessEqual(num2, 9)


    def test_data_augmentation(self):
        """Test data augmentation."""
        # Generate batch
        inputs, targets = self.dataset.generate_batch()

        # Apply augmentation
        aug_inputs, aug_targets = self.dataset.augment_batch(inputs, targets)

        self.assertEqual(aug_inputs.shape, inputs.shape)
        self.assertEqual(aug_targets.shape, targets.shape)

        # Verify that augmented examples are still valid
        for i in range(2):  # Check a couple of examples
            # Find multiplication symbol
            times_pos = -1
            for j in range(self.dataset.max_seq_len):
                if aug_inputs[i, j].item() == TIMES_TOKEN:
                    times_pos = j
                    break

            if times_pos == -1:
                continue  # Skip if no multiplication symbol found

            # Extract first number
            num1_tokens = aug_inputs[i, :times_pos].tolist()
            num1 = int(''.join([str(d) for d in num1_tokens if d < 10]))

            # Find EOS token
            eos_pos = -1
            for j in range(times_pos + 1, self.dataset.max_seq_len):
                if aug_inputs[i, j].item() == EOS_TOKEN:
                    eos_pos = j
                    break

            if eos_pos == -1:
                continue  # Skip if no EOS token found

            # Extract second number
            num2_tokens = aug_inputs[i, times_pos+1:eos_pos].tolist()
            num2 = int(''.join([str(d) for d in num2_tokens if d < 10]))

            # Extract result
            result_eos_pos = -1
            for j in range(self.dataset.max_seq_len):
                if aug_targets[i, j].item() == EOS_TOKEN:
                    result_eos_pos = j
                    break

            if result_eos_pos == -1:
                continue  # Skip if no EOS token found in target

            result_tokens = aug_targets[i, :result_eos_pos].tolist()
            result = int(''.join([str(d) for d in result_tokens if d < 10]))

            # Check that result is correct
            self.assertEqual(result, num1 * num2)

    def test_get_example_str(self):
        """Test getting a string representation of an example."""
        # Create a simple example
        input_tokens = [1, 2, TIMES_TOKEN, 3, EOS_TOKEN, PAD_TOKEN, PAD_TOKEN]
        target_tokens = [3, 6, EOS_TOKEN, PAD_TOKEN, PAD_TOKEN, PAD_TOKEN, PAD_TOKEN]

        example_str = self.dataset.get_example_str(input_tokens, target_tokens)
        self.assertEqual(example_str, "12×3 = 36")

    def test_infinite_length(self):
        """Test that the dataset has infinite length."""
        # We can't directly compare with float('inf') in assertEqual
        # because 'float' object cannot be interpreted as an integer
        self.assertTrue(len(self.dataset) > 1000000)  # Just check it's very large


if __name__ == '__main__':
    unittest.main()
