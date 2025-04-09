"""
Tests for the training step functions.
"""

import unittest
import torch
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.ttm.models.ttm_model import TokenTuringMachine
from src.ttm.training.train_step import train_step, eval_step, get_example_predictions


class TestTrainStep(unittest.TestCase):
    """Test cases for the train_step function."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.seq_len = 4
        self.vocab_size = 10
        self.embedding_dim = 16
        self.memory_size = 8
        self.r = 4
        self.num_layers = 2
        self.num_heads = 4
        self.hidden_dim = 32

        # Create model
        self.model = TokenTuringMachine(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            memory_size=self.memory_size,
            r=self.r,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim
        )

        # Create optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        # Create loss function
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

        # Create batch with requires_grad=True
        self.inputs = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        self.targets = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        # Reshape logits to match expected shape for CrossEntropyLoss
        self.batch = {'input_ids': self.inputs, 'labels': self.targets.clone()}

        # Make sure model parameters require gradients
        for param in self.model.parameters():
            param.requires_grad = True

    def test_train_step(self):
        """Test the train_step function."""
        # Zero gradients
        self.optimizer.zero_grad()

        # Perform training step
        loss, memory, metrics = train_step(
            model=self.model,
            batch=self.batch,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            clip_grad_norm=1.0
        )

        # Check that loss is a scalar
        self.assertTrue(loss.dim() == 0)

        # Check that memory is returned
        self.assertIsNotNone(memory)
        self.assertEqual(memory.shape, (self.batch_size, self.memory_size, self.embedding_dim))

        # Check that metrics are returned
        self.assertIsInstance(metrics, dict)
        self.assertIn('lr', metrics)

    def test_train_step_with_accumulation(self):
        """Test the train_step function with gradient accumulation."""
        # Zero gradients
        self.optimizer.zero_grad()

        # Perform training step with accumulation
        loss1, memory1, _ = train_step(
            model=self.model,
            batch=self.batch,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            clip_grad_norm=1.0,
            accumulation_steps=2,
            current_step=0
        )

        # Perform second training step with accumulation
        loss2, memory2, _ = train_step(
            model=self.model,
            batch=self.batch,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            clip_grad_norm=1.0,
            accumulation_steps=2,
            current_step=1,
            memory=memory1
        )

        # Check that memory is updated
        self.assertFalse(torch.allclose(memory1, memory2))

    def test_train_step_with_clipping(self):
        """Test the train_step function with gradient clipping."""
        # Zero gradients
        self.optimizer.zero_grad()

        # Perform training step with clipping
        loss, memory, metrics = train_step(
            model=self.model,
            batch=self.batch,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            clip_grad_norm=1.0
        )

        # Check that loss and memory are returned
        self.assertIsNotNone(loss)
        self.assertIsNotNone(memory)


class TestEvalStep(unittest.TestCase):
    """Test cases for the eval_step function."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.seq_len = 4
        self.vocab_size = 10
        self.embedding_dim = 16
        self.memory_size = 8
        self.r = 4
        self.num_layers = 2
        self.num_heads = 4
        self.hidden_dim = 32

        # Create model
        self.model = TokenTuringMachine(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            memory_size=self.memory_size,
            r=self.r,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim
        )

        # Create loss function
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

        # Create batch
        self.inputs = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        self.targets = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        # Reshape logits to match expected shape for CrossEntropyLoss
        self.batch = {'input_ids': self.inputs, 'labels': self.targets.clone()}

    def test_eval_step(self):
        """Test the eval_step function."""
        # Set model to evaluation mode
        self.model.eval()

        # Perform evaluation step
        loss, memory, metrics = eval_step(
            model=self.model,
            batch=self.batch,
            loss_fn=self.loss_fn
        )

        # Check that loss is a scalar
        self.assertTrue(loss.dim() == 0)

        # Check that memory is returned
        self.assertIsNotNone(memory)
        self.assertEqual(memory.shape, (self.batch_size, self.memory_size, self.embedding_dim))

        # Check that metrics are returned
        self.assertIsInstance(metrics, dict)
        self.assertIn('position_accuracy', metrics)
        self.assertIn('sequence_accuracy', metrics)

    def test_eval_step_with_eos(self):
        """Test the eval_step function with EOS token."""
        # Set model to evaluation mode
        self.model.eval()

        # Create batch with EOS token
        eos_token_id = 2
        inputs = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        targets = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        targets[0, 2] = eos_token_id  # Add EOS token
        batch = {'input_ids': inputs, 'labels': targets}

        # Perform evaluation step
        loss, memory, metrics = eval_step(
            model=self.model,
            batch=batch,
            loss_fn=self.loss_fn,
            eos_token_id=eos_token_id
        )

        # Check that metrics are returned
        self.assertIsInstance(metrics, dict)
        self.assertIn('position_accuracy', metrics)
        self.assertIn('sequence_accuracy', metrics)


class TestGetExamplePredictions(unittest.TestCase):
    """Test cases for the get_example_predictions function."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.seq_len = 4
        self.vocab_size = 10
        self.embedding_dim = 16
        self.memory_size = 8
        self.r = 4
        self.num_layers = 2
        self.num_heads = 4
        self.hidden_dim = 32

        # Create model
        self.model = TokenTuringMachine(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            memory_size=self.memory_size,
            r=self.r,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim
        )

        # Create batch
        self.inputs = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        self.targets = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        # Reshape logits to match expected shape for CrossEntropyLoss
        self.batch = {'input_ids': self.inputs, 'labels': self.targets.clone()}

    def test_get_example_predictions(self):
        """Test the get_example_predictions function."""
        # Set model to evaluation mode
        self.model.eval()

        # Get example predictions
        examples = get_example_predictions(
            model=self.model,
            batch=self.batch,
            num_examples=1
        )

        # Check that examples are returned
        self.assertIsInstance(examples, list)
        self.assertEqual(len(examples), 1)

        # Check that example contains expected fields
        example = examples[0]
        self.assertIn('input_tokens', example)
        self.assertIn('target_tokens', example)
        self.assertIn('prediction_tokens', example)

    def test_get_example_predictions_with_tokenizer(self):
        """Test the get_example_predictions function with a tokenizer."""
        # Set model to evaluation mode
        self.model.eval()

        # Create simple tokenizer
        class SimpleTokenizer:
            def encode(self, text):
                return [ord(c) for c in text]

            def decode(self, tokens):
                return ''.join([chr(t) for t in tokens])

        tokenizer = SimpleTokenizer()

        # Get example predictions
        examples = get_example_predictions(
            model=self.model,
            batch=self.batch,
            tokenizer=tokenizer,
            num_examples=1
        )

        # Check that examples are returned
        self.assertIsInstance(examples, list)
        self.assertEqual(len(examples), 1)

        # Check that example contains expected fields
        example = examples[0]
        self.assertIn('input_tokens', example)
        self.assertIn('target_tokens', example)
        self.assertIn('prediction_tokens', example)
        self.assertIn('input_text', example)
        self.assertIn('target_text', example)
        self.assertIn('prediction_text', example)


if __name__ == '__main__':
    unittest.main()
