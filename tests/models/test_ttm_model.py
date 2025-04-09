"""
Tests for the Token Turing Machine (TTM) model.
"""

import unittest
import torch
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.ttm.models.ttm_model import (
    TokenEmbedding,
    OutputHead,
    TokenTuringMachine
)
from src.ttm.utils.masking import EOSCrossEntropyLoss
from src.ttm.utils.losses import TTMLoss, LabelSmoothingLoss, FocalLoss


class TestTokenEmbedding(unittest.TestCase):
    """Test cases for the TokenEmbedding class."""

    def setUp(self):
        """Set up test fixtures."""
        self.vocab_size = 1000
        self.embedding_dim = 64
        self.max_seq_len = 20
        self.batch_size = 2
        self.seq_len = 10
        self.embedding = TokenEmbedding(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            max_seq_len=self.max_seq_len
        )

        # Create random token indices
        self.tokens = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))

    def test_initialization(self):
        """Test that the embedding module is initialized correctly."""
        self.assertEqual(self.embedding.vocab_size, self.vocab_size)
        self.assertEqual(self.embedding.embedding_dim, self.embedding_dim)
        self.assertEqual(self.embedding.max_seq_len, self.max_seq_len)

    def test_forward(self):
        """Test forward pass."""
        embeddings = self.embedding(self.tokens)

        self.assertEqual(embeddings.shape, (self.batch_size, self.seq_len, self.embedding_dim))


class TestOutputHead(unittest.TestCase):
    """Test cases for the OutputHead class."""

    def setUp(self):
        """Set up test fixtures."""
        self.embedding_dim = 64
        self.vocab_size = 1000
        self.batch_size = 2
        self.seq_len = 10
        self.output_head = OutputHead(
            embedding_dim=self.embedding_dim,
            vocab_size=self.vocab_size
        )

        # Create random embeddings
        self.embeddings = torch.randn(self.batch_size, self.seq_len, self.embedding_dim)

    def test_initialization(self):
        """Test that the output head is initialized correctly."""
        self.assertEqual(self.output_head.embedding_dim, self.embedding_dim)
        self.assertEqual(self.output_head.vocab_size, self.vocab_size)

    def test_forward(self):
        """Test forward pass."""
        logits = self.output_head(self.embeddings)

        self.assertEqual(logits.shape, (self.batch_size, self.seq_len, self.vocab_size))


class TestTokenTuringMachine(unittest.TestCase):
    """Test cases for the TokenTuringMachine class."""

    def setUp(self):
        """Set up test fixtures."""
        self.vocab_size = 1000
        self.embedding_dim = 64
        self.memory_size = 8
        self.r = 4
        self.num_layers = 2
        self.num_heads = 4
        self.hidden_dim = 256
        self.batch_size = 2
        self.seq_len = 10

        # Create TTM model
        self.ttm = TokenTuringMachine(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            memory_size=self.memory_size,
            r=self.r,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
            padding_token=0,
            eos_token=1
        )

        # Create memory-less TTM model
        self.memory_less_ttm = TokenTuringMachine(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            memory_size=self.memory_size,
            r=self.r,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
            memory_less=True,
            padding_token=0,
            eos_token=1
        )

        # Create random token indices
        self.tokens = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))

    def test_initialization(self):
        """Test that the TTM model is initialized correctly."""
        self.assertEqual(self.ttm.vocab_size, self.vocab_size)
        self.assertEqual(self.ttm.embedding_dim, self.embedding_dim)
        self.assertEqual(self.ttm.memory_size, self.memory_size)
        self.assertEqual(self.ttm.r, self.r)
        self.assertFalse(self.ttm.memory_less)
        self.assertTrue(self.memory_less_ttm.memory_less)
        self.assertEqual(self.ttm.padding_token, 0)
        self.assertEqual(self.ttm.eos_token, 1)
        self.assertTrue(self.ttm.causal_attention)

    def test_initialize_memory(self):
        """Test initializing memory."""
        memory = self.ttm.initialize_memory(self.batch_size)

        self.assertEqual(memory.shape, (self.batch_size, self.memory_size, self.embedding_dim))

        # Check that memory is initialized with zeros
        zeros = torch.zeros_like(memory)
        self.assertTrue(torch.allclose(memory, zeros, atol=1e-6))

        # Check that memory-less model returns None
        memory_less = self.memory_less_ttm.initialize_memory(self.batch_size)
        self.assertIsNone(memory_less)

    def test_forward(self):
        """Test forward pass."""
        # Test with memory
        logits, new_memory = self.ttm(self.tokens)

        self.assertEqual(logits.shape, (self.batch_size, self.seq_len, self.vocab_size))
        self.assertEqual(new_memory.shape, (self.batch_size, self.memory_size, self.embedding_dim))

        # Test with memory-less model
        logits_memory_less, new_memory_memory_less = self.memory_less_ttm(self.tokens)

        self.assertEqual(logits_memory_less.shape, (self.batch_size, self.seq_len, self.vocab_size))
        self.assertIsNone(new_memory_memory_less)

    def test_forward_with_provided_memory(self):
        """Test forward pass with provided memory."""
        memory = torch.randn(self.batch_size, self.memory_size, self.embedding_dim)

        logits, new_memory = self.ttm(self.tokens, memory)

        self.assertEqual(logits.shape, (self.batch_size, self.seq_len, self.vocab_size))
        self.assertEqual(new_memory.shape, (self.batch_size, self.memory_size, self.embedding_dim))

    def test_generate(self):
        """Test token generation."""
        # Test with memory
        generated_tokens = self.ttm.generate(
            tokens=self.tokens,
            max_length=15,
            temperature=1.0
        )

        self.assertEqual(generated_tokens.shape, (self.batch_size, 15))

        # Check that the first seq_len tokens are the same as the input
        self.assertTrue(torch.allclose(generated_tokens[:, :self.seq_len], self.tokens, atol=1e-6))

        # Test with memory-less model
        generated_tokens_memory_less = self.memory_less_ttm.generate(
            tokens=self.tokens,
            max_length=15,
            temperature=1.0
        )

        self.assertEqual(generated_tokens_memory_less.shape, (self.batch_size, 15))

        # Check that the first seq_len tokens are the same as the input
        self.assertTrue(torch.allclose(generated_tokens_memory_less[:, :self.seq_len], self.tokens, atol=1e-6))

    def test_generate_with_top_k(self):
        """Test token generation with top-k sampling."""
        generated_tokens = self.ttm.generate(
            tokens=self.tokens,
            max_length=15,
            temperature=1.0,
            top_k=5
        )

        self.assertEqual(generated_tokens.shape, (self.batch_size, 15))

    def test_generate_with_top_p(self):
        """Test token generation with top-p sampling."""
        generated_tokens = self.ttm.generate(
            tokens=self.tokens,
            max_length=15,
            temperature=1.0,
            top_p=0.9
        )

        self.assertEqual(generated_tokens.shape, (self.batch_size, 15))

    def test_generate_with_eos(self):
        """Test token generation with EOS token."""
        eos_token = 42

        # Set one of the generated tokens to be the EOS token
        with unittest.mock.patch('torch.multinomial', return_value=torch.tensor([[eos_token], [eos_token]])):
            generated_tokens = self.ttm.generate(
                tokens=self.tokens,
                max_length=15,
                temperature=1.0,
                eos_token=eos_token
            )

        self.assertLessEqual(generated_tokens.shape[1], 15)
        self.assertTrue((generated_tokens == eos_token).any())

    def test_forward_with_eos_masking(self):
        """Test forward pass with EOS masking."""
        # Create tokens with EOS
        tokens_with_eos = torch.randint(2, self.vocab_size, (self.batch_size, self.seq_len))
        tokens_with_eos[0, 3] = 1  # EOS token at position 3 in first sequence
        tokens_with_eos[1, 5] = 1  # EOS token at position 5 in second sequence

        # Forward pass with EOS masking
        logits, new_memory = self.ttm(tokens_with_eos, mask_eos=True)

        self.assertEqual(logits.shape, (self.batch_size, self.seq_len, self.vocab_size))
        self.assertEqual(new_memory.shape, (self.batch_size, self.memory_size, self.embedding_dim))

    def test_create_loss_fn(self):
        """Test creating a loss function."""
        # Create cross-entropy loss function
        loss_fn = self.ttm.create_loss_fn(loss_type='cross_entropy')

        # Check that it's an instance of EOSCrossEntropyLoss
        self.assertIsInstance(loss_fn, EOSCrossEntropyLoss)

        # Create loss function for memory-less model
        memory_less_loss_fn = self.memory_less_ttm.create_loss_fn(loss_type='cross_entropy')

        # Check that it's an instance of EOSCrossEntropyLoss
        self.assertIsInstance(memory_less_loss_fn, EOSCrossEntropyLoss)

    def test_create_ttm_loss_fn(self):
        """Test creating a TTM loss function."""
        # Create TTM loss function
        loss_fn = self.ttm.create_loss_fn(
            loss_type='ttm',
            memory_loss_weight=0.1,
            attention_loss_weight=0.1
        )

        # Check that it's an instance of TTMLoss
        self.assertIsInstance(loss_fn, TTMLoss)

    def test_create_label_smoothing_loss_fn(self):
        """Test creating a label smoothing loss function."""
        # Create label smoothing loss function
        loss_fn = self.ttm.create_loss_fn(
            loss_type='label_smoothing',
            label_smoothing=0.1
        )

        # Check that it's an instance of LabelSmoothingLoss
        self.assertIsInstance(loss_fn, LabelSmoothingLoss)

    def test_create_focal_loss_fn(self):
        """Test creating a focal loss function."""
        # Create focal loss function
        loss_fn = self.ttm.create_loss_fn(
            loss_type='focal',
            focal_alpha=0.25,
            focal_gamma=2.0
        )

        # Check that it's an instance of FocalLoss
        self.assertIsInstance(loss_fn, FocalLoss)


if __name__ == '__main__':
    unittest.main()
