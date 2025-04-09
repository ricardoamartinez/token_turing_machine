"""
Tests for the transformer processing unit module.
"""

import unittest
import torch
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.ttm.models.transformer_unit import (
    MultiHeadAttention,
    FeedForward,
    TransformerLayer,
    TransformerEncoder,
    TransformerProcessingUnit
)


class TestMultiHeadAttention(unittest.TestCase):
    """Test cases for the MultiHeadAttention class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dim = 64
        self.num_heads = 4
        self.batch_size = 2
        self.seq_len = 10
        self.attn = MultiHeadAttention(
            dim=self.dim,
            num_heads=self.num_heads
        )
        
        # Create random input tensors
        self.q = torch.randn(self.batch_size, self.seq_len, self.dim)
        self.k = torch.randn(self.batch_size, self.seq_len, self.dim)
        self.v = torch.randn(self.batch_size, self.seq_len, self.dim)
    
    def test_initialization(self):
        """Test that the attention module is initialized correctly."""
        self.assertEqual(self.attn.dim, self.dim)
        self.assertEqual(self.attn.num_heads, self.num_heads)
        self.assertEqual(self.attn.head_dim, self.dim // self.num_heads)
        self.assertEqual(self.attn.scale, (self.dim // self.num_heads) ** -0.5)
    
    def test_forward(self):
        """Test forward pass."""
        output = self.attn(self.q, self.k, self.v)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.dim))
    
    def test_forward_with_mask(self):
        """Test forward pass with attention mask."""
        # Create a causal mask (lower triangular)
        mask = torch.tril(torch.ones(self.seq_len, self.seq_len))
        
        output = self.attn(self.q, self.k, self.v, attn_mask=mask)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.dim))
    
    def test_forward_with_padding_mask(self):
        """Test forward pass with key padding mask."""
        # Create a padding mask (1 = masked, 0 = not masked)
        padding_mask = torch.zeros(self.batch_size, self.seq_len, dtype=torch.bool)
        padding_mask[:, -2:] = True  # Mask the last two tokens
        
        output = self.attn(self.q, self.k, self.v, key_padding_mask=padding_mask)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.dim))


class TestFeedForward(unittest.TestCase):
    """Test cases for the FeedForward class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dim = 64
        self.hidden_dim = 256
        self.batch_size = 2
        self.seq_len = 10
        self.ff = FeedForward(
            dim=self.dim,
            hidden_dim=self.hidden_dim
        )
        
        # Create random input tensor
        self.x = torch.randn(self.batch_size, self.seq_len, self.dim)
    
    def test_initialization(self):
        """Test that the feed-forward module is initialized correctly."""
        self.assertEqual(self.ff.dim, self.dim)
        self.assertEqual(self.ff.hidden_dim, self.hidden_dim)
    
    def test_forward(self):
        """Test forward pass."""
        output = self.ff(self.x)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.dim))


class TestTransformerLayer(unittest.TestCase):
    """Test cases for the TransformerLayer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dim = 64
        self.num_heads = 4
        self.hidden_dim = 256
        self.batch_size = 2
        self.seq_len = 10
        self.layer = TransformerLayer(
            dim=self.dim,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim
        )
        
        # Create random input tensor
        self.x = torch.randn(self.batch_size, self.seq_len, self.dim)
    
    def test_initialization(self):
        """Test that the transformer layer is initialized correctly."""
        self.assertEqual(self.layer.dim, self.dim)
        self.assertTrue(self.layer.norm_first)  # Default value
    
    def test_forward(self):
        """Test forward pass."""
        output = self.layer(self.x)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.dim))
    
    def test_forward_with_mask(self):
        """Test forward pass with attention mask."""
        # Create a causal mask (lower triangular)
        mask = torch.tril(torch.ones(self.seq_len, self.seq_len))
        
        output = self.layer(self.x, attn_mask=mask)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.dim))
    
    def test_forward_with_padding_mask(self):
        """Test forward pass with key padding mask."""
        # Create a padding mask (1 = masked, 0 = not masked)
        padding_mask = torch.zeros(self.batch_size, self.seq_len, dtype=torch.bool)
        padding_mask[:, -2:] = True  # Mask the last two tokens
        
        output = self.layer(self.x, key_padding_mask=padding_mask)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.dim))


class TestTransformerEncoder(unittest.TestCase):
    """Test cases for the TransformerEncoder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dim = 64
        self.num_layers = 2
        self.num_heads = 4
        self.hidden_dim = 256
        self.batch_size = 2
        self.seq_len = 10
        self.encoder = TransformerEncoder(
            dim=self.dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim
        )
        
        # Create random input tensor
        self.x = torch.randn(self.batch_size, self.seq_len, self.dim)
    
    def test_initialization(self):
        """Test that the transformer encoder is initialized correctly."""
        self.assertEqual(self.encoder.dim, self.dim)
        self.assertEqual(self.encoder.num_layers, self.num_layers)
        self.assertEqual(len(self.encoder.layers), self.num_layers)
    
    def test_forward(self):
        """Test forward pass."""
        output = self.encoder(self.x)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.dim))


class TestTransformerProcessingUnit(unittest.TestCase):
    """Test cases for the TransformerProcessingUnit class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dim = 64
        self.num_layers = 2
        self.num_heads = 4
        self.hidden_dim = 256
        self.batch_size = 2
        self.seq_len = 10
        self.processor = TransformerProcessingUnit(
            dim=self.dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim
        )
        
        # Create random input tensor
        self.x = torch.randn(self.batch_size, self.seq_len, self.dim)
    
    def test_initialization(self):
        """Test that the transformer processing unit is initialized correctly."""
        self.assertEqual(self.processor.dim, self.dim)
        self.assertEqual(self.processor.transformer.num_layers, self.num_layers)
    
    def test_forward(self):
        """Test forward pass."""
        output = self.processor(self.x)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.dim))


if __name__ == '__main__':
    unittest.main()
