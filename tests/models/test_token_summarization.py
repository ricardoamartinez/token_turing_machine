"""
Tests for the token summarization module.
"""

import unittest
import torch
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.ttm.models.token_summarization import (
    MLPSummarizer,
    QuerySummarizer,
    PoolingSummarizer,
    token_summarize
)


class TestMLPSummarizer(unittest.TestCase):
    """Test cases for the MLPSummarizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.embedding_dim = 64
        self.batch_size = 2
        self.num_tokens = 10
        self.summarizer = MLPSummarizer(embedding_dim=self.embedding_dim)
        
        # Create random token embeddings
        self.tokens = torch.randn(self.batch_size, self.num_tokens, self.embedding_dim)
    
    def test_initialization(self):
        """Test that the summarizer is initialized correctly."""
        self.assertEqual(self.summarizer.embedding_dim, self.embedding_dim)
        self.assertEqual(self.summarizer.hidden_dim, 128)  # Default value
        self.assertEqual(self.summarizer.temperature, 1.0)  # Default value
    
    def test_compute_importance_weights(self):
        """Test computing importance weights."""
        weights = self.summarizer.compute_importance_weights(self.tokens)
        
        self.assertEqual(weights.shape, (self.batch_size, self.num_tokens, 1))
    
    def test_normalize_weights(self):
        """Test normalizing weights."""
        weights = torch.randn(self.batch_size, self.num_tokens, 1)
        norm_weights = self.summarizer.normalize_weights(weights)
        
        self.assertEqual(norm_weights.shape, (self.batch_size, self.num_tokens, 1))
        
        # Check that weights sum to 1 for each batch
        sums = norm_weights.sum(dim=1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-6))
    
    def test_weighted_sum(self):
        """Test computing weighted sum."""
        weights = torch.ones(self.batch_size, self.num_tokens, 1) / self.num_tokens
        summed = self.summarizer.weighted_sum(self.tokens, weights)
        
        self.assertEqual(summed.shape, (self.batch_size, 1, self.embedding_dim))
        
        # Check that summed is the average of tokens
        expected = self.tokens.mean(dim=1, keepdim=True)
        self.assertTrue(torch.allclose(summed, expected, atol=1e-6))
    
    def test_forward_k1(self):
        """Test forward pass with k=1."""
        summary = self.summarizer(self.tokens, k=1)
        
        self.assertEqual(summary.shape, (self.batch_size, 1, self.embedding_dim))
    
    def test_forward_k3(self):
        """Test forward pass with k=3."""
        summary = self.summarizer(self.tokens, k=3)
        
        self.assertEqual(summary.shape, (self.batch_size, 3, self.embedding_dim))
    
    def test_forward_k_greater_than_num_tokens(self):
        """Test forward pass with k > num_tokens."""
        summary = self.summarizer(self.tokens, k=15)
        
        self.assertEqual(summary.shape, (self.batch_size, 15, self.embedding_dim))
        
        # Check that the first num_tokens are the original tokens
        self.assertTrue(torch.allclose(summary[:, :self.num_tokens, :], self.tokens, atol=1e-6))
        
        # Check that the rest are zeros
        zeros = torch.zeros(self.batch_size, 15 - self.num_tokens, self.embedding_dim)
        self.assertTrue(torch.allclose(summary[:, self.num_tokens:, :], zeros, atol=1e-6))


class TestQuerySummarizer(unittest.TestCase):
    """Test cases for the QuerySummarizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.embedding_dim = 64
        self.batch_size = 2
        self.num_tokens = 10
        self.summarizer = QuerySummarizer(embedding_dim=self.embedding_dim)
        
        # Create random token embeddings
        self.tokens = torch.randn(self.batch_size, self.num_tokens, self.embedding_dim)
    
    def test_initialization(self):
        """Test that the summarizer is initialized correctly."""
        self.assertEqual(self.summarizer.embedding_dim, self.embedding_dim)
        self.assertEqual(self.summarizer.num_heads, 4)  # Default value
        self.assertEqual(self.summarizer.head_dim, self.embedding_dim // 4)
    
    def test_forward_k1(self):
        """Test forward pass with k=1."""
        summary = self.summarizer(self.tokens, k=1)
        
        self.assertEqual(summary.shape, (self.batch_size, 1, self.embedding_dim))
    
    def test_forward_k3(self):
        """Test forward pass with k=3."""
        summary = self.summarizer(self.tokens, k=3)
        
        self.assertEqual(summary.shape, (self.batch_size, 3, self.embedding_dim))
    
    def test_forward_k_greater_than_num_tokens(self):
        """Test forward pass with k > num_tokens."""
        summary = self.summarizer(self.tokens, k=15)
        
        self.assertEqual(summary.shape, (self.batch_size, 15, self.embedding_dim))
        
        # Check that the first num_tokens are the original tokens
        self.assertTrue(torch.allclose(summary[:, :self.num_tokens, :], self.tokens, atol=1e-6))
        
        # Check that the rest are zeros
        zeros = torch.zeros(self.batch_size, 15 - self.num_tokens, self.embedding_dim)
        self.assertTrue(torch.allclose(summary[:, self.num_tokens:, :], zeros, atol=1e-6))


class TestPoolingSummarizer(unittest.TestCase):
    """Test cases for the PoolingSummarizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.embedding_dim = 64
        self.batch_size = 2
        self.num_tokens = 10
        self.summarizer = PoolingSummarizer(embedding_dim=self.embedding_dim)
        
        # Create random token embeddings
        self.tokens = torch.randn(self.batch_size, self.num_tokens, self.embedding_dim)
    
    def test_initialization(self):
        """Test that the summarizer is initialized correctly."""
        self.assertEqual(self.summarizer.embedding_dim, self.embedding_dim)
        self.assertEqual(self.summarizer.pooling_type, 'avg')  # Default value
    
    def test_forward_k1(self):
        """Test forward pass with k=1."""
        summary = self.summarizer(self.tokens, k=1)
        
        self.assertEqual(summary.shape, (self.batch_size, 1, self.embedding_dim))
    
    def test_forward_k3(self):
        """Test forward pass with k=3."""
        summary = self.summarizer(self.tokens, k=3)
        
        self.assertEqual(summary.shape, (self.batch_size, 3, self.embedding_dim))
    
    def test_forward_k_greater_than_num_tokens(self):
        """Test forward pass with k > num_tokens."""
        summary = self.summarizer(self.tokens, k=15)
        
        self.assertEqual(summary.shape, (self.batch_size, 15, self.embedding_dim))
        
        # Check that the first num_tokens are the original tokens
        self.assertTrue(torch.allclose(summary[:, :self.num_tokens, :], self.tokens, atol=1e-6))
        
        # Check that the rest are zeros
        zeros = torch.zeros(self.batch_size, 15 - self.num_tokens, self.embedding_dim)
        self.assertTrue(torch.allclose(summary[:, self.num_tokens:, :], zeros, atol=1e-6))
    
    def test_max_pooling(self):
        """Test max pooling."""
        summarizer = PoolingSummarizer(embedding_dim=self.embedding_dim, pooling_type='max')
        summary = summarizer(self.tokens, k=1)
        
        self.assertEqual(summary.shape, (self.batch_size, 1, self.embedding_dim))


class TestTokenSummarize(unittest.TestCase):
    """Test cases for the token_summarize function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.embedding_dim = 64
        self.batch_size = 2
        self.num_tokens = 10
        
        # Create random token embeddings
        self.tokens = torch.randn(self.batch_size, self.num_tokens, self.embedding_dim)
    
    def test_mlp_summarization(self):
        """Test MLP-based summarization."""
        summary = token_summarize(self.tokens, k=3, method='mlp')
        
        self.assertEqual(summary.shape, (self.batch_size, 3, self.embedding_dim))
    
    def test_query_summarization(self):
        """Test query-based summarization."""
        summary = token_summarize(self.tokens, k=3, method='query')
        
        self.assertEqual(summary.shape, (self.batch_size, 3, self.embedding_dim))
    
    def test_pooling_summarization(self):
        """Test pooling-based summarization."""
        summary = token_summarize(self.tokens, k=3, method='pooling')
        
        self.assertEqual(summary.shape, (self.batch_size, 3, self.embedding_dim))
    
    def test_custom_summarizer(self):
        """Test using a custom summarizer."""
        summarizer = MLPSummarizer(embedding_dim=self.embedding_dim)
        summary = token_summarize(self.tokens, k=3, summarizer=summarizer)
        
        self.assertEqual(summary.shape, (self.batch_size, 3, self.embedding_dim))
    
    def test_invalid_method(self):
        """Test invalid summarization method."""
        with self.assertRaises(ValueError):
            token_summarize(self.tokens, k=3, method='invalid')


if __name__ == '__main__':
    unittest.main()
