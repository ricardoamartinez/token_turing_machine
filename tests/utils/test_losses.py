"""
Tests for the loss functions.
"""

import unittest
import torch
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.ttm.utils.losses import (
    TTMLoss,
    LabelSmoothingLoss,
    FocalLoss,
    create_loss_function
)
from src.ttm.utils.masking import EOSCrossEntropyLoss


class TestTTMLoss(unittest.TestCase):
    """Test cases for the TTMLoss class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.seq_len = 4
        self.vocab_size = 10
        self.embedding_dim = 16
        self.memory_size = 8
        self.num_heads = 2
        self.eos_token = 2
        
        # Create loss function
        self.loss_fn = TTMLoss(
            eos_token=self.eos_token,
            memory_loss_weight=0.1,
            attention_loss_weight=0.1
        )
        
        # Create logits and targets
        self.logits = torch.randn(self.batch_size, self.seq_len, self.vocab_size)
        self.targets = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        
        # Create memory states
        self.memory_before = torch.randn(self.batch_size, self.memory_size, self.embedding_dim)
        self.memory_after = torch.randn(self.batch_size, self.memory_size, self.embedding_dim)
        
        # Create attention weights
        self.attention_weights = torch.softmax(
            torch.randn(self.batch_size, self.num_heads, self.seq_len, self.seq_len),
            dim=-1
        )
    
    def test_initialization(self):
        """Test that the loss function is initialized correctly."""
        self.assertEqual(self.loss_fn.eos_token, self.eos_token)
        self.assertEqual(self.loss_fn.memory_loss_weight, 0.1)
        self.assertEqual(self.loss_fn.attention_loss_weight, 0.1)
        self.assertIsInstance(self.loss_fn.token_loss_fn, EOSCrossEntropyLoss)
    
    def test_compute_token_loss(self):
        """Test computing token prediction loss."""
        token_loss = self.loss_fn.compute_token_loss(self.logits, self.targets)
        
        self.assertTrue(token_loss.dim() == 0)  # Scalar
        self.assertTrue(token_loss.item() > 0)  # Loss should be positive
    
    def test_compute_memory_consistency_loss(self):
        """Test computing memory consistency loss."""
        memory_loss = self.loss_fn.compute_memory_consistency_loss(
            self.memory_before,
            self.memory_after
        )
        
        self.assertTrue(memory_loss.dim() == 0)  # Scalar
        self.assertTrue(memory_loss.item() > 0)  # Loss should be positive
    
    def test_compute_attention_entropy_loss(self):
        """Test computing attention entropy loss."""
        attention_loss = self.loss_fn.compute_attention_entropy_loss(
            self.attention_weights
        )
        
        self.assertTrue(attention_loss.dim() == 0)  # Scalar
        self.assertTrue(attention_loss.item() > 0)  # Loss should be positive
    
    def test_forward(self):
        """Test forward pass."""
        loss = self.loss_fn(
            self.logits,
            self.targets,
            self.memory_before,
            self.memory_after,
            self.attention_weights
        )
        
        self.assertTrue(loss.dim() == 0)  # Scalar
        self.assertTrue(loss.item() > 0)  # Loss should be positive
    
    def test_forward_with_reduction_none(self):
        """Test forward pass with reduction='none'."""
        loss_fn = TTMLoss(
            eos_token=self.eos_token,
            memory_loss_weight=0.1,
            attention_loss_weight=0.1,
            reduction='none'
        )
        
        losses = loss_fn(
            self.logits,
            self.targets,
            self.memory_before,
            self.memory_after,
            self.attention_weights
        )
        
        self.assertIsInstance(losses, dict)
        self.assertIn('total', losses)
        self.assertIn('token', losses)
        self.assertIn('memory', losses)
        self.assertIn('attention', losses)
    
    def test_forward_without_memory_and_attention(self):
        """Test forward pass without memory and attention."""
        loss = self.loss_fn(self.logits, self.targets)
        
        self.assertTrue(loss.dim() == 0)  # Scalar
        self.assertTrue(loss.item() > 0)  # Loss should be positive


class TestLabelSmoothingLoss(unittest.TestCase):
    """Test cases for the LabelSmoothingLoss class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.seq_len = 4
        self.vocab_size = 10
        self.smoothing = 0.1
        
        # Create loss function
        self.loss_fn = LabelSmoothingLoss(
            smoothing=self.smoothing
        )
        
        # Create logits and targets
        self.logits = torch.randn(self.batch_size, self.seq_len, self.vocab_size)
        self.targets = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
    
    def test_initialization(self):
        """Test that the loss function is initialized correctly."""
        self.assertEqual(self.loss_fn.smoothing, self.smoothing)
    
    def test_forward(self):
        """Test forward pass."""
        loss = self.loss_fn(self.logits, self.targets)
        
        self.assertTrue(loss.dim() == 0)  # Scalar
        self.assertTrue(loss.item() > 0)  # Loss should be positive
    
    def test_forward_with_reduction_none(self):
        """Test forward pass with reduction='none'."""
        loss_fn = LabelSmoothingLoss(
            smoothing=self.smoothing,
            reduction='none'
        )
        
        loss = loss_fn(self.logits, self.targets)
        
        self.assertEqual(loss.shape, (self.batch_size, self.seq_len))
    
    def test_forward_with_ignore_index(self):
        """Test forward pass with ignore_index."""
        ignore_index = 0
        loss_fn = LabelSmoothingLoss(
            smoothing=self.smoothing,
            ignore_index=ignore_index
        )
        
        # Create targets with ignore_index
        targets = self.targets.clone()
        targets[0, 0] = ignore_index
        
        loss = loss_fn(self.logits, targets)
        
        self.assertTrue(loss.dim() == 0)  # Scalar
        self.assertTrue(loss.item() > 0)  # Loss should be positive


class TestFocalLoss(unittest.TestCase):
    """Test cases for the FocalLoss class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.seq_len = 4
        self.vocab_size = 10
        self.alpha = 0.25
        self.gamma = 2.0
        
        # Create loss function
        self.loss_fn = FocalLoss(
            alpha=self.alpha,
            gamma=self.gamma
        )
        
        # Create logits and targets
        self.logits = torch.randn(self.batch_size, self.seq_len, self.vocab_size)
        self.targets = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
    
    def test_initialization(self):
        """Test that the loss function is initialized correctly."""
        self.assertEqual(self.loss_fn.alpha, self.alpha)
        self.assertEqual(self.loss_fn.gamma, self.gamma)
    
    def test_forward(self):
        """Test forward pass."""
        loss = self.loss_fn(self.logits, self.targets)
        
        self.assertTrue(loss.dim() == 0)  # Scalar
        self.assertTrue(loss.item() > 0)  # Loss should be positive
    
    def test_forward_with_reduction_none(self):
        """Test forward pass with reduction='none'."""
        loss_fn = FocalLoss(
            alpha=self.alpha,
            gamma=self.gamma,
            reduction='none'
        )
        
        loss = loss_fn(self.logits, self.targets)
        
        self.assertEqual(loss.shape, (self.batch_size, self.seq_len))
    
    def test_forward_with_ignore_index(self):
        """Test forward pass with ignore_index."""
        ignore_index = 0
        loss_fn = FocalLoss(
            alpha=self.alpha,
            gamma=self.gamma,
            ignore_index=ignore_index
        )
        
        # Create targets with ignore_index
        targets = self.targets.clone()
        targets[0, 0] = ignore_index
        
        loss = loss_fn(self.logits, targets)
        
        self.assertTrue(loss.dim() == 0)  # Scalar
        self.assertTrue(loss.item() > 0)  # Loss should be positive


class TestCreateLossFunction(unittest.TestCase):
    """Test cases for the create_loss_function function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.eos_token = 2
    
    def test_create_cross_entropy_loss(self):
        """Test creating a cross-entropy loss function."""
        loss_fn = create_loss_function(
            loss_type='cross_entropy',
            eos_token=self.eos_token
        )
        
        self.assertIsInstance(loss_fn, EOSCrossEntropyLoss)
    
    def test_create_cross_entropy_loss_without_eos(self):
        """Test creating a cross-entropy loss function without EOS token."""
        loss_fn = create_loss_function(
            loss_type='cross_entropy',
            eos_token=None
        )
        
        self.assertIsInstance(loss_fn, torch.nn.CrossEntropyLoss)
    
    def test_create_label_smoothing_loss(self):
        """Test creating a label smoothing loss function."""
        loss_fn = create_loss_function(
            loss_type='label_smoothing',
            label_smoothing=0.1
        )
        
        self.assertIsInstance(loss_fn, LabelSmoothingLoss)
    
    def test_create_focal_loss(self):
        """Test creating a focal loss function."""
        loss_fn = create_loss_function(
            loss_type='focal',
            focal_alpha=0.25,
            focal_gamma=2.0
        )
        
        self.assertIsInstance(loss_fn, FocalLoss)
    
    def test_create_ttm_loss(self):
        """Test creating a TTM loss function."""
        loss_fn = create_loss_function(
            loss_type='ttm',
            eos_token=self.eos_token,
            memory_loss_weight=0.1,
            attention_loss_weight=0.1
        )
        
        self.assertIsInstance(loss_fn, TTMLoss)
    
    def test_create_unknown_loss(self):
        """Test creating an unknown loss function."""
        with self.assertRaises(ValueError):
            create_loss_function(loss_type='unknown')


if __name__ == '__main__':
    unittest.main()
