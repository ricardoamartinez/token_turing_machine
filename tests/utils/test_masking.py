"""
Tests for the masking utilities.
"""

import unittest
import torch
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.ttm.utils.masking import (
    create_padding_mask,
    create_causal_mask,
    create_combined_mask,
    find_eos_positions,
    mask_after_eos,
    create_eos_loss_mask,
    apply_eos_loss_mask,
    EOSCrossEntropyLoss
)


class TestPaddingMask(unittest.TestCase):
    """Test cases for the padding mask functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.seq_len = 5
        self.padding_token = 0
        
        # Create token sequences with padding
        self.tokens = torch.tensor([
            [1, 2, 3, 0, 0],  # Sequence with padding at positions 3 and 4
            [4, 5, 0, 0, 0]   # Sequence with padding at positions 2, 3, and 4
        ])
    
    def test_create_padding_mask(self):
        """Test creating a padding mask."""
        mask = create_padding_mask(self.tokens, self.padding_token)
        
        self.assertEqual(mask.shape, (self.batch_size, self.seq_len))
        self.assertTrue(mask.dtype == torch.bool)
        
        # Check that the mask correctly identifies padding tokens
        expected_mask = torch.tensor([
            [False, False, False, True, True],
            [False, False, True, True, True]
        ])
        self.assertTrue(torch.all(mask == expected_mask))


class TestCausalMask(unittest.TestCase):
    """Test cases for the causal mask functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.seq_len = 4
    
    def test_create_causal_mask(self):
        """Test creating a causal mask."""
        mask = create_causal_mask(self.seq_len)
        
        self.assertEqual(mask.shape, (self.seq_len, self.seq_len))
        self.assertTrue(mask.dtype == torch.bool)
        
        # Check that the mask is lower triangular
        expected_mask = torch.tensor([
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True]
        ])
        self.assertTrue(torch.all(mask == expected_mask))


class TestCombinedMask(unittest.TestCase):
    """Test cases for the combined mask functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.seq_len = 4
        self.padding_token = 0
        
        # Create token sequences with padding
        self.tokens = torch.tensor([
            [1, 2, 0, 0],  # Sequence with padding at positions 2 and 3
            [3, 4, 5, 0]   # Sequence with padding at position 3
        ])
    
    def test_create_combined_mask_causal(self):
        """Test creating combined masks with causal masking."""
        attn_mask, key_padding_mask = create_combined_mask(
            self.tokens,
            self.padding_token,
            causal=True
        )
        
        self.assertEqual(attn_mask.shape, (self.seq_len, self.seq_len))
        self.assertEqual(key_padding_mask.shape, (self.batch_size, self.seq_len))
        
        # Check that the attention mask is causal
        expected_attn_mask = torch.tensor([
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True]
        ])
        self.assertTrue(torch.all(attn_mask == expected_attn_mask))
        
        # Check that the key padding mask correctly identifies padding tokens
        expected_key_padding_mask = torch.tensor([
            [False, False, True, True],
            [False, False, False, True]
        ])
        self.assertTrue(torch.all(key_padding_mask == expected_key_padding_mask))
    
    def test_create_combined_mask_non_causal(self):
        """Test creating combined masks without causal masking."""
        attn_mask, key_padding_mask = create_combined_mask(
            self.tokens,
            self.padding_token,
            causal=False
        )
        
        self.assertIsNone(attn_mask)
        self.assertEqual(key_padding_mask.shape, (self.batch_size, self.seq_len))
        
        # Check that the key padding mask correctly identifies padding tokens
        expected_key_padding_mask = torch.tensor([
            [False, False, True, True],
            [False, False, False, True]
        ])
        self.assertTrue(torch.all(key_padding_mask == expected_key_padding_mask))


class TestEOSHandling(unittest.TestCase):
    """Test cases for the EOS handling functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 3
        self.seq_len = 5
        self.eos_token = 2
        self.mask_token = 0
        
        # Create token sequences with EOS
        self.tokens = torch.tensor([
            [1, 2, 3, 4, 5],  # EOS at position 1
            [6, 7, 8, 2, 9],  # EOS at position 3
            [10, 11, 12, 13, 14]  # No EOS
        ])
    
    def test_find_eos_positions(self):
        """Test finding EOS positions."""
        positions = find_eos_positions(self.tokens, self.eos_token)
        
        self.assertEqual(positions.shape, (self.batch_size,))
        
        # Check that the positions are correct
        expected_positions = torch.tensor([1, 3, 5])  # 5 for no EOS
        self.assertTrue(torch.all(positions == expected_positions))
    
    def test_mask_after_eos(self):
        """Test masking tokens after EOS."""
        masked_tokens = mask_after_eos(self.tokens, self.eos_token, self.mask_token)
        
        self.assertEqual(masked_tokens.shape, self.tokens.shape)
        
        # Check that tokens after EOS are masked
        expected_masked_tokens = torch.tensor([
            [1, 2, 0, 0, 0],  # Masked after position 1
            [6, 7, 8, 2, 0],  # Masked after position 3
            [10, 11, 12, 13, 14]  # No masking (no EOS)
        ])
        self.assertTrue(torch.all(masked_tokens == expected_masked_tokens))
    
    def test_create_eos_loss_mask_include_eos(self):
        """Test creating an EOS loss mask including the EOS token."""
        mask = create_eos_loss_mask(self.tokens, self.eos_token, include_eos=True)
        
        self.assertEqual(mask.shape, self.tokens.shape)
        
        # Check that the mask includes tokens up to and including EOS
        expected_mask = torch.tensor([
            [True, True, False, False, False],  # Include up to position 1
            [True, True, True, True, False],  # Include up to position 3
            [True, True, True, True, True]  # Include all (no EOS)
        ])
        self.assertTrue(torch.all(mask == expected_mask))
    
    def test_create_eos_loss_mask_exclude_eos(self):
        """Test creating an EOS loss mask excluding the EOS token."""
        mask = create_eos_loss_mask(self.tokens, self.eos_token, include_eos=False)
        
        self.assertEqual(mask.shape, self.tokens.shape)
        
        # Check that the mask includes tokens strictly before EOS
        expected_mask = torch.tensor([
            [True, False, False, False, False],  # Include up to position 0
            [True, True, True, False, False],  # Include up to position 2
            [True, True, True, True, True]  # Include all (no EOS)
        ])
        self.assertTrue(torch.all(mask == expected_mask))
    
    def test_apply_eos_loss_mask_no_reduction(self):
        """Test applying an EOS loss mask with no reduction."""
        # Create a dummy loss tensor
        loss = torch.ones(self.batch_size, self.seq_len)
        
        masked_loss = apply_eos_loss_mask(
            loss,
            self.tokens,
            self.eos_token,
            include_eos=True,
            reduction='none'
        )
        
        self.assertEqual(masked_loss.shape, loss.shape)
        
        # Check that the loss is masked correctly
        expected_masked_loss = torch.tensor([
            [1.0, 1.0, 0.0, 0.0, 0.0],  # Masked after position 1
            [1.0, 1.0, 1.0, 1.0, 0.0],  # Masked after position 3
            [1.0, 1.0, 1.0, 1.0, 1.0]  # No masking (no EOS)
        ])
        self.assertTrue(torch.allclose(masked_loss, expected_masked_loss))
    
    def test_apply_eos_loss_mask_mean_reduction(self):
        """Test applying an EOS loss mask with mean reduction."""
        # Create a dummy loss tensor
        loss = torch.ones(self.batch_size, self.seq_len)
        
        masked_loss = apply_eos_loss_mask(
            loss,
            self.tokens,
            self.eos_token,
            include_eos=True,
            reduction='mean'
        )
        
        self.assertTrue(masked_loss.dim() == 0)  # Scalar
        
        # Check that the mean is computed correctly
        # Total non-masked elements: 2 + 4 + 5 = 11
        expected_mean = 11.0 / 11.0  # All ones, so mean is 1.0
        self.assertTrue(torch.allclose(masked_loss, torch.tensor(expected_mean)))
    
    def test_apply_eos_loss_mask_sum_reduction(self):
        """Test applying an EOS loss mask with sum reduction."""
        # Create a dummy loss tensor
        loss = torch.ones(self.batch_size, self.seq_len)
        
        masked_loss = apply_eos_loss_mask(
            loss,
            self.tokens,
            self.eos_token,
            include_eos=True,
            reduction='sum'
        )
        
        self.assertTrue(masked_loss.dim() == 0)  # Scalar
        
        # Check that the sum is computed correctly
        # Total non-masked elements: 2 + 4 + 5 = 11
        expected_sum = 11.0  # All ones, so sum is 11.0
        self.assertTrue(torch.allclose(masked_loss, torch.tensor(expected_sum)))


class TestEOSCrossEntropyLoss(unittest.TestCase):
    """Test cases for the EOSCrossEntropyLoss class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.seq_len = 4
        self.vocab_size = 10
        self.eos_token = 2
        
        # Create logits and targets
        self.logits = torch.randn(self.batch_size, self.seq_len, self.vocab_size)
        self.targets = torch.tensor([
            [1, 2, 3, 4],  # EOS at position 1
            [5, 6, 7, 2]   # EOS at position 3
        ])
    
    def test_eos_cross_entropy_loss_mean_reduction(self):
        """Test EOS cross entropy loss with mean reduction."""
        loss_fn = EOSCrossEntropyLoss(
            eos_token=self.eos_token,
            include_eos=True,
            reduction='mean'
        )
        
        loss = loss_fn(self.logits, self.targets)
        
        self.assertTrue(loss.dim() == 0)  # Scalar
    
    def test_eos_cross_entropy_loss_none_reduction(self):
        """Test EOS cross entropy loss with no reduction."""
        loss_fn = EOSCrossEntropyLoss(
            eos_token=self.eos_token,
            include_eos=True,
            reduction='none'
        )
        
        loss = loss_fn(self.logits, self.targets)
        
        self.assertEqual(loss.shape, (self.batch_size, self.seq_len))
        
        # Check that the loss is masked correctly
        # First sequence: positions 0 and 1 should have non-zero loss
        # Second sequence: positions 0, 1, 2, and 3 should have non-zero loss
        mask = torch.tensor([
            [True, True, False, False],
            [True, True, True, True]
        ])
        self.assertTrue(torch.all((loss > 0) == mask))


if __name__ == '__main__':
    unittest.main()
