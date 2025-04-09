"""
Tests for the tokenization module.
"""

import unittest
import torch
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.ttm.data.tokenization import (
    VOCAB_SIZE, DIGIT_TOKENS, TIMES_TOKEN, EOS_TOKEN, PAD_TOKEN,
    number_to_tokens, tokens_to_number, tokens_to_string,
    create_multiplication_example, pad_sequence,
    create_tensor_from_tokens, create_batch_from_examples
)

class TestTokenization(unittest.TestCase):
    """Test cases for the tokenization module."""
    
    def test_vocabulary_constants(self):
        """Test that vocabulary constants are defined correctly."""
        self.assertEqual(VOCAB_SIZE, 13)
        self.assertEqual(len(DIGIT_TOKENS), 10)
        self.assertEqual(TIMES_TOKEN, 10)
        self.assertEqual(EOS_TOKEN, 11)
        self.assertEqual(PAD_TOKEN, 12)
    
    def test_number_to_tokens(self):
        """Test conversion of numbers to tokens."""
        self.assertEqual(number_to_tokens(0), [0])
        self.assertEqual(number_to_tokens(9), [9])
        self.assertEqual(number_to_tokens(42), [4, 2])
        self.assertEqual(number_to_tokens(123), [1, 2, 3])
        self.assertEqual(number_to_tokens(7890), [7, 8, 9, 0])
    
    def test_tokens_to_number(self):
        """Test conversion of tokens to numbers."""
        self.assertEqual(tokens_to_number([0]), 0)
        self.assertEqual(tokens_to_number([9]), 9)
        self.assertEqual(tokens_to_number([4, 2]), 42)
        self.assertEqual(tokens_to_number([1, 2, 3]), 123)
        self.assertEqual(tokens_to_number([7, 8, 9, 0]), 7890)
        
        # Test invalid tokens
        with self.assertRaises(ValueError):
            tokens_to_number([10])  # TIMES_TOKEN
    
    def test_tokens_to_string(self):
        """Test conversion of tokens to strings."""
        self.assertEqual(tokens_to_string([4, 2]), "42")
        self.assertEqual(tokens_to_string([1, 2, TIMES_TOKEN, 3, 4]), "12×34")
        self.assertEqual(tokens_to_string([5, 6, EOS_TOKEN]), "56<EOS>")
        self.assertEqual(tokens_to_string([7, 8, PAD_TOKEN, PAD_TOKEN]), "78<PAD><PAD>")
        
        # Test invalid tokens
        with self.assertRaises(ValueError):
            tokens_to_string([13])  # Invalid token
    
    def test_create_multiplication_example(self):
        """Test creation of multiplication examples."""
        # Test simple example
        input_tokens, target_tokens = create_multiplication_example(12, 34, max_seq_len=10)
        
        # Check input format: num1 × num2 EOS PAD...
        self.assertEqual(input_tokens[:4], [1, 2, TIMES_TOKEN, 3])
        self.assertEqual(input_tokens[4:6], [4, EOS_TOKEN])
        self.assertTrue(all(t == PAD_TOKEN for t in input_tokens[6:]))
        
        # Check target format: result EOS PAD...
        self.assertEqual(target_tokens[:3], [4, 0, 8])  # 12 * 34 = 408
        self.assertEqual(target_tokens[3], EOS_TOKEN)
        self.assertTrue(all(t == PAD_TOKEN for t in target_tokens[4:]))
        
        # Test with different max_seq_len
        input_tokens, target_tokens = create_multiplication_example(5, 6, max_seq_len=5)
        self.assertEqual(len(input_tokens), 5)
        self.assertEqual(len(target_tokens), 5)
        
        # Test with large numbers
        input_tokens, target_tokens = create_multiplication_example(123, 456, max_seq_len=15)
        self.assertEqual(tokens_to_number(input_tokens[:3]), 123)
        self.assertEqual(input_tokens[3], TIMES_TOKEN)
        self.assertEqual(tokens_to_number(input_tokens[4:7]), 456)
    
    def test_pad_sequence(self):
        """Test padding of sequences."""
        # Test padding shorter sequence
        padded = pad_sequence([1, 2, 3], 5)
        self.assertEqual(padded, [1, 2, 3, PAD_TOKEN, PAD_TOKEN])
        
        # Test sequence of exact length
        padded = pad_sequence([4, 5, 6], 3)
        self.assertEqual(padded, [4, 5, 6])
        
        # Test truncating longer sequence
        padded = pad_sequence([7, 8, 9, 0], 2)
        self.assertEqual(padded, [7, 8])
    
    def test_create_tensor_from_tokens(self):
        """Test creation of tensors from tokens."""
        tokens = [1, 2, 3, PAD_TOKEN]
        tensor = create_tensor_from_tokens(tokens)
        
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.dtype, torch.long)
        self.assertEqual(tensor.tolist(), tokens)
    
    def test_create_batch_from_examples(self):
        """Test creation of batches from examples."""
        examples = [
            ([1, 2, TIMES_TOKEN, 3, 4, EOS_TOKEN, PAD_TOKEN], [4, 0, 8, EOS_TOKEN, PAD_TOKEN, PAD_TOKEN, PAD_TOKEN]),
            ([5, TIMES_TOKEN, 6, EOS_TOKEN, PAD_TOKEN, PAD_TOKEN, PAD_TOKEN], [3, 0, EOS_TOKEN, PAD_TOKEN, PAD_TOKEN, PAD_TOKEN, PAD_TOKEN])
        ]
        
        input_batch, target_batch = create_batch_from_examples(examples)
        
        self.assertIsInstance(input_batch, torch.Tensor)
        self.assertIsInstance(target_batch, torch.Tensor)
        self.assertEqual(input_batch.shape, (2, 7))
        self.assertEqual(target_batch.shape, (2, 7))
        self.assertEqual(input_batch[0].tolist(), examples[0][0])
        self.assertEqual(input_batch[1].tolist(), examples[1][0])
        self.assertEqual(target_batch[0].tolist(), examples[0][1])
        self.assertEqual(target_batch[1].tolist(), examples[1][1])

if __name__ == '__main__':
    unittest.main()
