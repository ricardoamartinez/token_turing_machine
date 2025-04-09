"""
Tests for the memory operations module.
"""

import unittest
import torch
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.ttm.models.memory_operations import (
    MemoryReadOperation,
    MemoryWriteOperation,
    MemoryModule
)


class TestMemoryReadOperation(unittest.TestCase):
    """Test cases for the MemoryReadOperation class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.embedding_dim = 64
        self.batch_size = 2
        self.memory_size = 8
        self.input_size = 4
        self.r = 6
        self.read_op = MemoryReadOperation(
            embedding_dim=self.embedding_dim,
            r=self.r,
            summarization_method='mlp'
        )
        
        # Create random memory and input tokens
        self.memory = torch.randn(self.batch_size, self.memory_size, self.embedding_dim)
        self.input_tokens = torch.randn(self.batch_size, self.input_size, self.embedding_dim)
    
    def test_initialization(self):
        """Test that the read operation is initialized correctly."""
        self.assertEqual(self.read_op.embedding_dim, self.embedding_dim)
        self.assertEqual(self.read_op.r, self.r)
        self.assertEqual(self.read_op.summarization_method, 'mlp')
    
    def test_concat_memory_input(self):
        """Test concatenating memory and input tokens."""
        combined = self.read_op.concat_memory_input(self.memory, self.input_tokens)
        
        self.assertEqual(combined.shape, (self.batch_size, self.memory_size + self.input_size, self.embedding_dim))
        
        # Check that the first part is memory and the second part is input
        self.assertTrue(torch.allclose(combined[:, :self.memory_size, :], self.memory, atol=1e-6))
        self.assertTrue(torch.allclose(combined[:, self.memory_size:, :], self.input_tokens, atol=1e-6))
    
    def test_add_positional_info(self):
        """Test adding positional embeddings."""
        memory_with_pos, input_with_pos = self.read_op.add_positional_info(self.memory, self.input_tokens)
        
        self.assertEqual(memory_with_pos.shape, self.memory.shape)
        self.assertEqual(input_with_pos.shape, self.input_tokens.shape)
        
        # Check that positional embeddings were added
        self.assertFalse(torch.allclose(memory_with_pos, self.memory, atol=1e-6))
        self.assertFalse(torch.allclose(input_with_pos, self.input_tokens, atol=1e-6))
    
    def test_forward(self):
        """Test forward pass."""
        read_tokens = self.read_op(self.memory, self.input_tokens)
        
        self.assertEqual(read_tokens.shape, (self.batch_size, self.r, self.embedding_dim))


class TestMemoryWriteOperation(unittest.TestCase):
    """Test cases for the MemoryWriteOperation class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.embedding_dim = 64
        self.batch_size = 2
        self.memory_size = 8
        self.write_size = 4
        
        # Create memory write operations with different methods
        self.summarization_write_op = MemoryWriteOperation(
            embedding_dim=self.embedding_dim,
            memory_size=self.memory_size,
            write_method='summarization',
            summarization_method='mlp'
        )
        
        self.erase_add_write_op = MemoryWriteOperation(
            embedding_dim=self.embedding_dim,
            memory_size=self.memory_size,
            write_method='erase_add',
            summarization_method='mlp'
        )
        
        self.concat_write_op = MemoryWriteOperation(
            embedding_dim=self.embedding_dim,
            memory_size=self.memory_size,
            write_method='concat',
            summarization_method='mlp'
        )
        
        # Create random memory and write tokens
        self.memory = torch.randn(self.batch_size, self.memory_size, self.embedding_dim)
        self.write_tokens = torch.randn(self.batch_size, self.write_size, self.embedding_dim)
    
    def test_initialization(self):
        """Test that the write operations are initialized correctly."""
        self.assertEqual(self.summarization_write_op.embedding_dim, self.embedding_dim)
        self.assertEqual(self.summarization_write_op.memory_size, self.memory_size)
        self.assertEqual(self.summarization_write_op.write_method, 'summarization')
        
        self.assertEqual(self.erase_add_write_op.write_method, 'erase_add')
        self.assertEqual(self.concat_write_op.write_method, 'concat')
    
    def test_summarization_write(self):
        """Test summarization-based write."""
        new_memory = self.summarization_write_op.summarization_write(self.memory, self.write_tokens)
        
        self.assertEqual(new_memory.shape, (self.batch_size, self.memory_size, self.embedding_dim))
    
    def test_erase_add_write(self):
        """Test erase-add write."""
        new_memory = self.erase_add_write_op.erase_add_write(self.memory, self.write_tokens)
        
        self.assertEqual(new_memory.shape, (self.batch_size, self.memory_size, self.embedding_dim))
    
    def test_concat_write(self):
        """Test concatenation-based write."""
        new_memory = self.concat_write_op.concat_write(self.memory, self.write_tokens)
        
        self.assertEqual(new_memory.shape, (self.batch_size, self.memory_size, self.embedding_dim))
        
        # Check that the new memory contains the most recent tokens
        combined = torch.cat([self.memory, self.write_tokens], dim=1)
        expected = combined[:, -self.memory_size:, :]
        self.assertTrue(torch.allclose(new_memory, expected, atol=1e-6))
    
    def test_forward_summarization(self):
        """Test forward pass with summarization write."""
        new_memory = self.summarization_write_op(self.memory, self.write_tokens)
        
        self.assertEqual(new_memory.shape, (self.batch_size, self.memory_size, self.embedding_dim))
    
    def test_forward_erase_add(self):
        """Test forward pass with erase-add write."""
        new_memory = self.erase_add_write_op(self.memory, self.write_tokens)
        
        self.assertEqual(new_memory.shape, (self.batch_size, self.memory_size, self.embedding_dim))
    
    def test_forward_concat(self):
        """Test forward pass with concatenation write."""
        new_memory = self.concat_write_op(self.memory, self.write_tokens)
        
        self.assertEqual(new_memory.shape, (self.batch_size, self.memory_size, self.embedding_dim))


class TestMemoryModule(unittest.TestCase):
    """Test cases for the MemoryModule class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.embedding_dim = 64
        self.batch_size = 2
        self.memory_size = 8
        self.input_size = 4
        self.r = 6
        self.memory_module = MemoryModule(
            embedding_dim=self.embedding_dim,
            memory_size=self.memory_size,
            r=self.r,
            summarization_method='mlp',
            write_method='summarization'
        )
        
        # Create random input tokens
        self.input_tokens = torch.randn(self.batch_size, self.input_size, self.embedding_dim)
    
    def test_initialization(self):
        """Test that the memory module is initialized correctly."""
        self.assertEqual(self.memory_module.embedding_dim, self.embedding_dim)
        self.assertEqual(self.memory_module.memory_size, self.memory_size)
        self.assertEqual(self.memory_module.r, self.r)
    
    def test_initialize_memory(self):
        """Test initializing memory."""
        memory = self.memory_module.initialize_memory(self.batch_size)
        
        self.assertEqual(memory.shape, (self.batch_size, self.memory_size, self.embedding_dim))
        
        # Check that memory is initialized with zeros
        zeros = torch.zeros_like(memory)
        self.assertTrue(torch.allclose(memory, zeros, atol=1e-6))
    
    def test_read(self):
        """Test reading from memory."""
        memory = self.memory_module.initialize_memory(self.batch_size)
        read_tokens = self.memory_module.read(memory, self.input_tokens)
        
        self.assertEqual(read_tokens.shape, (self.batch_size, self.r, self.embedding_dim))
    
    def test_write(self):
        """Test writing to memory."""
        memory = self.memory_module.initialize_memory(self.batch_size)
        new_memory = self.memory_module.write(memory, self.input_tokens)
        
        self.assertEqual(new_memory.shape, (self.batch_size, self.memory_size, self.embedding_dim))
    
    def test_forward(self):
        """Test forward pass."""
        memory = self.memory_module.initialize_memory(self.batch_size)
        read_tokens, new_memory = self.memory_module(memory, self.input_tokens)
        
        self.assertEqual(read_tokens.shape, (self.batch_size, self.r, self.embedding_dim))
        self.assertEqual(new_memory.shape, (self.batch_size, self.memory_size, self.embedding_dim))


if __name__ == '__main__':
    unittest.main()
