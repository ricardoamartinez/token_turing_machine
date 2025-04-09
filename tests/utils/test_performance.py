"""
Tests for the performance utilities.
"""

import unittest
import torch
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.ttm.models.ttm_model import TokenTuringMachine
from src.ttm.utils.performance import (
    measure_flops,
    measure_memory,
    benchmark_forward,
    benchmark_sequence_length,
    compare_cpu_cuda,
    jit_attention,
    jit_memory_read,
    jit_memory_write
)


class TestMeasureFlops(unittest.TestCase):
    """Test cases for the measure_flops function."""

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

        # Make sure all parameters require gradients
        for param in self.model.parameters():
            param.requires_grad = True

        # Create input shape
        self.input_shape = (self.batch_size, self.seq_len)

    def test_measure_flops(self):
        """Test measuring FLOPS."""
        # Measure FLOPS
        flops = measure_flops(self.model, self.input_shape)

        # Check that FLOPS is a positive integer
        self.assertGreater(flops, 0)
        self.assertEqual(type(flops), int)


class TestMeasureMemory(unittest.TestCase):
    """Test cases for the measure_memory function."""

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

        # Create input shape
        self.input_shape = (self.batch_size, self.seq_len)

    def test_measure_memory(self):
        """Test measuring memory usage."""
        # Measure memory usage
        memory_stats = measure_memory(self.model, self.input_shape)

        # Check that memory stats is a dictionary
        self.assertIsInstance(memory_stats, dict)

        # Check that memory stats contains expected keys
        self.assertIn('cpu_memory_initial', memory_stats)
        self.assertIn('cpu_memory_final', memory_stats)
        self.assertIn('cpu_memory_diff', memory_stats)

        # Check that memory usage is non-negative
        self.assertGreaterEqual(memory_stats['cpu_memory_initial'], 0)
        self.assertGreaterEqual(memory_stats['cpu_memory_final'], 0)


class TestBenchmarkForward(unittest.TestCase):
    """Test cases for the benchmark_forward function."""

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

        # Create input shape
        self.input_shape = (self.batch_size, self.seq_len)

    def test_benchmark_forward(self):
        """Test benchmarking forward pass."""
        # Benchmark forward pass
        results = benchmark_forward(
            model=self.model,
            input_shape=self.input_shape,
            num_iterations=2,
            warmup_iterations=1
        )

        # Check that results is a dictionary
        self.assertIsInstance(results, dict)

        # Check that results contains expected keys
        self.assertIn('total_time', results)
        self.assertIn('avg_time', results)
        self.assertIn('iterations', results)

        # Check that times are positive
        self.assertGreater(results['total_time'], 0)
        self.assertGreater(results['avg_time'], 0)

        # Check that iterations is correct
        self.assertEqual(results['iterations'], 2)


class TestBenchmarkSequenceLength(unittest.TestCase):
    """Test cases for the benchmark_sequence_length function."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.seq_lengths = [4, 8]
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

    def test_benchmark_sequence_length(self):
        """Test benchmarking for different sequence lengths."""
        # Benchmark sequence lengths
        results = benchmark_sequence_length(
            model=self.model,
            batch_size=self.batch_size,
            seq_lengths=self.seq_lengths,
            num_iterations=2,
            warmup_iterations=1
        )

        # Check that results is a dictionary
        self.assertIsInstance(results, dict)

        # Check that results contains entries for each sequence length
        for seq_len in self.seq_lengths:
            self.assertIn(seq_len, results)

            # Check that each entry is a dictionary
            self.assertIsInstance(results[seq_len], dict)

            # Check that each entry contains expected keys
            self.assertIn('avg_time', results[seq_len])
            self.assertIn('flops', results[seq_len])


class TestCompareCpuCuda(unittest.TestCase):
    """Test cases for the compare_cpu_cuda function."""

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

        # Create input shape
        self.input_shape = (self.batch_size, self.seq_len)

    def test_compare_cpu_cuda(self):
        """Test comparing CPU and CUDA performance."""
        # Skip test if CUDA is not available
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")

        # Compare CPU and CUDA
        results = compare_cpu_cuda(
            model=self.model,
            input_shape=self.input_shape,
            num_iterations=2,
            warmup_iterations=1
        )

        # Check that results is a dictionary
        self.assertIsInstance(results, dict)

        # Check that results contains entries for CPU and CUDA
        self.assertIn('cpu', results)
        self.assertIn('cuda', results)

        # Check that each entry is a dictionary
        self.assertIsInstance(results['cpu'], dict)
        self.assertIsInstance(results['cuda'], dict)

        # Check that each entry contains expected keys
        self.assertIn('avg_time', results['cpu'])
        self.assertIn('avg_time', results['cuda'])

        # Check that CUDA entry contains speedup
        self.assertIn('speedup', results['cuda'])

        # Check that speedup is positive
        self.assertGreater(results['cuda']['speedup'], 0)


class TestJitFunctions(unittest.TestCase):
    """Test cases for the JIT-compiled functions."""

    def test_jit_attention(self):
        """Test JIT-compiled attention function."""
        # Create inputs
        batch_size = 2
        num_heads = 4
        seq_len = 8
        head_dim = 16

        query = torch.randn(batch_size, num_heads, seq_len, head_dim)
        key = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim)
        mask = torch.zeros(batch_size, 1, seq_len, seq_len)

        # Run JIT-compiled function
        output = jit_attention(query, key, value, mask)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, num_heads, seq_len, head_dim))

    def test_jit_memory_read(self):
        """Test JIT-compiled memory read function."""
        # Create inputs
        batch_size = 2
        memory_size = 8
        embedding_dim = 16
        r = 4

        memory = torch.randn(batch_size, memory_size, embedding_dim)
        read_weights = torch.softmax(torch.randn(batch_size, r, memory_size), dim=-1)

        # Run JIT-compiled function
        read_vectors = jit_memory_read(memory, read_weights)

        # Check output shape
        self.assertEqual(read_vectors.shape, (batch_size, r, embedding_dim))

    def test_jit_memory_write(self):
        """Test JIT-compiled memory write function."""
        # Create inputs
        batch_size = 2
        memory_size = 8
        embedding_dim = 16
        r = 4

        memory = torch.randn(batch_size, memory_size, embedding_dim)
        write_weights = torch.softmax(torch.randn(batch_size, r, memory_size), dim=-1)
        write_vectors = torch.randn(batch_size, r, embedding_dim)

        # Run JIT-compiled function
        updated_memory = jit_memory_write(memory, write_weights, write_vectors)

        # Check output shape
        self.assertEqual(updated_memory.shape, (batch_size, memory_size, embedding_dim))


if __name__ == '__main__':
    unittest.main()
