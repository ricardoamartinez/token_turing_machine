"""
Tests for the optimizer utilities.
"""

import unittest
import torch
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.ttm.models.ttm_model import TokenTuringMachine
from src.ttm.training.optimizer import (
    create_optimizer,
    create_scheduler,
    get_grouped_parameters,
    get_parameter_names,
    get_ttm_optimizer
)


class TestCreateOptimizer(unittest.TestCase):
    """Test cases for the create_optimizer function."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 10)
        )

    def test_create_adam_optimizer(self):
        """Test creating an Adam optimizer."""
        optimizer = create_optimizer(
            model=self.model,
            optimizer_type='adam',
            learning_rate=1e-4,
            weight_decay=0.01
        )

        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertEqual(optimizer.defaults['lr'], 1e-4)
        self.assertEqual(optimizer.defaults['weight_decay'], 0.01)

    def test_create_adamw_optimizer(self):
        """Test creating an AdamW optimizer."""
        optimizer = create_optimizer(
            model=self.model,
            optimizer_type='adamw',
            learning_rate=1e-4,
            weight_decay=0.01
        )

        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(optimizer.defaults['lr'], 1e-4)
        self.assertEqual(optimizer.defaults['weight_decay'], 0.01)

    def test_create_sgd_optimizer(self):
        """Test creating an SGD optimizer."""
        optimizer = create_optimizer(
            model=self.model,
            optimizer_type='sgd',
            learning_rate=1e-4,
            weight_decay=0.01,
            momentum=0.9
        )

        self.assertIsInstance(optimizer, torch.optim.SGD)
        self.assertEqual(optimizer.defaults['lr'], 1e-4)
        self.assertEqual(optimizer.defaults['weight_decay'], 0.01)
        self.assertEqual(optimizer.defaults['momentum'], 0.9)

    def test_create_rmsprop_optimizer(self):
        """Test creating an RMSprop optimizer."""
        optimizer = create_optimizer(
            model=self.model,
            optimizer_type='rmsprop',
            learning_rate=1e-4,
            weight_decay=0.01,
            momentum=0.9
        )

        self.assertIsInstance(optimizer, torch.optim.RMSprop)
        self.assertEqual(optimizer.defaults['lr'], 1e-4)
        self.assertEqual(optimizer.defaults['weight_decay'], 0.01)
        self.assertEqual(optimizer.defaults['momentum'], 0.9)

    def test_create_optimizer_with_separate_decay(self):
        """Test creating an optimizer with separate weight decay."""
        optimizer = create_optimizer(
            model=self.model,
            optimizer_type='adamw',
            learning_rate=1e-4,
            weight_decay=0.01,
            separate_decay_parameters=True
        )

        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(optimizer.param_groups[0]['weight_decay'], 0.01)
        self.assertEqual(optimizer.param_groups[1]['weight_decay'], 0.0)

    def test_create_optimizer_with_unknown_type(self):
        """Test creating an optimizer with an unknown type."""
        with self.assertRaises(ValueError):
            create_optimizer(
                model=self.model,
                optimizer_type='unknown'
            )


class TestCreateScheduler(unittest.TestCase):
    """Test cases for the create_scheduler function."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 10)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def test_create_linear_scheduler(self):
        """Test creating a linear scheduler."""
        scheduler = create_scheduler(
            optimizer=self.optimizer,
            scheduler_type='linear',
            num_warmup_steps=100,
            num_training_steps=1000
        )

        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.LambdaLR)

    def test_create_cosine_scheduler(self):
        """Test creating a cosine scheduler."""
        scheduler = create_scheduler(
            optimizer=self.optimizer,
            scheduler_type='cosine',
            num_warmup_steps=100,
            num_training_steps=1000,
            num_cycles=0.5,
            min_lr=0.0
        )

        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.LambdaLR)

    def test_create_cosine_annealing_scheduler(self):
        """Test creating a cosine annealing scheduler."""
        scheduler = create_scheduler(
            optimizer=self.optimizer,
            scheduler_type='cosine_annealing',
            num_training_steps=1000,
            min_lr=0.0
        )

        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_create_plateau_scheduler(self):
        """Test creating a plateau scheduler."""
        scheduler = create_scheduler(
            optimizer=self.optimizer,
            scheduler_type='plateau',
            factor=0.1,
            patience=10,
            threshold=1e-4,
            min_lr=0.0
        )

        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_create_scheduler_with_unknown_type(self):
        """Test creating a scheduler with an unknown type."""
        with self.assertRaises(ValueError):
            create_scheduler(
                optimizer=self.optimizer,
                scheduler_type='unknown'
            )

    def test_create_scheduler_with_none_type(self):
        """Test creating a scheduler with None type."""
        scheduler = create_scheduler(
            optimizer=self.optimizer,
            scheduler_type=None
        )

        self.assertIsNone(scheduler)


class TestGetGroupedParameters(unittest.TestCase):
    """Test cases for the get_grouped_parameters function."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.LayerNorm(20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 10)
        )

    def test_get_grouped_parameters(self):
        """Test getting grouped parameters."""
        param_groups = get_grouped_parameters(
            model=self.model,
            weight_decay=0.01,
            no_decay_name_list=['bias', 'layer_norm', 'layernorm']
        )

        self.assertEqual(len(param_groups), 2)
        self.assertEqual(param_groups[0]['weight_decay'], 0.01)
        self.assertEqual(param_groups[1]['weight_decay'], 0.0)


class TestGetParameterNames(unittest.TestCase):
    """Test cases for the get_parameter_names function."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.LayerNorm(20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 10)
        )

    def test_get_parameter_names(self):
        """Test getting parameter names."""
        param_names = get_parameter_names(
            model=self.model,
            forbidden_layer_types=[]
        )

        self.assertIsInstance(param_names, list)
        self.assertTrue(len(param_names) > 0)

    def test_get_parameter_names_with_forbidden_layers(self):
        """Test getting parameter names with forbidden layers."""
        # First get all parameter names
        all_param_names = get_parameter_names(
            model=self.model,
            forbidden_layer_types=[]
        )

        # Then get parameter names excluding LayerNorm
        param_names = get_parameter_names(
            model=self.model,
            forbidden_layer_types=[torch.nn.LayerNorm]
        )

        self.assertIsInstance(param_names, list)

        # Check that some parameters are excluded
        self.assertLess(len(param_names), len(all_param_names))


class TestGetTTMOptimizer(unittest.TestCase):
    """Test cases for the get_ttm_optimizer function."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = TokenTuringMachine(
            vocab_size=100,
            embedding_dim=64,
            memory_size=8,
            r=4,
            num_layers=2,
            num_heads=4,
            hidden_dim=256
        )

    def test_get_ttm_optimizer(self):
        """Test getting a TTM optimizer."""
        optimizer = get_ttm_optimizer(
            model=self.model,
            learning_rate=1e-4,
            weight_decay=0.01,
            optimizer_type='adamw',
            separate_decay=True
        )

        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertTrue(len(optimizer.param_groups) > 1)

    def test_get_ttm_optimizer_without_separate_decay(self):
        """Test getting a TTM optimizer without separate decay."""
        optimizer = get_ttm_optimizer(
            model=self.model,
            learning_rate=1e-4,
            weight_decay=0.01,
            optimizer_type='adamw',
            separate_decay=False
        )

        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(len(optimizer.param_groups), 1)

    def test_get_ttm_optimizer_with_unknown_type(self):
        """Test getting a TTM optimizer with an unknown type."""
        with self.assertRaises(ValueError):
            get_ttm_optimizer(
                model=self.model,
                optimizer_type='unknown'
            )


if __name__ == '__main__':
    unittest.main()
