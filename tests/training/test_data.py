"""
Tests for the data utilities.
"""

import unittest
import torch
import sys
import os
import tempfile
import json

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.ttm.training.data import (
    SequenceDataset,
    CausalLanguageModelingDataset,
    MathDataset,
    create_dataloaders,
    load_json_data,
    load_text_data,
    split_data
)


class TestSequenceDataset(unittest.TestCase):
    """Test cases for the SequenceDataset class."""

    def setUp(self):
        """Set up test fixtures."""
        self.data = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11, 12],
            [13, 14, 15]
        ]
        self.seq_len = 6
        self.pad_token_id = 0
        self.eos_token_id = 2

    def test_initialization(self):
        """Test that the dataset is initialized correctly."""
        dataset = SequenceDataset(
            data=self.data,
            seq_len=self.seq_len,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id
        )

        self.assertEqual(len(dataset), len(self.data))

    def test_getitem(self):
        """Test getting an item from the dataset."""
        dataset = SequenceDataset(
            data=self.data,
            seq_len=self.seq_len,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id
        )

        item = dataset[0]

        self.assertIn('input_ids', item)
        self.assertIn('labels', item)
        self.assertEqual(item['input_ids'].shape, torch.Size([self.seq_len]))
        self.assertEqual(item['labels'].shape, torch.Size([self.seq_len]))

    def test_getitem_with_padding(self):
        """Test getting an item that needs padding."""
        dataset = SequenceDataset(
            data=self.data,
            seq_len=self.seq_len,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id
        )

        item = dataset[2]  # [13, 14, 15]

        self.assertEqual(item['input_ids'].shape, torch.Size([self.seq_len]))
        self.assertEqual(item['input_ids'][-1], self.pad_token_id)

    def test_getitem_with_truncation(self):
        """Test getting an item that needs truncation."""
        dataset = SequenceDataset(
            data=self.data,
            seq_len=self.seq_len,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id
        )

        item = dataset[1]  # [6, 7, 8, 9, 10, 11, 12]

        self.assertEqual(item['input_ids'].shape, torch.Size([self.seq_len]))
        self.assertEqual(item['input_ids'][-1], self.eos_token_id)

    def test_getitem_with_eos(self):
        """Test getting an item with EOS token."""
        dataset = SequenceDataset(
            data=self.data,
            seq_len=self.seq_len,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id
        )

        item = dataset[0]  # [1, 2, 3, 4, 5]

        self.assertEqual(item['input_ids'].shape, torch.Size([self.seq_len]))
        # Check that the sequence has been padded to seq_len
        self.assertTrue(item['input_ids'][5] in [self.pad_token_id, self.eos_token_id])
        # Check that the EOS token is present
        self.assertTrue(self.eos_token_id in item['input_ids'])


class TestCausalLanguageModelingDataset(unittest.TestCase):
    """Test cases for the CausalLanguageModelingDataset class."""

    def setUp(self):
        """Set up test fixtures."""
        self.data = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11, 12],
            [13, 14, 15]
        ]
        self.seq_len = 4
        self.stride = 2
        self.pad_token_id = 0
        self.eos_token_id = 2

    def test_initialization(self):
        """Test that the dataset is initialized correctly."""
        dataset = CausalLanguageModelingDataset(
            data=self.data,
            seq_len=self.seq_len,
            stride=self.stride,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id
        )

        self.assertTrue(len(dataset) > 0)

    def test_getitem(self):
        """Test getting an item from the dataset."""
        dataset = CausalLanguageModelingDataset(
            data=self.data,
            seq_len=self.seq_len,
            stride=self.stride,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id
        )

        item = dataset[0]

        self.assertIn('input_ids', item)
        self.assertIn('labels', item)
        self.assertEqual(item['input_ids'].shape, torch.Size([self.seq_len]))
        self.assertEqual(item['labels'].shape, torch.Size([self.seq_len]))

    def test_getitem_with_padding(self):
        """Test getting an item that needs padding."""
        dataset = CausalLanguageModelingDataset(
            data=[[1, 2]],
            seq_len=self.seq_len,
            stride=self.stride,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id
        )

        item = dataset[0]

        self.assertEqual(item['input_ids'].shape, torch.Size([self.seq_len]))
        self.assertEqual(item['input_ids'][-1], self.pad_token_id)
        self.assertEqual(item['labels'][-1], -100)  # Padding tokens are ignored in loss


class TestMathDataset(unittest.TestCase):
    """Test cases for the MathDataset class."""

    def setUp(self):
        """Set up test fixtures."""
        self.data = [
            {'question': [1, 2, 3], 'answer': [4, 5]},
            {'question': [6, 7], 'answer': [8, 9, 10]},
            {'question': [11, 12, 13, 14], 'answer': [15]}
        ]
        self.seq_len = 6
        self.pad_token_id = 0
        self.eos_token_id = 2

    def test_initialization(self):
        """Test that the dataset is initialized correctly."""
        dataset = MathDataset(
            data=self.data,
            seq_len=self.seq_len,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id
        )

        self.assertEqual(len(dataset), len(self.data))

    def test_getitem(self):
        """Test getting an item from the dataset."""
        dataset = MathDataset(
            data=self.data,
            seq_len=self.seq_len,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id
        )

        item = dataset[0]

        self.assertIn('input_ids', item)
        self.assertIn('labels', item)
        self.assertEqual(item['input_ids'].shape, torch.Size([self.seq_len]))
        self.assertEqual(item['labels'].shape, torch.Size([self.seq_len]))

    def test_getitem_with_padding(self):
        """Test getting an item that needs padding."""
        dataset = MathDataset(
            data=self.data,
            seq_len=self.seq_len,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id
        )

        item = dataset[0]  # {'question': [1, 2, 3], 'answer': [4, 5]}

        self.assertEqual(item['input_ids'].shape, torch.Size([self.seq_len]))
        # Check that the sequence has been padded or has EOS token at the end
        self.assertTrue(item['input_ids'][-1] in [self.pad_token_id, self.eos_token_id])

    def test_getitem_with_truncation(self):
        """Test getting an item that needs truncation."""
        dataset = MathDataset(
            data=[{'question': [1, 2, 3, 4], 'answer': [5, 6, 7, 8, 9]}],
            seq_len=self.seq_len,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id
        )

        item = dataset[0]

        self.assertEqual(item['input_ids'].shape, torch.Size([self.seq_len]))

    def test_getitem_with_eos(self):
        """Test getting an item with EOS token."""
        dataset = MathDataset(
            data=self.data,
            seq_len=self.seq_len,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id
        )

        item = dataset[0]  # {'question': [1, 2, 3], 'answer': [4, 5]}

        self.assertEqual(item['input_ids'].shape, torch.Size([self.seq_len]))
        # Check that the EOS token is present
        self.assertTrue(self.eos_token_id in item['input_ids'])


class TestCreateDataloaders(unittest.TestCase):
    """Test cases for the create_dataloaders function."""

    def setUp(self):
        """Set up test fixtures."""
        self.data = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11, 12],
            [13, 14, 15]
        ]
        self.seq_len = 6
        self.pad_token_id = 0
        self.eos_token_id = 2

        self.train_dataset = SequenceDataset(
            data=self.data,
            seq_len=self.seq_len,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id
        )

        self.val_dataset = SequenceDataset(
            data=self.data,
            seq_len=self.seq_len,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id
        )

    def test_create_dataloaders(self):
        """Test creating dataloaders."""
        train_dataloader, val_dataloader = create_dataloaders(
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )

        self.assertEqual(len(train_dataloader), 2)  # 3 examples with batch size 2
        self.assertEqual(len(val_dataloader), 2)  # 3 examples with batch size 2

    def test_create_dataloaders_without_val(self):
        """Test creating dataloaders without validation."""
        train_dataloader, val_dataloader = create_dataloaders(
            train_dataset=self.train_dataset,
            val_dataset=None,
            batch_size=2,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )

        self.assertEqual(len(train_dataloader), 2)  # 3 examples with batch size 2
        self.assertIsNone(val_dataloader)


class TestLoadJsonData(unittest.TestCase):
    """Test cases for the load_json_data function."""

    def setUp(self):
        """Set up test fixtures."""
        self.data = [
            {'text': 'Hello world', 'label': 1},
            {'text': 'Goodbye world', 'label': 0},
            {'text': 'Hello again', 'label': 1}
        ]

        # Create temporary JSON file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json')
        json.dump(self.data, self.temp_file)
        self.temp_file.close()

    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.temp_file.name)

    def test_load_json_data(self):
        """Test loading JSON data."""
        data = load_json_data(
            file_path=self.temp_file.name,
            tokenizer=None,
            max_examples=None,
            shuffle=False
        )

        self.assertEqual(len(data), len(self.data))
        self.assertEqual(data[0]['text'], self.data[0]['text'])

    def test_load_json_data_with_tokenizer(self):
        """Test loading JSON data with a tokenizer."""
        def tokenizer(text):
            return [ord(c) for c in text]

        data = load_json_data(
            file_path=self.temp_file.name,
            tokenizer=tokenizer,
            max_examples=None,
            shuffle=False
        )

        self.assertEqual(len(data), len(self.data))
        self.assertIn('tokens', data[0])
        self.assertEqual(data[0]['tokens'], tokenizer(self.data[0]['text']))

    def test_load_json_data_with_max_examples(self):
        """Test loading JSON data with a maximum number of examples."""
        data = load_json_data(
            file_path=self.temp_file.name,
            tokenizer=None,
            max_examples=2,
            shuffle=False
        )

        self.assertEqual(len(data), 2)


class TestLoadTextData(unittest.TestCase):
    """Test cases for the load_text_data function."""

    def setUp(self):
        """Set up test fixtures."""
        self.data = [
            'Hello world',
            'Goodbye world',
            'Hello again'
        ]

        # Create temporary text file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt')
        for line in self.data:
            self.temp_file.write(line + '\n')
        self.temp_file.close()

    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.temp_file.name)

    def test_load_text_data(self):
        """Test loading text data."""
        def tokenizer(text):
            return [ord(c) for c in text]

        data = load_text_data(
            file_path=self.temp_file.name,
            tokenizer=tokenizer,
            max_examples=None,
            shuffle=False
        )

        self.assertEqual(len(data), len(self.data))
        self.assertEqual(data[0], tokenizer(self.data[0]))

    def test_load_text_data_with_max_examples(self):
        """Test loading text data with a maximum number of examples."""
        def tokenizer(text):
            return [ord(c) for c in text]

        data = load_text_data(
            file_path=self.temp_file.name,
            tokenizer=tokenizer,
            max_examples=2,
            shuffle=False
        )

        self.assertEqual(len(data), 2)


class TestSplitData(unittest.TestCase):
    """Test cases for the split_data function."""

    def setUp(self):
        """Set up test fixtures."""
        self.data = list(range(100))

    def test_split_data(self):
        """Test splitting data."""
        train_data, val_data, test_data = split_data(
            data=self.data,
            val_ratio=0.1,
            test_ratio=0.1,
            shuffle=False
        )

        self.assertEqual(len(train_data), 80)
        self.assertEqual(len(val_data), 10)
        self.assertEqual(len(test_data), 10)

    def test_split_data_with_shuffle(self):
        """Test splitting data with shuffling."""
        train_data, val_data, test_data = split_data(
            data=self.data,
            val_ratio=0.1,
            test_ratio=0.1,
            shuffle=True
        )

        self.assertEqual(len(train_data), 80)
        self.assertEqual(len(val_data), 10)
        self.assertEqual(len(test_data), 10)

        # Check that the data is shuffled
        self.assertNotEqual(train_data, list(range(80)))


if __name__ == '__main__':
    unittest.main()
