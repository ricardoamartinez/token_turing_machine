"""
Data utilities for the Token Turing Machine (TTM) model.

This module provides utilities for loading and preprocessing data
for training the TTM model.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
import os
import json
import random


class SequenceDataset(Dataset):
    """Dataset for sequence modeling tasks."""
    
    def __init__(
        self,
        data: List[List[int]],
        seq_len: int,
        pad_token_id: int = 0,
        eos_token_id: Optional[int] = None
    ):
        """Initialize the sequence dataset.
        
        Args:
            data: List of token sequences
            seq_len: Maximum sequence length
            pad_token_id: Token ID for padding
            eos_token_id: Optional token ID for end-of-sequence
        """
        self.data = data
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
    
    def __len__(self) -> int:
        """Get the number of sequences in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sequence from the dataset.
        
        Args:
            idx: Index of the sequence
            
        Returns:
            Dictionary with input_ids and labels
        """
        # Get sequence
        sequence = self.data[idx]
        
        # Truncate sequence if needed
        if len(sequence) > self.seq_len:
            sequence = sequence[:self.seq_len]
        
        # Add EOS token if needed
        if self.eos_token_id is not None and sequence[-1] != self.eos_token_id:
            if len(sequence) < self.seq_len:
                sequence.append(self.eos_token_id)
            else:
                sequence[-1] = self.eos_token_id
        
        # Pad sequence if needed
        if len(sequence) < self.seq_len:
            sequence = sequence + [self.pad_token_id] * (self.seq_len - len(sequence))
        
        # Convert to tensors
        input_ids = torch.tensor(sequence, dtype=torch.long)
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }


class CausalLanguageModelingDataset(Dataset):
    """Dataset for causal language modeling tasks."""
    
    def __init__(
        self,
        data: List[List[int]],
        seq_len: int,
        stride: int = 1,
        pad_token_id: int = 0,
        eos_token_id: Optional[int] = None
    ):
        """Initialize the causal language modeling dataset.
        
        Args:
            data: List of token sequences
            seq_len: Maximum sequence length
            stride: Stride for sliding window
            pad_token_id: Token ID for padding
            eos_token_id: Optional token ID for end-of-sequence
        """
        self.data = data
        self.seq_len = seq_len
        self.stride = stride
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        
        # Create examples
        self.examples = []
        for sequence in data:
            # Add EOS token if needed
            if self.eos_token_id is not None and sequence[-1] != self.eos_token_id:
                sequence = sequence + [self.eos_token_id]
            
            # Create examples with sliding window
            for i in range(0, len(sequence) - 1, stride):
                # Get input and target
                input_ids = sequence[i:i+seq_len]
                
                # Pad input if needed
                if len(input_ids) < seq_len:
                    input_ids = input_ids + [pad_token_id] * (seq_len - len(input_ids))
                
                self.examples.append(input_ids)
    
    def __len__(self) -> int:
        """Get the number of examples in the dataset."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get an example from the dataset.
        
        Args:
            idx: Index of the example
            
        Returns:
            Dictionary with input_ids and labels
        """
        # Get example
        input_ids = self.examples[idx]
        
        # Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        
        # Set padding tokens to -100 in labels (ignored in loss)
        labels[labels == self.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }


class MathDataset(Dataset):
    """Dataset for mathematical reasoning tasks."""
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        seq_len: int,
        pad_token_id: int = 0,
        eos_token_id: Optional[int] = None
    ):
        """Initialize the math dataset.
        
        Args:
            data: List of examples with 'question' and 'answer' fields
            seq_len: Maximum sequence length
            pad_token_id: Token ID for padding
            eos_token_id: Optional token ID for end-of-sequence
        """
        self.data = data
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
    
    def __len__(self) -> int:
        """Get the number of examples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get an example from the dataset.
        
        Args:
            idx: Index of the example
            
        Returns:
            Dictionary with input_ids and labels
        """
        # Get example
        example = self.data[idx]
        question = example['question']
        answer = example['answer']
        
        # Add EOS token if needed
        if self.eos_token_id is not None and answer[-1] != self.eos_token_id:
            answer = answer + [self.eos_token_id]
        
        # Combine question and answer
        sequence = question + answer
        
        # Truncate sequence if needed
        if len(sequence) > self.seq_len:
            sequence = sequence[:self.seq_len]
        
        # Pad sequence if needed
        if len(sequence) < self.seq_len:
            sequence = sequence + [self.pad_token_id] * (self.seq_len - len(sequence))
        
        # Convert to tensors
        input_ids = torch.tensor(sequence, dtype=torch.long)
        
        # Create labels (shift right)
        labels = torch.full_like(input_ids, -100)  # -100 is ignored in loss
        labels[len(question)-1:-1] = input_ids[len(question):]
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create dataloaders for training and validation.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Optional validation dataset
        batch_size: Batch size
        shuffle: Whether to shuffle the training data
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for faster data transfer
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Create training dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Create validation dataloader if needed
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    return train_dataloader, val_dataloader


def load_json_data(
    file_path: str,
    tokenizer: Optional[Callable] = None,
    max_examples: Optional[int] = None,
    shuffle: bool = True
) -> List[Dict[str, Any]]:
    """Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        tokenizer: Optional tokenizer function
        max_examples: Maximum number of examples to load
        shuffle: Whether to shuffle the data
        
    Returns:
        List of examples
    """
    # Load data
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Shuffle data if needed
    if shuffle:
        random.shuffle(data)
    
    # Limit number of examples if needed
    if max_examples is not None:
        data = data[:max_examples]
    
    # Tokenize data if needed
    if tokenizer is not None:
        for example in data:
            if 'text' in example:
                example['tokens'] = tokenizer(example['text'])
            if 'question' in example:
                example['question_tokens'] = tokenizer(example['question'])
            if 'answer' in example:
                example['answer_tokens'] = tokenizer(example['answer'])
    
    return data


def load_text_data(
    file_path: str,
    tokenizer: Callable,
    max_examples: Optional[int] = None,
    shuffle: bool = True
) -> List[List[int]]:
    """Load data from a text file.
    
    Args:
        file_path: Path to the text file
        tokenizer: Tokenizer function
        max_examples: Maximum number of examples to load
        shuffle: Whether to shuffle the data
        
    Returns:
        List of token sequences
    """
    # Load data
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Shuffle data if needed
    if shuffle:
        random.shuffle(lines)
    
    # Limit number of examples if needed
    if max_examples is not None:
        lines = lines[:max_examples]
    
    # Tokenize data
    data = [tokenizer(line.strip()) for line in lines]
    
    return data


def split_data(
    data: List[Any],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = True
) -> Tuple[List[Any], List[Any], List[Any]]:
    """Split data into training, validation, and test sets.
    
    Args:
        data: List of examples
        val_ratio: Ratio of validation examples
        test_ratio: Ratio of test examples
        shuffle: Whether to shuffle the data
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    # Shuffle data if needed
    if shuffle:
        data = data.copy()
        random.shuffle(data)
    
    # Calculate split indices
    val_idx = int(len(data) * (1 - val_ratio - test_ratio))
    test_idx = int(len(data) * (1 - test_ratio))
    
    # Split data
    train_data = data[:val_idx]
    val_data = data[val_idx:test_idx]
    test_data = data[test_idx:]
    
    return train_data, val_data, test_data
