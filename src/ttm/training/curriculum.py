"""
Curriculum learning for the Token Turing Machine (TTM) model.

This module provides utilities for curriculum learning with the TTM model.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
import numpy as np
import random

from .data import MathDataset, create_dataloaders


class CurriculumDataset(Dataset):
    """Dataset for curriculum learning."""
    
    def __init__(
        self,
        stage: int = 0,
        max_digits: int = 5,
        seq_len: int = 32,
        pad_token_id: int = 0,
        eos_token_id: Optional[int] = None,
        tokenizer: Optional[Callable] = None
    ):
        """Initialize the curriculum dataset.
        
        Args:
            stage: Current curriculum stage
            max_digits: Maximum number of digits
            seq_len: Maximum sequence length
            pad_token_id: Token ID for padding
            eos_token_id: Optional token ID for end-of-sequence
            tokenizer: Optional tokenizer for encoding/decoding
        """
        self.stage = stage
        self.max_digits = max_digits
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.tokenizer = tokenizer
        
        # Set stage-specific parameters
        self.set_stage(stage)
        
        # Generate examples
        self.examples = self.generate_examples()
    
    def set_stage(self, stage: int) -> None:
        """Set the current curriculum stage.
        
        Args:
            stage: Curriculum stage
        """
        self.stage = stage
        
        # Set stage-specific parameters
        if stage == 0:
            # Stage 0: Single-digit addition
            self.num_digits_a = 1
            self.num_digits_b = 1
            self.operation = '+'
        elif stage == 1:
            # Stage 1: Two-digit addition
            self.num_digits_a = 2
            self.num_digits_b = 2
            self.operation = '+'
        elif stage == 2:
            # Stage 2: Three-digit addition
            self.num_digits_a = 3
            self.num_digits_b = 3
            self.operation = '+'
        elif stage == 3:
            # Stage 3: Single-digit multiplication
            self.num_digits_a = 1
            self.num_digits_b = 1
            self.operation = '*'
        elif stage == 4:
            # Stage 4: Two-digit by one-digit multiplication
            self.num_digits_a = 2
            self.num_digits_b = 1
            self.operation = '*'
        elif stage == 5:
            # Stage 5: Two-digit multiplication
            self.num_digits_a = 2
            self.num_digits_b = 2
            self.operation = '*'
        else:
            # Default: Use maximum digits
            self.num_digits_a = min(stage + 1, self.max_digits)
            self.num_digits_b = min(stage + 1, self.max_digits)
            self.operation = '*' if stage >= 3 else '+'
    
    def generate_examples(self, num_examples: int = 1000) -> List[Dict[str, Any]]:
        """Generate examples for the current stage.
        
        Args:
            num_examples: Number of examples to generate
            
        Returns:
            List of examples
        """
        examples = []
        
        for _ in range(num_examples):
            # Generate random numbers
            a = random.randint(10 ** (self.num_digits_a - 1), 10 ** self.num_digits_a - 1)
            b = random.randint(10 ** (self.num_digits_b - 1), 10 ** self.num_digits_b - 1)
            
            # Calculate result
            if self.operation == '+':
                result = a + b
            elif self.operation == '*':
                result = a * b
            else:
                raise ValueError(f"Unknown operation: {self.operation}")
            
            # Create question and answer
            question = f"{a} {self.operation} {b} = "
            answer = str(result)
            
            # Tokenize if tokenizer is provided
            if self.tokenizer is not None:
                question_tokens = self.tokenizer.encode(question)
                answer_tokens = self.tokenizer.encode(answer)
                
                # Add EOS token if needed
                if self.eos_token_id is not None and answer_tokens[-1] != self.eos_token_id:
                    answer_tokens.append(self.eos_token_id)
                
                example = {
                    'question': question,
                    'answer': answer,
                    'question_tokens': question_tokens,
                    'answer_tokens': answer_tokens
                }
            else:
                # Use character-level tokenization
                question_tokens = [ord(c) for c in question]
                answer_tokens = [ord(c) for c in answer]
                
                # Add EOS token if needed
                if self.eos_token_id is not None and answer_tokens[-1] != self.eos_token_id:
                    answer_tokens.append(self.eos_token_id)
                
                example = {
                    'question': question,
                    'answer': answer,
                    'question_tokens': question_tokens,
                    'answer_tokens': answer_tokens
                }
            
            examples.append(example)
        
        return examples
    
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
        example = self.examples[idx]
        
        # Get question and answer tokens
        question_tokens = example['question_tokens']
        answer_tokens = example['answer_tokens']
        
        # Combine question and answer
        sequence = question_tokens + answer_tokens
        
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
        labels[len(question_tokens)-1:-1] = input_ids[len(question_tokens):]
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }


def create_curriculum_dataloaders(
    stage: int,
    batch_size: int = 32,
    max_digits: int = 5,
    seq_len: int = 32,
    pad_token_id: int = 0,
    eos_token_id: Optional[int] = None,
    tokenizer: Optional[Callable] = None,
    train_ratio: float = 0.8,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """Create dataloaders for curriculum learning.
    
    Args:
        stage: Current curriculum stage
        batch_size: Batch size
        max_digits: Maximum number of digits
        seq_len: Maximum sequence length
        pad_token_id: Token ID for padding
        eos_token_id: Optional token ID for end-of-sequence
        tokenizer: Optional tokenizer for encoding/decoding
        train_ratio: Ratio of training examples
        shuffle: Whether to shuffle the data
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for faster data transfer
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Create dataset
    dataset = CurriculumDataset(
        stage=stage,
        max_digits=max_digits,
        seq_len=seq_len,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        tokenizer=tokenizer
    )
    
    # Split dataset
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size]
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_dataloader, val_dataloader


class MultiplicationDataset(Dataset):
    """Dataset for multiplication tasks."""
    
    def __init__(
        self,
        num_digits_a: int = 2,
        num_digits_b: int = 2,
        seq_len: int = 32,
        pad_token_id: int = 0,
        eos_token_id: Optional[int] = None,
        tokenizer: Optional[Callable] = None,
        num_examples: int = 1000
    ):
        """Initialize the multiplication dataset.
        
        Args:
            num_digits_a: Number of digits for the first number
            num_digits_b: Number of digits for the second number
            seq_len: Maximum sequence length
            pad_token_id: Token ID for padding
            eos_token_id: Optional token ID for end-of-sequence
            tokenizer: Optional tokenizer for encoding/decoding
            num_examples: Number of examples to generate
        """
        self.num_digits_a = num_digits_a
        self.num_digits_b = num_digits_b
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.tokenizer = tokenizer
        
        # Generate examples
        self.examples = self.generate_examples(num_examples)
    
    def generate_examples(self, num_examples: int) -> List[Dict[str, Any]]:
        """Generate multiplication examples.
        
        Args:
            num_examples: Number of examples to generate
            
        Returns:
            List of examples
        """
        examples = []
        
        for _ in range(num_examples):
            # Generate random numbers
            a = random.randint(10 ** (self.num_digits_a - 1), 10 ** self.num_digits_a - 1)
            b = random.randint(10 ** (self.num_digits_b - 1), 10 ** self.num_digits_b - 1)
            
            # Calculate result
            result = a * b
            
            # Create question and answer
            question = f"{a} * {b} = "
            answer = str(result)
            
            # Tokenize if tokenizer is provided
            if self.tokenizer is not None:
                question_tokens = self.tokenizer.encode(question)
                answer_tokens = self.tokenizer.encode(answer)
                
                # Add EOS token if needed
                if self.eos_token_id is not None and answer_tokens[-1] != self.eos_token_id:
                    answer_tokens.append(self.eos_token_id)
                
                example = {
                    'question': question,
                    'answer': answer,
                    'question_tokens': question_tokens,
                    'answer_tokens': answer_tokens
                }
            else:
                # Use character-level tokenization
                question_tokens = [ord(c) for c in question]
                answer_tokens = [ord(c) for c in answer]
                
                # Add EOS token if needed
                if self.eos_token_id is not None and answer_tokens[-1] != self.eos_token_id:
                    answer_tokens.append(self.eos_token_id)
                
                example = {
                    'question': question,
                    'answer': answer,
                    'question_tokens': question_tokens,
                    'answer_tokens': answer_tokens
                }
            
            examples.append(example)
        
        return examples
    
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
        example = self.examples[idx]
        
        # Get question and answer tokens
        question_tokens = example['question_tokens']
        answer_tokens = example['answer_tokens']
        
        # Combine question and answer
        sequence = question_tokens + answer_tokens
        
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
        labels[len(question_tokens)-1:-1] = input_ids[len(question_tokens):]
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }
