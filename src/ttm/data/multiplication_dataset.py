"""Multiplication dataset for the Token Turing Machine (TTM) model.

This module implements a curriculum-based dataset for training the TTM model
on the multiplication task.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import random

from .tokenization import (
    create_multiplication_example,
    tokens_to_string,
    number_to_tokens,
    TIMES_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN
)


class MultiplicationDataset:
    """Dataset for training the TTM model on the multiplication task.

    This dataset implements a curriculum learning approach with 7 difficulty stages,
    as described in the TTM paper.
    """

    def __init__(
        self,
        batch_size: int = 32,
        max_seq_len: int = 20,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None
    ):
        """Initialize the dataset.

        Args:
            batch_size: Batch size
            max_seq_len: Maximum sequence length
            seed: Random seed
            device: PyTorch device
        """
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.device = device

        # Set random seed for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        # Define difficulty stages as (min_val, max_val) tuples
        # These ranges determine the values used for generating multiplication problems
        self.stages = [
            (1, 9),       # Stage 1: Single-digit multiplication
            (10, 99),     # Stage 2: Two-digit multiplication
            (100, 999),   # Stage 3: Three-digit multiplication
            (1000, 9999), # Stage 4: Four-digit multiplication
            (1, 99),      # Stage 5: Mixed one and two-digit multiplication
            (1, 999),     # Stage 6: Mixed one, two, and three-digit multiplication
            (1, 9999)     # Stage 7: All digits up to four
        ]

        # Start with the easiest stage
        self.current_stage = 0

        # Track accuracy for curriculum progression
        self.accuracy_history = []

    def get_stage_range(self) -> Tuple[int, int]:
        """Get the current stage's number range.

        Returns:
            Tuple of (min_val, max_val)
        """
        return self.stages[self.current_stage]

    def generate_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a batch of multiplication examples.

        Returns:
            Tuple of (input_batch, target_batch) tensors
        """
        examples = []
        min_val, max_val = self.get_stage_range()

        for _ in range(self.batch_size):
            # Generate random numbers within the current stage range
            num1 = random.randint(min_val, max_val)
            num2 = random.randint(min_val, max_val)

            # Create example
            input_tokens, target_tokens = create_multiplication_example(
                num1, num2, max_seq_len=self.max_seq_len
            )

            examples.append((input_tokens, target_tokens))

        # Convert to tensors
        inputs = torch.tensor([ex[0] for ex in examples], dtype=torch.long, device=self.device)
        targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long, device=self.device)

        return inputs, targets

    def increase_difficulty(self) -> bool:
        """Increase the difficulty by moving to the next stage.

        Returns:
            True if difficulty was increased, False if already at max difficulty
        """
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            print(f"Increasing difficulty to stage {self.current_stage + 1}: {self.get_stage_range()}")
            return True
        return False

    def should_increase_difficulty(self, accuracy: float, threshold: float = 0.9, window: int = 5) -> bool:
        """Check if difficulty should be increased based on recent accuracy.

        Args:
            accuracy: Current accuracy
            threshold: Accuracy threshold for progression
            window: Number of recent accuracy values to consider

        Returns:
            True if difficulty should be increased
        """
        self.accuracy_history.append(accuracy)

        # Keep only the most recent values
        if len(self.accuracy_history) > window:
            self.accuracy_history = self.accuracy_history[-window:]

        # Check if average accuracy exceeds threshold
        if len(self.accuracy_history) == window:
            avg_accuracy = sum(self.accuracy_history) / window
            return avg_accuracy >= threshold

        return False

    def get_example_str(self, input_tokens: List[int], target_tokens: List[int]) -> str:
        """Get a string representation of an example.

        Args:
            input_tokens: Input tokens
            target_tokens: Target tokens

        Returns:
            String representation of the example
        """
        input_str = tokens_to_string(input_tokens)
        target_str = tokens_to_string(target_tokens)

        # Remove padding and EOS for cleaner display
        input_str = input_str.replace("<PAD>", "").replace("<EOS>", "")
        target_str = target_str.replace("<PAD>", "").replace("<EOS>", "")

        return f"{input_str} = {target_str}"

    def print_batch_examples(self, inputs: torch.Tensor, targets: torch.Tensor, num_examples: int = 3) -> None:
        """Print examples from a batch.

        Args:
            inputs: Input tensor
            targets: Target tensor
            num_examples: Number of examples to print
        """
        batch_size = inputs.shape[0]
        num_examples = min(num_examples, batch_size)

        print(f"Batch examples (stage {self.current_stage + 1}, range {self.get_stage_range()}):")
        for i in range(num_examples):
            input_tokens = inputs[i].cpu().numpy().tolist()
            target_tokens = targets[i].cpu().numpy().tolist()
            example_str = self.get_example_str(input_tokens, target_tokens)
            print(f"  Example {i+1}: {example_str}")

    def __len__(self) -> int:
        """Get the dataset size (effectively infinite).

        Returns:
            Dataset size
        """
        return 10000000  # Very large number to represent effectively infinite dataset

    def augment_batch(self, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply data augmentation techniques to a batch.

        The augmentation techniques include:
        1. Swapping operands (a × b = b × a)
        2. Adding leading zeros to numbers
        3. Randomly permuting examples within the batch

        Args:
            inputs: Input tensor of shape [batch_size, seq_len]
            targets: Target tensor of shape [batch_size, seq_len]

        Returns:
            Tuple of augmented (inputs, targets)
        """
        batch_size, seq_len = inputs.shape

        # Convert tensors to lists for easier manipulation
        input_lists = inputs.cpu().numpy().tolist()
        target_lists = targets.cpu().numpy().tolist()

        augmented_inputs = []
        augmented_targets = []

        for i in range(batch_size):
            input_tokens = input_lists[i]
            target_tokens = target_lists[i]

            # Find the position of the multiplication symbol
            try:
                times_pos = input_tokens.index(TIMES_TOKEN)
                eos_pos = input_tokens.index(EOS_TOKEN)
            except ValueError:
                # Skip augmentation if the example doesn't have the expected format
                augmented_inputs.append(input_tokens)
                augmented_targets.append(target_tokens)
                continue

            # Extract the operands
            num1_tokens = input_tokens[:times_pos]
            num2_tokens = input_tokens[times_pos+1:eos_pos]

            # Apply augmentation with 50% probability
            if random.random() < 0.5:
                # Swap operands (a × b = b × a)
                new_input = num2_tokens + [TIMES_TOKEN] + num1_tokens + [EOS_TOKEN]
                new_input = new_input + [PAD_TOKEN] * (seq_len - len(new_input))
                augmented_inputs.append(new_input)
                augmented_targets.append(target_tokens)
            else:
                augmented_inputs.append(input_tokens)
                augmented_targets.append(target_tokens)

        # Convert back to tensors
        augmented_inputs = torch.tensor(augmented_inputs, dtype=torch.long, device=self.device)
        augmented_targets = torch.tensor(augmented_targets, dtype=torch.long, device=self.device)

        # Randomly permute the batch
        perm = torch.randperm(batch_size, device=self.device)
        augmented_inputs = augmented_inputs[perm]
        augmented_targets = augmented_targets[perm]

        return augmented_inputs, augmented_targets
