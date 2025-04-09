"""
Tokenization module for the Token Turing Machine (TTM) model.

This module defines the vocabulary and tokenization functions for the TTM model
focused on the multiplication task.
"""

import torch
import numpy as np
from typing import List, Tuple, Union

# Define vocabulary constants
VOCAB_SIZE = 13
DIGIT_TOKENS = list(range(10))  # 0-9 for digits
TIMES_TOKEN = 10  # Multiplication symbol
EOS_TOKEN = 11    # End of sequence
PAD_TOKEN = 12    # Padding

# Display symbols for special tokens
TIMES_SYMBOL = "×"
EOS_SYMBOL = "<EOS>"
PAD_SYMBOL = "<PAD>"

def number_to_tokens(number: int) -> List[int]:
    """Convert a number to a sequence of digit tokens.

    Args:
        number: Integer to convert

    Returns:
        List of digit tokens representing the number
    """
    if number == 0:
        return [0]

    tokens = []
    while number > 0:
        tokens.append(number % 10)
        number //= 10

    # Reverse to get correct order (most significant digit first)
    return tokens[::-1]

def tokens_to_number(tokens: List[int]) -> int:
    """Convert a sequence of digit tokens to a number.

    Args:
        tokens: List of digit tokens

    Returns:
        Integer represented by the tokens
    """
    number = 0
    for token in tokens:
        if token not in DIGIT_TOKENS:
            raise ValueError(f"Invalid digit token: {token}")
        number = number * 10 + token
    return number

def tokens_to_string(tokens: List[int]) -> str:
    """Convert a sequence of tokens to a human-readable string.

    Args:
        tokens: List of tokens

    Returns:
        String representation of the tokens
    """
    result = []
    for token in tokens:
        if token in DIGIT_TOKENS:
            result.append(str(token))
        elif token == TIMES_TOKEN:
            result.append(TIMES_SYMBOL)
        elif token == EOS_TOKEN:
            result.append(EOS_SYMBOL)
        elif token == PAD_TOKEN:
            result.append(PAD_SYMBOL)
        else:
            raise ValueError(f"Invalid token: {token}")

    return "".join(result)

def create_multiplication_example(num1: int, num2: int, max_seq_len: int = 20) -> Tuple[List[int], List[int]]:
    """Create a multiplication example with input and target sequences.

    Args:
        num1: First number in multiplication
        num2: Second number in multiplication
        max_seq_len: Maximum sequence length

    Returns:
        Tuple of (input_tokens, target_tokens)
    """
    # Convert numbers to tokens
    num1_tokens = number_to_tokens(num1)
    num2_tokens = number_to_tokens(num2)

    # Create input sequence: num1 × num2 EOS PAD...
    input_tokens = num1_tokens + [TIMES_TOKEN] + num2_tokens + [EOS_TOKEN]

    # Create target sequence: result EOS PAD...
    result = num1 * num2
    result_tokens = number_to_tokens(result) + [EOS_TOKEN]

    # Pad sequences to max_seq_len
    input_tokens = pad_sequence(input_tokens, max_seq_len)
    target_tokens = pad_sequence(result_tokens, max_seq_len)

    return input_tokens, target_tokens

def pad_sequence(tokens: List[int], max_len: int) -> List[int]:
    """Pad a token sequence to the specified length.

    Args:
        tokens: List of tokens
        max_len: Maximum sequence length

    Returns:
        Padded sequence
    """
    if len(tokens) > max_len:
        return tokens[:max_len]

    return tokens + [PAD_TOKEN] * (max_len - len(tokens))

def create_tensor_from_tokens(tokens: List[int], device: torch.device = None) -> torch.Tensor:
    """Convert a list of tokens to a PyTorch tensor.

    Args:
        tokens: List of tokens
        device: PyTorch device

    Returns:
        PyTorch tensor
    """
    return torch.tensor(tokens, dtype=torch.long, device=device)

def create_batch_from_examples(examples: List[Tuple[List[int], List[int]]], device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a batch from a list of examples.

    Args:
        examples: List of (input_tokens, target_tokens) tuples
        device: PyTorch device

    Returns:
        Tuple of (input_batch, target_batch) tensors
    """
    inputs = [ex[0] for ex in examples]
    targets = [ex[1] for ex in examples]

    input_batch = torch.tensor(inputs, dtype=torch.long, device=device)
    target_batch = torch.tensor(targets, dtype=torch.long, device=device)

    return input_batch, target_batch

def get_token_name(token: int) -> str:
    """Get the name of a token.

    Args:
        token: Token ID

    Returns:
        Name of the token
    """
    if token in DIGIT_TOKENS:
        return str(token)
    elif token == TIMES_TOKEN:
        return "TIMES"
    elif token == EOS_TOKEN:
        return "EOS"
    elif token == PAD_TOKEN:
        return "PAD"
    else:
        raise ValueError(f"Invalid token: {token}")

def print_example(input_tokens: List[int], target_tokens: List[int]) -> None:
    """Print a multiplication example in a human-readable format.

    Args:
        input_tokens: Input sequence tokens
        target_tokens: Target sequence tokens
    """
    # Extract numbers and result from tokens
    input_str = tokens_to_string(input_tokens)
    target_str = tokens_to_string(target_tokens)

    # Remove padding and EOS for cleaner display
    input_str = input_str.replace(PAD_SYMBOL, "").replace(EOS_SYMBOL, "")
    target_str = target_str.replace(PAD_SYMBOL, "").replace(EOS_SYMBOL, "")

    print(f"Input:  {input_str}")
    print(f"Target: {target_str}")

    # Verify the calculation
    parts = input_str.split(TIMES_SYMBOL)
    if len(parts) == 2:
        num1 = int(parts[0])
        num2 = int(parts[1])
        expected = num1 * num2
        actual = int(target_str)
        assert expected == actual, f"Calculation error: {num1} × {num2} = {expected}, but got {actual}"


def tokens_to_labels(token_indices: Union[List[int], torch.Tensor, np.ndarray]) -> List[str]:
    """Convert token indices to human-readable labels.

    Args:
        token_indices: List of token indices

    Returns:
        List of token labels
    """
    # Convert to list if it's a tensor or numpy array
    if isinstance(token_indices, torch.Tensor):
        token_indices = token_indices.cpu().numpy().tolist()
    elif isinstance(token_indices, np.ndarray):
        token_indices = token_indices.tolist()

    # Convert each token to its string representation
    labels = []
    for token in token_indices:
        if token in DIGIT_TOKENS:
            labels.append(str(token))
        elif token == TIMES_TOKEN:
            labels.append(TIMES_SYMBOL)
        elif token == EOS_TOKEN:
            labels.append(EOS_SYMBOL)
        elif token == PAD_TOKEN:
            labels.append(PAD_SYMBOL)
        else:
            labels.append(f"<UNK:{token}>")

    return labels
