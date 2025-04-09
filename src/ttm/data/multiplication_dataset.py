"""Multiplication dataset for TTM.

This module implements a curriculum-based multiplication dataset for training TTM.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, List, Dict, Any


class MultiplicationDataset:
    """Curriculum-based multiplication dataset.
    
    This dataset generates multiplication problems with increasing difficulty.
    """
    
    def __init__(self, batch_size: int = 32, seq_len: int = 12):
        """Initialize the dataset.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
        """
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = 13  # 0-9 digits + '×' + EOS + padding
        self.current_stage = 0
        self.EOS_TOKEN = 11   # End of sequence token
        self.PAD_TOKEN = 12   # Padding token
        self.TIMES_TOKEN = 10 # Multiplication symbol
        self.stages = [
            (1, 2),     # Stage 0: Very simple multiplication (1-2)
            (1, 3),     # Stage 1: Simple multiplication (1-3)
            (1, 5),     # Stage 2: Basic multiplication tables (1-5)
            (1, 9),     # Stage 3: Full single digit multiplication
            (10, 20),   # Stage 4: Learning to handle carries
            (20, 50),   # Stage 5: Two digit by one digit
            (50, 99)    # Stage 6: Two digit by two digit
        ]
    
    def generate_batch(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate a batch of multiplication problems.
        
        Returns:
            Tuple of (inputs, targets) where:
                inputs: Input tokens of shape [batch_size, seq_len]
                targets: Target tokens of shape [batch_size, seq_len]
        """
        inputs = np.full((self.batch_size, self.seq_len), self.PAD_TOKEN, dtype=np.int32)
        targets = np.full((self.batch_size, self.seq_len), self.PAD_TOKEN, dtype=np.int32)
        
        min_val, max_val = self.stages[self.current_stage]
        
        for i in range(self.batch_size):
            # Generate numbers based on current stage
            if self.current_stage <= 3:  # Single digit stages
                num1 = np.random.randint(min_val, max_val + 1)
                num2 = np.random.randint(min_val, max_val + 1)
            else:  # Multi-digit stages
                num1 = np.random.randint(min_val, max_val + 1)
                num2 = np.random.randint(1, 10)  # Keep second number single digit initially
            
            result = num1 * num2
            
            # Convert to digit sequences
            num1_seq = [int(d) for d in str(num1)]
            num2_seq = [int(d) for d in str(num2)]
            result_seq = [int(d) for d in str(result)]
            
            # Create input sequence: num1 × num2 EOS PAD...
            input_seq = num1_seq + [self.TIMES_TOKEN] + num2_seq + [self.EOS_TOKEN]
            input_seq.extend([self.PAD_TOKEN] * (self.seq_len - len(input_seq)))
            
            # Create target sequence: result EOS PAD...
            target_seq = result_seq + [self.EOS_TOKEN]
            target_seq.extend([self.PAD_TOKEN] * (self.seq_len - len(target_seq)))
            
            # Set sequences
            inputs[i] = input_seq
            targets[i] = target_seq
        
        return jnp.array(inputs), jnp.array(targets)
    
    def increase_difficulty(self) -> bool:
        """Progress to the next stage of difficulty.
        
        Returns:
            True if successfully progressed, False if already at max difficulty
        """
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            return True
        return False
    
    def augment_batch(self, inputs: jnp.ndarray, targets: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply data augmentation to a batch.
        
        This implements the data augmentation techniques mentioned in the TTM paper.
        
        Args:
            inputs: Input tokens of shape [batch_size, seq_len]
            targets: Target tokens of shape [batch_size, seq_len]
            
        Returns:
            Tuple of (augmented_inputs, augmented_targets)
        """
        # Convert to numpy for easier manipulation
        inputs_np = np.array(inputs)
        targets_np = np.array(targets)
        
        # Apply random permutation to batch
        perm = np.random.permutation(len(inputs_np))
        inputs_np = inputs_np[perm]
        targets_np = targets_np[perm]
        
        # Apply mixup augmentation to a portion of the batch
        mixup_portion = 0.2
        mixup_count = int(mixup_portion * len(inputs_np))
        
        if mixup_count > 1:
            # Select indices for mixup
            mixup_indices = np.random.choice(len(inputs_np), mixup_count, replace=False)
            
            # Create mixed examples
            for idx in mixup_indices:
                # Select another random example
                other_idx = np.random.choice(len(inputs_np))
                while other_idx == idx:
                    other_idx = np.random.choice(len(inputs_np))
                
                # Mix inputs (keep structure intact)
                # Find the multiplication symbol in both sequences
                times_pos1 = np.where(inputs_np[idx] == self.TIMES_TOKEN)[0][0]
                times_pos2 = np.where(inputs_np[other_idx] == self.TIMES_TOKEN)[0][0]
                
                # Swap the first number
                inputs_np[idx, :times_pos1] = inputs_np[other_idx, :times_pos2]
                
                # Recompute the target
                # Extract the numbers
                num1 = int(''.join([str(d) for d in inputs_np[idx, :times_pos1]]))
                
                # Find the EOS token
                eos_pos = np.where(inputs_np[idx] == self.EOS_TOKEN)[0][0]
                num2 = int(''.join([str(d) for d in inputs_np[idx, times_pos1+1:eos_pos]]))
                
                # Compute result
                result = num1 * num2
                result_seq = [int(d) for d in str(result)]
                
                # Create new target sequence
                target_seq = np.full(self.seq_len, self.PAD_TOKEN, dtype=np.int32)
                target_seq[:len(result_seq)] = result_seq
                target_seq[len(result_seq)] = self.EOS_TOKEN
                
                targets_np[idx] = target_seq
        
        return jnp.array(inputs_np), jnp.array(targets_np)
