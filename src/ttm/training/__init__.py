"""
Training utilities for the Token Turing Machine (TTM) model.
"""

from .optimizer import (
    create_optimizer,
    create_scheduler,
    get_grouped_parameters,
    get_parameter_names,
    get_ttm_optimizer
)

from .trainer import TTMTrainer

from .data import (
    SequenceDataset,
    CausalLanguageModelingDataset,
    MathDataset,
    create_dataloaders,
    load_json_data,
    load_text_data,
    split_data
)

from .train_step import (
    train_step,
    eval_step,
    get_example_predictions
)

from .train_loop import (
    train_loop,
    evaluate,
    save_checkpoint,
    load_checkpoint,
    train_with_curriculum
)

from .curriculum import (
    CurriculumDataset,
    create_curriculum_dataloaders,
    MultiplicationDataset
)

__all__ = [
    'create_optimizer',
    'create_scheduler',
    'get_grouped_parameters',
    'get_parameter_names',
    'get_ttm_optimizer',
    'TTMTrainer',
    'SequenceDataset',
    'CausalLanguageModelingDataset',
    'MathDataset',
    'create_dataloaders',
    'load_json_data',
    'load_text_data',
    'split_data',
    'train_step',
    'eval_step',
    'get_example_predictions',
    'train_loop',
    'evaluate',
    'save_checkpoint',
    'load_checkpoint',
    'train_with_curriculum',
    'CurriculumDataset',
    'create_curriculum_dataloaders',
    'MultiplicationDataset'
]
