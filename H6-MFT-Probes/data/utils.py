"""
Utility functions for MFRC dataset processing and model operations.
"""

import logging
from typing import Tuple, Dict, Optional
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
from nnsight import LanguageModel

from .MFRCDataProcessingPipeline import MFRCConfig, prepare_mfrc_data
from .model_config import ModelConfig, ModelManager

logger = logging.getLogger(__name__)

def initialize_model_and_dataset(
    model_config: Optional[ModelConfig] = None,
    mfrc_config: Optional[MFRCConfig] = None
) -> Tuple[LanguageModel, DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Initialize both model and dataset with the given configurations.
    
    Args:
        model_config: Configuration for the model
        mfrc_config: Configuration for the dataset
        
    Returns:
        Tuple containing:
        - model: Initialized NNSight model
        - train_loader: DataLoader for training data
        - val_loader: DataLoader for validation data
        - test_loader: DataLoader for test data
        - label_mapping: Dictionary mapping labels to indices
    """
    # Initialize model manager
    model_manager = ModelManager(model_config)
    
    # Get model and tokenizer
    model = model_manager.initialize_model()
    tokenizer = model_manager.get_tokenizer()
    
    # Initialize dataset
    train_loader, val_loader, test_loader, label_mapping = initialize_mfrc_dataset(
        tokenizer=tokenizer,
        config=mfrc_config
    )
    
    return model, train_loader, val_loader, test_loader, label_mapping

def initialize_mfrc_dataset(
    tokenizer: PreTrainedTokenizer,
    config: Optional[MFRCConfig] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Initialize and prepare MFRC dataset with the given configuration.
    
    Args:
        tokenizer: The tokenizer to use for text processing
        config: Optional configuration for dataset preparation
        
    Returns:
        Tuple containing:
        - train_loader: DataLoader for training data
        - val_loader: DataLoader for validation data
        - test_loader: DataLoader for test data
        - label_mapping: Dictionary mapping labels to indices
    """
    logger.info("Initializing MFRC dataset...")
    
    if config is None:
        config = MFRCConfig()
    
    try:
        # Load and prepare the dataset
        train_loader, val_loader, test_loader, label_mapping = prepare_mfrc_data(
            tokenizer=tokenizer,
            config=config
        )
        
        # Log dataset information
        logger.info("\nDataset Information:")
        logger.info(f"Number of unique classes: {len(label_mapping)}")
        logger.info("\nLabel mapping details:")
        for label, idx in sorted(label_mapping.items()):
            logger.info(f"  - {label} (index {idx})")
        
        logger.info("\nDataloader sizes:")
        logger.info(f"Training batches: {len(train_loader)} ({len(train_loader.dataset)} samples)")
        logger.info(f"Validation batches: {len(val_loader)} ({len(val_loader.dataset)} samples)")
        logger.info(f"Test batches: {len(test_loader)} ({len(test_loader.dataset)} samples)")
        
        return train_loader, val_loader, test_loader, label_mapping
        
    except Exception as e:
        logger.error(f"Error initializing MFRC dataset: {str(e)}")
        raise 