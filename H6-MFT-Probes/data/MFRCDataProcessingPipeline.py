"""
MFRC Dataset Preparation Module
Handles loading, preprocessing, and batching of the MFRC dataset for LLM probing studies.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MFRCConfig:
    """Configuration for MFRC dataset preparation."""
    max_length: int = 256
    batch_size: int = 16
    num_workers: int = 4
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    cache_dir: str = './cache'
    min_samples_per_class: int = 2

class MFRCDataset(Dataset):
    """Custom Dataset for MFRC data handling."""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512
    ):
        """Initialize dataset with texts and labels."""
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get tokenized text and label for an index."""
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class MFRCProcessor:
    """Handles MFRC dataset processing and preparation."""
    

    def __init__(self, config: MFRCConfig):
        """Initialize processor with configuration."""
        self.config = config
        self.label_mapping = None
        self.reverse_mapping = None
        
    def is_single_foundation(self, label: str) -> bool:
        """Check if the label represents a single moral foundation."""
        return ',' not in label

    def load_dataset(self) -> pd.DataFrame:
        """Load MFRC dataset from Hugging Face and filter for Everyday Morality bucket."""
        logger.info("Loading MFRC dataset...")
        dataset = load_dataset("USC-MOLA-Lab/MFRC", cache_dir=self.config.cache_dir)
        df = pd.DataFrame(dataset['train'])
        
        # Filter for Everyday Morality bucket
        everyday_morality_mask = df['bucket'] == 'Everyday Morality'
        df = df[everyday_morality_mask].copy()
        
        # Filter for single-label instances
        single_label_mask = df['annotation'].apply(self.is_single_foundation)
        df_single = df[single_label_mask].copy()
        df_multi = df[~single_label_mask].copy()
        
        # Log dataset statistics
        total_samples = len(df)
        single_label_samples = len(df_single)
        multi_label_samples = len(df_multi)
        
        logger.info("\nDataset Statistics (Everyday Morality bucket):")
        logger.info(f"Total samples: {total_samples}")
        logger.info(f"Single-label samples: {single_label_samples} ({single_label_samples/total_samples*100:.2f}%)")
        logger.info(f"Multi-label samples: {multi_label_samples} ({multi_label_samples/total_samples*100:.2f}%)")
        
        # Log single-label distribution
        unique_labels = df_single['annotation'].unique()
        logger.info(f"\nSingle-label distribution ({len(unique_labels)} categories):")
        for label in sorted(unique_labels):
            count = len(df_single[df_single['annotation'] == label])
            logger.info(f"  - {label}: {count} samples")
        
        # Log multi-label combinations (for information)
        logger.info(f"\nMulti-label combinations (excluded from dataset):")
        multi_label_counts = df_multi['annotation'].value_counts()
        for label, count in multi_label_counts.items():
            logger.info(f"  - {label}: {count} samples")
        
        return df_single

    def create_label_mapping(self, labels: List[str]) -> Dict[str, int]:
        """Create a mapping from text labels to numeric indices."""
        # Count occurrences of each label
        label_counts = Counter(labels)
        
        # Log original distribution
        logger.info("\nLabel distribution:")
        for label, count in sorted(label_counts.items()):
            logger.info(f"  - {label}: {count} samples")
        
        # Filter labels that have at least min_samples_per_class samples
        valid_labels = [
            label for label, count in label_counts.items() 
            if count >= self.config.min_samples_per_class
        ]
        
        if not valid_labels:
            raise ValueError(
                f"No labels have at least {self.config.min_samples_per_class} samples!"
            )
        
        # Create and store the mapping
        self.label_mapping = {label: idx for idx, label in enumerate(sorted(valid_labels))}
        self.reverse_mapping = {idx: label for label, idx in self.label_mapping.items()}
        
        logger.info("\nLabel mapping:")
        for label, idx in sorted(self.label_mapping.items()):
            logger.info(f"  - {label} -> {idx}")
        
        return self.label_mapping

    def prepare_splits(self, df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[int], List[int], List[int]]:
        """
        Prepare train/val/test splits while handling classes with few samples.
        """
        # Filter the dataframe to only include rows with valid labels
        valid_df = df[df['annotation'].isin(self.label_mapping.keys())].copy()
        
        if len(valid_df) == 0:
            raise ValueError("No valid samples remaining after filtering!")
        
        texts = valid_df['text'].tolist()
        labels = [self.label_mapping[label] for label in valid_df['annotation']]
        
        # Log data statistics
        logger.info(f"Total samples after filtering: {len(texts)}")
        label_counts = Counter(labels)
        logger.info(f"Class distribution after filtering: {dict(label_counts)}")
        
        try:
            # Attempt stratified split
            train_texts, test_texts, train_labels, test_labels = train_test_split(
                texts, labels,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=labels
            )
            
            # Split remaining data into train/val
            val_size_adjusted = self.config.val_size / (1 - self.config.test_size)
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                train_texts, train_labels,
                test_size=val_size_adjusted,
                random_state=self.config.random_state,
                stratify=train_labels
            )
            
        except ValueError as e:
            logger.warning(f"Stratified split failed: {str(e)}")
            logger.warning("Falling back to random split")
            
            # Fallback to random split
            train_texts, test_texts, train_labels, test_labels = train_test_split(
                texts, labels,
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )
            
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                train_texts, train_labels,
                test_size=val_size_adjusted,
                random_state=self.config.random_state
            )
        
        # Log split sizes
        logger.info(f"Split sizes:")
        logger.info(f"Train: {len(train_texts)}")
        logger.info(f"Val: {len(val_texts)}")
        logger.info(f"Test: {len(test_texts)}")
        
        return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels

    def create_dataloaders(
        self,
        train_texts: List[str],
        val_texts: List[str],
        test_texts: List[str],
        train_labels: List[int],
        val_labels: List[int],
        test_labels: List[int],
        tokenizer: PreTrainedTokenizer
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create DataLoaders for train/val/test sets."""
        train_dataset = MFRCDataset(train_texts, train_labels, tokenizer, self.config.max_length)
        val_dataset = MFRCDataset(val_texts, val_labels, tokenizer, self.config.max_length)
        test_dataset = MFRCDataset(test_texts, test_labels, tokenizer, self.config.max_length)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )

        return train_loader, val_loader, test_loader

def prepare_mfrc_data(
    tokenizer: PreTrainedTokenizer,
    config: Optional[MFRCConfig] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """Main function to prepare MFRC dataset."""
    if config is None:
        config = MFRCConfig()

    processor = MFRCProcessor(config)
    
    try:
        # Load and process data
        df = processor.load_dataset()
        logger.info(f"Loaded dataset with {len(df)} samples")
        
        # Create label mapping (this will filter out invalid classes)
        label_mapping = processor.create_label_mapping(df['annotation'])
        logger.info(f"Created label mapping with {len(label_mapping)} classes")
        
        # Create splits
        train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = \
            processor.prepare_splits(df)
        
        # Create dataloaders
        train_loader, val_loader, test_loader = processor.create_dataloaders(
            train_texts, val_texts, test_texts,
            train_labels, val_labels, test_labels,
            tokenizer
        )
        
        logger.info(f"Successfully created dataloaders")
        logger.info(f"Training batches: {len(train_loader)}")
        logger.info(f"Validation batches: {len(val_loader)}")
        logger.info(f"Test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader, label_mapping
        
    except Exception as e:
        logger.error(f"Error preparing MFRC data: {str(e)}")
        raise