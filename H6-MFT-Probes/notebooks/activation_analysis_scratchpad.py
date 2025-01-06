# %%
"""
Scratchpad for analyzing saved activation files.
"""

import logging
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# %%
def analyze_saved_activations(activation_dir: str = "../data/activations"):
    """Analyze the saved activation files to check what data we're storing.
    
    Args:
        activation_dir: Directory containing the saved activation files
    """
    activation_path = Path(activation_dir)
    
    # Analyze each split (train/val/test)
    for split in ['train', 'val', 'test']:
        split_path = activation_path / split
        if not split_path.exists():
            logger.warning(f"Split directory not found: {split_path}")
            continue
            
        logger.info(f"\nAnalyzing {split} split activations...")
        
        # Get all layer files
        layer_files = list(split_path.glob("*.pt"))
        logger.info(f"Found {len(layer_files)} layer files")
        
        # Analyze each layer file
        for layer_file in sorted(layer_files):
            logger.info(f"\nAnalyzing {layer_file.name}...")
            
            # Load data
            data = torch.load(layer_file)
            
            # Basic information about the data
            logger.info("\nData contents:")
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    logger.info(f"- {key}: tensor of shape {value.shape}")
                elif isinstance(value, list):
                    logger.info(f"- {key}: list of length {len(value)}")
                else:
                    logger.info(f"- {key}: {type(value)}")
            
            # Analyze activations
            activations = data['activations']
            logger.info(f"\nActivation analysis:")
            logger.info(f"Shape: {activations.shape}")
            logger.info(f"Type: {activations.dtype}")
            logger.info(f"Number of samples: {activations.shape[0]}")
            logger.info(f"Hidden dimension: {activations.shape[1]}")
            
            
            # Label distribution
            if 'labels' in data:
                label_counts = Counter(data['labels'])
                logger.info("\nLabel distribution:")
                for label, count in sorted(label_counts.items()):
                    logger.info(f"- Label {label}: {count} samples ({count/len(data['labels'])*100:.2f}%)")
            
            # Text examples
            if 'texts' in data:
                logger.info("\nText examples (first 3):")
                for i, text in enumerate(data['texts'][:3]):
                    logger.info(f"Sample {i}: {text[:100]}...")

            # Activation examples
            if 'activations' in data:
                logger.info("\nActivation examples (first 3):")
                activations = data['activations'][:3]
                logger.info(f"Activations shape: {activations.shape}")
                for i, activation in enumerate(activations):
                    logger.info(f"Activation {i} shape: {activation.shape}")
            
            # Memory cleanup
            del data
            torch.cuda.empty_cache()

# %%
if __name__ == "__main__":
    analyze_saved_activations() 
# %%
