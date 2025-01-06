# %%
"""
Script to train logistic probes for each moral foundation category.
"""

import logging
from pathlib import Path
from models.probe_trainer import ProbeTrainer
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Train logistic probes for all layers and moral foundation categories."""
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    if device == "cuda":
        # Log GPU info
        logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        logger.info(f"Memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    # Initialize trainer with optimized parameters for logistic regression
    trainer = ProbeTrainer(
        activation_dir="data/activations",
        probe_dir="data/probes",
        num_classes=8,  # Number of moral foundations
        batch_size=256,  # Large batch size for stable gradients
        learning_rate=0.001,  # Standard learning rate for logistic regression
        num_epochs=50,  # Sufficient epochs with early stopping
        device=device
    )
    
    # Train logistic probes for all layers
    trainer.train_all_probes()
    
    # Final GPU cleanup
    if device == "cuda":
        torch.cuda.empty_cache()
        logger.info("Cleared CUDA cache")

if __name__ == "__main__":
    main() 
# %%

