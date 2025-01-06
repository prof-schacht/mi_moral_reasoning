# %%
import logging
from pathlib import Path
from data.model_config import ModelConfig
from data.MFRCDataProcessingPipeline import MFRCConfig
from data.utils import initialize_model_and_dataset
from models.activation_extractor import ActivationExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# %%
def main():
    # 1. Create configurations
    model_config = ModelConfig(
        model_name="google/gemma-2-9b-it",
        device_map="cuda:0",
        max_length=768,
        batch_size=1
    )
    
    mfrc_config = MFRCConfig(
        max_length=768,
        batch_size=1,
        num_workers=4,
        test_size=0.1,
        val_size=0.1,
        random_state=42,
        cache_dir='./cache',
        min_samples_per_class=2
    )
    
    # 2. Initialize model and dataset
    model, train_loader, val_loader, test_loader, label_mapping = initialize_model_and_dataset(
        model_config=model_config,
        mfrc_config=mfrc_config
    )
    
    # 3. Initialize activation extractor
    extractor = ActivationExtractor(model)
    
    # 4. Create directory for saving activations
    save_dir = Path("./data/activations")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 5. Extract and save activations for each split
    logger.info("Extracting activations from training set...")
    extractor.process_dataset(train_loader, str(save_dir / "train"))
    
    logger.info("Extracting activations from validation set...")
    extractor.process_dataset(val_loader, str(save_dir / "val"))
    
    logger.info("Extracting activations from test set...")
    extractor.process_dataset(test_loader, str(save_dir / "test"))
    
    return model, train_loader, val_loader, test_loader, label_mapping

# %%
if __name__ == "__main__":
    model, train_loader, val_loader, test_loader, label_mapping = main()

# %%
