"""
Model configuration and initialization utilities.
"""

from dataclasses import dataclass
from typing import Optional, Union
from transformers import PreTrainedTokenizer, AutoTokenizer
from nnsight import LanguageModel

@dataclass
class ModelConfig:
    """Configuration for model setup and activation extraction."""
    model_name: str = "google/gemma-2b-it"
    device_map: str = "cuda:0"  # or "auto" for automatic device mapping
    max_length: int = 256
    batch_size: int = 16
    cache_dir: str = "./cache"

class ModelManager:
    """Handles model and tokenizer initialization and management."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.model = None
        self.tokenizer = None
    
    def initialize_model(self) -> LanguageModel:
        """Initialize the language model with NNSight wrapper."""
        if self.model is None:
            self.model = LanguageModel(
                self.config.model_name,
                device_map=self.config.device_map
            )
        return self.model
    
    def get_tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer for the current model."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir
            )
        return self.tokenizer 