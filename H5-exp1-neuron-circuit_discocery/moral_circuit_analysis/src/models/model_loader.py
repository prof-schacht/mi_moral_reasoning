import torch
from transformer_lens import HookedTransformer
from typing import Optional
from .model_config import ModelConfig

class ModelLoader:
    @staticmethod
    def load_model(config: Optional[ModelConfig] = None) -> HookedTransformer:
        """
        Load a transformer model based on the provided configuration.
        Falls back to default config if none provided.
        """
        if config is None:
            config = ModelConfig()
            
        # Set device based on CUDA availability
        if config.device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            config.device = "cpu"
            
        try:
            model = HookedTransformer.from_pretrained(
                config.model_name,
                device=config.device,
                dtype=getattr(torch, config.dtype),
                default_padding_side=config.default_padding_side,
                center_writing_weights=config.center_writing_weights,
                center_unembed=config.center_unembed,
                fold_ln=config.fold_ln,
                move_to_device=config.move_to_device
            )
            print(f"Successfully loaded model {config.model_name} on {config.device}")
            return model
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise 