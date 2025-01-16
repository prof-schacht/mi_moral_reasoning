from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    model_name: str = 'google/gemma-2-9b-it'
    device: str = "cuda"  # Will fall back to CPU if CUDA not available
    dtype: str = "float32"
    default_padding_side: str = "right"
    center_writing_weights: bool = True
    center_unembed: bool = True
    fold_ln: bool = True
    move_to_device: bool = True
    
    # Optional API configurations
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    
    def to_dict(self):
        """Convert config to dictionary, excluding None values"""
        return {k: v for k, v in self.__dict__.items() if v is not None} 