import pickle
from typing import List, Tuple, Optional
import os
import json

def load_moral_pairs(file_path: str) -> List[Tuple[str, str]]:
    """Load moral/immoral text pairs from a file."""
    if file_path.endswith('.pkl'):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)
            return [(pair['moral'], pair['immoral']) for pair in data]
    else:
        raise ValueError("Unsupported file format. Use .pkl or .json")

def save_results(results: dict, file_path: str) -> None:
    """Save analysis results to a file."""
    with open(file_path, 'wb') as f:
        pickle.dump(results, f)

def load_results(file_path: str) -> Optional[dict]:
    """Load previously saved analysis results."""
    if not os.path.exists(file_path):
        return None
        
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def load_validation_texts(file_path: str) -> List[str]:
    """Load validation texts for neuron analysis."""
    if file_path.endswith('.txt'):
        with open(file_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data['texts'] if isinstance(data, dict) else data
    else:
        raise ValueError("Unsupported file format. Use .txt or .json") 