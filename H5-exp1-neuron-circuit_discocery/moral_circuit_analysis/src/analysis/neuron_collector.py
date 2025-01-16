import torch
import networkx as nx
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
from tqdm.auto import tqdm
from collections import defaultdict
from transformer_lens import HookedTransformer

@dataclass
class CoactivationPattern:
    neurons: Set[Tuple[int, int]]  # Set of (layer, neuron) pairs
    strength: float                # Co-activation strength
    frequency: int                 # How often this pattern occurs
    exemplars: List[str]          # Example texts triggering this pattern

class NeuronActivationCollector:
    def __init__(self, model: HookedTransformer,
                 activation_threshold: float = 0.7,
                 correlation_threshold: float = 0.6,
                 batch_size: int = 32):
        self.model = model
        self.device = model.device
        self.n_layers = model.cfg.n_layers
        self.n_neurons = model.cfg.d_mlp
        
        self.activation_threshold = activation_threshold
        self.correlation_threshold = correlation_threshold
        self.batch_size = batch_size
        self.neuron_graph = nx.Graph()
        
        print(f"Initialized collector with {self.n_layers} layers and {self.n_neurons} neurons per layer")
        
    def get_mlp_activations_with_tokens(self, text: str) -> Dict[Tuple[int, int], List[Tuple[str, float]]]:
        """Get activations for all neurons with corresponding tokens."""
        tokens = self.model.to_tokens(text)
        token_texts = [self.model.to_string(token) for token in tokens[0]]
        
        all_activations = defaultdict(list)
        
        _, cache = self.model.run_with_cache(tokens)
        
        for layer_idx in range(self.n_layers):
            mlp_acts = cache['post', layer_idx, 'mlp'].squeeze(0)
            
            for neuron_idx in range(self.n_neurons):
                neuron_activations = mlp_acts[:, neuron_idx].detach().cpu().numpy()
                token_activations = list(zip(token_texts, neuron_activations.tolist()))
                all_activations[(layer_idx, neuron_idx)].extend(token_activations)
        
        return dict(all_activations)

    def get_all_activations_batch(self, texts: List[str]) -> torch.Tensor:
        """Get activations for all neurons across all texts efficiently using batching."""
        all_activations = []
        
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Processing texts"):
            batch_texts = texts[i:i + self.batch_size]
            batch_tokens = [self.model.to_tokens(text) for text in batch_texts]
            
            # Pad sequences in batch
            max_len = max(tokens.shape[1] for tokens in batch_tokens)
            padded_tokens = torch.zeros((len(batch_texts), max_len), dtype=torch.long, device=self.device)
            for j, tokens in enumerate(batch_tokens):
                padded_tokens[j, :tokens.shape[1]] = tokens[0]
            
            # Get activations for batch
            with torch.no_grad():
                _, cache = self.model.run_with_cache(padded_tokens)
                
                batch_activations = []
                for layer_idx in range(self.n_layers):
                    mlp_acts = cache['post', layer_idx, 'mlp']
                    batch_activations.append(mlp_acts)
                
                batch_stack = torch.stack(batch_activations, dim=1)
                all_activations.append(batch_stack)
        
        return torch.cat(all_activations, dim=0)

    def compute_coactivation_matrix(self, activations: torch.Tensor) -> torch.Tensor:
        """Compute co-activation matrix across all neurons."""
        n_samples, n_layers, seq_len, n_neurons = activations.shape
        total_neurons = n_layers * n_neurons
        
        flat_activations = (activations
                          .permute(0, 2, 1, 3)
                          .reshape(-1, total_neurons))
        
        active_neurons = (flat_activations > self.activation_threshold).float()
        
        print("Computing correlation matrix...")
        coactivation_matrix = torch.zeros((total_neurons, total_neurons), device=self.device)
        
        try:
            chunk_size = min(1000, total_neurons // 2)
            for i in tqdm(range(0, total_neurons, chunk_size)):
                end_idx = min(i + chunk_size, total_neurons)
                chunk = active_neurons[:, i:end_idx].T @ active_neurons
                chunk /= active_neurons.shape[0]  # Normalize
                coactivation_matrix[i:end_idx] = chunk
                
            mask = torch.abs(coactivation_matrix) > self.correlation_threshold
            coactivation_matrix = coactivation_matrix * mask.float()
            coactivation_matrix = (coactivation_matrix + coactivation_matrix.t()) / 2
            
        except RuntimeError as e:
            print(f"GPU Error: {e}")
            print("Falling back to CPU computation...")
            active_neurons = active_neurons.cpu()
            coactivation_matrix = coactivation_matrix.cpu()
            
            for i in tqdm(range(0, total_neurons, chunk_size)):
                end_idx = min(i + chunk_size, total_neurons)
                chunk = active_neurons[:, i:end_idx].T @ active_neurons
                chunk /= active_neurons.shape[0]
                coactivation_matrix[i:end_idx] = chunk
            
            if self.device != "cpu":
                coactivation_matrix = coactivation_matrix.to(self.device)
        
        return coactivation_matrix 