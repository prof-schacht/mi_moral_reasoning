import torch
import numpy as np
from typing import List, Tuple, Dict
from scipy.stats import pearsonr

def calculate_neuron_consistency(activations: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """
    Calculate how consistently each neuron activates above threshold.
    
    Args:
        activations: Tensor of shape [n_samples, n_layers, n_neurons]
        threshold: Activation threshold
        
    Returns:
        Tensor of shape [n_layers, n_neurons] with consistency scores
    """
    return (activations > threshold).float().mean(dim=0)

def calculate_layer_importance(activation_diffs: torch.Tensor) -> List[Tuple[int, float]]:
    """
    Calculate importance score for each layer based on activation differences.
    
    Args:
        activation_diffs: Tensor of shape [n_layers, n_neurons] with mean activation differences
        
    Returns:
        List of (layer_idx, importance_score) tuples, sorted by importance
    """
    layer_importance = []
    for layer in range(activation_diffs.shape[0]):
        importance = torch.abs(activation_diffs[layer]).mean().item()
        layer_importance.append((layer, importance))
    
    return sorted(layer_importance, key=lambda x: x[1], reverse=True)

def calculate_neuron_correlations(activations: torch.Tensor) -> torch.Tensor:
    """
    Calculate correlation matrix between neurons.
    
    Args:
        activations: Tensor of shape [n_samples, n_total_neurons]
        
    Returns:
        Correlation matrix of shape [n_total_neurons, n_total_neurons]
    """
    n_samples, n_neurons = activations.shape
    correlations = torch.zeros((n_neurons, n_neurons), device=activations.device)
    
    # Calculate correlations in chunks to avoid memory issues
    chunk_size = 1000
    for i in range(0, n_neurons, chunk_size):
        end_i = min(i + chunk_size, n_neurons)
        chunk_i = activations[:, i:end_i].cpu().numpy()
        
        for j in range(0, n_neurons, chunk_size):
            end_j = min(j + chunk_size, n_neurons)
            chunk_j = activations[:, j:end_j].cpu().numpy()
            
            for ii in range(chunk_i.shape[1]):
                for jj in range(chunk_j.shape[1]):
                    corr, _ = pearsonr(chunk_i[:, ii], chunk_j[:, jj])
                    correlations[i + ii, j + jj] = corr
    
    return correlations

def calculate_component_metrics(components: List[set], n_layers: int) -> Dict:
    """
    Calculate various metrics for neuron components.
    
    Args:
        components: List of sets of nodes (layer, neuron pairs)
        n_layers: Total number of layers
        
    Returns:
        Dictionary with various component metrics
    """
    metrics = {
        'n_components': len(components),
        'component_sizes': [],
        'avg_layer': [],
        'layer_spread': [],
        'density': []
    }
    
    for component in components:
        # Size
        metrics['component_sizes'].append(len(component))
        
        # Layer statistics
        layers = [int(node.split('N')[0][1:]) for node in component]
        metrics['avg_layer'].append(np.mean(layers))
        metrics['layer_spread'].append(np.std(layers))
        
        # Density (proportion of layers covered)
        unique_layers = len(set(layers))
        metrics['density'].append(unique_layers / n_layers)
    
    # Add summary statistics
    metrics.update({
        'avg_component_size': np.mean(metrics['component_sizes']),
        'max_component_size': max(metrics['component_sizes']),
        'avg_layer_spread': np.mean(metrics['layer_spread']),
        'avg_density': np.mean(metrics['density'])
    })
    
    return metrics 