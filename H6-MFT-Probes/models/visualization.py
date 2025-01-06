"""
Visualization utilities for probe analysis.
"""

import logging
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

def load_probe_results(probe_dir: str) -> Dict[str, Dict]:
    """Load all probe results from the probe directory."""
    probe_dir = Path(probe_dir)
    results = {}
    
    for probe_file in sorted(probe_dir.glob("*_logistic_probes.pt")):
        layer_name = probe_file.stem.replace("_logistic_probes", "")
        results[layer_name] = torch.load(probe_file)
    
    return results

def extract_metrics(results: Dict[str, Dict]) -> Tuple[Dict[str, np.ndarray], List[str], List[int]]:
    """Extract metrics for each layer and class."""
    metrics = {
        'f1': [],
        'precision': [],
        'recall': [],
        'accuracy': [],
        'avg_prob': []
    }
    
    # Get ordered lists of layers and classes
    layers = sorted(results.keys())
    classes = sorted(results[layers[0]].keys())
    
    # Extract metrics for each layer and class
    for layer in layers:
        layer_metrics = {k: [] for k in metrics.keys()}
        for class_idx in classes:
            class_metrics = results[layer][class_idx]['metrics']
            for metric_name in metrics.keys():
                layer_metrics[metric_name].append(class_metrics[metric_name])
        
        for metric_name in metrics.keys():
            metrics[metric_name].append(layer_metrics[metric_name])
    
    # Convert to numpy arrays
    for metric_name in metrics.keys():
        metrics[metric_name] = np.array(metrics[metric_name])
    
    return metrics, layers, classes

def plot_layer_metrics(
    probe_dir: str,
    output_dir: str = "figures",
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """Plot metrics across layers for each moral foundation category."""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load results
    results = load_probe_results(probe_dir)
    metrics, layers, classes = extract_metrics(results)
    
    # Set style
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    
    # Colors for different classes
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    
    # Plot each metric
    for metric_name, metric_values in metrics.items():
        plt.figure(figsize=figsize)
        
        # Plot lines for each class
        for class_idx in range(len(classes)):
            plt.plot(
                range(len(layers)),
                metric_values[:, class_idx],
                marker='o',
                label=f'Class {class_idx}',
                color=colors[class_idx],
                linewidth=2,
                markersize=8,
                alpha=0.7
            )
        
        # Customize plot
        plt.title(f'{metric_name.replace("_", " ").title()} across Layers', fontsize=14, pad=20)
        plt.xlabel('Layer Index', fontsize=12)
        plt.ylabel(metric_name.replace("_", " ").title(), fontsize=12)
        plt.xticks(
            range(len(layers)),
            [f"Layer {i}" for i in range(len(layers))],
            rotation=45
        )
        plt.legend(
            title='Moral Foundation',
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            fontsize=10,
            title_fontsize=12
        )
        
        # Add grid
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_dir / f'{metric_name}_across_layers.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create heatmap for each metric
    for metric_name, metric_values in metrics.items():
        plt.figure(figsize=figsize)
        
        # Create heatmap
        im = plt.imshow(metric_values.T, aspect='auto', cmap='viridis')
        
        # Add colorbar
        plt.colorbar(im)
        
        # Add text annotations
        for i in range(metric_values.shape[0]):
            for j in range(metric_values.shape[1]):
                text = plt.text(
                    i, j,
                    f'{metric_values[i, j]:.3f}',
                    ha='center',
                    va='center',
                    color='white' if metric_values[i, j] > np.mean(metric_values) else 'black'
                )
        
        # Customize plot
        plt.title(f'{metric_name.replace("_", " ").title()} Heatmap', fontsize=14, pad=20)
        plt.xlabel('Layer', fontsize=12)
        plt.ylabel('Moral Foundation', fontsize=12)
        
        # Set ticks
        plt.xticks(
            range(len(layers)),
            [f"Layer {i}" for i in range(len(layers))],
            rotation=45
        )
        plt.yticks(
            range(len(classes)),
            [f"Class {i}" for i in range(len(classes))]
        )
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_dir / f'{metric_name}_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Saved visualization plots to {output_dir}")

def plot_class_distribution(probe_dir: str, output_dir: str = "figures") -> None:
    """Plot class distribution and training statistics across moral foundations."""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load results
    results = load_probe_results(probe_dir)
    first_layer = next(iter(results.values()))
    
    # Extract class statistics
    classes = sorted(first_layer.keys())
    train_counts = []
    val_counts = []
    
    for class_idx in classes:
        train_counts.append(first_layer[class_idx]['config']['train_counts'])
        val_counts.append(first_layer[class_idx]['config']['val_counts'])
    
    # Set style
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    
    # Plot training sample distribution
    plt.figure(figsize=(12, 6))
    x = np.arange(len(classes))
    width = 0.35
    
    # Plot training samples
    plt.bar(
        x - width/2,
        [c['positive'] for c in train_counts],
        width,
        label='Positive Samples',
        color='#2ecc71',
        alpha=0.7
    )
    plt.bar(
        x + width/2,
        [c['negative'] for c in train_counts],
        width,
        label='Negative Samples (Random)',
        color='#e74c3c',
        alpha=0.7
    )
    
    plt.title('Balanced Training Sample Distribution', fontsize=14, pad=20)
    plt.xlabel('Moral Foundation', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(x, [f'Class {i}' for i in classes])
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, counts in enumerate(train_counts):
        plt.text(i - width/2, counts['positive'], str(counts['positive']), ha='center', va='bottom')
        plt.text(i + width/2, counts['negative'], str(counts['negative']), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'balanced_training_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot validation sample distribution
    plt.figure(figsize=(12, 6))
    
    plt.bar(
        x - width/2,
        [c['positive'] for c in val_counts],
        width,
        label='Positive Samples',
        color='#2ecc71',
        alpha=0.7
    )
    plt.bar(
        x + width/2,
        [c['negative'] for c in val_counts],
        width,
        label='Negative Samples (Random)',
        color='#e74c3c',
        alpha=0.7
    )
    
    plt.title('Balanced Validation Sample Distribution', fontsize=14, pad=20)
    plt.xlabel('Moral Foundation', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(x, [f'Class {i}' for i in classes])
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, counts in enumerate(val_counts):
        plt.text(i - width/2, counts['positive'], str(counts['positive']), ha='center', va='bottom')
        plt.text(i + width/2, counts['negative'], str(counts['negative']), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'balanced_validation_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot total samples per class
    plt.figure(figsize=(12, 6))
    total_samples = [c['total'] for c in train_counts]
    bars = plt.bar(x, total_samples, color='#3498db', alpha=0.7)
    plt.title('Total Training Samples per Foundation', fontsize=14, pad=20)
    plt.xlabel('Moral Foundation', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(x, [f'Class {i}' for i in classes])
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{int(height)}',
            ha='center',
            va='bottom'
        )
    
    plt.tight_layout()
    plt.savefig(output_dir / 'total_samples_per_class.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved distribution plots to {output_dir}") 