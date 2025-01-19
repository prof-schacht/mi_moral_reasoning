import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Tuple, Dict
import pandas as pd

def plot_moral_neuron_analysis(results: Dict, moral_pairs: List[Tuple[str, str]], save_path: str = None, dimension: str = None, model_name: str = None):
    """
    Create comprehensive visualizations for moral neuron analysis.
    
    Args:
        results: Dictionary containing analysis results from MoralBehaviorAnalyzer
        moral_pairs: List of (moral_text, immoral_text) pairs used in analysis
        save_path: Optional path to save the plots
        dimension: The moral dimension being analyzed
        model_name: Name of the model being analyzed
    """
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Add title with model name if provided
    if model_name:
        fig.suptitle(f'Moral Neuron Analysis for {model_name}', fontsize=16, y=1.02)
    
    # 1. Distribution of moral neurons across layers
    ax1 = plt.subplot(2, 2, 1)
    plot_layer_distribution(results['moral_neurons'], results['immoral_neurons'], ax1, dimension, model_name)
    
    # 2. Consistency levels of moral neurons
    ax2 = plt.subplot(2, 2, 2)
    plot_consistency_distribution(results, ax2, dimension, model_name)
    
    # 3. Sample-wise means for important moral neurons
    ax3 = plt.subplot(2, 2, 3)
    plot_sample_means(results['moral_neurons'], results['sample_wise_means'], ax3, dimension, model_name)
    
    # 4. Most consistent positions visualization
    ax4 = plt.subplot(2, 2, 4)
    plot_consistent_positions(results['moral_neurons'], results['position_consistency'], moral_pairs, ax4, dimension, model_name)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_layer_distribution(moral_neurons: List[Tuple[int, int]], 
                          immoral_neurons: List[Tuple[int, int]], 
                          ax: plt.Axes, dimension: str, model_name: str = None):
    """Plot distribution of moral and immoral neurons across layers."""
    # Count neurons per layer
    layer_counts = {}
    for layer, _ in moral_neurons:
        layer_counts[layer] = layer_counts.get(layer, {'moral': 0, 'immoral': 0})
        layer_counts[layer]['moral'] += 1
    for layer, _ in immoral_neurons:
        layer_counts[layer] = layer_counts[layer] = layer_counts.get(layer, {'moral': 0, 'immoral': 0})
        layer_counts[layer]['immoral'] += 1
    
    # Prepare data for plotting
    layers = sorted(layer_counts.keys())
    moral_counts = [layer_counts[l]['moral'] for l in layers]
    immoral_counts = [layer_counts[l]['immoral'] for l in layers]
    
    # Create stacked bar plot
    width = 0.35
    ax.bar(layers, moral_counts, width, label='Moral', color='green', alpha=0.6)
    ax.bar(layers, immoral_counts, width, bottom=moral_counts, label='Immoral', color='red', alpha=0.6)
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Number of Neurons')
    title = f'Distribution of Moral/Immoral Neurons Across Layers for {dimension}'
    if model_name:
        title = f'{title}\n{model_name}'
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_consistency_distribution(results: Dict, ax: plt.Axes, dimension: str, model_name: str = None):
    """Plot consistency distribution of neurons."""
    consistency_data = pd.DataFrame(results.get('consistency_distribution', {}).items(),
                                  columns=['Consistency', 'Count'])
    
    sns.barplot(data=consistency_data, x='Consistency', y='Count', ax=ax)
    ax.set_xlabel('Consistency Level')
    ax.set_ylabel('Number of Neurons')
    title = f'Distribution of Neuron Consistency Levels for {dimension}'
    if model_name:
        title = f'{title}\n{model_name}'
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)

def plot_sample_means(moral_neurons: List[Tuple[int, int]], 
                     sample_wise_means: Dict[Tuple[int, int], List[float]], 
                     ax: plt.Axes, dimension: str, model_name: str = None):
    """Plot sample-wise means for each moral neuron."""
    num_neurons = len(moral_neurons)
    num_samples = len(next(iter(sample_wise_means.values())))
    
    # Create heatmap data
    heatmap_data = np.zeros((num_neurons, num_samples))
    neuron_labels = []
    
    for i, (layer, neuron) in enumerate(moral_neurons):
        heatmap_data[i] = sample_wise_means[(layer, neuron)]
        neuron_labels.append(f"L{layer}N{neuron}")
    
    sns.heatmap(heatmap_data, ax=ax, cmap='RdYlGn', center=0,
                xticklabels=[f"Sample {i+1}" for i in range(num_samples)],
                yticklabels=neuron_labels)
    title = f'Sample-wise Activation Differences for Moral Neurons for {dimension}'
    if model_name:
        title = f'{title}\n{model_name}'
    ax.set_title(title)
    ax.set_xlabel('Samples')
    ax.set_ylabel('Neurons')

def plot_consistent_positions(moral_neurons: List[Tuple[int, int]], 
                            position_consistency: Dict[Tuple[int, int], List[Tuple[int, float]]], 
                            moral_pairs: List[Tuple[str, str]], 
                            ax: plt.Axes, dimension: str, model_name: str = None):
    """Visualize most consistent positions for moral neurons."""
    # Get example text for position context
    example_text = moral_pairs[0][0]
    tokens = example_text.split()
    
    # Create position importance heatmap
    num_neurons = len(moral_neurons)
    max_pos = max(pos for neuron_data in position_consistency.values() 
                 for pos, _ in neuron_data)
    heatmap_data = np.zeros((num_neurons, max_pos + 1))
    neuron_labels = []
    
    for i, (layer, neuron) in enumerate(moral_neurons):
        for pos, cons in position_consistency[(layer, neuron)]:
            heatmap_data[i, pos] = cons
        neuron_labels.append(f"L{layer}N{neuron}")
    
    sns.heatmap(heatmap_data, ax=ax, cmap='viridis',
                xticklabels=[f"Pos {i}" for i in range(max_pos + 1)],
                yticklabels=neuron_labels)
    title = f'Position-wise Consistency for Moral Neurons for {dimension}'
    if model_name:
        title = f'{title}\n{model_name}'
    ax.set_title(title)
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Neurons')

    # Add token labels if available
    if len(tokens) > max_pos:
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(np.arange(max_pos + 1) + 0.5)
        ax2.set_xticklabels(tokens[:max_pos + 1], rotation=45, ha='left') 