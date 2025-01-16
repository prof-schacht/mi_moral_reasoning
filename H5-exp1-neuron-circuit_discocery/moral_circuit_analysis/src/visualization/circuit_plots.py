import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
from collections import defaultdict
from matplotlib.ticker import MaxNLocator

def plot_moral_circuits(results: Dict) -> plt.Figure:
    """Visualize the moral decision circuits."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot 1: Layer importance
    layers, importance = zip(*results['layer_importance'])
    ax1.bar(layers, importance)
    ax1.set_title('Layer Importance in Moral Decisions')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Importance Score')
    
    # Plot 2: Neuron distribution
    moral_layers = [l for l, _ in results['moral_neurons']]
    immoral_layers = [l for l, _ in results['immoral_neurons']]
    
    bins = range(-1, max(max(moral_layers), max(immoral_layers)) + 2)
    ax2.hist([moral_layers, immoral_layers], label=['Moral', 'Immoral'],
             bins=bins, alpha=0.6)
    ax2.set_title('Distribution of Moral/Immoral Neurons Across Layers')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Number of Neurons')
    ax2.legend()
    
    # Plot 3: Absolute count of moral/immoral neurons per layer
    moral_counts = defaultdict(int)
    immoral_counts = defaultdict(int)
    
    for layer, _ in results['moral_neurons']:
        moral_counts[layer] += 1
    for layer, _ in results['immoral_neurons']:
        immoral_counts[layer] += 1
        
    layers = sorted(set(moral_counts.keys()) | set(immoral_counts.keys()))
    moral_values = [moral_counts[l] for l in layers]
    immoral_values = [immoral_counts[l] for l in layers]
    
    width = 0.35
    ax3.bar([x - width/2 for x in layers], moral_values, width, label='Moral')
    ax3.bar([x + width/2 for x in layers], immoral_values, width, label='Immoral')
    
    # Add total count labels
    for i, layer in enumerate(layers):
        total = moral_counts[layer] + immoral_counts[layer]
        if total > 0:
            ax3.text(layer, max(moral_counts[layer], immoral_counts[layer]),
                    f'Total: {total}', ha='center', va='bottom')
    
    ax3.set_title('Absolute Count of Moral/Immoral Neurons per Layer')
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Number of Neurons')
    ax3.legend()
    ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    return fig

def plot_moral_circuits_with_descriptions(results: Dict, descriptions: Dict) -> plt.Figure:
    """Visualize moral circuits with neuron descriptions."""
    fig = plot_moral_circuits(results)
    
    # Add textbox with descriptions
    desc_text = "Key Neuron Descriptions:\n\n"
    for (layer, neuron), desc in descriptions.items():
        desc_text += f"Layer {layer} Neuron {neuron}:\n"
        desc_text += f"{desc}\n\n"
    
    fig.text(1.1, 0.5, desc_text,
            fontsize=8, va='center', ha='left',
            bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig 