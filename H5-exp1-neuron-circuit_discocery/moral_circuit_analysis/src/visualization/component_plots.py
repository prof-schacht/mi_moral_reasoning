import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import networkx as nx

def plot_component_layer_distribution(components: List[nx.Graph], n_layers: int) -> plt.Figure:
    """Plot the distribution of neurons across layers for each component."""
    plt.figure(figsize=(12, 6))
    
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99FFCC', '#FFB366', '#99FF99']
    
    # Calculate layer distribution for each component
    for idx, component in enumerate(components):
        layer_counts = np.zeros(n_layers)
        for node in component:
            layer = int(node.split('N')[0][1:])
            layer_counts[layer] += 1
            
        # Plot distribution with dashed lines
        plt.plot(range(n_layers), layer_counts, '--', 
                color=colors[idx % len(colors)], 
                alpha=0.7)
        
        # Plot dots for non-zero values
        non_zero_mask = layer_counts > 0
        plt.plot(np.arange(n_layers)[non_zero_mask], 
                layer_counts[non_zero_mask], 'o',
                color=colors[idx % len(colors)],
                label=f'Component {idx + 1}',
                markersize=8)
    
    plt.xlabel('Layer')
    plt.ylabel('Number of Neurons')
    plt.title('Component Distribution Across Layers')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    return plt.gcf()

def plot_component_boxplots(components: List[nx.Graph], n_layers: int) -> plt.Figure:
    """Create boxplots showing layer distribution for each component."""
    plt.figure(figsize=(12, 6))
    
    # Collect layer numbers for each component
    component_layers = []
    labels = []
    
    for idx, component in enumerate(components):
        layers = [int(node.split('N')[0][1:]) for node in component]
        component_layers.append(layers)
        labels.append(f'Component {idx + 1}')
    
    plt.boxplot(component_layers, labels=labels)
    plt.ylabel('Layer')
    plt.title('Layer Distribution by Component')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    return plt.gcf()

def plot_component_heatmap(components: List[nx.Graph], n_layers: int, n_neurons: int) -> plt.Figure:
    """Create a heatmap showing neuron activation patterns for each component."""
    plt.figure(figsize=(15, 8))
    
    # Create matrix of neuron activations
    activation_matrix = np.zeros((len(components), n_layers))
    
    for comp_idx, component in enumerate(components):
        for node in component:
            layer = int(node.split('N')[0][1:])
            activation_matrix[comp_idx, layer] += 1
    
    plt.imshow(activation_matrix, aspect='auto', cmap='YlOrRd')
    plt.colorbar(label='Number of Neurons')
    
    plt.xlabel('Layer')
    plt.ylabel('Component')
    plt.title('Component Neuron Distribution Heatmap')
    
    # Add component labels
    plt.yticks(range(len(components)), [f'Component {i+1}' for i in range(len(components))])
    
    return plt.gcf()

def plot_component_summary(components: List[nx.Graph], n_layers: int, n_neurons: int) -> None:
    """Create a comprehensive summary of component distributions."""
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Layer distribution plot
    plt.subplot(2, 2, 1)
    plot_component_layer_distribution(components, n_layers)
    plt.title('Component Distribution Across Layers')
    
    # 2. Boxplot
    plt.subplot(2, 2, 2)
    plot_component_boxplots(components, n_layers)
    plt.title('Layer Distribution by Component')
    
    # 3. Heatmap
    plt.subplot(2, 2, 3)
    plot_component_heatmap(components, n_layers, n_neurons)
    plt.title('Component Neuron Distribution Heatmap')
    
    # 4. Component size comparison
    plt.subplot(2, 2, 4)
    sizes = [len(comp) for comp in components]
    plt.bar(range(len(components)), sizes)
    plt.xlabel('Component')
    plt.ylabel('Number of Neurons')
    plt.title('Component Sizes')
    plt.xticks(range(len(components)), [f'Component {i+1}' for i in range(len(components))])
    
    plt.tight_layout()
    return fig 