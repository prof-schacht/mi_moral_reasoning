import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List
import numpy as np

def plot_neuron_network(results: Dict) -> nx.Graph:
    """Create a network visualization of moral/immoral neurons and their connections."""
    G = nx.Graph()
    
    # Add nodes for all neurons
    all_neurons = [(n, 'moral') for n in results['moral_neurons']] + \
                 [(n, 'immoral') for n in results['immoral_neurons']]
    
    for neuron, ntype in all_neurons:
        G.add_node(f"L{neuron[0]}N{neuron[1]}", 
                  color='blue' if ntype == 'moral' else 'red',
                  type=ntype,
                  layer=neuron[0])
    
    # Add edges based on activation patterns
    activation_diffs = results['activation_differences']
    nodes = list(G.nodes())
    
    for i, node1 in enumerate(nodes):
        layer1, neuron1 = map(int, node1[1:].split('N'))
        for node2 in nodes[i+1:]:
            layer2, neuron2 = map(int, node2[1:].split('N'))
            
            # Only connect neurons within 2 layers
            if abs(layer1 - layer2) <= 2:
                act1 = activation_diffs[layer1, neuron1]
                act2 = activation_diffs[layer2, neuron2]
                similarity = 1 / (1 + torch.mean(torch.abs(act1 - act2)).item())
                
                if similarity > 0.3:
                    G.add_edge(node1, node2, weight=similarity)
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Draw nodes
    node_colors = [G.nodes[node]['color'] for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.7)
    
    # Draw edges
    edge_weights = [G[u][v]['weight'] * 2 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5)
    
    # Add labels
    nx.draw_networkx_labels(G, pos)
    
    # Add legend
    moral_patch = plt.Rectangle((0, 0), 1, 1, fc="blue", alpha=0.7, label="Moral Neurons")
    immoral_patch = plt.Rectangle((0, 0), 1, 1, fc="red", alpha=0.7, label="Immoral Neurons")
    plt.legend(handles=[moral_patch, immoral_patch])
    
    plt.title("Network of Moral and Immoral Neurons")
    plt.axis('off')
    plt.tight_layout()
    
    # Print network statistics
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    print(f"Number of connected components: {nx.number_connected_components(G)}")
    
    return G

def plot_neuron_components(results: Dict) -> nx.Graph:
    """Create a network visualization with different colors for each connected component."""
    G = plot_neuron_network(results)
    components = list(nx.connected_components(G))
    
    plt.figure(figsize=(15, 10))
    
    # Use distinct colors for components
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99FFCC', '#FFB366', '#99FF99']
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Draw each component
    for idx, component in enumerate(components):
        subgraph = G.subgraph(component)
        color = colors[idx % len(colors)]
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos,
                             nodelist=list(component),
                             node_color=color,
                             node_size=500,
                             alpha=0.7)
        
        # Draw edges
        edge_list = list(subgraph.edges())
        if edge_list:
            edge_weights = [subgraph[u][v]['weight'] * 2 for u, v in edge_list]
            nx.draw_networkx_edges(G, pos,
                                 edgelist=edge_list,
                                 width=edge_weights,
                                 alpha=0.5)
    
    # Add labels
    nx.draw_networkx_labels(G, pos)
    
    # Add legend for components
    legend_elements = [plt.Rectangle((0, 0), 1, 1, fc=colors[i], alpha=0.7,
                                   label=f'Component {i+1}')
                      for i in range(len(components))]
    plt.legend(handles=legend_elements)
    
    plt.title("Network of Moral Neurons - Colored by Connected Components")
    plt.axis('off')
    plt.tight_layout()
    
    # Print component information
    print("\nComponent Analysis:")
    for idx, component in enumerate(components):
        print(f"\nComponent {idx + 1} ({len(component)} neurons):")
        print("Neurons:", ', '.join(sorted(list(component))))
        
        layers = [int(node.split('N')[0][1:]) for node in component]
        avg_layer = sum(layers) / len(layers)
        print(f"Average layer: {avg_layer:.2f}")
    
    return G 