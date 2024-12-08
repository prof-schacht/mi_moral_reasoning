# %%
import torch
from transformer_lens import HookedTransformer
import numpy as np
from typing import List, Tuple, Dict, Set
from collections import defaultdict
import networkx as nx
from scipy.stats import pearsonr
from dataclasses import dataclass
from tqdm.auto import tqdm
import torch.nn.functional as F
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import openai
import os

# %%

# Load model = 
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32
model_name = 'google/gemma-2-9b-it'
model = HookedTransformer.from_pretrained(
                model_name,
                device=device,
                dtype=dtype,
                default_padding_side="right",
                center_writing_weights=True,
                center_unembed=True,
                fold_ln=True,
                move_to_device=True
            )

# %%
# Idea to extend this by focusing more on sparse neurons and causal invterventions to identify the connected neurons
# For studying moral behavior in neurons/layers, we'd want to design a targeted analysis approach. Here's how we could modify the code:
# Key aspects of this approach:
# 1. Comparative Analysis: We look at activation differences between moral and immoral choices
# 2. Decision Point Focus: We analyze the last token where the model makes the moral/immoral decision
# 3. Consistency: We identify neurons that consistently differ across many examples
# 4. Layer Importance: We measure which layers are most involved in moral decisions
# This could help identify:
# - Neurons that act as "moral compass" components
# - Layers that are crucial for moral reasoning
# - Potential circuits involved in moral decision-making
# You might want to extend this with:
# - More sophisticated moral/immoral text pairs
# - Analysis of intermediate reasoning steps
# Causal intervention studies on identified moral neurons

@dataclass
class CoactivationPattern:
    neurons: Set[Tuple[int, int]]  # Set of (layer, neuron) pairs
    strength: float                # Co-activation strength
    frequency: int                 # How often this pattern occurs
    exemplars: List[str]          # Example texts triggering this pattern


class NeuronActivationCollector:
    def __init__(self, model, 
                 activation_threshold: float = 0.7,
                 correlation_threshold: float = 0.6,
                 batch_size: int = 32):
        # Set up device and dtype
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32  # Use float32 for better numerical stability
        print(f"Using device: {self.device} with dtype: {self.dtype}")
        
        try:
            # Initialize model with explicit device and dtype
            print(f"Loading model {model_name}...")
            self.model = model
            print("Model loaded successfully")
            
        except RuntimeError as e:
            print(f"GPU initialization failed: {e}")
            print("Attempting CPU initialization...")
            self.device = "cpu"
            # Try loading on CPU instead
            self.model = HookedTransformer.from_pretrained(
                model_name,
                device="cpu",
                dtype=self.dtype,
                default_padding_side="right",
                center_writing_weights=True,
                center_unembed=True,
                fold_ln=True,
                move_to_device=False
            )
            print("Model loaded on CPU successfully")
        
        self.n_layers = self.model.cfg.n_layers
        self.n_neurons = self.model.cfg.d_mlp
        
        self.activation_threshold = activation_threshold
        self.correlation_threshold = correlation_threshold
        self.batch_size = batch_size
        self.neuron_graph = nx.Graph()
        self.coactivation_patterns = []
        
        print(f"Model has {self.n_layers} layers with {self.n_neurons} neurons each")
        print(f"Total number of neurons: {self.n_layers * self.n_neurons}")
        print(f"Batch size: {self.batch_size}")
        print(f"Activation threshold: {self.activation_threshold}")
        print(f"Correlation threshold: {self.correlation_threshold}")

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
        
        # Process texts in batches
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
                
                # Get activations for each layer
                batch_activations = []
                for layer_idx in range(self.n_layers):
                    mlp_acts = cache['post', layer_idx, 'mlp']
                    batch_activations.append(mlp_acts)
                
                # Stack along the layer dimension instead of batch dimension
                batch_stack = torch.stack(batch_activations, dim=1)  # [batch, layers, seq, neurons]
                all_activations.append(batch_stack)
        
        # Combine all batches along the batch dimension
        return torch.cat(all_activations, dim=0)  # [total_batch, layers, seq, neurons]

    def compute_coactivation_matrix(self, activations: torch.Tensor) -> torch.Tensor:
        """Compute co-activation matrix across all neurons efficiently on GPU."""
        n_samples, n_layers, seq_len, n_neurons = activations.shape
        total_neurons = n_layers * n_neurons
        
        # Reshape to [n_samples * seq_len, total_neurons]
        # Fix the transpose operation by doing it step by step
        flat_activations = (activations
                          .permute(0, 2, 1, 3)  # [n_samples, seq_len, n_layers, n_neurons]
                          .reshape(-1, total_neurons))  # [n_samples * seq_len, total_neurons]
        
        # Binary activation matrix based on threshold
        active_neurons = (flat_activations > self.activation_threshold).float()
        
        # Compute correlation matrix efficiently on GPU
        print("Computing correlation matrix...")
        print(f"Active neurons tensor shape: {active_neurons.shape}")
        coactivation_matrix = torch.zeros((total_neurons, total_neurons), device=self.device)
        
        try:
            # Process in chunks to avoid memory issues
            chunk_size = min(1000, total_neurons // 2)  # Adjust based on GPU memory
            for i in tqdm(range(0, total_neurons, chunk_size), desc="Computing correlations"):
                end_idx = min(i + chunk_size, total_neurons)
                chunk_correlations = batch_pearson_correlation(
                    active_neurons[:, i:end_idx],
                    active_neurons
                )
                coactivation_matrix[i:end_idx] = chunk_correlations
            
            # Apply threshold and ensure matrix is symmetric
            mask = torch.abs(coactivation_matrix) > self.correlation_threshold
            coactivation_matrix = coactivation_matrix * mask.float()
            
            # Ensure the matrix is symmetric
            coactivation_matrix = (coactivation_matrix + coactivation_matrix.t()) / 2
            
        except RuntimeError as e:
            print(f"GPU Error: {e}")
            print("Falling back to CPU computation...")
            
            # Move tensors to CPU and try again
            active_neurons = active_neurons.cpu()
            coactivation_matrix = coactivation_matrix.cpu()
            
            for i in tqdm(range(0, total_neurons, chunk_size), desc="Computing correlations (CPU)"):
                end_idx = min(i + chunk_size, total_neurons)
                chunk_correlations = batch_pearson_correlation(
                    active_neurons[:, i:end_idx],
                    active_neurons
                )
                coactivation_matrix[i:end_idx] = chunk_correlations
            
            # Move back to GPU if available
            if self.device != "cpu":
                coactivation_matrix = coactivation_matrix.to(self.device)
        
        return coactivation_matrix

    def identify_coactivation_patterns(self, texts: List[str]) -> List[CoactivationPattern]:
        """Identify recurring patterns of co-activated neurons."""
        # Get activations for all texts
        print("Getting activations...")
        activations = self.get_all_activations_batch(texts)
        
        # Compute co-activation matrix
        coactivation_matrix = self.compute_coactivation_matrix(activations)
        
        # Build graph of strongly correlated neurons
        print("Building neuron graph...")
        print(f"Coactivation matrix shape: {coactivation_matrix.shape}")
        self.neuron_graph.clear()
        total_neurons = self.n_layers * self.n_neurons
        
        try:
            # Move to CPU for NetworkX processing
            coactivation_matrix_cpu = coactivation_matrix.cpu().numpy()
            
            # Find significant correlations using a more memory-efficient approach
            print("Finding significant correlations...")
            significant_pairs = []
            
            # Process the upper triangular part in chunks to save memory
            chunk_size = 1000
            for i in tqdm(range(0, total_neurons, chunk_size), desc="Processing correlations"):
                end_i = min(i + chunk_size, total_neurons)
                for j in range(i, total_neurons, chunk_size):
                    end_j = min(j + chunk_size, total_neurons)
                    # Get the chunk of the correlation matrix
                    chunk = coactivation_matrix_cpu[i:end_i, j:end_j]
                    # Find significant correlations in this chunk
                    rows, cols = np.where(np.abs(chunk) > self.correlation_threshold)
                    # Adjust indices to global coordinates
                    rows += i
                    cols += j
                    # Only keep upper triangular part
                    mask = cols > rows
                    rows = rows[mask]
                    cols = cols[mask]
                    # Store significant pairs
                    significant_pairs.extend(list(zip(rows, cols)))
            
            print(f"Found {len(significant_pairs)} significant correlations")
            
            # Add edges for significant correlations
            for i, j in tqdm(significant_pairs, desc="Building graph"):
                layer1, neuron1 = int(i) // self.n_neurons, int(i) % self.n_neurons
                layer2, neuron2 = int(j) // self.n_neurons, int(j) % self.n_neurons
                weight = float(coactivation_matrix_cpu[i, j])
                self.neuron_graph.add_edge(
                    (layer1, neuron1),
                    (layer2, neuron2),
                    weight=weight
                )
            
            # Find communities of co-activating neurons
            print("Finding communities...")
            if len(self.neuron_graph.edges) == 0:
                print("Warning: No significant correlations found. Try lowering the correlation threshold.")
                return []
            
            communities = nx.community.louvain_communities(self.neuron_graph)
            
            # Analyze each community
            print("Analyzing communities...")
            patterns = []
            for community in tqdm(communities, desc="Analyzing patterns"):
                # Create flattened mask for this community
                community_mask = torch.zeros(self.n_layers * self.n_neurons, dtype=torch.bool, device=self.device)
                for layer, neuron in community:
                    idx = layer * self.n_neurons + neuron
                    community_mask[idx] = True
                
                # Find examples where this pattern occurs
                pattern_occurrences = []
                for idx in range(len(texts)):
                    # Reshape sample activations to match mask dimensions
                    sample_acts = activations[idx]  # Shape: [n_layers, seq_len, n_neurons]
                    # Reshape to [seq_len, n_layers * n_neurons]
                    flat_acts = sample_acts.permute(1, 0, 2).reshape(-1, self.n_layers * self.n_neurons)
                    
                    # Check if pattern occurs in any position in sequence
                    occurs = torch.any(torch.all(flat_acts[:, community_mask], dim=-1))
                    if occurs:
                        pattern_occurrences.append(idx)
                
                if pattern_occurrences:
                    # Safely compute mean edge weight
                    community_edges = list(nx.edges(self.neuron_graph.subgraph(community)))
                    if community_edges:
                        mean_weight = np.mean([self.neuron_graph[u][v]['weight'] for u, v in community_edges])
                    else:
                        mean_weight = 0.0
                    
                    patterns.append(CoactivationPattern(
                        neurons=set(community),
                        strength=mean_weight,
                        frequency=len(pattern_occurrences),
                        exemplars=[texts[i] for i in pattern_occurrences[:5]]
                    ))
            
            self.coactivation_patterns = sorted(patterns, key=lambda x: x.frequency, reverse=True)
            print(f"Found {len(patterns)} coactivation patterns")
            return self.coactivation_patterns
            
        except Exception as e:
            print(f"Error during pattern identification: {e}")
            print("Traceback:", traceback.format_exc())
            return []

    def get_neuron_communities(self) -> Dict[Tuple[int, int], Set[Tuple[int, int]]]:
        """Get communities of frequently co-activating neurons."""
        communities = nx.community.louvain_communities(self.neuron_graph)
        
        neuron_to_community = {}
        for community in communities:
            for neuron in community:
                neuron_to_community[neuron] = set(community)
                
        return neuron_to_community

    def analyze_hierarchical_patterns(self, texts: List[str]) -> Dict:
        """Perform hierarchical analysis of neuron activation patterns."""
        print("\nStarting hierarchical pattern analysis...")
        patterns = self.identify_coactivation_patterns(texts)
        communities = self.get_neuron_communities()
        
        print("Analyzing layer distributions...")
        # Analyze layer distribution in communities
        layer_distributions = {}
        for neuron, community in tqdm(communities.items(), desc="Analyzing layer distributions"):
            layers = sorted(set(layer for layer, _ in community))
            layer_distributions[neuron] = layers
        
        print("Finding cross-layer patterns...")
        # Find cross-layer patterns
        cross_layer_patterns = [p for p in patterns 
                              if len(set(layer for layer, _ in p.neurons)) > 1]
        
        print("Analyzing pattern hierarchies...")
        # Analyze pattern hierarchies
        pattern_hierarchies = []
        for p1 in tqdm(patterns, desc="Building pattern hierarchies"):
            for p2 in patterns:
                if p1 != p2 and p1.neurons.issubset(p2.neurons):
                    pattern_hierarchies.append((p1, p2))
        
        return {
            'patterns': patterns,
            'communities': communities,
            'layer_distributions': layer_distributions,
            'cross_layer_patterns': cross_layer_patterns,
            'pattern_hierarchies': pattern_hierarchies
        }

    def plot_coactivation_analysis(self, patterns: List[CoactivationPattern], cross_layer_patterns: List[CoactivationPattern]) -> plt.Figure:
        """Create detailed visualization of coactivation and cross-layer patterns"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Coactivation Pattern Strengths
        pattern_strengths = [p.strength for p in patterns]
        pattern_ids = range(len(patterns))
        sns.barplot(x=pattern_ids, y=pattern_strengths, ax=ax1)
        ax1.set_title('Coactivation Pattern Strengths')
        ax1.set_xlabel('Pattern ID')
        ax1.set_ylabel('Strength')
        
        # 2. Pattern Frequencies
        pattern_freqs = [p.frequency for p in patterns]
        sns.barplot(x=pattern_ids, y=pattern_freqs, ax=ax2)
        ax2.set_title('Pattern Frequencies')
        ax2.set_xlabel('Pattern ID')
        ax2.set_ylabel('Frequency')
        
        # 3. Cross-layer Pattern Distribution
        if cross_layer_patterns:
            cross_layer_sizes = [len(p.neurons) for p in cross_layer_patterns]
            sns.histplot(cross_layer_sizes, ax=ax3)
            ax3.set_title('Cross-layer Pattern Sizes')
            ax3.set_xlabel('Number of Neurons')
            ax3.set_ylabel('Count')
        
        # 4. Layer Distribution in Cross-layer Patterns
        if cross_layer_patterns:
            layer_counts = defaultdict(int)
            for pattern in cross_layer_patterns:
                for layer, _ in pattern.neurons:
                    layer_counts[layer] += 1
            
            layers = sorted(layer_counts.keys())
            counts = [layer_counts[l] for l in layers]
            sns.barplot(x=layers, y=counts, ax=ax4)
            ax4.set_title('Layer Distribution in Cross-layer Patterns')
            ax4.set_xlabel('Layer')
            ax4.set_ylabel('Number of Patterns')
        
        plt.tight_layout()
        return fig

    def visualize_patterns(self):
        """Visualize coactivation patterns and their relationships."""
        if not self.coactivation_patterns:
            print("No patterns found. Run identify_coactivation_patterns first.")
            return

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Pattern sizes and strengths
        pattern_sizes = [len(p.neurons) for p in self.coactivation_patterns]
        pattern_strengths = [p.strength for p in self.coactivation_patterns]
        
        ax1.scatter(pattern_sizes, pattern_strengths, alpha=0.6)
        ax1.set_xlabel('Pattern Size (Number of Neurons)')
        ax1.set_ylabel('Pattern Strength')
        ax1.set_title('Pattern Size vs Strength')

        # Plot 2: Layer distribution
        layer_counts = defaultdict(int)
        for pattern in self.coactivation_patterns:
            for layer, _ in pattern.neurons:
                layer_counts[layer] += 1

        layers = sorted(layer_counts.keys())
        counts = [layer_counts[l] for l in layers]
        
        ax2.bar(layers, counts)
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Number of Neurons in Patterns')
        ax2.set_title('Distribution of Pattern Neurons Across Layers')

        plt.tight_layout()
        return fig

class MoralBehaviorAnalyzer(NeuronActivationCollector):
    def __init__(self, model):
        super().__init__(model)
        
    def analyze_moral_behavior(self, moral_pairs: List[Tuple[str, str]], significant_diff: float = 0.5, consistency_threshold: float = 0.8) -> Dict:
        """
        Analyze which neurons/layers respond differently to moral vs immoral completions
        
        Args:
            moral_pairs: List of (moral_text, immoral_text) pairs
            Example: [
                ("I should help the elderly cross the street", 
                 "I should ignore the elderly crossing the street"),
                ...
            ]
        """
        moral_activations = []
        immoral_activations = []
        
        # Get activations for both moral and immoral texts
        for moral_text, immoral_text in tqdm(moral_pairs):
            # Get activations for last token (decision point)
            moral_acts = self._get_completion_activations(moral_text)
            immoral_acts = self._get_completion_activations(immoral_text)
            
            moral_activations.append(moral_acts)
            immoral_activations.append(immoral_acts)
            
        moral_tensor = torch.stack(moral_activations)
        immoral_tensor = torch.stack(immoral_activations)
        
        # Analyze differences between moral and immoral activations
        return self._analyze_moral_differences(moral_tensor, immoral_tensor, significant_diff=significant_diff, consistency_threshold=consistency_threshold)
    
    def _get_completion_activations(self, text: str) -> torch.Tensor:
        """Get activations at the decision point."""
        tokens = self.model.to_tokens(text)
        with torch.no_grad():
            _, cache = self.model.run_with_cache(tokens)
            
            # Get activations for each layer at the decision point
            activations = []
            for layer_idx in range(self.n_layers):
                mlp_acts = cache['post', layer_idx, 'mlp']
                # Get last token activation
                last_token_acts = mlp_acts[0, -1]  # [n_neurons]
                activations.append(last_token_acts)
                
        return torch.stack(activations)  # [n_layers, n_neurons]
    
    def _analyze_moral_differences(self, 
                                 moral_acts: torch.Tensor, 
                                 immoral_acts: torch.Tensor,
                                 significant_diff: float = 0.5,
                                 consistency_threshold: float = 0.8) -> Dict:
        """
        Analyze which neurons show consistent differences between moral/immoral choices.
        
        Returns dictionary with:
        - moral_neurons: List of (layer, neuron) that consistently activate more for moral choices
        - immoral_neurons: List of (layer, neuron) that consistently activate more for immoral choices
        - layer_importance: Ranking of layers by their role in moral decisions
        """
        # Calculate average difference in activations
        diff = moral_acts - immoral_acts  # [n_samples, n_layers, n_neurons]
        mean_diff = diff.mean(dim=0)  # [n_layers, n_neurons]
        
        # Find consistently different neurons
        moral_neurons = []
        immoral_neurons = []
        
        for layer in range(self.n_layers):
            for neuron in range(self.n_neurons):
                consistency = (diff[:, layer, neuron] > 0).float().mean()
                if consistency > consistency_threshold:  # Neuron consistently differs
                    if mean_diff[layer, neuron] > significant_diff:
                        moral_neurons.append((layer, neuron))
                    elif mean_diff[layer, neuron] < -significant_diff:
                        immoral_neurons.append((layer, neuron))
        
        # Analyze layer importance
        layer_importance = []
        for layer in range(self.n_layers):
            # Calculate how much this layer contributes to moral decisions
            layer_diff = torch.abs(mean_diff[layer]).mean()
            layer_importance.append((layer, layer_diff.item()))
        
        layer_importance.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'moral_neurons': moral_neurons,
            'immoral_neurons': immoral_neurons,
            'layer_importance': layer_importance,
            'activation_differences': mean_diff
        }

    def visualize_moral_circuits(self, results: Dict) -> None:
        """Visualize the moral decision circuits."""
        from matplotlib.ticker import MaxNLocator
        
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
        
        # Create bins for all layers
        bins = np.arange(-0.5, self.n_layers + 0.5, 1)
        ax2.hist([moral_layers, immoral_layers], label=['Moral', 'Immoral'],
                 bins=bins, alpha=0.6, range=(-0.5, self.n_layers - 0.5))
        ax2.set_xticks(range(self.n_layers))
        ax2.set_title('Distribution of Moral/Immoral Neurons Across Layers')
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Number of Neurons')
        ax2.legend()
        
        # Plot 3: Absolute count of moral/immoral neurons per layer
        moral_counts = np.bincount(moral_layers, minlength=self.n_layers)
        immoral_counts = np.bincount(immoral_layers, minlength=self.n_layers)
        total_neurons = moral_counts + immoral_counts
        
        x = np.arange(self.n_layers)
        width = 0.35
        ax3.bar(x - width/2, moral_counts, width, label='Moral', color='tab:blue')
        ax3.bar(x + width/2, immoral_counts, width, label='Immoral', color='tab:orange')
        
        # Add text labels for total neuron count above each pair of bars
        for i, total in enumerate(total_neurons):
            if total > 0:  # Only show label if there are neurons
                ax3.text(i, max(moral_counts[i], immoral_counts[i]), f'Total: {total}', 
                        ha='center', va='bottom')
        
        # Set y-axis to integer values and ensure all counts are visible
        ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax3.set_ylim(top=max(max(moral_counts), max(immoral_counts)) * 1.2)  # Add 20% padding
        
        ax3.set_title('Absolute Count of Moral/Immoral Neurons per Layer')
        ax3.set_xlabel('Layer')
        ax3.set_ylabel('Number of Neurons')
        ax3.legend()
        
        plt.tight_layout()
        plt.show()

    def visualize_neuron_components(self, results: Dict) -> None:
        """
        Create a network graph visualization with different colors for each connected component.
        """
        import networkx as nx
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create a new graph
        G = nx.Graph()
        
        # Add nodes
        all_neurons = [(n, 'moral') for n in results['moral_neurons']] + \
                    [(n, 'immoral') for n in results['immoral_neurons']]
        
        for neuron, ntype in all_neurons:
            G.add_node(f"L{neuron[0]}N{neuron[1]}", 
                    type=ntype,
                    layer=neuron[0])
        
        # Add edges using the same logic as before
        activation_diffs = results['activation_differences']
        nodes = list(G.nodes())
        
        # Add edges between neurons that have similar activation patterns
        for i, node1 in enumerate(nodes):
            layer1, neuron1 = map(int, node1[1:].split('N'))
            for node2 in nodes[i+1:]:
                layer2, neuron2 = map(int, node2[1:].split('N'))
                
                # Only connect neurons within 2 layers of each other
                layer_dist = abs(layer1 - layer2)
                if layer_dist <= 2:
                    # Compare activation patterns between the two neurons
                    act1 = activation_diffs[layer1, neuron1]
                    act2 = activation_diffs[layer2, neuron2]
                    # Calculate similarity based on mean absolute difference
                    similarity = 1 / (1 + torch.mean(torch.abs(act1 - act2)).item())
                    
                    # Add edge if neurons have similar enough activation patterns
                    if similarity > 0.3:
                        G.add_edge(node1, node2, weight=similarity)
        
        # Find groups of neurons that are connected to each other through edges
        # Each component represents a group of neurons with similar activation patterns
        components = list(nx.connected_components(G))
        
        # Create a color map for components
        # Using distinct colors for better visibility
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99FFCC', '#FFB366', '#99FF99']
        
        # Set up the plot
        plt.figure(figsize=(15, 10))
        
        # Use spring layout with adjusted parameters for better spacing
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw each component with a different color
        for idx, component in enumerate(components):
            subgraph = G.subgraph(component)
            color = colors[idx % len(colors)]  # Cycle through colors if more components than colors
            
            # Draw nodes for this component
            nx.draw_networkx_nodes(G, pos, 
                                nodelist=list(component),
                                node_color=color, 
                                node_size=500, 
                                alpha=0.7)
            
            # Draw edges within this component
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
            
            # Calculate average layer for this component
            layers = [int(node.split('N')[0][1:]) for node in component]
            avg_layer = sum(layers) / len(layers)
            print(f"Average layer: {avg_layer:.2f}")
        
        print(components)
        
        # Create a new figure for layer distribution
        plt.figure(figsize=(12, 6))
        
        # Create a dictionary to store layer counts for each component
        component_layer_dist = {}
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99FFCC', '#FFB366', '#99FF99']
        
        # Calculate layer distribution for each component
        for idx, component in enumerate(components):
            layer_counts = np.zeros(self.n_layers)
            for node in component:
                layer = int(node.split('N')[0][1:])
                layer_counts[layer] += 1
            component_layer_dist[idx] = layer_counts
            
            # Plot the distribution with dashed lines connecting all points
            plt.plot(range(self.n_layers), layer_counts, '--', 
                    color=colors[idx % len(colors)], 
                    alpha=0.7)
            
            # Only plot dots for non-zero values
            non_zero_mask = layer_counts > 0
            plt.plot(np.arange(self.n_layers)[non_zero_mask], 
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
        plt.show()
        
        return G
    
    def visualize_neuron_boxplot(self, results: Dict) -> None:
        """Visualize neuron layer distribution  differences as boxplots per component."""
        pass

    def generate_text(self, input_text: str, max_new_tokens: int = 10, temperature: float = 1.0) -> str:
        """
        Generate text continuation from input text.
        
        Args:
            input_text: The input text to continue from
            max_new_tokens: Number of new tokens to generate (default: 10)
            temperature: Sampling temperature (default: 1.0)
            
        Returns:
            str: Only the newly generated text (without input text)
        """
        print(f"Input text: {input_text}")
        
        # Get input tokens
        input_tokens = self.model.to_tokens(input_text)
        input_token_length = input_tokens.shape[1]
        
        # Generate tokens
        generated_tokens = self.model.generate(
            input_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        
        # Extract only the new tokens (excluding input tokens)
        new_tokens = generated_tokens[0][input_token_length:]
        
        # Convert only the new tokens to readable text
        generated_text = self.model.to_string(new_tokens)
        print(f"\nGenerated continuation: {generated_text}")
        
        return generated_text
    def visualize_neuron_network(self, results: Dict) -> None:
        """
        Create an improved network graph visualization of moral/immoral neurons and their connections
        based on layer proximity and activation patterns.
        """
        import networkx as nx
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create a new graph
        G = nx.Graph()
        
        # Combine all neurons and mark their type
        all_neurons = [(n, 'moral') for n in results['moral_neurons']] + \
                    [(n, 'immoral') for n in results['immoral_neurons']]
        
        # Add nodes
        for neuron, ntype in all_neurons:
            G.add_node(f"L{neuron[0]}N{neuron[1]}", 
                    color='blue' if ntype == 'moral' else 'red',
                    type=ntype,
                    layer=neuron[0])
        
        # Add edges based on:
        # 1. Layer proximity
        # 2. Activation pattern similarity
        activation_diffs = results['activation_differences']
        nodes = list(G.nodes())
        
        for i, node1 in enumerate(nodes):
            layer1, neuron1 = map(int, node1[1:].split('N'))
            for node2 in nodes[i+1:]:
                layer2, neuron2 = map(int, node2[1:].split('N'))
                
                # Calculate layer distance
                layer_dist = abs(layer1 - layer2)
                
                # Only connect neurons within 2 layers of each other
                if layer_dist <= 2:
                    # Calculate activation similarity
                    act1 = activation_diffs[layer1, neuron1]
                    act2 = activation_diffs[layer2, neuron2]
                    
                    # Use mean absolute difference as similarity measure
                    similarity = 1 / (1 + torch.mean(torch.abs(act1 - act2)).item())
                    
                    # Add edge if similarity is significant
                    if similarity > 0.3:  # Adjust threshold as needed
                        G.add_edge(node1, node2, weight=similarity)
        
        # Set up the plot
        plt.figure(figsize=(15, 10))
        
        # Use a hierarchical layout based on layers
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw nodes
        node_colors = [G.nodes[node]['color'] for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.7)
        
        # Draw edges with width based on weight
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
        plt.show()
        
        # Print some network statistics
        print(f"Number of edges: {G.number_of_edges()}")
        print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
        print(f"Number of connected components: {nx.number_connected_components(G)}")
        
        return G



# %%
class MoralNeuronDescriber(MoralBehaviorAnalyzer):
    def __init__(self, model_name: str = 'gpt-4o', llm_name: str = "gpt-4o"):
        super().__init__(model_name)
        self.llm_name = llm_name
        # Initialize OpenAI client for descriptions
        self.client = openai.OpenAI()
        
    def describe_moral_neurons(self, results: Dict) -> Dict[Tuple[int, int], str]:
        """Generate descriptions for moral/immoral neurons using LLM."""
        descriptions = {}
        
        # Combine moral and immoral neurons for description
        all_neurons = results['moral_neurons'] + results['immoral_neurons']
        
        for layer, neuron in tqdm(all_neurons, desc="Describing neurons"):
            # Get exemplars for this neuron
            exemplars = self._get_neuron_exemplars(layer, neuron)
            
            # Generate description using LLM
            description = self._generate_neuron_description(
                layer, 
                neuron,
                exemplars,
                is_moral=(layer, neuron) in results['moral_neurons']
            )
            
            descriptions[(layer, neuron)] = description
            
        return descriptions
    
    def _get_neuron_exemplars(self, layer: int, neuron: int, num_exemplars: int = 5) -> List[Dict]:
        """Get top activating examples for a neuron."""
        exemplars = []
        
        # Run forward pass on validation dataset
        tokens = self.model.to_tokens(self.validation_texts)
        with torch.no_grad():
            _, cache = self.model.run_with_cache(tokens)
            activations = cache['post', layer, 'mlp'][..., neuron]
            
            # Get indices of top activating tokens
            top_indices = torch.topk(activations.flatten(), num_exemplars).indices
            batch_indices = top_indices // activations.size(1)
            token_indices = top_indices % activations.size(1)
            
            # Collect exemplars with activation values
            for batch_idx, token_idx in zip(batch_indices, token_indices):
                text = self.validation_texts[batch_idx]
                activation = activations[batch_idx, token_idx].item()
                token = self.model.to_string(tokens[batch_idx, token_idx])
                
                exemplars.append({
                    'text': text,
                    'token': token,
                    'activation': activation
                })
                
        return exemplars
    
    def _generate_neuron_description(self, layer: int, neuron: int, 
                                   exemplars: List[Dict], is_moral: bool) -> str:
        """Generate natural language description of neuron behavior using LLM."""
        
        # Construct prompt for LLM
        prompt = f"""Describe the behavior of a neuron in layer {layer} of a language model that responds to moral/ethical concepts.

The neuron appears to be {'positively' if is_moral else 'negatively'} associated with moral behavior.

Here are some example texts and tokens where this neuron strongly activates:

"""
        # Add exemplars to prompt
        for ex in exemplars:
            prompt += f"Text: {ex['text']}\n"
            prompt += f"Token: {ex['token']}\n"
            prompt += f"Activation: {ex['activation']:.3f}\n\n"
            
        prompt += "\nBased on these examples, provide a concise description of what concept or pattern this neuron appears to detect:"
        
        try:
            # Get description from LLM
            response = self.client.chat.completions.create(
                model=self.llm_name,
                messages=[
                    {"role": "system", "content": "You are an expert at interpreting neural network behavior."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating description: {e}")
            return "Description generation failed"
            
    def visualize_moral_circuits_with_descriptions(self, results: Dict, descriptions: Dict):
        """Extend visualization to include neuron descriptions."""
        # First create the base visualization
        fig = self.visualize_moral_circuits(results)
        
        # Add textbox with key neuron descriptions
        fig.text(1.1, 0.5, self._format_descriptions(descriptions),
                fontsize=8, va='center', ha='left',
                bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def _format_descriptions(self, descriptions: Dict) -> str:
        """Format neuron descriptions for visualization."""
        text = "Key Neuron Descriptions:\n\n"
        for (layer, neuron), desc in descriptions.items():
            text += f"Layer {layer} Neuron {neuron}:\n"
            text += f"{desc}\n\n"
        return text
# %%


# load list_moral_tupels
with open('../data/cm_train_contrastive_tupels.pkl', 'rb') as f:
    list_moral_tupels = pickle.load(f)



moral_pairs = list_moral_tupels
    # ("I should help the elderly cross the street. Evaluate this statement in terms of morality.", 
#      "I should ignore the elderly crossing the street. Evaluate this statement in terms of morality."),
#      ("I found a wallet on the street with 100 EURs in it. Also the person-id. My bike is broken, which cost me 90 EUR. The 100 EUR would help me to fix my bike. I will keep the wallet. Rate this behavior in moral correct to immoral. (Give no explanation)", 
#       "I found a wallet on the street with 100 EURs in it. Also the person-id. My bike is broken, which cost me 90 EUR. The 100 EUR would help me to fix my bike. I will return the wallet. Rate this behavior in moral correct to immoral. (Give no explanation)")
# ]

# %%
# Load moral_circuit_results.pkl if exists  
if os.path.exists('moral_circuit_results.pkl'):
    with open('moral_circuit_results.pkl', 'rb') as f:
        results = pickle.load(f)


# %%
moral_analyzer = MoralBehaviorAnalyzer(model=model)
# %%
# Print the output of the model for the 10 tokens after generateion
moral_analyzer.generate_text(moral_pairs[1][1], max_new_tokens=30)

# %%
results = moral_analyzer.analyze_moral_behavior(moral_pairs, significant_diff=0.2, consistency_threshold=0.5)

# %% 
# Save the results
with open('moral_circuit_results.pkl', 'wb') as f:
    pickle.dump(results, f)


# %%
moral_analyzer.visualize_moral_circuits(results)
# %%

moral_analyzer.visualize_neuron_components(results)

# Descriptions
# %%
moral_describer = MoralNeuronDescriber(model_name='google/gemma-2-9b-it', llm_name='gpt-4o')
# %%
descriptions = moral_describer.describe_moral_neurons(results)
# %%
moral_describer.visualize_moral_circuits_with_descriptions(results, descriptions)
# %%

moral_analyzer.visualize_neuron_network(results)

# %%
