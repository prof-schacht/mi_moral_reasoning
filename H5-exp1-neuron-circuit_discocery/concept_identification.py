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

@dataclass
class CoactivationPattern:
    neurons: Set[Tuple[int, int]]  # Set of (layer, neuron) pairs
    strength: float                # Co-activation strength
    frequency: int                 # How often this pattern occurs
    exemplars: List[str]          # Example texts triggering this pattern

def batch_pearson_correlation(x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
    """Compute pearson correlation between batches of vectors efficiently on GPU."""
    if y is None:
        y = x
    
    # Center the data
    x = x - x.mean(dim=0, keepdim=True)
    if y is not x:
        y = y - y.mean(dim=0, keepdim=True)
    
    # Compute correlation matrix
    x_std = torch.sqrt(torch.sum(x**2, dim=0, keepdim=True))
    y_std = torch.sqrt(torch.sum(y**2, dim=0, keepdim=True))
    
    corr = torch.mm(x.t(), y) / (x_std.t() @ y_std)
    return corr

class NeuronActivationCollector:
    def __init__(self, model_name: str = 'gpt2', 
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
            self.model = HookedTransformer.from_pretrained(
                model_name,
                device=self.device,
                dtype=self.dtype,
                default_padding_side="right",
                center_writing_weights=True,
                center_unembed=True,
                fold_ln=True,
                move_to_device=True
            )
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
                
                all_activations.append(torch.stack(batch_activations))
        
        # Combine all batches
        return torch.cat(all_activations, dim=0)

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

# Example usage
if __name__ == "__main__":
    # Set CUDA device options for better error handling
    torch.backends.cuda.matmul.allow_tf32 = False  # For better numerical precision
    torch.backends.cudnn.allow_tf32 = False
    
    # Initialize collector with smaller batch size for stability
    try:
        collector = NeuronActivationCollector('gpt2', batch_size=16)
    except Exception as e:
        print(f"Error initializing collector: {e}")
        print("Trying with CPU...")
        collector = NeuronActivationCollector('gpt2', batch_size=8)
    
    # Example texts
    texts = [
        "The cat sat on the mat.",
        "Python is a programming language.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models process data.",
        "OpenAI released GPT-4 in 2023."
    ]

    # Perform hierarchical analysis
    analysis_results = collector.analyze_hierarchical_patterns(texts)

    # Print results
    print("\nAnalysis Results:")
    print(f"Found {len(analysis_results['patterns'])} co-activation patterns")
    print(f"Found {len(analysis_results['cross_layer_patterns'])} cross-layer patterns")
    print(f"Found {len(analysis_results['pattern_hierarchies'])} hierarchical relationships")

    # Visualize patterns
    fig = collector.visualize_patterns()
    plt.show()

    # Print some example patterns
    print("\nExample Patterns:")
    for i, pattern in enumerate(collector.coactivation_patterns[:3]):  # Show first 3 patterns
        print(f"\nPattern {i+1}:")
        print(f"Size: {len(pattern.neurons)} neurons")
        print(f"Strength: {pattern.strength:.3f}")
        print(f"Frequency: {pattern.frequency}")
        print("Example text triggering this pattern:", pattern.exemplars[0] if pattern.exemplars else "No examples")

# %%
