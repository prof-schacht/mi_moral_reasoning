import torch
from typing import List, Tuple, Dict, Optional
from tqdm.auto import tqdm
from .neuron_collector import NeuronActivationCollector
from transformer_lens import HookedTransformer
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class TemporalPattern:
    layer: int
    neuron: int
    pattern_type: str  # 'build-up', 'spike', 'sustained', etc.
    start_token: int
    duration: int
    peak_activation: float
    context: str

class MoralBehaviorAnalyzer(NeuronActivationCollector):
    def __init__(self, model: HookedTransformer):
        super().__init__(model)
        
    def analyze_moral_behavior(self, moral_pairs: List[Tuple[str, str]], 
                             significant_diff: float = 0.005,
                             consistency_threshold: float = 0.55,
                             temporal_window: int = 5) -> Dict:
        """
        Analyze sequence-wide moral behavior patterns.
        
        Args:
            moral_pairs: List of (moral_text, immoral_text) pairs
            significant_diff: Threshold for considering activation differences significant
            consistency_threshold: Threshold for neuron response consistency
            temporal_window: Window size for detecting temporal patterns
            
        Returns:
            Dictionary containing analysis results including temporal patterns
        """
        moral_activations = []
        immoral_activations = []
        moral_lengths = []
        immoral_lengths = []
        
        # Get sequence-wide activations for both moral and immoral texts
        for moral_text, immoral_text in tqdm(moral_pairs, desc="Analyzing moral pairs"):
            moral_acts, moral_len = self._get_sequence_activations(moral_text)
            immoral_acts, immoral_len = self._get_sequence_activations(immoral_text)
            
            moral_activations.append(moral_acts)
            immoral_activations.append(immoral_acts)
            moral_lengths.append(moral_len)
            immoral_lengths.append(immoral_len)
            
        return self._analyze_moral_differences(
            moral_pairs=moral_pairs,
            moral_acts=moral_activations,
            immoral_acts=immoral_activations,
            moral_lengths=moral_lengths,
            immoral_lengths=immoral_lengths,
            significant_diff=significant_diff,
            consistency_threshold=consistency_threshold,
            temporal_window=temporal_window
        )
    
    def _get_sequence_activations(self, text: str) -> Tuple[torch.Tensor, int]:
        """Get activations for the entire sequence."""
        tokens = self.model.to_tokens(text)
        seq_length = tokens.shape[1]
        
        with torch.no_grad():
            _, cache = self.model.run_with_cache(tokens)
            
            # Get activations for all layers and all positions
            # Shape: [n_layers, seq_length, n_neurons]
            activations = torch.stack([
                cache['post', layer_idx, 'mlp'][0]
                for layer_idx in range(self.n_layers)
            ])
                
        return activations, seq_length
    
    def _detect_temporal_patterns(self, 
                                activations: torch.Tensor,
                                layer: int,
                                neuron: int,
                                window_size: int = 5,
                                threshold: float = 0.5) -> List[TemporalPattern]:
        """
        Detect temporal activation patterns for a specific neuron.
        
        Args:
            activations: Tensor of shape [seq_length, n_neurons]
            layer: Layer index
            neuron: Neuron index
            window_size: Size of window for pattern detection
            threshold: Activation threshold for pattern detection
        """
        patterns = []
        neuron_acts = activations[layer, :, neuron].cpu().numpy()
        seq_length = len(neuron_acts)
        
        # Detect build-up patterns (steadily increasing activation)
        for i in range(seq_length - window_size):
            window = neuron_acts[i:i + window_size]
            if np.all(np.diff(window) > 0) and max(window) > threshold:
                patterns.append(TemporalPattern(
                    layer=layer,
                    neuron=neuron,
                    pattern_type='build-up',
                    start_token=i,
                    duration=window_size,
                    peak_activation=float(max(window)),
                    context=f"tokens_{i}_to_{i+window_size}"
                ))
        
        # Detect spike patterns (sudden high activation)
        for i in range(1, seq_length - 1):
            if (neuron_acts[i] > threshold and 
                neuron_acts[i] > neuron_acts[i-1] * 1.5 and
                neuron_acts[i] > neuron_acts[i+1] * 1.5):
                patterns.append(TemporalPattern(
                    layer=layer,
                    neuron=neuron,
                    pattern_type='spike',
                    start_token=i,
                    duration=1,
                    peak_activation=float(neuron_acts[i]),
                    context=f"token_{i}"
                ))
        
        # Detect sustained activation patterns
        i = 0
        while i < seq_length:
            if neuron_acts[i] > threshold:
                duration = 1
                while i + duration < seq_length and neuron_acts[i + duration] > threshold:
                    duration += 1
                if duration >= window_size:
                    patterns.append(TemporalPattern(
                        layer=layer,
                        neuron=neuron,
                        pattern_type='sustained',
                        start_token=i,
                        duration=duration,
                        peak_activation=float(max(neuron_acts[i:i+duration])),
                        context=f"tokens_{i}_to_{i+duration}"
                    ))
                i += duration
            i += 1
            
        return patterns
    
    def _analyze_moral_differences(self, 
                                moral_pairs: List[Tuple[str, str]],
                                moral_acts: List[torch.Tensor],
                                immoral_acts: List[torch.Tensor],
                                moral_lengths: List[int],
                                immoral_lengths: List[int],
                                significant_diff: float = 0.005,
                                consistency_threshold: float = 0.55,
                                temporal_window: int = 5) -> Dict:
        """
        Analyze sequence-wide differences between moral/immoral choices.
        
        Args:
            moral_pairs: Original moral/immoral text pairs
            moral_acts: List of moral activation tensors
            immoral_acts: List of immoral activation tensors
            moral_lengths: List of moral sequence lengths
            immoral_lengths: List of immoral sequence lengths
            significant_diff: Threshold for considering differences significant
            consistency_threshold: Threshold for neuron response consistency
            temporal_window: Window size for temporal pattern detection
        
        Returns dictionary with:
        - moral_neurons: List of (layer, neuron) that consistently activate more for moral choices
        - immoral_neurons: List of (layer, neuron) that consistently activate more for immoral choices
        - temporal_patterns: Dictionary of temporal activation patterns
        - key_trigger_points: Points where moral/immoral paths diverge significantly
        - layer_importance: Ranking of layers by their role in moral decisions
        """
        # Pad sequences to max length for comparison
        max_length = max(max(moral_lengths), max(immoral_lengths))
        padded_moral = torch.zeros(len(moral_acts), self.n_layers, max_length, self.n_neurons, device=self.device)
        padded_immoral = torch.zeros_like(padded_moral)
        
        for i, (moral, immoral) in enumerate(zip(moral_acts, immoral_acts)):
            padded_moral[i, :, :moral_lengths[i]] = moral
            padded_immoral[i, :, :immoral_lengths[i]] = immoral
        
        # Compute differences across sequences
        diff = padded_moral - padded_immoral
        mean_diff = diff.mean(dim=0)  # Average across samples
        
        # Enhanced debug information
        max_diff = torch.abs(mean_diff).max().item()
        mean_abs_diff = torch.abs(mean_diff).mean().item()
        print(f"\nActivation difference statistics:")
        print(f"Maximum difference: {max_diff:.6f}")
        print(f"Mean absolute difference: {mean_abs_diff:.6f}")
        
        # Distribution of differences
        all_diffs = mean_diff.flatten()
        percentiles = torch.tensor([0, 25, 50, 75, 100], device=self.device)
        diff_percentiles = torch.quantile(all_diffs, percentiles.float() / 100)
        
        # Move to CPU for printing
        diff_percentiles = diff_percentiles.cpu()
        print("\nDifference distribution percentiles:")
        for p, v in zip(percentiles.cpu(), diff_percentiles):
            print(f"{p}th percentile: {v:.6f}")
        
        # Find consistently different neurons
        moral_neurons = []
        immoral_neurons = []
        temporal_patterns = defaultdict(list)
        sample_wise_means = {}
        position_consistency = {}
        
        # Track statistics about why neurons are rejected
        consistency_fails = 0
        significance_fails = 0
        high_consistency_count = 0
        
        # Enhanced consistency analysis
        consistency_distribution = defaultdict(int)
        significance_distribution = defaultdict(int)
        
        for layer in range(self.n_layers):
            for neuron in range(self.n_neurons):
                neuron_diff = diff[:, layer, :, neuron]
                consistency = (neuron_diff > 0).float().mean()
                mean_neuron_diff = mean_diff[layer, :, neuron].mean()
                
                # Track consistency distribution in bins of 0.1
                consistency_bin = round(float(consistency) * 10) / 10
                consistency_distribution[consistency_bin] += 1
                
                # Track significance distribution in bins of 0.001
                significance_bin = round(float(abs(mean_neuron_diff)) * 1000) / 1000
                significance_distribution[significance_bin] += 1
                
                # Track statistics
                if consistency > consistency_threshold:
                    high_consistency_count += 1
                    if abs(mean_neuron_diff) < significant_diff:
                        significance_fails += 1
                else:
                    consistency_fails += 1
                
                # Only show debug information for neurons that pass both thresholds
                if consistency > consistency_threshold and abs(mean_neuron_diff) > significant_diff:
                    neuron_type = "Moral" if mean_neuron_diff > significant_diff else "Immoral"
                    print(f"\n{neuron_type} Neuron ({layer}, {neuron}):")
                    print(f"Mean difference: {mean_neuron_diff:.6f}")
                    print(f"Consistency: {consistency:.6f}")
                    
                    # Analyze per-position behavior
                    pos_consistency = [(pos, (neuron_diff[:, pos] > 0).float().mean().item())
                                    for pos in range(neuron_diff.shape[1])]
                    most_consistent_pos = sorted(pos_consistency, key=lambda x: abs(x[1]-0.5), reverse=True)[:3]
                    print("Most consistent positions:", most_consistent_pos)
                    
                    # Store position consistency for visualization
                    position_consistency[(layer, neuron)] = pos_consistency
                    
                    # Print and store sample-wise statistics
                    sample_means = neuron_diff.mean(dim=1).cpu()
                    sample_wise_means[(layer, neuron)] = sample_means.tolist()
                    print(f"Sample-wise means: {sample_means.tolist()}")
                
                if consistency > consistency_threshold:
                    if mean_neuron_diff > significant_diff:
                        moral_neurons.append((layer, neuron))
                    elif mean_neuron_diff < -significant_diff:
                        immoral_neurons.append((layer, neuron))
        
        # Print enhanced statistics
        print("\nConsistency Distribution:")
        for consistency, count in sorted(consistency_distribution.items()):
            print(f"Consistency {consistency:.1f}-{consistency+0.1:.1f}: {count} neurons")
        
        print("\nSignificance Distribution:")
        significant_counts = sorted(
            [(sig, count) for sig, count in significance_distribution.items() if sig > 0.001],
            key=lambda x: x[0]
        )
        for significance, count in significant_counts[:10]:  # Show top 10 significance levels
            print(f"Difference {significance:.4f}: {count} neurons")
        
        # Print summary statistics
        total_neurons = self.n_layers * self.n_neurons
        print(f"\nNeuron filtering statistics:")
        print(f"Total neurons: {total_neurons}")
        print(f"Neurons with high consistency (>{consistency_threshold}): {high_consistency_count}")
        print(f"Neurons failed consistency check: {consistency_fails}")
        print(f"Neurons failed significance check: {significance_fails}")
        
        # Find key trigger points with token context
        position_importance = []
        token_contexts = []
        
        # Get sample tokens for context
        sample_moral = moral_acts[0] if moral_acts else None
        if sample_moral is not None:
            # Get tokens for the first moral text
            sample_tokens = self.model.to_tokens(moral_pairs[0][0])
            token_texts = [self.model.to_string(token) for token in sample_tokens[0]]
            
            for pos in range(max_length):
                pos_diff = torch.abs(mean_diff[:, pos, :]).mean()
                if not torch.isnan(pos_diff):  # Ignore padded positions
                    position_importance.append((pos, pos_diff.item()))
                    
                    # Get context around this position
                    start_idx = max(0, pos - 2)
                    end_idx = min(len(token_texts), pos + 3)
                    context = token_texts[start_idx:end_idx]
                    token_contexts.append({
                        'position': pos,
                        'token': token_texts[pos] if pos < len(token_texts) else '<pad>',
                        'context': ' '.join(context)
                    })
        
        position_importance.sort(key=lambda x: x[1], reverse=True)
        key_trigger_points = position_importance[:5]  # Top 5 important positions
        
        # Print trigger point contexts
        print("\nKey trigger points with context:")
        for pos, diff_val in key_trigger_points:
            context = next((ctx for ctx in token_contexts if ctx['position'] == pos), None)
            if context:
                print(f"Position {pos} (diff: {diff_val:.6f}):")
                print(f"Token: '{context['token']}'")
                print(f"Context: '...{context['context']}...'")
                print()

        # Compute layer importance
        layer_importance = []
        for layer in range(self.n_layers):
            layer_diff = torch.abs(mean_diff[layer]).mean()
            layer_importance.append((layer, layer_diff.item()))
        
        layer_importance.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'moral_neurons': moral_neurons,
            'immoral_neurons': immoral_neurons,
            'temporal_patterns': dict(temporal_patterns),
            'key_trigger_points': key_trigger_points,
            'key_trigger_contexts': token_contexts[:5],
            'layer_importance': layer_importance,
            'activation_differences': mean_diff,
            'sample_wise_means': sample_wise_means,
            'position_consistency': position_consistency,
            'consistency_distribution': dict(consistency_distribution),
            'significance_distribution': dict(significance_distribution)
        }
        
    def generate_text(self, input_text: str, max_new_tokens: int = 10, temperature: float = 1.0) -> str:
        """Generate text continuation from input text."""
        input_tokens = self.model.to_tokens(input_text)
        input_token_length = input_tokens.shape[1]
        
        generated_tokens = self.model.generate(
            input_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        
        new_tokens = generated_tokens[0][input_token_length:]
        generated_text = self.model.to_string(new_tokens)
        
        return generated_text 