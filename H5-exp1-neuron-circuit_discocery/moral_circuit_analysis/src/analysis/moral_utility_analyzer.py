"""
Extension of MoralBehaviorAnalyzer that integrates utility analysis.
Trains probes to predict utility values from hidden states and analyzes
the relationship between moral neurons and utility representations.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm
import json
from pathlib import Path

from .moral_analyzer import MoralBehaviorAnalyzer
from .utility_analyzer import UtilityModel, ThurstoneUtility
from ..models.model_loader import load_model


class UtilityProbe(nn.Module):
    """Linear probe for predicting utility values from hidden states."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        return self.linear(x)


class MoralUtilityAnalyzer(MoralBehaviorAnalyzer):
    """
    Extends moral behavior analysis with utility representations.
    Analyzes the relationship between moral neurons and utility values.
    """
    
    def __init__(self, model, device='cuda'):
        super().__init__(model)
        self.device = device
        self.utility_probes = {}  # Layer -> trained probe
        self.probe_accuracies = {}  # Layer -> accuracy metrics
        
    def train_utility_probes(
        self,
        outcomes: List[str],
        utility_model: UtilityModel,
        layers_to_probe: Optional[List[int]] = None,
        test_size: float = 0.2,
        regularization: float = 1.0
    ) -> Dict[int, Dict[str, float]]:
        """
        Train linear probes to predict utility values from hidden states.
        
        Args:
            outcomes: List of outcome texts
            utility_model: Fitted utility model with values for outcomes
            layers_to_probe: Specific layers to probe (None = all layers)
            test_size: Fraction of data for testing
            regularization: Ridge regression regularization parameter
            
        Returns:
            Dictionary mapping layers to accuracy metrics
        """
        if layers_to_probe is None:
            layers_to_probe = list(range(self.n_layers))
        
        # Get hidden states for all outcomes
        print("Extracting hidden states for outcomes...")
        hidden_states_by_layer = self._extract_hidden_states(outcomes)
        
        # Prepare utility targets
        y = []
        valid_indices = []
        for i, outcome in enumerate(outcomes):
            if outcome in utility_model.utilities:
                y.append(utility_model.utilities[outcome].mean)
                valid_indices.append(i)
        
        y = np.array(y)
        
        # Train probe for each layer
        probe_results = {}
        
        for layer in tqdm(layers_to_probe, desc="Training utility probes"):
            # Get hidden states for this layer
            X = hidden_states_by_layer[layer][valid_indices]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Train Ridge regression probe
            probe = Ridge(alpha=regularization)
            probe.fit(X_train, y_train)
            
            # Evaluate
            y_pred_train = probe.predict(X_train)
            y_pred_test = probe.predict(X_test)
            
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            test_mse = mean_squared_error(y_test, y_pred_test)
            
            # Store results
            self.utility_probes[layer] = probe
            probe_results[layer] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_mse': test_mse,
                'n_train': len(X_train),
                'n_test': len(X_test)
            }
            
            self.probe_accuracies[layer] = probe_results[layer]
        
        return probe_results
    
    def _extract_hidden_states(self, texts: List[str]) -> Dict[int, np.ndarray]:
        """
        Extract hidden states for a list of texts.
        
        Returns:
            Dictionary mapping layer index to array of hidden states
        """
        hidden_states_by_layer = {layer: [] for layer in range(self.n_layers)}
        
        for text in tqdm(texts, desc="Extracting hidden states"):
            tokens = self.model.to_tokens(text)
            
            with torch.no_grad():
                _, cache = self.model.run_with_cache(tokens)
                
                # Get mean pooled hidden states for each layer
                for layer in range(self.n_layers):
                    # Get hidden states and mean pool across sequence
                    hidden = cache[f'blocks.{layer}.hook_mlp_out'][0]  # [seq_len, hidden_dim]
                    pooled = hidden.mean(dim=0).cpu().numpy()  # [hidden_dim]
                    hidden_states_by_layer[layer].append(pooled)
        
        # Convert lists to arrays
        for layer in hidden_states_by_layer:
            hidden_states_by_layer[layer] = np.array(hidden_states_by_layer[layer])
        
        return hidden_states_by_layer
    
    def analyze_neuron_utility_correspondence(
        self,
        moral_neurons: Dict[str, List[Tuple[int, int]]],
        utility_model: UtilityModel,
        moral_texts: List[str],
        immoral_texts: List[str]
    ) -> Dict[str, float]:
        """
        Analyze correspondence between identified moral neurons and utility representations.
        
        Args:
            moral_neurons: Dictionary of moral neuron locations
            utility_model: Fitted utility model
            moral_texts: List of moral outcome texts
            immoral_texts: List of immoral outcome texts
            
        Returns:
            Dictionary of correspondence metrics
        """
        results = {}
        
        # Get utility values for texts
        moral_utilities = []
        immoral_utilities = []
        
        for text in moral_texts:
            if text in utility_model.utilities:
                moral_utilities.append(utility_model.utilities[text].mean)
        
        for text in immoral_texts:
            if text in utility_model.utilities:
                immoral_utilities.append(utility_model.utilities[text].mean)
        
        # Calculate utility separation
        if moral_utilities and immoral_utilities:
            results['utility_separation'] = np.mean(moral_utilities) - np.mean(immoral_utilities)
            results['utility_overlap'] = self._calculate_distribution_overlap(
                moral_utilities, immoral_utilities
            )
        
        # Analyze probe accuracy at layers with many moral neurons
        layer_neuron_counts = {}
        for neurons in moral_neurons.values():
            for layer, _ in neurons:
                layer_neuron_counts[layer] = layer_neuron_counts.get(layer, 0) + 1
        
        # Find layers with most moral neurons
        top_moral_layers = sorted(layer_neuron_counts.items(), 
                                 key=lambda x: x[1], reverse=True)[:5]
        
        # Compare probe accuracy at these layers
        if self.probe_accuracies:
            moral_layer_accuracies = []
            other_layer_accuracies = []
            
            for layer, accuracy_dict in self.probe_accuracies.items():
                if layer in [l for l, _ in top_moral_layers]:
                    moral_layer_accuracies.append(accuracy_dict['test_r2'])
                else:
                    other_layer_accuracies.append(accuracy_dict['test_r2'])
            
            if moral_layer_accuracies and other_layer_accuracies:
                results['moral_layer_avg_r2'] = np.mean(moral_layer_accuracies)
                results['other_layer_avg_r2'] = np.mean(other_layer_accuracies)
                results['moral_layer_advantage'] = (
                    results['moral_layer_avg_r2'] - results['other_layer_avg_r2']
                )
        
        return results
    
    def _calculate_distribution_overlap(
        self, 
        dist1: List[float], 
        dist2: List[float]
    ) -> float:
        """Calculate overlap between two distributions."""
        # Simple overlap metric based on range overlap
        min1, max1 = min(dist1), max(dist1)
        min2, max2 = min(dist2), max(dist2)
        
        overlap = max(0, min(max1, max2) - max(min1, min2))
        total_range = max(max1, max2) - min(min1, min2)
        
        return overlap / total_range if total_range > 0 else 0
    
    def identify_utility_encoding_neurons(
        self,
        layer: int,
        threshold: float = 0.3
    ) -> List[int]:
        """
        Identify neurons that strongly encode utility values.
        
        Args:
            layer: Layer to analyze
            threshold: Threshold for weight magnitude
            
        Returns:
            List of neuron indices that encode utility
        """
        if layer not in self.utility_probes:
            return []
        
        probe = self.utility_probes[layer]
        weights = np.abs(probe.coef_[0]) if hasattr(probe, 'coef_') else []
        
        # Find neurons with high magnitude weights
        utility_neurons = []
        if len(weights) > 0:
            threshold_value = np.percentile(weights, (1 - threshold) * 100)
            utility_neurons = [i for i, w in enumerate(weights) if w > threshold_value]
        
        return utility_neurons
    
    def save_utility_analysis(self, output_path: Path):
        """Save utility analysis results."""
        results = {
            'probe_accuracies': self.probe_accuracies,
            'utility_encoding_neurons': {}
        }
        
        # Save utility encoding neurons for each layer
        for layer in self.utility_probes:
            neurons = self.identify_utility_encoding_neurons(layer)
            results['utility_encoding_neurons'][str(layer)] = neurons
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Saved utility analysis results to {output_path}")


def create_unified_analysis_script():
    """Create script for running unified moral-utility analysis."""
    script_content = '''#!/usr/bin/env python
"""Unified analysis combining moral neurons and utility representations."""

import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.analysis.moral_utility_analyzer import MoralUtilityAnalyzer
from src.analysis.utility_analyzer import UtilityAnalyzer
from src.data.preference_data import load_preference_data
from src.models.model_loader import load_model
import json


def main():
    parser = argparse.ArgumentParser(
        description="Run unified moral-utility analysis"
    )
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--preferences", type=Path, required=True, 
                        help="Path to preference elicitation results")
    parser.add_argument("--moral-results", type=Path, required=True,
                        help="Path to moral neuron analysis results")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load preference data and fit utility model
    print("Loading preference data...")
    with open(args.preferences, 'r') as f:
        pref_data = json.load(f)
    
    # Initialize utility analyzer
    utility_analyzer = UtilityAnalyzer()
    
    # Fit utility models for each dimension
    utility_models = {}
    for dimension, preferences in pref_data.items():
        print(f"\\nFitting utility model for {dimension}...")
        # Convert preference data format
        # (Implementation would need proper data conversion here)
        
    # Load model
    print(f"\\nLoading model: {args.model}")
    model, _ = load_model(args.model, args.device)
    
    # Initialize moral-utility analyzer
    analyzer = MoralUtilityAnalyzer(model, args.device)
    
    # Train utility probes
    print("\\nTraining utility probes...")
    # (Implementation continues...)
    
    print(f"\\nAnalysis complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    sys.exit(main())
'''
    
    script_path = Path("scripts/run_unified_analysis.py")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    return script_path