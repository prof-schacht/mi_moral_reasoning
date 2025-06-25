"""
Enhanced ablation analysis that incorporates utility-based evaluation metrics.
Tests how ablating moral neurons affects utility functions and preference coherence.
"""

import torch
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
import json
from pathlib import Path

from .ablation import AblationAnalysis
from .utility_analyzer import UtilityAnalyzer, UtilityModel
from .preference_elicitor import PreferenceElicitor
from ..data.preference_data import PreferenceScenario, PreferenceData


@dataclass
class UtilityAblationResult:
    """Results from utility-based ablation analysis."""
    dimension: str
    cluster: str
    ablation_value: float
    
    # Original utility metrics
    original_accuracy: float
    original_transitivity: float
    original_completeness: float
    
    # Ablated utility metrics
    ablated_accuracy: float
    ablated_transitivity: float
    ablated_completeness: float
    
    # Change metrics
    accuracy_change: float
    transitivity_change: float
    completeness_change: float
    
    # Preference distribution changes
    preference_shifts: Dict[str, float]  # Scenario -> shift magnitude
    avg_preference_shift: float
    max_preference_shift: float
    
    # Utility distortion
    utility_correlation: float  # Correlation between original and ablated utilities
    utility_rmse: float  # RMSE between original and ablated utilities


class UtilityAblationAnalysis(AblationAnalysis):
    """
    Extended ablation analysis that measures effects on utility representations.
    """
    
    def __init__(self, model_name: str, model=None, tokenizer=None, device='cuda'):
        super().__init__(model_name, model, tokenizer, device)
        self.preference_elicitor = PreferenceElicitor(model_name, device)
        self.utility_analyzer = UtilityAnalyzer()
    
    def run_utility_ablation_analysis(
        self,
        preference_data: PreferenceData,
        neurons_to_ablate: List[Tuple[int, int]],
        original_utility_model: UtilityModel,
        ablation_value: float = -20.0,
        dimension: str = "unknown",
        cluster: str = "unknown",
        num_samples_per_scenario: int = 4
    ) -> UtilityAblationResult:
        """
        Run ablation analysis with utility-based evaluation.
        
        Args:
            preference_data: Preference scenarios to test
            neurons_to_ablate: List of (layer, neuron) tuples to ablate
            original_utility_model: Utility model before ablation
            ablation_value: Value to set ablated neurons to
            dimension: Moral dimension being tested
            cluster: Neuron cluster being ablated
            num_samples_per_scenario: Samples for preference elicitation
            
        Returns:
            UtilityAblationResult with comprehensive metrics
        """
        print(f"Running utility ablation analysis for {dimension} - {cluster}")
        print(f"Ablating {len(neurons_to_ablate)} neurons with value {ablation_value}")
        
        # Apply ablation
        self._apply_ablation(neurons_to_ablate, ablation_value)
        
        # Re-elicit preferences with ablated model
        print("Re-eliciting preferences with ablated model...")
        ablated_preferences = []
        
        for scenario in tqdm(preference_data.scenarios[:50], desc="Eliciting preferences"):
            pref = self.preference_elicitor.elicit_aggregated_preference(
                scenario,
                num_samples=num_samples_per_scenario
            )
            ablated_preferences.append(pref)
        
        # Fit new utility model
        print("Fitting utility model to ablated preferences...")
        ablated_utility_model = self.utility_analyzer.fit_thurstonian_model(
            ablated_preferences
        )
        
        # Calculate preference shifts
        preference_shifts = self._calculate_preference_shifts(
            preference_data.scenarios[:50],
            original_utility_model,
            ablated_utility_model
        )
        
        # Calculate utility distortion
        utility_correlation, utility_rmse = self._calculate_utility_distortion(
            original_utility_model,
            ablated_utility_model
        )
        
        # Remove ablation
        self._remove_ablation(neurons_to_ablate)
        
        # Create result
        result = UtilityAblationResult(
            dimension=dimension,
            cluster=cluster,
            ablation_value=ablation_value,
            
            # Original metrics
            original_accuracy=original_utility_model.accuracy,
            original_transitivity=original_utility_model.transitivity_score,
            original_completeness=original_utility_model.completeness_score,
            
            # Ablated metrics
            ablated_accuracy=ablated_utility_model.accuracy,
            ablated_transitivity=ablated_utility_model.transitivity_score,
            ablated_completeness=ablated_utility_model.completeness_score,
            
            # Changes
            accuracy_change=ablated_utility_model.accuracy - original_utility_model.accuracy,
            transitivity_change=ablated_utility_model.transitivity_score - original_utility_model.transitivity_score,
            completeness_change=ablated_utility_model.completeness_score - original_utility_model.completeness_score,
            
            # Preference shifts
            preference_shifts=preference_shifts,
            avg_preference_shift=np.mean(list(preference_shifts.values())),
            max_preference_shift=max(preference_shifts.values()) if preference_shifts else 0,
            
            # Utility distortion
            utility_correlation=utility_correlation,
            utility_rmse=utility_rmse
        )
        
        return result
    
    def _apply_ablation(self, neurons: List[Tuple[int, int]], value: float):
        """Apply ablation to specified neurons."""
        def ablation_hook(activation, hook, layer_idx, neuron_indices):
            # Set specified neurons to ablation value
            for neuron_idx in neuron_indices:
                activation[:, :, neuron_idx] = value
            return activation
        
        # Group neurons by layer
        neurons_by_layer = {}
        for layer, neuron in neurons:
            if layer not in neurons_by_layer:
                neurons_by_layer[layer] = []
            neurons_by_layer[layer].append(neuron)
        
        # Add hooks
        self.ablation_hooks = []
        for layer, neuron_indices in neurons_by_layer.items():
            hook_fn = lambda act, hook, l=layer, n=neuron_indices: ablation_hook(act, hook, l, n)
            hook = self.model.blocks[layer].mlp.hook_post.register_forward_hook(hook_fn)
            self.ablation_hooks.append(hook)
    
    def _remove_ablation(self, neurons: List[Tuple[int, int]]):
        """Remove ablation hooks."""
        if hasattr(self, 'ablation_hooks'):
            for hook in self.ablation_hooks:
                hook.remove()
            self.ablation_hooks = []
    
    def _calculate_preference_shifts(
        self,
        scenarios: List[PreferenceScenario],
        original_model: UtilityModel,
        ablated_model: UtilityModel
    ) -> Dict[str, float]:
        """Calculate how preferences shifted due to ablation."""
        shifts = {}
        
        for scenario in scenarios:
            # Get original preference probability
            orig_prob = original_model.get_preference_probability(
                scenario.option_a, scenario.option_b
            )
            
            # Get ablated preference probability
            ablated_prob = ablated_model.get_preference_probability(
                scenario.option_a, scenario.option_b
            )
            
            # Calculate shift magnitude
            shift = abs(orig_prob - ablated_prob)
            shifts[f"{scenario.option_a[:30]}...vs...{scenario.option_b[:30]}"] = shift
        
        return shifts
    
    def _calculate_utility_distortion(
        self,
        original_model: UtilityModel,
        ablated_model: UtilityModel
    ) -> Tuple[float, float]:
        """Calculate correlation and RMSE between original and ablated utilities."""
        # Get common outcomes
        common_outcomes = set(original_model.utilities.keys()) & set(ablated_model.utilities.keys())
        
        if not common_outcomes:
            return 0.0, float('inf')
        
        # Extract utility values
        orig_values = []
        ablated_values = []
        
        for outcome in common_outcomes:
            orig_values.append(original_model.utilities[outcome].mean)
            ablated_values.append(ablated_model.utilities[outcome].mean)
        
        orig_values = np.array(orig_values)
        ablated_values = np.array(ablated_values)
        
        # Calculate correlation
        correlation = np.corrcoef(orig_values, ablated_values)[0, 1]
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((orig_values - ablated_values) ** 2))
        
        return correlation, rmse
    
    def save_utility_ablation_results(
        self,
        results: List[UtilityAblationResult],
        output_path: Path
    ):
        """Save utility ablation results to JSON."""
        data = []
        
        for result in results:
            data.append({
                'dimension': result.dimension,
                'cluster': result.cluster,
                'ablation_value': result.ablation_value,
                
                'original_metrics': {
                    'accuracy': result.original_accuracy,
                    'transitivity': result.original_transitivity,
                    'completeness': result.original_completeness
                },
                
                'ablated_metrics': {
                    'accuracy': result.ablated_accuracy,
                    'transitivity': result.ablated_transitivity,
                    'completeness': result.ablated_completeness
                },
                
                'changes': {
                    'accuracy_change': result.accuracy_change,
                    'transitivity_change': result.transitivity_change,
                    'completeness_change': result.completeness_change
                },
                
                'preference_analysis': {
                    'avg_shift': result.avg_preference_shift,
                    'max_shift': result.max_preference_shift,
                    'n_scenarios': len(result.preference_shifts)
                },
                
                'utility_distortion': {
                    'correlation': result.utility_correlation,
                    'rmse': result.utility_rmse
                }
            })
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved utility ablation results to {output_path}")


def create_utility_ablation_script():
    """Create script for running utility-based ablation analysis."""
    script_path = Path("scripts/run_utility_ablation.py")
    script_content = '''#!/usr/bin/env python
"""Run utility-based ablation analysis."""

import argparse
from pathlib import Path
import json
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.analysis.utility_ablation import UtilityAblationAnalysis
from src.analysis.utility_analyzer import UtilityAnalyzer
from src.data.preference_data import load_preference_data


def main():
    parser = argparse.ArgumentParser(
        description="Run utility-based ablation analysis"
    )
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--neurons", type=Path, required=True, 
                        help="Path to neuron identification results")
    parser.add_argument("--preferences", type=Path, required=True,
                        help="Path to preference data")
    parser.add_argument("--utility-model", type=Path, required=True,
                        help="Path to fitted utility model")
    parser.add_argument("--dimension", type=str, required=True,
                        help="Moral dimension to analyze")
    parser.add_argument("--cluster", type=str, required=True,
                        help="Neuron cluster to ablate")
    parser.add_argument("--ablation-value", type=float, default=-20.0,
                        help="Ablation value")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output path for results")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    preference_data = load_preference_data(args.preferences)
    
    # Load utility model
    utility_analyzer = UtilityAnalyzer()
    original_utility_model = utility_analyzer.load_utility_model(args.utility_model)
    
    # Load neurons to ablate
    with open(args.neurons, 'r') as f:
        neuron_data = json.load(f)
    
    # Extract neurons for specified cluster
    # (Implementation would depend on neuron data format)
    neurons_to_ablate = []  # Placeholder
    
    # Initialize ablation analyzer
    analyzer = UtilityAblationAnalysis(args.model, device=args.device)
    
    # Run analysis
    result = analyzer.run_utility_ablation_analysis(
        preference_data=preference_data,
        neurons_to_ablate=neurons_to_ablate,
        original_utility_model=original_utility_model,
        ablation_value=args.ablation_value,
        dimension=args.dimension,
        cluster=args.cluster
    )
    
    # Save results
    analyzer.save_utility_ablation_results([result], args.output)
    
    print(f"\\nAnalysis complete! Results saved to {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    return script_path