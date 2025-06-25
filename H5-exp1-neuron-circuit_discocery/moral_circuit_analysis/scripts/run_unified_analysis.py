#!/usr/bin/env python
"""Unified analysis combining moral neurons and utility representations."""

import argparse
from pathlib import Path
import json
import pickle
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.analysis.moral_utility_analyzer import MoralUtilityAnalyzer
from src.analysis.utility_analyzer import UtilityAnalyzer
from src.analysis.preference_elicitor import AggregatedPreference
from src.data.preference_data import PreferenceScenario, load_preference_data
from src.models.model_loader import load_model


def load_moral_neuron_results(path: Path):
    """Load moral neuron analysis results."""
    if path.suffix == '.pkl':
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        with open(path, 'r') as f:
            return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Run unified moral-utility analysis"
    )
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--preferences", type=Path, required=True, 
                        help="Path to preference elicitation results")
    parser.add_argument("--utility-models", type=Path, required=True,
                        help="Path to directory with fitted utility models")
    parser.add_argument("--moral-results", type=Path, required=True,
                        help="Path to moral neuron analysis results")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model: {args.model}")
    model, tokenizer = load_model(args.model, args.device)
    
    # Initialize moral-utility analyzer
    analyzer = MoralUtilityAnalyzer(model, args.device)
    
    # Load utility models
    print("\nLoading utility models...")
    utility_models = {}
    for model_file in args.utility_models.glob("*_utility_model.json"):
        dimension = model_file.stem.replace("_utility_model", "")
        utility_analyzer = UtilityAnalyzer()
        utility_models[dimension] = utility_analyzer.load_utility_model(model_file)
        print(f"  - Loaded utility model for {dimension}")
    
    # Load moral neuron results
    print("\nLoading moral neuron results...")
    moral_results = load_moral_neuron_results(args.moral_results)
    
    # Extract moral neurons by dimension
    moral_neurons_by_dimension = {}
    if isinstance(moral_results, dict):
        if 'moral_neurons' in moral_results:
            # Format: {dimension: [(layer, neuron), ...]}
            for key, neurons in moral_results['moral_neurons'].items():
                if isinstance(neurons, list):
                    moral_neurons_by_dimension[key] = neurons
    
    # For each dimension, analyze neuron-utility correspondence
    all_results = {}
    
    for dimension in utility_models:
        if dimension not in moral_neurons_by_dimension:
            print(f"\nSkipping {dimension} - no moral neurons found")
            continue
            
        print(f"\nAnalyzing {dimension}...")
        
        # Get outcomes for this dimension
        outcomes = list(utility_models[dimension].utilities.keys())
        
        # Train utility probes
        print("  - Training utility probes...")
        probe_results = analyzer.train_utility_probes(
            outcomes=outcomes,
            utility_model=utility_models[dimension],
            test_size=0.2
        )
        
        # Find best probe layer
        best_layer = max(probe_results.items(), key=lambda x: x[1]['test_r2'])[0]
        print(f"  - Best probe layer: {best_layer} (R² = {probe_results[best_layer]['test_r2']:.3f})")
        
        # Identify utility-encoding neurons
        utility_neurons = analyzer.identify_utility_encoding_neurons(best_layer)
        print(f"  - Found {len(utility_neurons)} utility-encoding neurons at layer {best_layer}")
        
        # Get moral neurons for this dimension
        moral_neurons = moral_neurons_by_dimension[dimension]
        moral_neurons_at_layer = [n for l, n in moral_neurons if l == best_layer]
        
        # Calculate overlap
        overlap = len(set(utility_neurons) & set(moral_neurons_at_layer))
        
        results = {
            'dimension': dimension,
            'probe_results': probe_results,
            'best_probe_layer': best_layer,
            'best_probe_r2': probe_results[best_layer]['test_r2'],
            'n_utility_neurons': len(utility_neurons),
            'n_moral_neurons': len(moral_neurons),
            'n_moral_neurons_at_best_layer': len(moral_neurons_at_layer),
            'neuron_overlap': overlap,
            'overlap_ratio': overlap / len(moral_neurons_at_layer) if moral_neurons_at_layer else 0
        }
        
        all_results[dimension] = results
        
        # Save detailed results for this dimension
        dim_output_path = args.output_dir / f"{dimension}_unified_analysis.json"
        with open(dim_output_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Save utility probe analysis
    analyzer.save_utility_analysis(args.output_dir / "utility_probe_analysis.json")
    
    # Save summary
    summary = {
        'model': args.model,
        'dimensions_analyzed': list(all_results.keys()),
        'avg_probe_r2': np.mean([r['best_probe_r2'] for r in all_results.values()]),
        'avg_overlap_ratio': np.mean([r['overlap_ratio'] for r in all_results.values()]),
        'dimension_results': all_results
    }
    
    summary_path = args.output_dir / "unified_analysis_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nAnalysis complete! Results saved to {args.output_dir}")
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    for dim, results in all_results.items():
        print(f"\n{dim}:")
        print(f"  - Utility probe R²: {results['best_probe_r2']:.3f}")
        print(f"  - Moral neurons: {results['n_moral_neurons']}")
        print(f"  - Utility neurons: {results['n_utility_neurons']}")
        print(f"  - Overlap: {results['neuron_overlap']} ({results['overlap_ratio']:.1%})")
    
    return 0


if __name__ == "__main__":
    # Add numpy import for summary calculations
    import numpy as np
    sys.exit(main())