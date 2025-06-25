#!/usr/bin/env python
"""Script for fitting Thurstonian utility models to preference data."""

import argparse
from pathlib import Path
import json
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.analysis.utility_analyzer import UtilityAnalyzer
from src.analysis.preference_elicitor import AggregatedPreference
from src.data.preference_data import PreferenceScenario


def load_elicitation_results(path: Path):
    """Load preference elicitation results."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Convert to AggregatedPreference objects
    results = {}
    for dimension, prefs in data.items():
        preferences = []
        for pref_data in prefs:
            scenario = PreferenceScenario(
                dimension=pref_data['scenario']['dimension'],
                option_a=pref_data['scenario']['option_a'],
                option_b=pref_data['scenario']['option_b'],
                context=pref_data['scenario'].get('context')
            )
            
            pref = AggregatedPreference(
                scenario=scenario,
                prefer_a_count=pref_data['prefer_a_count'],
                prefer_b_count=pref_data['prefer_b_count'],
                total_count=pref_data['total_count'],
                prefer_a_probability=pref_data['prefer_a_probability'],
                prefer_b_probability=pref_data['prefer_b_probability'],
                avg_confidence=pref_data['avg_confidence']
            )
            preferences.append(pref)
        
        results[dimension] = preferences
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Fit Thurstonian utility models to preference data"
    )
    parser.add_argument(
        "--preferences",
        type=Path,
        required=True,
        help="Path to preference elicitation results"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for utility models"
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=0.1,
        help="Regularization parameter for utility fitting"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=1000,
        help="Maximum optimization iterations"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load preference data
    print(f"Loading preference data from {args.preferences}")
    elicitation_results = load_elicitation_results(args.preferences)
    
    # Initialize utility analyzer
    analyzer = UtilityAnalyzer(regularization=args.regularization)
    
    # Fit utility models for each dimension
    all_results = {}
    
    for dimension, preferences in elicitation_results.items():
        print(f"\nFitting utility model for dimension: {dimension}")
        print(f"  - Number of preference scenarios: {len(preferences)}")
        
        # Fit the model
        utility_model = analyzer.fit_thurstonian_model(
            preferences,
            max_iterations=args.max_iterations
        )
        
        # Save the model
        output_path = args.output_dir / f"{dimension}_utility_model.json"
        analyzer.save_utility_model(utility_model, output_path)
        
        # Print summary
        print(f"  - Model accuracy: {utility_model.accuracy:.3f}")
        print(f"  - Transitivity score: {utility_model.transitivity_score:.3f}")
        print(f"  - Completeness score: {utility_model.completeness_score:.3f}")
        print(f"  - Number of unique outcomes: {len(utility_model.utilities)}")
        
        all_results[dimension] = {
            'accuracy': utility_model.accuracy,
            'transitivity_score': utility_model.transitivity_score,
            'completeness_score': utility_model.completeness_score,
            'n_outcomes': len(utility_model.utilities)
        }
    
    # Save summary
    summary_path = args.output_dir / "utility_models_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nAll utility models fitted successfully!")
    print(f"Results saved to {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())