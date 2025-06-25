#!/usr/bin/env python
"""Script for eliciting preferences from language models."""

import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.analysis.preference_elicitor import PreferenceElicitor
from src.data.preference_data import load_preference_data


def main():
    parser = argparse.ArgumentParser(description="Elicit preferences from language models")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--data", type=Path, required=True, help="Path to preference data")
    parser.add_argument("--output", type=Path, required=True, help="Output path for results")
    parser.add_argument("--dimension", type=str, help="Specific dimension to process")
    parser.add_argument("--max-scenarios", type=int, help="Maximum scenarios to process")
    parser.add_argument("--samples-per-scenario", type=int, default=4, 
                        help="Number of samples per scenario")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Check if input exists
    if not args.data.exists():
        print(f"Error: Preference data file {args.data} does not exist")
        return 1
    
    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Load preference data
    print(f"Loading preference data from {args.data}")
    preference_data = load_preference_data(args.data)
    
    # Initialize elicitor
    print(f"Initializing preference elicitor with model: {args.model}")
    elicitor = PreferenceElicitor(args.model, args.device)
    
    # Elicit preferences
    print("\nEliciting preferences...")
    results = elicitor.elicit_preferences_for_dataset(
        preference_data,
        args.output,
        dimension=args.dimension,
        max_scenarios=args.max_scenarios,
        num_samples_per_scenario=args.samples_per_scenario
    )
    
    print(f"\nElicitation complete! Results saved to {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())