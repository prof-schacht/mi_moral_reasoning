#!/usr/bin/env python3

import os
import sys
import torch
import json
import argparse
from datetime import datetime
from typing import List, Dict

# # Analyze specific models across all dimensions
# python scripts/analyze_models.py --models "google/gemma-2b" "google/gemma-7b"

# # Analyze specific models and dimensions
# python scripts/analyze_models.py --models "google/gemma-2b" "google/gemma-7b" --dimensions "care" "fairness"

# # Specify custom results directory
# python scripts/analyze_models.py --models "google/gemma-2b" --results-dir "/path/to/results"

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer_lens import HookedTransformer
from src.analysis.moral_analyzer import MoralBehaviorAnalyzer
from data.mft_dim import get_moral_keys, generate_mc_prompts, get_moral_statements
from src.visualization.moral_neuron_viz import plot_moral_neuron_analysis

def analyze_model(model_name: str, dimensions: List[str], results_dir: str, precision: str = None):
    """
    Analyze a single model across all specified moral dimensions.
    
    Args:
        model_name: Name of the model to analyze
        dimensions: List of moral dimensions to analyze
        results_dir: Directory to save results
        
    Returns:
        Nothing
    """
    print(f"\nAnalyzing model: {model_name}")
    
    try:
        # Load model
        print(f"Loading model {model_name}...")
        if precision is not None:
            model = HookedTransformer.from_pretrained(model_name, dtype=precision)
        else:   
            model = HookedTransformer.from_pretrained(model_name)
        
        # Initialize analyzer
        analyzer = MoralBehaviorAnalyzer(model)
        
        # Analyze each dimension
        for dimension in dimensions:
            print(f"\nAnalyzing dimension: {dimension}")
            
            try:
                # Data list of moral statements and immoral statements
                moral_statements = get_moral_statements(dimension=dimension, moral=True)
                immoral_statements = get_moral_statements(dimension=dimension, moral=False)

                moral_pairs = [(statement["statement"], immoral_statements[i]["statement"]) for i, statement in enumerate(moral_statements)]

                # Run analysis
                results = analyzer.analyze_moral_behavior(
                    moral_pairs,
                    temporal_window=5
                )
                
                # Create model-specific directory
                model_dir = os.path.join(results_dir, model_name.replace("/", "-").replace(".", "-"))
                os.makedirs(model_dir, exist_ok=True)
                
                # Save key results in text format
                timestamp = datetime.now().strftime("%Y-%m-%d")
                model_name_safe = model_name.replace("/", "-").replace(".", "-")
                precision_name = precision if precision is not None else "fp16"
                results_file = os.path.join(model_dir, f"{timestamp}_{model_name_safe}_{precision_name}_moral-{dimension}_results.txt")
                
                with open(results_file, 'w') as f:
                    f.write(f"Model: {model_name}\n")
                    f.write(f"Dimension: {dimension}\n")
                    f.write(f"Timestamp: {timestamp}\n\n")
                    f.write(f"Precision: {precision}\n")

                    f.write("Moral Neurons:\n")
                    for neuron in results.get("moral_neurons", []):
                        f.write(f"{neuron}\n")
                    f.write("\n")
                    
                    f.write("Immoral Neurons:\n")
                    for neuron in results.get("immoral_neurons", []):
                        f.write(f"{neuron}\n")
                    f.write("\n")

                    f.write("Neuron Consistency:\n")
                    for neuron in results.get("neuron_consistency", []):
                        f.write(f"{neuron}\n")
                    f.write("\n")
                    
                    f.write("Layer Importance:\n")
                    for layer in results.get("layer_importance", []):
                        f.write(f"{layer}\n")
                    f.write("\n")
                    
                    f.write("Key Trigger Points:\n")
                    for point in results.get("key_trigger_points", []):
                        f.write(f"{point}\n")
                    
                print(f"Results saved to {results_file}")
                
                # Try to save visualization
                try:
                    save_path = os.path.join(model_dir, f"{timestamp}_{model_name_safe}_moral-{dimension}_neuron-analysis.png")
                    plot_moral_neuron_analysis(results, moral_pairs, save_path=save_path, dimension=dimension, model_name=model_name_safe)
                    print(f"Plot saved to {save_path}")
                except Exception as plot_error:
                    print(f"Warning: Could not create plot for dimension {dimension}: {str(plot_error)}")
                
            except Exception as dim_error:
                print(f"Warning: Error analyzing dimension {dimension}: {str(dim_error)}")
                continue  # Continue with next dimension
        
        # Clean up model to free memory
        del model
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error loading model {model_name}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Run moral analysis across multiple models and dimensions")
    parser.add_argument(
        "--models", 
        nargs="+", 
        required=True,
        help="List of model names to analyze"
    )
    parser.add_argument(
        "--dimensions", 
        nargs="+", 
        default=None,
        help="List of moral dimensions to analyze. If not specified, analyzes all dimensions."
    )
    parser.add_argument(
        "--results-dir", 
        default="./results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--precision",
        default=None,
        help="Precision to use for model"
    )
    
    args = parser.parse_args()
    
    # Get all dimensions if not specified
    if args.dimensions is None:
        args.dimensions = get_moral_keys()
    
    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)
    
    
    # Analyze each model
    for model_name in args.models:
        analyze_model(model_name, args.dimensions, args.results_dir, args.precision)
    
 
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 