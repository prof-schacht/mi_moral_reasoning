import argparse
import os
from src.models.model_loader import ModelLoader
from src.models.model_config import ModelConfig
from src.analysis.ablation import AblationAnalyzer
from src.utils.data_loader import load_moral_pairs, load_results, save_results
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Run ablation analysis on moral neurons')
    
    parser.add_argument('--model_name', type=str, default='google/gemma-2-9b-it',
                      help='Name of the model to analyze')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to moral/immoral text pairs')
    parser.add_argument('--results_path', type=str, required=True,
                      help='Path to saved analysis results')
    parser.add_argument('--output_dir', type=str, default='results/ablation',
                      help='Directory to save ablation results')
    parser.add_argument('--ablation_value', type=float, default=None,
                      help='Value to set ablated neurons to (default: 0)')
    
    return parser.parse_args()

def plot_ablation_results(results: dict, output_dir: str):
    """Create visualizations of ablation results."""
    # 1. Response Changes Plot
    plt.figure(figsize=(10, 6))
    moral_changes = [x[0] for x in results['response_changes']]
    immoral_changes = [x[1] for x in results['response_changes']]
    
    plt.boxplot([moral_changes, immoral_changes], labels=['Moral', 'Immoral'])
    plt.title('Response Changes After Ablation')
    plt.ylabel('Change Magnitude')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'response_changes.png'))
    plt.close()
    
    # 2. Agreement Changes
    plt.figure(figsize=(8, 6))
    agreements = [
        results['moral_agreement_original'],
        results['moral_agreement_ablated']
    ]
    plt.boxplot(agreements, labels=['Original', 'Ablated'])
    plt.title('Moral Agreement Before and After Ablation')
    plt.ylabel('Agreement Score')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'agreement_changes.png'))
    plt.close()
    
    # 3. Statistics Summary
    plt.figure(figsize=(12, 6))
    stats = {k: v for k, v in results.items() if isinstance(v, (int, float))}
    plt.bar(range(len(stats)), list(stats.values()))
    plt.xticks(range(len(stats)), list(stats.keys()), rotation=45, ha='right')
    plt.title('Ablation Statistics Summary')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ablation_stats.png'))
    plt.close()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model {args.model_name}...")
    config = ModelConfig(model_name=args.model_name)
    model = ModelLoader.load_model(config)
    
    # Load data
    print(f"Loading data from {args.data_path}...")
    moral_pairs = load_moral_pairs(args.data_path)
    
    # Load previous analysis results
    print(f"Loading analysis results from {args.results_path}...")
    analysis_results = load_results(args.results_path)
    if analysis_results is None:
        raise ValueError(f"No results found at {args.results_path}")
    
    # Initialize ablation analyzer
    analyzer = AblationAnalyzer(model)
    
    # Get neurons to ablate
    moral_neurons = analysis_results['moral_neurons']
    immoral_neurons = analysis_results['immoral_neurons']
    
    # Run ablation analysis for moral neurons
    print("\nAnalyzing moral neuron ablation...")
    moral_results = analyzer.analyze_ablation_impact(
        moral_pairs,
        moral_neurons,
        args.ablation_value
    )
    
    # Run ablation analysis for immoral neurons
    print("\nAnalyzing immoral neuron ablation...")
    immoral_results = analyzer.analyze_ablation_impact(
        moral_pairs,
        immoral_neurons,
        args.ablation_value
    )
    
    # Save results
    ablation_results = {
        'moral_neuron_ablation': moral_results,
        'immoral_neuron_ablation': immoral_results,
        'ablated_moral_neurons': moral_neurons,
        'ablated_immoral_neurons': immoral_neurons,
        'ablation_value': args.ablation_value
    }
    
    results_path = os.path.join(args.output_dir, 'ablation_results.pkl')
    save_results(ablation_results, results_path)
    print(f"\nSaved ablation results to {results_path}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_ablation_results(moral_results, os.path.join(args.output_dir, 'moral_ablation'))
    plot_ablation_results(immoral_results, os.path.join(args.output_dir, 'immoral_ablation'))
    
    # Save text summary
    summary_path = os.path.join(args.output_dir, 'ablation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Ablation Analysis Summary\n")
        f.write("=======================\n\n")
        
        f.write("Moral Neuron Ablation\n")
        f.write("-----------------\n")
        for key, value in moral_results.items():
            if isinstance(value, (int, float)):
                f.write(f"{key}: {value:.4f}\n")
        
        f.write("\nImmoral Neuron Ablation\n")
        f.write("-------------------\n")
        for key, value in immoral_results.items():
            if isinstance(value, (int, float)):
                f.write(f"{key}: {value:.4f}\n")
    
    print(f"Saved summary to {summary_path}")
    print("Ablation analysis complete!")

if __name__ == '__main__':
    main() 