import argparse
import os
from src.models.model_loader import ModelLoader
from src.models.model_config import ModelConfig
from src.analysis.moral_analyzer import MoralBehaviorAnalyzer
from src.analysis.neuron_describer import MoralNeuronDescriber
from src.utils.data_loader import load_moral_pairs, save_results, load_results
from src.visualization import circuit_plots, network_plots, component_plots

def parse_args():
    parser = argparse.ArgumentParser(description='Run moral neuron analysis')
    
    parser.add_argument('--model_name', type=str, default='google/gemma-2-9b-it',
                      help='Name of the model to analyze')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to moral/immoral text pairs')
    parser.add_argument('--results_dir', type=str, default='results',
                      help='Directory to save results')
    parser.add_argument('--significant_diff', type=float, default=0.5,
                      help='Threshold for significant activation differences')
    parser.add_argument('--consistency_threshold', type=float, default=0.8,
                      help='Threshold for neuron response consistency')
    parser.add_argument('--generate_descriptions', action='store_true',
                      help='Whether to generate neuron descriptions using LLM')
    parser.add_argument('--llm_name', type=str, default='gpt-4',
                      help='Name of LLM to use for descriptions')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model {args.model_name}...")
    config = ModelConfig(model_name=args.model_name)
    model = ModelLoader.load_model(config)
    
    # Load data
    print(f"Loading data from {args.data_path}...")
    moral_pairs = load_moral_pairs(args.data_path)
    
    # Initialize analyzer
    analyzer = MoralBehaviorAnalyzer(model)
    
    # Run analysis
    print("Running moral behavior analysis...")
    results = analyzer.analyze_moral_behavior(
        moral_pairs,
        significant_diff=args.significant_diff,
        consistency_threshold=args.consistency_threshold
    )
    
    # Generate descriptions if requested
    if args.generate_descriptions:
        print("Generating neuron descriptions...")
        describer = MoralNeuronDescriber(model, llm_name=args.llm_name)
        descriptions = describer.describe_moral_neurons(results)
        results['descriptions'] = descriptions
    
    # Save results
    results_path = os.path.join(args.results_dir, 'moral_circuit_results.pkl')
    save_results(results, results_path)
    print(f"Saved results to {results_path}")
    
    # Generate visualizations
    print("Generating visualizations...")
    
    # Circuit plots
    fig = circuit_plots.plot_moral_circuits(results)
    fig.savefig(os.path.join(args.results_dir, 'moral_circuits.png'))
    
    if 'descriptions' in results:
        fig = circuit_plots.plot_moral_circuits_with_descriptions(results, results['descriptions'])
        fig.savefig(os.path.join(args.results_dir, 'moral_circuits_with_descriptions.png'))
    
    # Network plots
    G = network_plots.plot_neuron_network(results)
    plt.savefig(os.path.join(args.results_dir, 'neuron_network.png'))
    
    G = network_plots.plot_neuron_components(results)
    plt.savefig(os.path.join(args.results_dir, 'neuron_components.png'))
    
    # Component plots
    components = list(nx.connected_components(G))
    fig = component_plots.plot_component_summary(
        components,
        n_layers=model.cfg.n_layers,
        n_neurons=model.cfg.d_mlp
    )
    fig.savefig(os.path.join(args.results_dir, 'component_summary.png'))
    
    print("Analysis complete!")

if __name__ == '__main__':
    main() 