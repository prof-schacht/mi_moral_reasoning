import argparse
import os
from src.models.model_loader import ModelLoader
from src.models.model_config import ModelConfig
from src.analysis.neuron_describer import MoralNeuronDescriber
from src.utils.data_loader import load_results, save_results

def parse_args():
    parser = argparse.ArgumentParser(description='Generate descriptions for moral neurons')
    
    parser.add_argument('--model_name', type=str, default='google/gemma-2-9b-it',
                      help='Name of the model to analyze')
    parser.add_argument('--results_path', type=str, required=True,
                      help='Path to saved analysis results')
    parser.add_argument('--llm_name', type=str, default='gpt-4',
                      help='Name of LLM to use for descriptions')
    parser.add_argument('--output_dir', type=str, default='results',
                      help='Directory to save results')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model {args.model_name}...")
    config = ModelConfig(model_name=args.model_name)
    model = ModelLoader.load_model(config)
    
    # Load previous results
    print(f"Loading results from {args.results_path}...")
    results = load_results(args.results_path)
    if results is None:
        raise ValueError(f"No results found at {args.results_path}")
    
    # Initialize describer
    describer = MoralNeuronDescriber(model, llm_name=args.llm_name)
    
    # Generate descriptions
    print("Generating neuron descriptions...")
    descriptions = describer.describe_moral_neurons(results)
    
    # Add descriptions to results
    results['descriptions'] = descriptions
    
    # Save updated results
    output_path = os.path.join(args.output_dir, 'moral_circuit_results_with_descriptions.pkl')
    save_results(results, output_path)
    print(f"Saved results with descriptions to {output_path}")
    
    # Save descriptions to text file for easy reading
    txt_path = os.path.join(args.output_dir, 'neuron_descriptions.txt')
    with open(txt_path, 'w') as f:
        f.write("Moral Neuron Descriptions\n")
        f.write("=======================\n\n")
        
        f.write("Moral Neurons\n")
        f.write("-----------\n")
        for neuron in results['moral_neurons']:
            if neuron in descriptions:
                f.write(f"\nLayer {neuron[0]}, Neuron {neuron[1]}:\n")
                f.write(descriptions[neuron])
                f.write("\n")
        
        f.write("\nImmoral Neurons\n")
        f.write("-------------\n")
        for neuron in results['immoral_neurons']:
            if neuron in descriptions:
                f.write(f"\nLayer {neuron[0]}, Neuron {neuron[1]}:\n")
                f.write(descriptions[neuron])
                f.write("\n")
    
    print(f"Saved readable descriptions to {txt_path}")
    print("Description generation complete!")

if __name__ == '__main__':
    main() 