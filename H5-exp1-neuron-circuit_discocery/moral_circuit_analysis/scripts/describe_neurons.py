#!/usr/bin/env python3

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Tuple
from datetime import datetime
import torch
from tqdm import tqdm

# Call to start the script: python scripts/describe_neurons.py --model "google/gemma-2-9b-it" --neurons results/google-gemma-2-9b-it/2025-01-22_google-gemma-2-9b-it_fp16_moral-fairness_moral_neurons.json --dimension fairness

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer_lens import HookedTransformer
from src.analysis.neuron_describer_oai_v3 import ImprovedNeuronEvaluator
from data.mft_dim import get_moral_statements, get_neutral_statements
import os
from dotenv import load_dotenv
import random
import pandas as pd

def load_neurons_from_file(file_path: str) -> List[Tuple[int, int]]:
    """Load list of (layer, neuron) tuples from a JSON file."""
    with open(file_path, 'r') as f:
        neurons = json.load(f)
    return [(int(layer), int(neuron)) for layer, neuron in neurons]

def main():
    parser = argparse.ArgumentParser(description='Analyze neurons in bulk using the ImprovedNeuronEvaluator')
    parser.add_argument('--model', type=str, required=True, help='Name of the model to analyze')
    parser.add_argument('--neurons', type=str, required=True, help='Path to JSON file containing list of [layer, neuron] pairs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--num-top-sequences', type=int, default=5, help='Number of top activating sequences to analyze')
    parser.add_argument('--output-dir', type=str, default='results/neuron_describer_logs', help='Directory to store results')
    parser.add_argument('--llm-name', type=str, default='gpt-4o', help='Name of the LLM to use for analysis')
    parser.add_argument('--dimension', type=str, default='care', help='Dimension of Moral Foundation Theory to analyze')
    args = parser.parse_args()

    # Load environment variables (for API keys)
    load_dotenv("../.env")

    # Load model
    print(f"Loading model {args.model}...")
    model = HookedTransformer.from_pretrained(args.model)
    
    dimension = args.dimension
    print(f"Analyzing dimension: {dimension}")
    # Load neurons to analyze
    neurons = load_neurons_from_file(args.neurons)
    print(f"Loaded {len(neurons)} neurons to analyze")

    # Get moral and neutral statements
    moral_statements = get_moral_statements()
    moral_statements = [statement["statement"] for statement in moral_statements]
    neutral_statements = get_neutral_statements()[:5]

    # Initialize evaluator
    evaluator = ImprovedNeuronEvaluator(
        model=model,
        llm_name=args.llm_name,
        num_top_sequences=args.num_top_sequences,
        batch_size=args.batch_size,
        api_key=os.getenv('OPENAI_API_KEY'),
        log_dir=args.output_dir,
        dimension=dimension
    )

    # Create output directory
    timestamp = datetime.now().strftime("%Y-%m-%d")
    model_name = args.model.split('/')[-1].replace('.', '-')
    #model_name = args.model.replace('/', '-').replace('.', '-')
    output_dir = Path(args.output_dir) / model_name / dimension
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize results list for DataFrame
    results = []

    # Analyze each neuron
    for layer, neuron_idx in tqdm(neurons, desc="Analyzing neurons"):
        try:
            print(f"Analyzing neuron L{layer}-N{neuron_idx}")
            # Get top activating sequences
            top_activations = evaluator.get_top_activating_sequences(
                layer=layer,
                neuron_idx=neuron_idx,
                texts=moral_statements
            )
            top_texts = [t.text for t in top_activations]
            
            # Create random texts excluding top activating ones
            random_texts = random.sample([t for t in moral_statements if t not in top_activations], 5)

            # Analyze neuron
            result = evaluator.evaluate_neuron(
                layer=layer,
                neuron_idx=neuron_idx,
                texts=top_texts,
                random_texts=neutral_statements,
                revise=True
            )

            # Extract top activating tokens and their activations
            top_tokens = [
                f"{act.token} ({act.activation:.3f})"
                for act in result.top_activations
            ]

            # Store results
            results.append({
                'Layer': layer,
                'Neuron': neuron_idx,
                'Description': result.explanation.split('\n')[0],  # First line of explanation
                'Score': result.score,
                'Revision_Score': result.revision_score,
                'Top_Activating_Tokens': ' | '.join(top_tokens)
            })

        except Exception as e:
            print(f"Error analyzing neuron (L{layer}-N{neuron_idx}): {str(e)}")
            results.append({
                'Layer': layer,
                'Neuron': neuron_idx,
                'Description': f"Error: {str(e)}",
                'Score': None,
                'Revision_Score': None,
                'Top_Activating_Tokens': None
            })

    # Create DataFrame and save
    df = pd.DataFrame(results)
    output_file = output_dir / f"{timestamp}_{model_name}_moral-{dimension}_neuron-analysis_summary.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSummary saved to: {output_file}")

if __name__ == '__main__':
    main() 