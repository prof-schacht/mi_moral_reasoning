#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
import sys
from datetime import datetime
from typing import List, Tuple, Dict

import torch
from transformer_lens import HookedTransformer
from openai import OpenAI
from dotenv import load_dotenv
import random
load_dotenv(dotenv_path="../.env")

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.analysis.ablation import AblationAnalyzer
from data.mft_dim import get_moral_statements, get_neutral_statements, moral_foundations

ANALYSIS_PROMPT = """
You are analyzing results from a neuron ablation study in a large language model (LLM) that investigates moral behavior circuits. The study examines how disabling specific neurons affects the model's responses to moral and {immoral_neutral} scenarios.

Experimental Setup:
- The study compares the model's responses before and after ablating (temporarily disabling) specific neurons
- The model was given pairs of scenarios: one moral and one {immoral_neutral} version
- For each pair, we measure:
  1. How much the model's response changes after ablation (response_changes)
  2. How similarly the model treats moral vs {immoral_neutral} scenarios (moral_agreement)
  3. The overall impact on the model's moral reasoning capabilities

The measurements use cosine similarity where:
- Response changes: 0 means no change, 1 means completely different response
- Moral agreement: 0 means completely different treatment, 1 means identical treatment

Here are the results from ablating the target neurons:

Response Changes (moral_scenario, {immoral_neutral}_scenario):
{response_changes}

Model's ability to distinguish between moral/{immoral_neutral} scenarios:
- Original moral/{immoral_neutral} agreement scores: {moral_agreement_original}
- Ablated moral/{immoral_neutral} agreement scores: {moral_agreement_ablated}

Summary Statistics:
- Average change in moral responses: {avg_moral_change:.3f} (±{std_moral_change:.3f})
- Average change in {immoral_neutral} responses: {avg_immoral_change:.3f} (±{std_immoral_change:.3f})
- Original average moral/{immoral_neutral} agreement: {original_agreement:.3f}
- Ablated average moral/{immoral_neutral} agreement: {ablated_agreement:.3f}
- Overall change in moral/{immoral_neutral} agreement: {agreement_change:.3f}

Please analyze these results and provide:
1. What do the response changes tell us about the ablated neurons' role in moral/{immoral_neutral} processing?
2. How does ablation affect the model's ability to distinguish between moral and {immoral_neutral} scenarios?
3. Are there any notable patterns or outliers in the data?
4. What conclusions can we draw about these neurons' contribution to the model's moral reasoning capabilities?

Please support your interpretation with specific numbers from the results.
"""

def load_neurons(file_path: str) -> List[Tuple[int, int]]:
    """
    Load neurons to ablate from JSON.
    Converts string layer and neuron numbers to integers.
    
    Example input JSON: [["26", "2140"], ["2", "47"]]
    Returns: [(26, 2140), (2, 47)]
    """
    with open(file_path, 'r') as f:
        neurons_str = json.load(f)
    # Convert string pairs to integer tuples
    return [(int(layer), int(neuron)) for layer, neuron in neurons_str]

def get_llm_analysis(results: Dict, api_key: str, immoral_neutral: str) -> str:
    """Get GPT-4 analysis of ablation results."""
    client = OpenAI(api_key=api_key)
    
    # Format the prompt with results
    formatted_prompt = ANALYSIS_PROMPT.format(
        response_changes=results['response_changes'],
        moral_agreement_original=results['moral_agreement_original'],
        moral_agreement_ablated=results['moral_agreement_ablated'],
        avg_moral_change=results['avg_moral_change'],
        std_moral_change=results['std_moral_change'],
        avg_immoral_change=results['avg_immoral_change'],
        std_immoral_change=results['std_immoral_change'],
        original_agreement=results['original_agreement'],
        ablated_agreement=results['ablated_agreement'],
        agreement_change=results['agreement_change'],
        immoral_neutral=immoral_neutral
    )
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": formatted_prompt}]
    )
    
    return response.choices[0].message.content

def main():
    # python scripts/run_ablation_analysis.py --model "google/gemma-2-9b-it" --neurons ./results/google-gemma-2-9b-it/2025-01-22_google-gemma-2-9b-it_fp16_moral-care_neuron_moral_ethic_cl1.json --results-dir ./results/ablation/ --dimension "care" --llm_explainer True --max_new_tokens 70 --temperature 0 --ablation_value -20 --device "cuda:2" --neuron_cluster "1"
    parser = argparse.ArgumentParser(description='Run ablation analysis on moral circuits')
    parser.add_argument('--model', type=str, required=True, help='Name of the model to analyze')
    parser.add_argument('--neurons', type=str, required=True, help='Path to JSON file containing neurons to ablate')
    parser.add_argument('--results-dir', type=str, required=True, help='Directory to store results')
    parser.add_argument('--dimension', type=str, required=True, help='Dimension/aspect being analyzed')
    parser.add_argument('--llm_explainer', type=bool, required=True, help='Whether to use GPT-4 to explain the results')
    parser.add_argument('--max_new_tokens', type=int, required=True, help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, required=True, help='Temperature for the model')
    parser.add_argument('--ablation_value', type=float, required=True, help='Value to ablate the neurons by')
    parser.add_argument('--device', type=str, required=True, default='cuda', help='Device to run the model on')
    parser.add_argument('--neuron_cluster', type=str, required=True, default=1, help='Cluster to run the model on')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model {args.model}...")
    model = HookedTransformer.from_pretrained(args.model, device=args.device if torch.cuda.is_available() else 'cpu')
    
    # Initialize analyzer
    analyzer = AblationAnalyzer(model)
    
    # Load neurons and statement pairs
    neurons = load_neurons(args.neurons)

    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = args.model.split('/')[-1].replace('-', '_')
    results_base_dir = Path(args.results_dir) / model_name / args.dimension
    results_base_dir.mkdir(parents=True, exist_ok=True)
    
    
    # Get moral_immoral_pairs
    moral_immoral_pairs = moral_foundations[args.dimension]
    
    print("Dataset moral_immoral_pairs:")
    print(moral_immoral_pairs[:5])
    
    # Get moral_neutral_pairs
    moral_statements = [stmt["statement"] for stmt in get_moral_statements(dimension=args.dimension, moral=True)]
    neutral_statements = get_neutral_statements()
    
    # Create a list of (moral, neutral) statement pairs
    moral_neutral_pairs = []
    for moral_statement in moral_statements:
        neutral_statement = random.choice(neutral_statements)
        moral_neutral_pairs.append((moral_statement, neutral_statement))

    print("Dataset moral_neutral_pairs:")
    print(moral_neutral_pairs[:5])
    # Run analyses
    analyses = {
        'moral_vs_immoral': (moral_immoral_pairs, 'immoral'),
        'moral_vs_neutral': (moral_neutral_pairs, 'neutral')
    }
    
    for analysis_name, (pairs, comparison_type) in analyses.items():
        print(f"\nRunning {analysis_name} analysis...")
        results = analyzer.analyze_ablation_impact(pairs, neurons, ablation_value=args.ablation_value, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        
        # Save raw results
        results_file = results_base_dir / f"{timestamp}_{model_name}_{args.dimension}_cl{args.neuron_cluster}_{analysis_name}_{comparison_type}_ablation_value_{str(args.ablation_value)}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {results_file}")
        
        if args.llm_explainer == True:
            # Get and save GPT-4 analysis
            print("Getting GPT-4 analysis...")
            analysis = get_llm_analysis(results, OPENAI_API_KEY, comparison_type)
            analysis_file = results_base_dir / f"{timestamp}_{model_name}_{args.dimension}_cl{args.neuron_cluster}_{analysis_name}_{comparison_type}_ablation_value_{str(args.ablation_value)}_LLM_explanation.txt"
            with open(analysis_file, 'w') as f:
                f.write(analysis)
            print(f"Saved analysis to {analysis_file}")

if __name__ == '__main__':
    main() 