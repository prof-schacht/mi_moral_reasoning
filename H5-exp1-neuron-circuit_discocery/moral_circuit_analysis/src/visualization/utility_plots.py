"""
Visualization functions for utility analysis and moral-utility integration.
Creates plots for utility landscapes, preference graphs, and neuron-utility mappings.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import pandas as pd
from matplotlib.patches import Rectangle
import matplotlib.cm as cm

from ..analysis.utility_analyzer import UtilityModel, ThurstoneUtility
from ..analysis.preference_elicitor import AggregatedPreference
from ..data.preference_data import PreferenceScenario


def plot_utility_landscape(
    utility_model: UtilityModel,
    dimension: str,
    output_path: Optional[Path] = None,
    top_n: int = 20
):
    """
    Plot utility landscape showing relative utilities of outcomes.
    
    Args:
        utility_model: Fitted utility model
        dimension: Moral dimension name
        output_path: Where to save the plot
        top_n: Number of top/bottom outcomes to show
    """
    # Extract utilities
    outcomes = list(utility_model.utilities.keys())
    utilities = [(o, utility_model.utilities[o]) for o in outcomes]
    
    # Sort by utility mean
    utilities.sort(key=lambda x: x[1].mean, reverse=True)
    
    # Select top and bottom outcomes
    if len(utilities) > 2 * top_n:
        selected = utilities[:top_n] + utilities[-top_n:]
    else:
        selected = utilities
    
    # Prepare data for plotting
    labels = []
    means = []
    stds = []
    
    for outcome, util in selected:
        # Truncate long outcomes
        label = outcome[:50] + "..." if len(outcome) > 50 else outcome
        labels.append(label)
        means.append(util.mean)
        stds.append(util.std)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create bar plot with error bars
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, means, xerr=stds, capsize=5)
    
    # Color bars based on value
    colors = ['green' if m > 0 else 'red' for m in means]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
        bar.set_alpha(0.7)
    
    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Utility Value', fontsize=12)
    ax.set_title(f'Utility Landscape - {dimension}', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax.grid(axis='x', alpha=0.3)
    
    # Add model metrics
    metrics_text = (f"Accuracy: {utility_model.accuracy:.3f}\n"
                   f"Transitivity: {utility_model.transitivity_score:.3f}\n"
                   f"Completeness: {utility_model.completeness_score:.3f}")
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_preference_graph(
    preferences: List[AggregatedPreference],
    utility_model: Optional[UtilityModel] = None,
    output_path: Optional[Path] = None,
    max_nodes: int = 30
):
    """
    Plot preference graph showing relationships between outcomes.
    
    Args:
        preferences: List of aggregated preferences
        utility_model: Optional utility model for node coloring
        output_path: Where to save the plot
        max_nodes: Maximum number of nodes to display
    """
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes for unique outcomes
    outcomes = set()
    for pref in preferences:
        outcomes.add(pref.scenario.option_a)
        outcomes.add(pref.scenario.option_b)
    
    # Limit nodes if too many
    if len(outcomes) > max_nodes:
        # Select based on utility if available
        if utility_model:
            sorted_outcomes = sorted(outcomes, 
                                   key=lambda x: abs(utility_model.utilities.get(x, ThurstoneUtility(x, 0, 1)).mean),
                                   reverse=True)
            outcomes = set(sorted_outcomes[:max_nodes])
        else:
            outcomes = set(list(outcomes)[:max_nodes])
    
    # Add nodes
    for outcome in outcomes:
        G.add_node(outcome[:30] + "..." if len(outcome) > 30 else outcome)
    
    # Add edges based on preferences
    for pref in preferences:
        if pref.scenario.option_a in outcomes and pref.scenario.option_b in outcomes:
            a_label = pref.scenario.option_a[:30] + "..." if len(pref.scenario.option_a) > 30 else pref.scenario.option_a
            b_label = pref.scenario.option_b[:30] + "..." if len(pref.scenario.option_b) > 30 else pref.scenario.option_b
            
            if pref.prefer_a_probability > 0.5:
                G.add_edge(a_label, b_label, weight=pref.preference_strength)
            else:
                G.add_edge(b_label, a_label, weight=pref.preference_strength)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Position nodes using spring layout
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Color nodes based on utility if available
    if utility_model:
        node_colors = []
        for node in G.nodes():
            # Find full outcome name
            full_outcome = None
            for outcome in outcomes:
                if node == (outcome[:30] + "..." if len(outcome) > 30 else outcome):
                    full_outcome = outcome
                    break
            
            if full_outcome and full_outcome in utility_model.utilities:
                util_value = utility_model.utilities[full_outcome].mean
                node_colors.append(util_value)
            else:
                node_colors.append(0)
        
        # Normalize colors
        vmin = min(node_colors) if node_colors else -1
        vmax = max(node_colors) if node_colors else 1
        node_colors = [(c - vmin) / (vmax - vmin) if vmax > vmin else 0.5 for c in node_colors]
    else:
        node_colors = 'lightblue'
    
    # Draw graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap='RdYlGn',
                          node_size=3000, ax=ax, vmin=0, vmax=1)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    
    # Draw edges with varying thickness based on preference strength
    edges = G.edges(data=True)
    weights = [e[2]['weight'] for e in edges]
    nx.draw_networkx_edges(G, pos, width=[w*3 for w in weights],
                          alpha=0.5, edge_color='gray', arrows=True,
                          arrowsize=20, ax=ax)
    
    ax.set_title('Preference Graph', fontsize=16, fontweight='bold')
    ax.axis('off')
    
    # Add colorbar if utility model provided
    if utility_model and isinstance(node_colors, list):
        sm = cm.ScalarMappable(cmap='RdYlGn', 
                              norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Utility Value', fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_neuron_utility_mapping(
    probe_results: Dict[int, Dict[str, float]],
    moral_neurons_by_layer: Dict[int, List[int]],
    utility_neurons_by_layer: Dict[int, List[int]],
    output_path: Optional[Path] = None
):
    """
    Plot the relationship between moral neurons and utility representations across layers.
    
    Args:
        probe_results: Utility probe accuracy by layer
        moral_neurons_by_layer: Moral neurons at each layer
        utility_neurons_by_layer: Utility-encoding neurons at each layer
        output_path: Where to save the plot
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    layers = sorted(probe_results.keys())
    
    # Plot 1: Probe accuracy by layer
    r2_scores = [probe_results[l]['test_r2'] for l in layers]
    ax1.plot(layers, r2_scores, 'b-', linewidth=2, marker='o')
    ax1.set_ylabel('Utility Probe R²', fontsize=12)
    ax1.set_title('Utility Representation Analysis Across Layers', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, max(r2_scores) * 1.1 if r2_scores else 1)
    
    # Plot 2: Neuron counts by layer
    moral_counts = [len(moral_neurons_by_layer.get(l, [])) for l in layers]
    utility_counts = [len(utility_neurons_by_layer.get(l, [])) for l in layers]
    
    width = 0.35
    x = np.array(layers)
    ax2.bar(x - width/2, moral_counts, width, label='Moral Neurons', alpha=0.7, color='orange')
    ax2.bar(x + width/2, utility_counts, width, label='Utility Neurons', alpha=0.7, color='green')
    ax2.set_ylabel('Neuron Count', fontsize=12)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Plot 3: Overlap analysis
    overlaps = []
    overlap_ratios = []
    
    for layer in layers:
        moral_set = set(moral_neurons_by_layer.get(layer, []))
        utility_set = set(utility_neurons_by_layer.get(layer, []))
        
        if moral_set:
            overlap = len(moral_set & utility_set)
            overlap_ratio = overlap / len(moral_set)
        else:
            overlap = 0
            overlap_ratio = 0
        
        overlaps.append(overlap)
        overlap_ratios.append(overlap_ratio)
    
    ax3.bar(layers, overlap_ratios, alpha=0.7, color='purple')
    ax3.set_ylabel('Overlap Ratio', fontsize=12)
    ax3.set_xlabel('Layer', fontsize=12)
    ax3.set_ylim(0, 1.1)
    ax3.grid(alpha=0.3)
    
    # Add text annotations for overlap counts
    for i, (layer, count) in enumerate(zip(layers, overlaps)):
        if count > 0:
            ax3.text(layer, overlap_ratios[i] + 0.02, str(count), 
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_utility_ablation_effects(
    ablation_results: List[Dict],
    output_path: Optional[Path] = None
):
    """
    Plot the effects of ablation on utility metrics.
    
    Args:
        ablation_results: List of utility ablation results
        output_path: Where to save the plot
    """
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(ablation_results)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Accuracy changes
    ax = axes[0, 0]
    dims = df['dimension'].unique()
    x = np.arange(len(dims))
    width = 0.25
    
    for i, cluster in enumerate(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster]
        values = [cluster_data[cluster_data['dimension'] == d]['changes'].iloc[0]['accuracy_change'] 
                 if len(cluster_data[cluster_data['dimension'] == d]) > 0 else 0 
                 for d in dims]
        ax.bar(x + i*width, values, width, label=cluster, alpha=0.7)
    
    ax.set_ylabel('Accuracy Change', fontsize=12)
    ax.set_title('Effect on Utility Model Accuracy', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(dims, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 2: Transitivity changes
    ax = axes[0, 1]
    for i, cluster in enumerate(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster]
        values = [cluster_data[cluster_data['dimension'] == d]['changes'].iloc[0]['transitivity_change'] 
                 if len(cluster_data[cluster_data['dimension'] == d]) > 0 else 0 
                 for d in dims]
        ax.bar(x + i*width, values, width, label=cluster, alpha=0.7)
    
    ax.set_ylabel('Transitivity Change', fontsize=12)
    ax.set_title('Effect on Preference Transitivity', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(dims, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 3: Average preference shift
    ax = axes[1, 0]
    shift_data = []
    for _, row in df.iterrows():
        shift_data.append({
            'Dimension': row['dimension'],
            'Cluster': row['cluster'],
            'Avg Shift': row['preference_analysis']['avg_shift']
        })
    
    shift_df = pd.DataFrame(shift_data)
    pivot_shift = shift_df.pivot(index='Dimension', columns='Cluster', values='Avg Shift')
    pivot_shift.plot(kind='bar', ax=ax, alpha=0.7)
    
    ax.set_ylabel('Average Preference Shift', fontsize=12)
    ax.set_title('Preference Distribution Changes', fontsize=12, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 4: Utility correlation
    ax = axes[1, 1]
    corr_data = []
    for _, row in df.iterrows():
        corr_data.append({
            'Dimension': row['dimension'],
            'Cluster': row['cluster'],
            'Correlation': row['utility_distortion']['correlation']
        })
    
    corr_df = pd.DataFrame(corr_data)
    pivot_corr = corr_df.pivot(index='Dimension', columns='Cluster', values='Correlation')
    pivot_corr.plot(kind='bar', ax=ax, alpha=0.7)
    
    ax.set_ylabel('Utility Correlation', fontsize=12)
    ax.set_title('Utility Preservation After Ablation', fontsize=12, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Utility-Based Ablation Analysis Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_utility_visualization_script():
    """Create script for generating utility visualizations."""
    script_path = Path("scripts/generate_utility_visualizations.py")
    script_content = '''#!/usr/bin/env python
"""Generate visualizations for utility analysis results."""

import argparse
from pathlib import Path
import json
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.visualization.utility_plots import (
    plot_utility_landscape,
    plot_preference_graph,
    plot_neuron_utility_mapping,
    plot_utility_ablation_effects
)
from src.analysis.utility_analyzer import UtilityAnalyzer
from src.analysis.preference_elicitor import AggregatedPreference
from src.data.preference_data import PreferenceScenario


def main():
    parser = argparse.ArgumentParser(
        description="Generate utility analysis visualizations"
    )
    parser.add_argument("--utility-models", type=Path, required=True,
                        help="Directory containing utility models")
    parser.add_argument("--preferences", type=Path, required=True,
                        help="Path to preference elicitation results")
    parser.add_argument("--probe-results", type=Path, 
                        help="Path to utility probe analysis results")
    parser.add_argument("--ablation-results", type=Path,
                        help="Path to utility ablation results")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    utility_analyzer = UtilityAnalyzer()
    
    # Generate utility landscapes
    print("Generating utility landscapes...")
    for model_file in args.utility_models.glob("*_utility_model.json"):
        dimension = model_file.stem.replace("_utility_model", "")
        utility_model = utility_analyzer.load_utility_model(model_file)
        
        output_path = args.output_dir / f"{dimension}_utility_landscape.png"
        plot_utility_landscape(utility_model, dimension, output_path)
        print(f"  - Generated landscape for {dimension}")
    
    # Generate preference graphs
    if args.preferences.exists():
        print("\\nGenerating preference graphs...")
        with open(args.preferences, 'r') as f:
            pref_data = json.load(f)
        
        for dimension, prefs in pref_data.items():
            # Convert to preference objects
            preferences = []
            for p in prefs[:50]:  # Limit for visualization
                scenario = PreferenceScenario(
                    dimension=p['scenario']['dimension'],
                    option_a=p['scenario']['option_a'],
                    option_b=p['scenario']['option_b']
                )
                pref = AggregatedPreference(
                    scenario=scenario,
                    prefer_a_count=p['prefer_a_count'],
                    prefer_b_count=p['prefer_b_count'],
                    total_count=p['total_count'],
                    prefer_a_probability=p['prefer_a_probability'],
                    prefer_b_probability=p['prefer_b_probability'],
                    avg_confidence=p['avg_confidence']
                )
                preferences.append(pref)
            
            # Load corresponding utility model if available
            model_path = args.utility_models / f"{dimension}_utility_model.json"
            if model_path.exists():
                utility_model = utility_analyzer.load_utility_model(model_path)
            else:
                utility_model = None
            
            output_path = args.output_dir / f"{dimension}_preference_graph.png"
            plot_preference_graph(preferences, utility_model, output_path)
            print(f"  - Generated preference graph for {dimension}")
    
    # Generate neuron-utility mapping if probe results available
    if args.probe_results and args.probe_results.exists():
        print("\\nGenerating neuron-utility mapping...")
        with open(args.probe_results, 'r') as f:
            probe_data = json.load(f)
        
        # Extract probe accuracies and neuron info
        probe_accuracies = probe_data.get('probe_accuracies', {})
        utility_neurons = probe_data.get('utility_encoding_neurons', {})
        
        # Convert string keys to int
        probe_accuracies = {int(k): v for k, v in probe_accuracies.items()}
        utility_neurons = {int(k): v for k, v in utility_neurons.items()}
        
        # TODO: Load moral neurons from analysis results
        moral_neurons = {}  # Placeholder
        
        output_path = args.output_dir / "neuron_utility_mapping.png"
        plot_neuron_utility_mapping(probe_accuracies, moral_neurons, utility_neurons, output_path)
        print("  - Generated neuron-utility mapping")
    
    # Generate ablation effects visualization
    if args.ablation_results and args.ablation_results.exists():
        print("\\nGenerating ablation effects visualization...")
        with open(args.ablation_results, 'r') as f:
            ablation_data = json.load(f)
        
        output_path = args.output_dir / "utility_ablation_effects.png"
        plot_utility_ablation_effects(ablation_data, output_path)
        print("  - Generated ablation effects plot")
    
    print(f"\\nAll visualizations saved to {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    script_path.chmod(0o755)
    
    return script_path