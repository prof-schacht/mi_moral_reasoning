"""
Extension functions for the Flask app to support utility analysis views.
Add these functions to the main app.py file.
"""

import json
from pathlib import Path
from flask import render_template


def load_utility_model_data(model_name, dimension):
    """Load utility model data for a specific dimension."""
    # Try multiple possible locations
    possible_paths = [
        RESULTS_DIR / "utility_models" / f"{dimension}_utility_model.json",
        RESULTS_DIR / model_name / "utility_models" / f"{dimension}_utility_model.json",
        RESULTS_DIR / model_name.replace('-', '_') / "utility_models" / f"{dimension}_utility_model.json"
    ]
    
    for path in possible_paths:
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Format for template
            return {
                'accuracy': data.get('accuracy', 0),
                'transitivity_score': data.get('transitivity_score', 0),
                'completeness_score': data.get('completeness_score', 0),
                'n_outcomes': len(data.get('utilities', {}))
            }
    
    return None


def load_probe_results(model_name, dimension):
    """Load utility probe analysis results."""
    possible_paths = [
        RESULTS_DIR / "unified_analysis" / f"{dimension}_unified_analysis.json",
        RESULTS_DIR / model_name / "unified_analysis" / f"{dimension}_unified_analysis.json",
        RESULTS_DIR / model_name.replace('-', '_') / "unified_analysis" / f"{dimension}_unified_analysis.json"
    ]
    
    for path in possible_paths:
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
            
            return {
                'best_layer': data.get('best_probe_layer'),
                'best_r2': data.get('best_probe_r2', 0),
                'n_utility_neurons': data.get('n_utility_neurons', 0),
                'overlap_ratio': int(data.get('overlap_ratio', 0) * 100)
            }
    
    return None


def load_utility_ablation_results(model_name, dimension):
    """Load utility-based ablation results."""
    possible_paths = [
        RESULTS_DIR / "utility_ablation" / model_name / dimension / "ablation_results.json",
        RESULTS_DIR / model_name / "utility_ablation" / dimension / "ablation_results.json",
        RESULTS_DIR / model_name.replace('-', '_') / "utility_ablation" / dimension / "ablation_results.json"
    ]
    
    for path in possible_paths:
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Format for template
            results = []
            for item in data:
                results.append({
                    'cluster': item['cluster'],
                    'ablation_value': item['ablation_value'],
                    'accuracy_change': item['changes']['accuracy_change'],
                    'transitivity_change': item['changes']['transitivity_change'],
                    'avg_preference_shift': item['preference_analysis']['avg_shift'],
                    'utility_correlation': item['utility_distortion']['correlation']
                })
            
            return results
    
    return None


def find_utility_visualizations(model_name, dimension):
    """Find paths to utility visualization images."""
    viz_paths = {}
    
    # Check various possible locations
    possible_dirs = [
        RESULTS_DIR / "utility_visualizations",
        RESULTS_DIR / model_name / "utility_visualizations",
        RESULTS_DIR / model_name.replace('-', '_') / "utility_visualizations"
    ]
    
    for viz_dir in possible_dirs:
        if viz_dir.exists():
            # Look for utility landscape
            landscape_path = viz_dir / f"{dimension}_utility_landscape.png"
            if landscape_path.exists():
                # Convert to relative path for static serving
                viz_paths['landscape'] = str(landscape_path.relative_to(RESULTS_DIR))
            
            # Look for preference graph
            graph_path = viz_dir / f"{dimension}_preference_graph.png"
            if graph_path.exists():
                viz_paths['graph'] = str(graph_path.relative_to(RESULTS_DIR))
    
    return viz_paths


# Add this route to the main app.py file
@app.route('/model/<model_name>/dimension/<dimension>/utility')
def utility_analysis_view(model_name, dimension):
    """View utility analysis results for a specific dimension."""
    # Load utility model data
    utility_model = load_utility_model_data(model_name, dimension)
    if not utility_model:
        return render_template('error.html', 
                             message="No utility model found for this dimension"), 404
    
    # Load probe results if available
    probe_results = load_probe_results(model_name, dimension)
    
    # Load ablation results if available
    ablation_results = load_utility_ablation_results(model_name, dimension)
    
    # Find visualizations
    viz_paths = find_utility_visualizations(model_name, dimension)
    
    return render_template('utility_analysis.html',
                         model_name=model_name,
                         dimension=dimension,
                         utility_model=utility_model,
                         probe_results=probe_results,
                         ablation_results=ablation_results,
                         utility_landscape_path=viz_paths.get('landscape'),
                         preference_graph_path=viz_paths.get('graph'))


# Add link to utility analysis in the dimension view
# This should be added to the dimension.html template
def add_utility_analysis_link(model_name, dimension):
    """Check if utility analysis is available and return link."""
    utility_model = load_utility_model_data(model_name, dimension)
    if utility_model:
        return url_for('utility_analysis_view', model_name=model_name, dimension=dimension)
    return None