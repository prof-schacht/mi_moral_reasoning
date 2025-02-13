from flask import Flask, render_template, jsonify, send_from_directory, url_for
import os
import json
from pathlib import Path
import pandas as pd

# Get the absolute path to the application directory
APP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = APP_DIR.parent / "results"

# Initialize Flask app with explicit template and static folders
app = Flask(__name__,
           template_folder=str(APP_DIR / "templates"),
           static_folder=str(APP_DIR / "static"))

# Enable debug mode and configure
app.config['DEBUG'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Add a route to serve files from results directory
@app.route('/results/<path:filename>')
def serve_results(filename):
    """Serve files from the results directory."""
    try:
        return send_from_directory(str(RESULTS_DIR), filename)
    except Exception as e:
        print(f"Error serving file {filename}: {e}")
        return str(e), 403

def get_available_models():
    """Get list of all models that have been analyzed."""
    models = set()
    
    # Check all possible locations for model results
    if RESULTS_DIR.exists():
        # Check main results directory
        models.update(d.name for d in RESULTS_DIR.iterdir() if d.is_dir())
        
        # Check ablation directory
        ablation_dir = RESULTS_DIR / "ablation"
        if ablation_dir.exists():
            # Convert underscore names back to hyphenated format
            ablation_models = [d.name.replace('_', '-') for d in ablation_dir.iterdir() if d.is_dir()]
            models.update(ablation_models)
            print(f"Found ablation models: {ablation_models}")
        
        # Check neuron describer directory
        describer_dir = RESULTS_DIR / "neuron_describer_logs"
        if describer_dir.exists():
            # Convert underscore names back to hyphenated format
            describer_models = [d.name.replace('_', '-') for d in describer_dir.iterdir() if d.is_dir()]
            models.update(describer_models)
            print(f"Found describer models: {describer_models}")
    
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Total models found: {models}")
    return sorted(list(models))

def get_model_dimensions(model_name):
    """Get available moral dimensions for a given model."""
    dimensions = set()
    
    # Convert model name format for ablation and describer directories
    model_name_underscore = model_name.replace('-', '_')
    print(f"Looking for dimensions with model name: {model_name_underscore}")
    
    # Check ablation directory
    ablation_dir = RESULTS_DIR / "ablation" / model_name_underscore
    print(f"Checking ablation directory: {ablation_dir}")
    if ablation_dir.exists():
        ablation_dims = [d.name for d in ablation_dir.iterdir() if d.is_dir()]
        dimensions.update(ablation_dims)
        print(f"Found ablation dimensions: {ablation_dims}")
    else:
        print(f"Ablation directory does not exist: {ablation_dir}")
    
    # Check neuron_describer_logs directory
    describer_dir = RESULTS_DIR / "neuron_describer_logs" / model_name_underscore
    print(f"Checking describer directory: {describer_dir}")
    if describer_dir.exists():
        describer_dims = [d.name for d in describer_dir.iterdir() if d.is_dir()]
        dimensions.update(describer_dims)
        print(f"Found describer dimensions: {describer_dims}")
    else:
        print(f"Describer directory does not exist: {describer_dir}")
    
    print(f"Total dimensions found for {model_name}: {dimensions}")
    return sorted(list(dimensions))

def get_neuron_descriptions(model_name, dimension):
    """Get neuron descriptions for a given model and dimension."""
    model_name_underscore = model_name.replace('-', '_')
    
    # Look for neuron description files in the specific dimension directory
    desc_dir = RESULTS_DIR / "neuron_describer_logs" / model_name_underscore / dimension
    
    if not desc_dir.exists():
        print(f"Neuron description directory not found at {desc_dir}")
        return None
        
    # Look for the neuron analysis summary file with the specific dimension
    pattern = f"*{model_name_underscore}_moral-{dimension}_neuron-analysis_summary.csv"
    csv_files = list(desc_dir.parent.glob(pattern))
    
    if not csv_files:
        print(f"No neuron description summary file found matching pattern {pattern}")
        return None
        
    # Use the most recent CSV file if multiple exist
    latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading neuron descriptions from {latest_csv}")
    
    try:
        return pd.read_csv(latest_csv)
    except Exception as e:
        print(f"Error reading neuron descriptions CSV: {e}")
        return None

def get_ablation_results(model_name, dimension):
    """Get ablation results for a given model and dimension."""
    model_name_underscore = model_name.replace('-', '_')
    ablation_dir = RESULTS_DIR / "ablation" / model_name_underscore / dimension
    if not ablation_dir.exists():
        print(f"Warning: Ablation directory not found at {ablation_dir}")  # Debug print
        return []
    
    results = []
    for result_file in ablation_dir.glob("*_results.json"):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # Get corresponding LLM explanation if it exists
            explanation_file = result_file.parent / f"{result_file.stem.replace('_results', '')}_LLM_explanation.txt"
            explanation = ""
            if explanation_file.exists():
                with open(explanation_file, 'r') as f:
                    explanation = f.read()
            
            # Check for visualizations
            vis_dir = ablation_dir / "visualizations"
            vis_files = []
            if vis_dir.exists():
                pattern = f"{result_file.stem.split('_results')[0]}*.png"
                vis_files = [str(f.relative_to(RESULTS_DIR)) for f in vis_dir.glob(pattern)]
                print(f"Found visualization files: {vis_files}")
            
            # Parse the filename for a meaningful title
            title = parse_ablation_filename(result_file.stem)
            
            results.append({
                'timestamp': result_file.stem.split('_')[0],
                'title': title,
                'data': data,
                'explanation': explanation,
                'visualizations': vis_files
            })
        except Exception as e:
            print(f"Error processing result file {result_file}: {e}")  # Debug print
            continue
    
    return sorted(results, key=lambda x: x['timestamp'], reverse=True)

# Add route to serve visualization files
@app.route('/visualizations/<path:filepath>')
def serve_visualization(filepath):
    """Serve visualization files from the results directory."""
    try:
        # Get the directory containing the file
        file_path = RESULTS_DIR / filepath
        directory = file_path.parent
        filename = file_path.name
        print(f"Serving visualization: {directory} / {filename}")
        return send_from_directory(str(directory), filename)
    except Exception as e:
        print(f"Error serving visualization {filepath}: {e}")
        return str(e), 404

# Update the template to use the new visualization route
@app.template_filter('vis_url')
def visualization_url_filter(filepath):
    """Convert visualization filepath to URL."""
    return url_for('serve_visualization', filepath=filepath)

@app.route('/')
def index():
    """Main page showing available models."""
    try:
        models = get_available_models()
        print(f"Rendering index.html with models: {models}")
        print(f"Template folder: {app.template_folder}")
        print(f"Available templates: {os.listdir(app.template_folder)}")
        return render_template('index.html', models=models)
    except Exception as e:
        print(f"Error rendering index page: {e}")
        return f"""
        <html>
            <head><title>Error</title></head>
            <body>
                <h1>Error Loading Page</h1>
                <p>Error: {str(e)}</p>
                <p>Template Directory: {app.template_folder}</p>
                <p>Available Templates: {os.listdir(app.template_folder)}</p>
                <p>Current Working Directory: {os.getcwd()}</p>
                <p>APP_DIR: {APP_DIR}</p>
            </body>
        </html>
        """

@app.route('/model/<model_name>')
def model_overview(model_name):
    """Overview page for a specific model."""
    try:
        dimensions = get_model_dimensions(model_name)
        print(f"Rendering model.html for {model_name} with dimensions: {dimensions}")
        print(f"Template folder: {app.template_folder}")
        print(f"Available templates: {os.listdir(app.template_folder)}")
        return render_template('model.html', 
                             model_name=model_name, 
                             dimensions=dimensions)
    except Exception as e:
        print(f"Error rendering model page: {e}")
        return f"""
        <html>
            <head><title>Error</title></head>
            <body>
                <h1>Error Loading Model Page</h1>
                <p>Model: {model_name}</p>
                <p>Error: {str(e)}</p>
                <p>Template Directory: {app.template_folder}</p>
                <p>Available Templates: {os.listdir(app.template_folder)}</p>
                <p>Current Working Directory: {os.getcwd()}</p>
                <p>APP_DIR: {APP_DIR}</p>
            </body>
        </html>
        """

@app.route('/model/<model_name>/<dimension>')
def dimension_results(model_name, dimension):
    """Results page for a specific model and moral dimension."""
    try:
        descriptions = get_neuron_descriptions(model_name, dimension)
        ablation_results = get_ablation_results(model_name, dimension)
        print(f"Rendering dimension.html for {model_name}/{dimension}")
        return render_template('dimension.html',
                            model_name=model_name,
                            dimension=dimension,
                            descriptions=descriptions,
                            ablation_results=ablation_results)
    except Exception as e:
        print(f"Error rendering dimension page: {e}")
        return f"""
        <html>
            <head><title>Error</title></head>
            <body>
                <h1>Error Loading Dimension Page</h1>
                <p>Model: {model_name}</p>
                <p>Dimension: {dimension}</p>
                <p>Error: {str(e)}</p>
                <p>Template Directory: {app.template_folder}</p>
                <p>Available Templates: {os.listdir(app.template_folder)}</p>
            </body>
        </html>
        """

@app.route('/api/neuron_descriptions/<model_name>/<dimension>')
def api_neuron_descriptions(model_name, dimension):
    """API endpoint for neuron descriptions."""
    try:
        descriptions = get_neuron_descriptions(model_name, dimension)
        if descriptions is None:
            return jsonify([])
        return jsonify(descriptions.to_dict(orient='records'))
    except Exception as e:
        print(f"Error in API endpoint: {e}")
        return jsonify({"error": str(e)}), 500

def parse_ablation_filename(filename):
    """Parse ablation filename to get meaningful information."""
    try:
        # Example filename: 20250212_215918_gemma_2_9b_it_care_cl1_moral_vs_immoral_immoral_ablation_value_10.0_results
        parts = filename.split('_')
        
        # Extract date (first 8 digits)
        date = parts[0]
        formatted_date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
        
        # Find cluster part
        cluster_idx = next(i for i, part in enumerate(parts) if part.startswith('cl'))
        cluster_num = parts[cluster_idx][2:]
        
        # Find moral vs part and what follows
        vs_idx = parts.index('vs')
        comparison = parts[vs_idx + 1]  # immoral or neutral
        
        # Find ablation value
        value_idx = parts.index('value')
        value = parts[value_idx + 1]
        
        return {
            'date': formatted_date,
            'cluster': cluster_num,
            'comparison': comparison,
            'ablation_value': value
        }
    except Exception as e:
        print(f"Error parsing filename {filename}: {e}")
        return {
            'date': '',
            'cluster': '',
            'comparison': filename,
            'ablation_value': ''
        }

if __name__ == '__main__':
    print(f"Starting Flask app with:")
    print(f"- Template directory: {app.template_folder}")
    print(f"- Static directory: {app.static_folder}")
    print(f"- Results directory: {RESULTS_DIR}")
    print(f"- Current working directory: {os.getcwd()}")
    print(f"- Available templates: {os.listdir(app.template_folder)}")
    
    # Create symbolic link for results in static directory if it doesn't exist
    results_static_link = Path(app.static_folder) / "results"
    if not results_static_link.exists():
        os.symlink(str(RESULTS_DIR), str(results_static_link), target_is_directory=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000) 