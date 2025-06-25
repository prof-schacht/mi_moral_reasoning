# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a moral circuit analysis framework for large language models that identifies and analyzes neural circuits involved in moral decision-making. The project uses transformer_lens to analyze neurons responding to moral/immoral content across different moral dimensions (care, liberty, sanctity, authority, loyalty, fairness) based on Moral Foundation Theory.

## Key Commands

### Running Main Analysis
```bash
# Analyze models for moral neurons
python scripts/analyze_models.py --models "google/gemma-2b" --dimensions "care" "fairness"

# Alternative with run_analysis.py
python scripts/run_analysis.py \
    --model_name google/gemma-2-9b-it \
    --data_path data/moral_pairs.json \
    --results_dir results \
    --significant_diff 0.5 \
    --consistency_threshold 0.8
```

### Ablation Analysis
```bash
# Run ablation study
python scripts/run_ablation_analysis.py \
    --model_name google/gemma-2-9b-it \
    --dimension care \
    --cluster cl1 \
    --comparison moral_vs_immoral \
    --ablation_value -20.0 \
    --output_dir results/ablation

# Summarize ablation results
python scripts/summarize_ablation_results.py
```

### Neuron Description Generation
```bash
python scripts/describe_neurons.py --model google/gemma-2-9b-it --neurons neurons.json

# Alternative with generate_descriptions.py
python scripts/generate_descriptions.py \
    --model_name google/gemma-2-9b-it \
    --results_path results/moral_circuit_results.pkl \
    --llm_name gpt-4 \
    --output_dir results/descriptions
```

### Running Tests
```bash
python tests/test_visualization.py
# or if pytest is installed:
pytest tests/
```

### Web Reports Interface
```bash
cd reports
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

## Architecture Overview

The codebase follows a modular architecture with clear separation of concerns:

### Core Analysis Pipeline
1. **Model Loading** (`src/models/`) - Loads transformer models via transformer_lens
2. **Neuron Collection** (`src/analysis/neuron_collector.py`) - Collects neuron activations on moral/immoral prompts
3. **Moral Analysis** (`src/analysis/moral_analyzer.py`) - Identifies neurons with significant moral/immoral activation differences
4. **Circuit Discovery** - Analyzes co-activation patterns to identify moral reasoning circuits
5. **Ablation Studies** (`src/analysis/ablation.py`) - Tests causal relationships by ablating neuron clusters
6. **Visualization** (`src/visualization/`) - Creates circuit, network, and component visualizations

### Key Dependencies
- **transformer_lens** - For accessing model internals and neuron activations
- **torch** - Deep learning framework
- **openai** - For generating neuron descriptions via GPT-4
- **flask** - Web framework for results viewing interface

### Results Structure
Results are organized by model → dimension → analysis type:
```
results/
├── [model_name]/
│   ├── neuron_describer_logs/
│   │   └── [dimension]/
│   │       └── *_neuron-analysis_summary.csv
│   └── ablation/
│       └── [dimension]/
│           ├── *_results.json
│           ├── *_LLM_explanation.txt
│           └── visualizations/
```

### Important Design Patterns
- Scripts in `scripts/` are entry points that use modules from `src/`
- Analysis results are stored as pickle and JSON files for reusability
- Visualization functions generate both data and plots
- The Flask app reads from the standardized results directory structure
- Moral dimensions are defined in `data/mft_dim.py` following Moral Foundation Theory

## Development Notes

- No formal linting or formatting tools are configured - follow existing code style
- The project uses absolute imports from the `src` package
- OPENAI_API_KEY environment variable required for neuron descriptions
- Results can be large - the ablation analysis generates many files per dimension
- The web interface expects specific directory structure in results/