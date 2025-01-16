# Moral Circuit Analysis

This project provides tools for analyzing moral behavior in large language models by identifying and studying neural circuits involved in moral decision-making.

## Features

- Identification of neurons that consistently respond to moral/immoral content
- Analysis of neuron co-activation patterns and circuit formation
- Visualization of moral circuits and their distribution across model layers
- Generation of natural language descriptions for moral neurons using LLMs
- Ablation analysis to study causal relationships in moral circuits

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/moral-circuit-analysis.git
cd moral-circuit-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Main Analysis

```bash
python scripts/run_analysis.py \
    --model_name google/gemma-2-9b-it \
    --data_path data/moral_pairs.json \
    --results_dir results \
    --significant_diff 0.5 \
    --consistency_threshold 0.8
```

### Generating Neuron Descriptions

```bash
python scripts/generate_descriptions.py \
    --model_name google/gemma-2-9b-it \
    --results_path results/moral_circuit_results.pkl \
    --llm_name gpt-4 \
    --output_dir results/descriptions
```

### Running Ablation Analysis

```bash
python scripts/generate_ablation.py \
    --model_name google/gemma-2-9b-it \
    --data_path data/moral_pairs.json \
    --results_path results/moral_circuit_results.pkl \
    --output_dir results/ablation
```

## Project Structure

```
moral_circuit_analysis/
├── README.md
├── requirements.txt
├── setup.py
├── src/
│   ├── models/
│   │   ├── model_loader.py        # Model loading utilities
│   │   └── model_config.py        # Model configurations
│   ├── analysis/
│   │   ├── neuron_collector.py    # Base neuron analysis
│   │   ├── moral_analyzer.py      # Moral behavior analysis
│   │   ├── neuron_describer.py    # Neuron description generation
│   │   └── ablation.py           # Ablation analysis
│   ├── visualization/
│   │   ├── circuit_plots.py       # Circuit visualization
│   │   ├── network_plots.py       # Network visualization
│   │   └── component_plots.py     # Component visualization
│   └── utils/
│       ├── data_loader.py         # Data loading utilities
│       └── metrics.py             # Analysis metrics
├── scripts/
│   ├── run_analysis.py           # Main analysis script
│   ├── generate_descriptions.py   # Description generation
│   └── generate_ablation.py      # Ablation analysis
└── data/
    └── README.md                 # Data documentation
```

## Data Format

The analysis expects moral/immoral text pairs in either JSON or pickle format:

```json
[
    {
        "moral": "I should help the elderly cross the street",
        "immoral": "I should ignore the elderly crossing the street"
    },
    ...
]
```

## Results

The analysis produces several outputs:

1. Identified moral and immoral neurons
2. Layer importance rankings
3. Circuit visualizations
4. Neuron descriptions (if enabled)
5. Ablation analysis results

Results are saved in the specified output directory with visualizations and detailed statistics.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{moral_circuit_analysis,
    title = {Moral Circuit Analysis},
    author = {Sigurd Schacht},
    year = {2025},
    url = {https://github.com/prof-schacht/moral_circuit_analysis}
}
``` 