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

The ablation analysis helps understand how different neuron clusters affect the model's moral judgments. The analysis can be run in two steps:

1. First, run the ablation analysis:
```bash
python scripts/run_ablation_analysis.py \
    --model_name google/gemma-2-9b-it \
    --dimension care \
    --cluster cl1 \
    --comparison moral_vs_immoral \
    --ablation_value -20.0 \
    --output_dir results/ablation
```

Parameters:
- `model_name`: Name of the model to analyze
- `dimension`: Moral dimension to analyze (care, liberty, sanctity)
- `cluster`: Which neuron cluster to ablate (cl1, cl2, cl3)
- `comparison`: Type of comparison (moral_vs_immoral or moral_vs_neutral)
- `ablation_value`: Strength of the ablation (-20.0 to 20.0)
- `output_dir`: Directory to save results

2. Then, summarize the results:
```bash
python scripts/summarize_ablation_results.py
```

This will create a summary table of all ablation results in `results/ablation/ablation_summary.csv`.

## Ablation Results Format

The ablation analysis produces results with the following metrics:

- `avg_moral_pred_change`: Average change in moral prediction scores
- `std_moral_pred_change`: Standard deviation of moral prediction changes
- `avg_immoral_pred_change`: Average change in immoral prediction scores
- `std_immoral_pred_change`: Standard deviation of immoral prediction changes
- `moral_effect_size`: Effect size for moral predictions
- `immoral_effect_size`: Effect size for immoral predictions

## Latest Ablation Results

Below is a summary of our latest ablation analysis results for the Gemma 2.9B model across different moral dimensions:

| Dimension | Cluster | Comparison | Ablation | Moral Score Impact | Immoral Score Impact | Moral Effect | Immoral Effect |
|-----------|---------|------------|----------|-------------------|---------------------|--------------|----------------|
| Care | cl1 | moral_vs_immoral | -20.000 | 0.015 | -0.013 | 0.137 | -0.063 |
| Care | cl1 | moral_vs_neutral | -20.000 | 0.015 | 0.465 | 0.137 | 0.936 |
| Liberty | cl2 | moral_vs_immoral | -20.000 | 0.010 | 0.004 | 0.083 | 0.014 |
| Liberty | cl2 | moral_vs_neutral | -20.000 | 0.010 | -0.009 | 0.083 | -0.030 |
| Sanctity | cl1 | moral_vs_immoral | -20.000 | -0.786 | -0.764 | -2.145 | -1.949 |
| Sanctity | cl1 | moral_vs_neutral | -20.000 | -0.786 | -0.149 | -2.145 | -0.467 |

Key findings:
- Care neurons (cl1) show positive moral effects with minimal impact on immoral judgments
- Liberty neurons (cl2) demonstrate balanced effects across moral and immoral scenarios
- Sanctity neurons (cl1) show strong negative effects, suggesting critical role in moral judgment

For the complete results table, see `results/ablation/ablation_summary.csv`.

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
│   ├── run_ablation_analysis.py  # Ablation analysis script
│   ├── summarize_ablation_results.py # Results summarization
│   └── describe_neurons.py       # Neuron description generation
└── data/
    └── README.md                 # Data documentation
```

## Results Directory Structure

After running the analysis, your results directory will look like this:

```
results/
└── ablation/
    └── gemma_2_9b_it/
        ├── care/
        │   ├── visualizations/
        │   │   ├── probe_distribution.png
        │   │   ├── probe_separation.png
        │   │   └── probe_trajectory.png
        │   └── *_results.json
        ├── liberty/
        │   └── ...
        ├── sanctity/
        │   └── ...
        └── ablation_summary.csv
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