# Moral Circuit Analysis with Utility Engineering

Welcome to the documentation for the Moral Circuit Analysis framework, now enhanced with Utility Engineering capabilities. This project combines mechanistic interpretability with behavioral analysis to understand moral reasoning in large language models.

## Overview

This framework provides a comprehensive approach to understanding how LLMs process moral information by:

1. **Identifying Moral Neurons**: Finding neurons that respond differently to moral vs. immoral content
2. **Discovering Moral Circuits**: Analyzing co-activation patterns to identify moral reasoning circuits
3. **Extracting Utility Functions**: Using preference elicitation to build coherent value representations
4. **Connecting Mechanisms to Behavior**: Linking neural activations to high-level moral preferences

## Key Features

### 🧠 Mechanistic Analysis
- Neuron activation analysis across moral dimensions
- Circuit discovery through co-activation patterns
- Temporal pattern detection in moral processing
- Natural language descriptions of neuron behavior

### 📊 Utility Engineering
- Forced-choice preference elicitation
- Thurstonian utility model fitting
- Transitivity and completeness analysis
- Utility probe training on hidden states

### 🔬 Unified Framework
- Integration of neuron-level and utility-level analysis
- Cross-validation between mechanisms and behavior
- Enhanced ablation studies with utility metrics
- Comprehensive visualization suite

## Quick Example

```python
# Convert moral pairs to preferences
python scripts/convert_moral_to_preferences.py \
    --input data/moral_pairs.json \
    --output data/preferences.json \
    --include-cross-dimension

# Elicit preferences from model
python scripts/elicit_preferences.py \
    --model google/gemma-2b \
    --data data/preferences.json \
    --output results/preferences_elicited.json

# Fit utility models
python scripts/fit_utility_models.py \
    --preferences results/preferences_elicited.json \
    --output-dir results/utility_models

# Run unified analysis
python scripts/run_unified_analysis.py \
    --model google/gemma-2b \
    --preferences results/preferences_elicited.json \
    --utility-models results/utility_models \
    --moral-results results/moral_analysis.pkl \
    --output-dir results/unified_analysis
```

## Research Questions

This framework helps answer:

- Do LLMs develop coherent moral value systems?
- How do neural mechanisms give rise to moral preferences?
- Can we predict utility values from neuron activations?
- How do moral circuits affect preference coherence?
- What is the relationship between mechanistic and behavioral representations?

## Documentation Structure

- **[Getting Started](getting-started/installation.md)**: Installation and setup instructions
- **[Core Concepts](concepts/overview.md)**: Understanding the theoretical foundations
- **[Analysis Pipeline](pipeline/data-preparation.md)**: Step-by-step analysis guides
- **[API Reference](api/data.md)**: Detailed API documentation
- **[Tutorials](tutorials/basic-analysis.md)**: Hands-on examples and workflows
- **[Results](results/interpretation.md)**: Understanding and visualizing outputs

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{moral_circuit_utility_2025,
    title = {Moral Circuit Analysis with Utility Engineering},
    author = {Your Name},
    year = {2025},
    url = {https://github.com/prof-schacht/moral_circuit_analysis}
}
```

## License

This project is licensed under the MIT License. See LICENSE file for details.