# Moral Circuit Analysis Web Reports

This web application provides an interactive interface to explore the results of moral circuit analysis experiments, including neuron identification, descriptions, and ablation studies.

## Features

- Browse results by model and moral dimension
- View detailed neuron descriptions and their properties
- Explore ablation study results with visualizations
- Read LLM-generated explanations of the findings
- Interactive data visualization

## Directory Structure

```
reports/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── static/            # Static files (CSS, JS, images)
├── templates/         # HTML templates
│   ├── base.html     # Base template
│   ├── index.html    # Model list
│   ├── model.html    # Model overview
│   └── dimension.html # Detailed results
└── README.md         # This file
```

## Setup

1. Create a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

4. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## Usage

1. On the home page, select a model from the available options
2. Choose a moral dimension (e.g., care, fairness) to analyze
3. Explore the results:
   - Neuron descriptions and properties
   - Ablation study results
   - Visualizations and analyses

## Data Structure

The application expects the following directory structure for the results:

```
results/
├── model_name/
│   ├── neuron_describer_logs/
│   │   └── dimension/
│   │       └── *_neuron-analysis_summary.csv
│   └── ablation/
│       └── dimension/
│           ├── *_results.json
│           ├── *_LLM_explanation.txt
│           └── visualizations/
│               └── *.png
```

## Contributing

Feel free to submit issues and enhancement requests! 