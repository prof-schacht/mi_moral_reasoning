# Installation

This guide will help you set up the Moral Circuit Analysis framework with Utility Engineering extensions.

## Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended for large models)
- At least 16GB RAM (32GB recommended)
- 10GB free disk space for models and results

## Basic Installation

### 1. Clone the Repository

```bash
git clone https://github.com/prof-schacht/moral_circuit_analysis.git
cd moral_circuit_analysis
```

### 2. Create Virtual Environment

=== "Linux/macOS"
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

=== "Windows"
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Dependencies Overview

The framework relies on several key packages:

### Core Dependencies
- **torch** (>=2.0.0): Deep learning framework
- **transformer_lens** (>=1.2.0): For accessing model internals
- **transformers**: Hugging Face transformers library

### Analysis Libraries
- **numpy**: Numerical computations
- **scipy**: Scientific computing and optimization
- **scikit-learn**: Machine learning utilities
- **pandas**: Data manipulation

### Visualization
- **matplotlib**: Plotting library
- **seaborn**: Statistical visualizations
- **networkx**: Graph analysis and visualization

### Utility Analysis
- **openai** (>=1.0.0): For neuron descriptions (requires API key)
- **python-dotenv**: Environment variable management

## Environment Setup

### 1. Create `.env` File

Create a `.env` file in the project root for API keys:

```bash
OPENAI_API_KEY=your-openai-api-key-here
```

### 2. Configure Model Cache

Set the Hugging Face cache directory (optional):

```bash
export HF_HOME=/path/to/your/cache
```

## Verification

Verify your installation:

```python
# test_installation.py
import torch
import transformer_lens
from src.models.model_loader import load_model

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Transformer Lens version: {transformer_lens.__version__}")

# Try loading a small model
model, tokenizer = load_model("gpt2", device="cpu")
print("✓ Installation successful!")
```

Run the test:

```bash
python test_installation.py
```

## Documentation Dependencies

To build and serve the documentation locally:

```bash
pip install mkdocs mkdocs-material mkdocs-mermaid2
```

Serve documentation:

```bash
mkdocs serve
```

## Web Interface Dependencies

For the results visualization web interface:

```bash
cd reports
pip install -r requirements.txt
```

## Troubleshooting

### CUDA Issues

If you encounter CUDA errors:

1. Check PyTorch CUDA version:
   ```python
   import torch
   print(torch.version.cuda)
   ```

2. Reinstall PyTorch with correct CUDA version:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

### Memory Issues

For large models, you may need to:

1. Use CPU instead of GPU:
   ```bash
   python scripts/analyze_models.py --device cpu
   ```

2. Reduce batch size in analysis scripts

3. Use model quantization or smaller models

### Missing Dependencies

If you encounter import errors:

```bash
pip install --upgrade -r requirements.txt
```

## Next Steps

Once installation is complete:

1. [Quick Start Guide](quickstart.md) - Run your first analysis
2. [Project Structure](structure.md) - Understand the codebase organization
3. [Data Preparation](../pipeline/data-preparation.md) - Prepare your moral data