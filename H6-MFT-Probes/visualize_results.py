# %%
"""
Script to generate visualizations of probe performance across layers and moral foundations.
"""

import logging
from models.visualization import plot_layer_metrics, plot_class_distribution

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Generate visualizations of probe performance."""
    # Set paths
    probe_dir = "data/probes"
    output_dir = "figures"
    
    # Generate plots
    logger.info("Generating layer-wise metric plots...")
    plot_layer_metrics(probe_dir, output_dir)
    
    logger.info("Generating class distribution plots...")
    plot_class_distribution(probe_dir, output_dir)
    
    logger.info("Visualization complete!")

if __name__ == "__main__":
    main()
# %% 