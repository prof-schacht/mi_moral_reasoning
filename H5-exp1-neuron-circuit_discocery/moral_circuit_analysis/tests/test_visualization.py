import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.analysis.visualization import ProbeVisualizer

def test_plot_legends(ax_or_fig, expected_labels, subplot_specific_labels=None):
    """Helper function to verify plot legends contain expected labels.
    
    Args:
        ax_or_fig: Either a matplotlib Axes object or Figure object
        expected_labels: List of expected legend labels for single plot or all subplots
        subplot_specific_labels: Dict mapping subplot index to its expected labels
    """
    if hasattr(ax_or_fig, 'axes'):  # It's a Figure
        axes = ax_or_fig.axes
    else:  # It's an Axes
        axes = [ax_or_fig]
        
    if subplot_specific_labels:
        # Check each subplot against its specific expected labels
        for idx, ax in enumerate(axes):
            if idx in subplot_specific_labels and ax.get_legend() is not None:
                legend_texts = [text.get_text() for text in ax.get_legend().get_texts()]
                for label in subplot_specific_labels[idx]:
                    assert label in legend_texts, f"Expected label '{label}' not found in subplot {idx} legend: {legend_texts}"
    else:
        # Check all axes against the same expected labels
        for ax in axes:
            if ax.get_legend() is not None:
                legend_texts = [text.get_text() for text in ax.get_legend().get_texts()]
                for label in expected_labels:
                    assert label in legend_texts, f"Expected label '{label}' not found in legend: {legend_texts}"

def test_visualizer():
    """Test the ProbeVisualizer class with dummy data for both immoral and neutral comparisons."""
    print("Testing ProbeVisualizer...")
    
    # Create dummy data
    np.random.seed(42)  # For reproducibility
    
    # Generate some random predictions
    n_samples = 10
    moral_predictions = [
        (np.random.random(), np.random.random()) 
        for _ in range(n_samples)
    ]
    comparison_predictions = [
        (np.random.random(), np.random.random()) 
        for _ in range(n_samples)
    ]
    
    # Create temporary directory for test outputs
    test_dir = Path("test_outputs")
    test_dir.mkdir(exist_ok=True)
    
    try:
        # Test for both comparison types
        for comparison_type in ["immoral", "neutral"]:
            print(f"\nTesting visualizations for {comparison_type} comparison...")
            
            # Initialize visualizer
            visualizer = ProbeVisualizer()
            
            # Test trajectory plot
            print(f"Testing trajectory plot for {comparison_type}...")
            traj_fig = visualizer.create_trajectory_plot(
                moral_predictions, 
                comparison_predictions,
                comparison_type=comparison_type
            )
            test_plot_legends(traj_fig, ['Moral Statements', f'{comparison_type.capitalize()} Statements'])
            traj_fig.savefig(test_dir / f"test_trajectory_{comparison_type}.png")
            plt.close(traj_fig)
            
            # Test separation plot
            print(f"Testing separation plot for {comparison_type}...")
            sep_fig = visualizer.create_separation_plot(
                moral_predictions, 
                comparison_predictions,
                comparison_type=comparison_type
            )
            test_plot_legends(sep_fig, ['Original', 'Ablated'])
            assert sep_fig.axes[0].get_ylabel() == f"{comparison_type.capitalize()} Statement Score"
            sep_fig.savefig(test_dir / f"test_separation_{comparison_type}.png")
            plt.close(sep_fig)
            
            # Test distribution plot
            print(f"Testing distribution plot for {comparison_type}...")
            dist_fig = visualizer.create_distribution_plot(
                moral_predictions, 
                comparison_predictions,
                comparison_type=comparison_type
            )
            # Test legends for both subplots separately
            subplot_labels = {
                0: ['Moral', comparison_type.capitalize()],  # First subplot
                1: ['Moral (Ablated)', f'{comparison_type.capitalize()} (Ablated)']  # Second subplot
            }
            test_plot_legends(dist_fig, None, subplot_labels)
            dist_fig.savefig(test_dir / f"test_distribution_{comparison_type}.png")
            plt.close(dist_fig)
            
            # Test create_all_plots
            print(f"Testing create_all_plots for {comparison_type}...")
            visualizer.create_all_plots(
                moral_predictions=moral_predictions,
                immoral_predictions=comparison_predictions,
                save_dir=test_dir,
                prefix=f"test_all_{comparison_type}",
                comparison_type=comparison_type
            )
            
        print("\nAll tests completed successfully!")
        print(f"Test plots saved to: {test_dir.absolute()}")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        raise
        
if __name__ == "__main__":
    test_visualizer() 