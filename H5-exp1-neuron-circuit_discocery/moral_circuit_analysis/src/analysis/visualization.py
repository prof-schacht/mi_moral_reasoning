import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path

class ProbeVisualizer:
    def __init__(self):
        # Set style using seaborn's default style
        sns.set_theme(style="whitegrid")
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (10, 8)
        plt.rcParams['figure.dpi'] = 100
    
    def create_trajectory_plot(self, 
                             moral_predictions: List[Tuple[float, float]], 
                             immoral_predictions: List[Tuple[float, float]], 
                             comparison_type: str = "immoral",
                             title: str = "Moral Assessment Trajectories") -> plt.Figure:
        """Create trajectory plot showing how probe scores change after ablation."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot trajectories
        for orig, abl in moral_predictions:
            ax.arrow(orig, 0.1, abl - orig, 0, 
                    head_width=0.02, head_length=0.01, fc='blue', ec='blue', alpha=0.5)
            
        for orig, abl in immoral_predictions:
            ax.arrow(orig, -0.1, abl - orig, 0,
                    head_width=0.02, head_length=0.01, fc='red', ec='red', alpha=0.5)
        
        # Add labels and styling
        ax.set_xlabel("Probe Score (Moral Assessment)")
        ax.set_ylabel("Statement Type")
        ax.set_title(title)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.5, 0.5)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3)
        
        # Add legend with dynamic labels
        ax.scatter([], [], c='blue', label='Moral Statements')
        ax.scatter([], [], c='red', label=f'{comparison_type.capitalize()} Statements')
        ax.legend()
        
        return fig
    
    def create_separation_plot(self,
                             moral_predictions: List[Tuple[float, float]],
                             immoral_predictions: List[Tuple[float, float]],
                             comparison_type: str = "immoral",
                             title: str = None) -> plt.Figure:
        """Create scatter plot showing moral vs immoral separation."""
        if title is None:
            title = f"Moral vs {comparison_type.capitalize()} Separation"
            
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extract original and ablated scores
        moral_orig = [orig for orig, _ in moral_predictions]
        moral_abl = [abl for _, abl in moral_predictions]
        immoral_orig = [orig for orig, _ in immoral_predictions]
        immoral_abl = [abl for _, abl in immoral_predictions]
        
        # Plot original and ablated points
        ax.scatter(moral_orig, immoral_orig, c='blue', label='Original', alpha=0.6)
        ax.scatter(moral_abl, immoral_abl, c='red', label='Ablated', alpha=0.6)
        
        # Draw arrows between pairs
        for (mo, ma), (io, ia) in zip(zip(moral_orig, moral_abl), 
                                     zip(immoral_orig, immoral_abl)):
            ax.arrow(mo, io, ma - mo, ia - io,
                    head_width=0.02, head_length=0.01, fc='gray', ec='gray', alpha=0.3)
        
        # Add diagonal line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        
        # Add labels and styling
        ax.set_xlabel("Moral Statement Score")
        ax.set_ylabel(f"{comparison_type.capitalize()} Statement Score")
        ax.set_title(title)
        ax.legend()
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        
        return fig
    
    def create_distribution_plot(self,
                               moral_predictions: List[Tuple[float, float]],
                               immoral_predictions: List[Tuple[float, float]],
                               comparison_type: str = "immoral",
                               title: str = "Score Distributions") -> plt.Figure:
        """Create distribution plot showing probe score distributions."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Extract scores
        moral_orig = [orig for orig, _ in moral_predictions]
        moral_abl = [abl for _, abl in moral_predictions]
        immoral_orig = [orig for orig, _ in immoral_predictions]
        immoral_abl = [abl for _, abl in immoral_predictions]
        
        # Plot original distributions
        sns.kdeplot(data=moral_orig, ax=ax1, label='Moral', color='blue')
        sns.kdeplot(data=immoral_orig, ax=ax1, label=f'{comparison_type.capitalize()}', color='red')
        ax1.set_title("Original Score Distribution")
        ax1.legend()
        
        # Plot ablated distributions
        sns.kdeplot(data=moral_abl, ax=ax2, label='Moral (Ablated)', color='blue')
        sns.kdeplot(data=immoral_abl, ax=ax2, label=f'{comparison_type.capitalize()} (Ablated)', color='red')
        ax2.set_title("Ablated Score Distribution")
        ax2.legend()
        
        # Add labels
        fig.suptitle(title)
        for ax in [ax1, ax2]:
            ax.set_xlabel("Probe Score")
            ax.set_ylabel("Density")
        
        plt.tight_layout()
        return fig
    
    def create_all_plots(self,
                        moral_predictions: List[Tuple[float, float]],
                        immoral_predictions: List[Tuple[float, float]],
                        save_dir: Path,
                        prefix: str,
                        comparison_type: str = "immoral"):
        """Create and save all visualization plots."""
        # Create plots
        traj_fig = self.create_trajectory_plot(moral_predictions, immoral_predictions, comparison_type)
        sep_fig = self.create_separation_plot(moral_predictions, immoral_predictions, comparison_type)
        dist_fig = self.create_distribution_plot(moral_predictions, immoral_predictions, comparison_type)
        
        # Save plots
        traj_fig.savefig(save_dir / f"{prefix}_trajectory.png")
        sep_fig.savefig(save_dir / f"{prefix}_separation.png")
        dist_fig.savefig(save_dir / f"{prefix}_distribution.png")
        
        # Close figures to free memory
        plt.close(traj_fig)
        plt.close(sep_fig)
        plt.close(dist_fig) 