# %%
import torch
import numpy as np
from transformer_lens import HookedTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from einops import rearrange, reduce, repeat


# %%
@dataclass
class ScenarioVariation:
    """Class to hold variations of a moral scenario"""
    base_scenario: str
    variations: List[str]

class VarianceVisualizer:
    """Dedicated class for visualization of variance analysis results"""
    
    def __init__(self, style: str = "darkgrid"):
        """Initialize visualizer with style settings"""
        sns.set_style(style)
        plt.rcParams['figure.figsize'] = [12, 8]
        
    def plot_layer_variance_progression(self, results: Dict[str, np.ndarray]) -> plt.Figure:
        """Create detailed layer variance progression plot"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Overall layer variance with trend
        x = np.arange(len(results['layer_variances']))
        sns.regplot(x=x, y=results['layer_variances'], 
                   scatter=True, ax=ax1)
        ax1.set_title('Layer Variance Progression')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Variance')
        
        # 2. Attention vs MLP variance comparison
        sns.boxplot(data=results['component_variances'], ax=ax2)
        ax2.set_title('Attention vs MLP Variance')
        ax2.set_xlabel('Component Type')
        ax2.set_ylabel('Variance')
        
        # 3. Layer-wise correlation matrix
        sns.heatmap(results['layer_correlations'], 
                   cmap='coolwarm', center=0, ax=ax3)
        ax3.set_title('Layer-wise Correlation Matrix')
        
        # 4. Cumulative variance explained
        sns.lineplot(data=results['cumulative_variance'], ax=ax4)
        ax4.set_title('Cumulative Variance Explained')
        ax4.set_xlabel('Number of Components')
        ax4.set_ylabel('Cumulative Variance Ratio')
        
        plt.tight_layout()
        return fig
    
    def plot_activation_patterns(self, results: Dict[str, torch.Tensor]) -> plt.Figure:
        """Plot detailed activation patterns"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Residual stream evolution
        residual_data = results['residual_stream'].squeeze().cpu().numpy()
        sns.heatmap(residual_data, 
                    ax=axes[0,0], cmap='viridis')
        axes[0,0].set_title('Residual Stream Evolution')
        
        # 2. Attention pattern evolution
        attention_data = results['attention_patterns'].mean(dim=[0, 1]).cpu().numpy()
        sns.heatmap(attention_data, 
                    ax=axes[0,1], cmap='viridis')
        axes[0,1].set_title('Average Attention Patterns')
        
        # 3. MLP activation distributions
        mlp_data = results['mlp_acts'].cpu().numpy().flatten()
        sns.kdeplot(data=mlp_data, ax=axes[1,0])
        axes[1,0].set_title('MLP Activation Distribution')
        
        # 4. Head importance analysis
        head_importance_data = results['head_importance'].cpu().numpy()
        sns.barplot(data=head_importance_data, ax=axes[1,1])
        axes[1,1].set_title('Attention Head Importance')
        
        plt.tight_layout()
        return fig

class LayerVarianceAnalyzer:
    def __init__(self, model_name: str = "gpt2-xl"):
        """Initialize with HookedTransformer"""
        self.model = HookedTransformer.from_pretrained(model_name)
        self.visualizer = VarianceVisualizer()
        
    def get_activation_names(self) -> List[str]:
        """Get names of activations we want to track"""
        return [
            'hook_resid_pre',      # Residual stream before layer
            'hook_attn_out',       # Output of attention
            'hook_mlp_out',        # Output of MLP
            'hook_resid_post',     # Residual stream after layer
            'attn.hook_pattern'    # Attention patterns
        ]
        
    def collect_activations(self, text: str) -> Dict[str, torch.Tensor]:
        """Collect activations using HookedTransformer without padding"""
        activation_names = self.get_activation_names()
        
        # Create hooks for all layers
        hooks = []
        for name in activation_names:
            for layer in range(self.model.cfg.n_layers):
                hooks.append((f"blocks.{layer}.{name}", name))
        
        # Run model with hooks
        tokens = self.model.to_tokens(text)
        _, cache = self.model.run_with_cache(
            tokens,
            names_filter=lambda name: any(hook[1] in name for hook in hooks)
        )
        
        return cache

    def compute_enhanced_metrics(
        self, 
        activation_caches: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """Compute enhanced set of variance metrics using only valid tokens"""
        n_layers = self.model.cfg.n_layers
        
        results = {
            'layer_variances': [],
            'component_variances': {'attention': [], 'mlp': []},
            'layer_correlations': torch.zeros(n_layers, n_layers),
            'residual_stream': [],
            'attention_patterns': [],
            'mlp_acts': [],
            'head_importance': []
        }
        
        # Process each layer
        for layer in range(n_layers):
            layer_acts = []
            attn_acts = []
            mlp_acts = []
            
            # Process each variation
            for cache in activation_caches:
                # Get activations
                resid = cache[f'blocks.{layer}.hook_resid_pre']
                attn = cache[f'blocks.{layer}.hook_attn_out']
                mlp = cache[f'blocks.{layer}.hook_mlp_out']
                
                # Get attention patterns directly from cache
                attn_pattern = cache[f'blocks.{layer}.attn.hook_pattern']
                
                # Get only the last token's activations for comparison
                # Mean over all tokens (mean(dim=1))
                # Last token only ([:, -1, :])
                # First token only ([:, 0, :])

                layer_acts.append(resid.mean(dim=1))
                attn_acts.append(attn.mean(dim=1))
                mlp_acts.append(mlp.mean(dim=1))
                
            # Stack activations
            layer_acts = torch.stack(layer_acts)
            attn_acts = torch.stack(attn_acts)
            mlp_acts = torch.stack(mlp_acts)
            
            # Store results
            results['layer_variances'].append(layer_acts.var().item())
            results['component_variances']['attention'].append(attn_acts.var().item())
            results['component_variances']['mlp'].append(mlp_acts.var().item())
            
            # Store activations for visualization
            results['residual_stream'].append(layer_acts.mean(0))
            # Get attention patterns from cache
            results['attention_patterns'].append(
                cache[f'blocks.{layer}.attn.hook_pattern'].mean(0)  # Average over batch
            )
            results['mlp_acts'].append(mlp_acts)
            
            # Compute head importance using norm of attention outputs
            head_output = attn_acts  # Already have attention outputs
            results['head_importance'].append(head_output.norm(dim=-1).mean(-1))
        
        # Convert lists to tensors
        results['residual_stream'] = torch.stack(results['residual_stream'])
        results['attention_patterns'] = torch.stack(results['attention_patterns'])
        results['mlp_acts'] = torch.stack(results['mlp_acts'])
        results['head_importance'] = torch.stack(results['head_importance'])
        
        # Compute layer correlations
        for i in range(n_layers):
            for j in range(n_layers):
                # Stack the two vectors into a 2D tensor
                correlation_input = torch.stack([
                    results['residual_stream'][i].flatten(),
                    results['residual_stream'][j].flatten()
                ])
                # Calculate correlation matrix and get the off-diagonal element
                corr = torch.corrcoef(correlation_input)[0,1]
                results['layer_correlations'][i,j] = corr
        
        # PCA for cumulative variance
        flat_acts = results['residual_stream'].reshape(-1, results['residual_stream'].shape[-1])
        pca = PCA()
        flat_acts_cpu = flat_acts.cpu().numpy()
        pca.fit(flat_acts_cpu)
        results['cumulative_variance'] = np.cumsum(pca.explained_variance_ratio_)
        
        return results

    def generate_scenario_variations(self, base_scenario: str, n_variations: int = 5) -> List[str]:
        """Generate variations of the base scenario"""
        variations = [
            base_scenario,  # Original
            base_scenario.replace("intentionally", "deliberately"),  # Synonym variation
            base_scenario.replace("The chairman", "The CEO"),  # Role variation
            base_scenario.replace("environment", "ecosystem"),  # Object variation
            "When asked about the situation, " + base_scenario,  # Context variation
        ]
        return variations[:n_variations]

    def statistical_tests(self, results: Dict) -> Dict:
        """Perform statistical tests on the variance patterns"""
        stats_results = {}
        
        # 1. Test for trend in layer variances
        x = np.arange(len(results['layer_variances']))
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            x, results['layer_variances']
        )
        stats_results['variance_trend'] = {
            'slope': slope,
            'p_value': p_value,
            'r_squared': r_value**2
        }
        
        # 2. Compare attention vs MLP variances
        t_stat, t_p = stats.ttest_ind(
            results['component_variances']['attention'],
            results['component_variances']['mlp']
        )
        stats_results['component_comparison'] = {
            't_statistic': t_stat,
            'p_value': t_p
        }
        
        return stats_results

    def analyze_scenario(self, scenario: ScenarioVariation) -> Dict:
        """Run complete analysis for a scenario and its variations"""
        # Collect activations
        activation_caches = []
        for text in scenario.variations:
            cache = self.collect_activations(text)
            activation_caches.append(cache)
        
        # Compute metrics
        results = self.compute_enhanced_metrics(activation_caches)
        
        # Statistical tests
        stats_results = self.statistical_tests(results)
        
        # Generate visualizations
        variance_fig = self.visualizer.plot_layer_variance_progression(results)
        patterns_fig = self.visualizer.plot_activation_patterns(results)
        
        return {
            'metrics': results,
            'statistics': stats_results,
            'figures': {
                'variance_progression': variance_fig,
                'activation_patterns': patterns_fig
            }
        }

# %%
# Example usage:
if __name__ == "__main__":
    # %%
    
    # Define base scenario
    base_scenario = """
    The vice-president of a company went to the chairman of the board and said, 
    'We are thinking of starting a new program. It will help us increase profits, 
    but it will also harm the environment.' The chairman answered, 'I don't care 
    at all about harming the environment. I just want to make as much profit as 
    I can. Let's start the new program.' They started the new program. Sure 
    enough, the environment was harmed.
    """
    
    # %%
    # Initialize analyzer
    analyzer = LayerVarianceAnalyzer()
    # %%
    # Generate variations
    variations = analyzer.generate_scenario_variations(base_scenario)
    # %%
    # Create scenario object
    scenario = ScenarioVariation(
        base_scenario=base_scenario,
        variations=variations
    )
    # %%
    # Run analysis
    results = analyzer.analyze_scenario(scenario)
    # %%
    # Print statistical results
    print("\nStatistical Analysis:")
    print("Variance Trend:")
    print(f"  Slope: {results['statistics']['variance_trend']['slope']:.4f}")
    print(f"  P-value: {results['statistics']['variance_trend']['p_value']:.4f}")
    print(f"  R-squared: {results['statistics']['variance_trend']['r_squared']:.4f}")
    
    print("\nComponent Comparison (Attention vs MLP):")
    print(f"  T-statistic: {results['statistics']['component_comparison']['t_statistic']:.4f}")
    print(f"  P-value: {results['statistics']['component_comparison']['p_value']:.4f}")
    
    # Show visualizations
    plt.show()
# %%
