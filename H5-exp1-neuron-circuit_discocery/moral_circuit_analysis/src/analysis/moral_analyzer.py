import torch
from typing import List, Tuple, Dict
from tqdm.auto import tqdm
from .neuron_collector import NeuronActivationCollector
from transformer_lens import HookedTransformer

class MoralBehaviorAnalyzer(NeuronActivationCollector):
    def __init__(self, model: HookedTransformer):
        super().__init__(model)
        
    def analyze_moral_behavior(self, moral_pairs: List[Tuple[str, str]], 
                             significant_diff: float = 0.5, 
                             consistency_threshold: float = 0.8) -> Dict:
        """
        Analyze which neurons/layers respond differently to moral vs immoral completions.
        
        Args:
            moral_pairs: List of (moral_text, immoral_text) pairs
            significant_diff: Threshold for considering activation differences significant
            consistency_threshold: Threshold for neuron response consistency
            
        Returns:
            Dictionary containing analysis results
        """
        moral_activations = []
        immoral_activations = []
        
        # Get activations for both moral and immoral texts
        for moral_text, immoral_text in tqdm(moral_pairs, desc="Analyzing moral pairs"):
            moral_acts = self._get_completion_activations(moral_text)
            immoral_acts = self._get_completion_activations(immoral_text)
            
            moral_activations.append(moral_acts)
            immoral_activations.append(immoral_acts)
            
        moral_tensor = torch.stack(moral_activations)
        immoral_tensor = torch.stack(immoral_activations)
        
        return self._analyze_moral_differences(
            moral_tensor, 
            immoral_tensor, 
            significant_diff=significant_diff,
            consistency_threshold=consistency_threshold
        )
    
    def _get_completion_activations(self, text: str) -> torch.Tensor:
        """Get activations at the decision point (last token)."""
        tokens = self.model.to_tokens(text)
        with torch.no_grad():
            _, cache = self.model.run_with_cache(tokens)
            
            activations = []
            for layer_idx in range(self.n_layers):
                mlp_acts = cache['post', layer_idx, 'mlp']
                last_token_acts = mlp_acts[0, -1]
                activations.append(last_token_acts)
                
        return torch.stack(activations)
    
    def _analyze_moral_differences(self, 
                                moral_acts: torch.Tensor, 
                                immoral_acts: torch.Tensor,
                                significant_diff: float = 0.5,
                                consistency_threshold: float = 0.8) -> Dict:
        """
        Analyze which neurons show consistent differences between moral/immoral choices.
        
        Returns dictionary with:
        - moral_neurons: List of (layer, neuron) that consistently activate more for moral choices
        - immoral_neurons: List of (layer, neuron) that consistently activate more for immoral choices
        - layer_importance: Ranking of layers by their role in moral decisions
        - activation_differences: Tensor of mean activation differences
        """
        diff = moral_acts - immoral_acts
        mean_diff = diff.mean(dim=0)
        
        moral_neurons = []
        immoral_neurons = []
        
        for layer in range(self.n_layers):
            for neuron in range(self.n_neurons):
                consistency = (diff[:, layer, neuron] > 0).float().mean()
                if consistency > consistency_threshold:
                    if mean_diff[layer, neuron] > significant_diff:
                        moral_neurons.append((layer, neuron))
                    elif mean_diff[layer, neuron] < -significant_diff:
                        immoral_neurons.append((layer, neuron))
        
        layer_importance = []
        for layer in range(self.n_layers):
            layer_diff = torch.abs(mean_diff[layer]).mean()
            layer_importance.append((layer, layer_diff.item()))
        
        layer_importance.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'moral_neurons': moral_neurons,
            'immoral_neurons': immoral_neurons,
            'layer_importance': layer_importance,
            'activation_differences': mean_diff
        }
        
    def generate_text(self, input_text: str, max_new_tokens: int = 10, temperature: float = 1.0) -> str:
        """Generate text continuation from input text."""
        input_tokens = self.model.to_tokens(input_text)
        input_token_length = input_tokens.shape[1]
        
        generated_tokens = self.model.generate(
            input_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        
        new_tokens = generated_tokens[0][input_token_length:]
        generated_text = self.model.to_string(new_tokens)
        
        return generated_text 