import torch
from typing import List, Tuple, Dict, Optional
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
import numpy as np

class AblationAnalyzer:
    def __init__(self, model: HookedTransformer):
        self.model = model
        self.device = model.cfg.device
        self.n_layers = model.cfg.n_layers
        self.n_neurons = model.cfg.d_mlp
        
    def ablate_neurons(self, 
                      text: str,
                      neurons: List[Tuple[int, int]],
                      ablation_value: Optional[float] = None) -> str:
        """
        Generate text with specified neurons ablated (set to ablation_value).
        
        Args:
            text: Input text
            neurons: List of (layer, neuron) pairs to ablate
            ablation_value: Value to set neurons to (None for zero)
        """
        def ablation_hook(activation: torch.Tensor, hook, neurons=neurons, value=ablation_value):
            for layer, neuron in neurons:
                if hook.layer() == layer:
                    #print(f"Ablating neuron {neuron} in layer {layer}")
                    activation[..., neuron] = value if value is not None else 0.0
            return activation
        
        # Add hooks for each layer that has neurons to ablate
        layers_to_hook = set(layer for layer, _ in neurons)
        
        # Add hooks directly to the model
        for layer in layers_to_hook:
            hook_point = self.model.blocks[layer].mlp.hook_post
            hook_point.add_hook(ablation_hook)
            print(f"Added hook to layer {layer}")
        
        try:
            # Generate text with hooks in place
            tokens = self.model.to_tokens(text)
            with torch.no_grad():
                output = self.model.generate(
                    tokens,
                    max_new_tokens=50,
                    temperature=1.0
                )
        finally:
            # Clean up all hooks
            self.model.reset_hooks()
            
        return self.model.to_string(output[0])
    
    def analyze_ablation_impact(self,
                              moral_pairs: List[Tuple[str, str]],
                              neurons: List[Tuple[int, int]],
                              ablation_value: Optional[float] = None) -> Dict:
        """
        Analyze how ablating specific neurons affects model's moral behavior.
        
        Args:
            moral_pairs: List of (moral_text, immoral_text) pairs
            neurons: List of (layer, neuron) pairs to ablate
            ablation_value: Value to set neurons to (None for zero)
            
        Returns:
            Dictionary with ablation analysis results
        """
        results = {
            'original_responses': [],
            'ablated_responses': [],
            'response_changes': [],
            'moral_agreement_original': [],
            'moral_agreement_ablated': []
        }
        
        for moral_text, immoral_text in tqdm(moral_pairs, desc="Analyzing ablation impact"):
            # Get original responses
            orig_moral = self.generate_text(moral_text)
            orig_immoral = self.generate_text(immoral_text)
            
            # Get responses with ablated neurons
            ablated_moral = self.ablate_neurons(moral_text, neurons, ablation_value)
            ablated_immoral = self.ablate_neurons(immoral_text, neurons, ablation_value)
            
            # Store responses
            results['original_responses'].append((orig_moral, orig_immoral))
            results['ablated_responses'].append((ablated_moral, ablated_immoral))
            
            # Analyze changes
            moral_change = self._compute_response_change(orig_moral, ablated_moral)
            immoral_change = self._compute_response_change(orig_immoral, ablated_immoral)
            results['response_changes'].append((moral_change, immoral_change))
            
            # Analyze moral agreement
            results['moral_agreement_original'].append(
                self._compute_moral_agreement(orig_moral, orig_immoral)
            )
            results['moral_agreement_ablated'].append(
                self._compute_moral_agreement(ablated_moral, ablated_immoral)
            )
        
        # Compute summary statistics
        results.update(self._compute_ablation_statistics(results))
        
        return results
    
    def _compute_response_change(self, original: str, ablated: str) -> float:
        """
        Compute how much the response changed after ablation using cosine similarity.
        
        Args:
            original: Original model response
            ablated: Response with neurons ablated
            
        Returns:
            Float between 0-1 indicating amount of change (1 = completely different)
        """
        # Get logits for both responses
        with torch.no_grad():
            tokens_orig = self.model.to_tokens(original)
            logits_orig = self.model(tokens_orig)
            
            tokens_abl = self.model.to_tokens(ablated) 
            logits_abl = self.model(tokens_abl)
            
            # Ensure tensors have same shape by padding shorter one
            max_len = max(logits_orig.size(1), logits_abl.size(1))
            if logits_orig.size(1) < max_len:
                logits_orig = torch.nn.functional.pad(logits_orig, (0, 0, 0, max_len - logits_orig.size(1)))
            if logits_abl.size(1) < max_len:
                logits_abl = torch.nn.functional.pad(logits_abl, (0, 0, 0, max_len - logits_abl.size(1)))
            
            # Flatten the logits while preserving batch dimension
            similarity = torch.nn.functional.cosine_similarity(
                logits_orig.view(logits_orig.size(0), -1),
                logits_abl.view(logits_abl.size(0), -1),
                dim=1
            ).mean().item()
        
        return 1.0 - similarity
    
    def _compute_moral_agreement(self, response1: str, response2: str) -> float:
        """
        Compute degree of moral agreement between two responses.
        
        Args:
            response1: First response to compare
            response2: Second response to compare
            
        Returns:
            Float between 0-1 indicating agreement (1 = complete agreement)
        """
        tokens1 = self.model.to_tokens(response1)
        tokens2 = self.model.to_tokens(response2)
        
        with torch.no_grad():
            logits1 = self.model(tokens1)
            logits2 = self.model(tokens2)
            
            # Ensure tensors have same shape by padding shorter one
            max_len = max(logits1.size(1), logits2.size(1))
            if logits1.size(1) < max_len:
                logits1 = torch.nn.functional.pad(logits1, (0, 0, 0, max_len - logits1.size(1)))
            if logits2.size(1) < max_len:
                logits2 = torch.nn.functional.pad(logits2, (0, 0, 0, max_len - logits2.size(1)))
            
            # Flatten the logits while preserving batch dimension
            agreement = torch.nn.functional.cosine_similarity(
                logits1.view(logits1.size(0), -1),
                logits2.view(logits2.size(0), -1),
                dim=1
            ).mean().item()
        
        return agreement
    
    def _compute_ablation_statistics(self, results: Dict) -> Dict:
        """Compute summary statistics for ablation results."""
        stats = {}
        
        # Average response changes
        moral_changes = [x[0] for x in results['response_changes']]
        immoral_changes = [x[1] for x in results['response_changes']]
        
        stats['avg_moral_change'] = np.mean(moral_changes)
        stats['std_moral_change'] = np.std(moral_changes)
        stats['avg_immoral_change'] = np.mean(immoral_changes)
        stats['std_immoral_change'] = np.std(immoral_changes)
        
        # Agreement changes
        orig_agreement = np.mean(results['moral_agreement_original'])
        ablated_agreement = np.mean(results['moral_agreement_ablated'])
        
        stats['original_agreement'] = orig_agreement
        stats['ablated_agreement'] = ablated_agreement
        stats['agreement_change'] = ablated_agreement - orig_agreement
        
        return stats
    
    def generate_text(self, text: str, max_new_tokens: int = 50) -> str:
        """Generate text continuation."""
        tokens = self.model.to_tokens(text)
        with torch.no_grad():
            output = self.model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                temperature=1.0
            )
        return self.model.to_string(output[0]) 