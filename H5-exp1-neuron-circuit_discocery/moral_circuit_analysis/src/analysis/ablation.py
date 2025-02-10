import torch
from typing import List, Tuple, Dict, Optional
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
import numpy as np
from sentence_transformers import SentenceTransformer

class AblationAnalyzer:
    def __init__(self, model: HookedTransformer):
        self.model = model
        self.device = model.cfg.device
        self.n_layers = model.cfg.n_layers
        self.n_neurons = model.cfg.d_mlp
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def ablate_neurons(self, 
                      text: str,
                      neurons: List[Tuple[int, int]],
                      ablation_value: Optional[float] = 0.0,
                      max_new_tokens: Optional[int] = 50,
                      temperature: Optional[float] = 1.0) -> str:
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
                    max_new_tokens=max_new_tokens,
                    temperature=temperature
                )
        finally:
            # Clean up all hooks
            self.model.reset_hooks()
            
        # Return only the newly generated tokens
        original_text = self.model.to_string(tokens[0])
        full_output = self.model.to_string(output[0])
        return full_output[len(original_text):]
    
    def analyze_ablation_impact(self,
                              moral_pairs: List[Tuple[str, str]],
                              neurons: List[Tuple[int, int]],
                              ablation_value: Optional[float] = 0.0,
                              max_new_tokens: Optional[int] = 50,
                              temperature: Optional[float] = 1.0) -> Dict:
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
            'moral_responses': [],     # List of (prompt, original_moral, ablated_moral) tuples
            'immoral_responses': [],   # List of (prompt, original_immoral, ablated_immoral) tuples
            'moral_similarities': [],  # List of (prompt_to_orig, prompt_to_ablated, orig_to_ablated) tuples
            'immoral_similarities': [], # List of (prompt_to_orig, prompt_to_ablated, orig_to_ablated) tuples
            'response_changes': []     # Keep this for backward compatibility
        }
        
        for moral_text, immoral_text in tqdm(moral_pairs, desc="Analyzing ablation impact"):
            # Get original and ablated responses
            orig_moral = self.generate_text(moral_text, max_new_tokens, temperature)
            orig_immoral = self.generate_text(immoral_text, max_new_tokens, temperature)
            ablated_moral = self.ablate_neurons(moral_text, neurons, ablation_value, max_new_tokens, temperature)
            ablated_immoral = self.ablate_neurons(immoral_text, neurons, ablation_value, max_new_tokens, temperature)
            
            # Store responses
            results['moral_responses'].append((moral_text, orig_moral, ablated_moral))
            results['immoral_responses'].append((immoral_text, orig_immoral, ablated_immoral))
            
            # Calculate similarities for moral responses
            moral_similarities = (
                self._compute_moral_agreement(moral_text, orig_moral),      # prompt to original
                self._compute_moral_agreement(moral_text, ablated_moral),   # prompt to ablated
                self._compute_moral_agreement(orig_moral, ablated_moral)    # original to ablated
            )
            results['moral_similarities'].append(moral_similarities)
            
            # Calculate similarities for immoral responses
            immoral_similarities = (
                self._compute_moral_agreement(immoral_text, orig_immoral),    # prompt to original
                self._compute_moral_agreement(immoral_text, ablated_immoral), # prompt to ablated
                self._compute_moral_agreement(orig_immoral, ablated_immoral)  # original to ablated
            )
            results['immoral_similarities'].append(immoral_similarities)
            
            # Store response changes for backward compatibility
            moral_change = 1.0 - moral_similarities[2]    # Convert similarity to change
            immoral_change = 1.0 - immoral_similarities[2]
            results['response_changes'].append((moral_change, immoral_change))
        
        # Compute summary statistics
        results.update(self._compute_ablation_statistics(results))
        
        return results
    
    def _compute_response_change(self, original: str, ablated: str) -> float:
        """
        Compute semantic difference between responses using sentence embeddings.
        
        Args:
            original: Original model response
            ablated: Response with neurons ablated
            
        Returns:
            Float between 0-1 indicating amount of change (1 = completely different)
        """
        # Get sentence embeddings
        emb_orig = self.embedding_model.encode(original, convert_to_tensor=True)
        emb_abl = self.embedding_model.encode(ablated, convert_to_tensor=True)
        
        # Calculate cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            emb_orig.unsqueeze(0),
            emb_abl.unsqueeze(0),
            dim=1
        ).mean().item()
        
        return 1.0 - similarity
    
    def _compute_moral_agreement(self, response1: str, response2: str) -> float:
        """
        Compute semantic similarity between responses using sentence embeddings.
        
        Args:
            response1: First response to compare
            response2: Second response to compare
            
        Returns:
            Float between 0-1 indicating agreement (1 = complete agreement)
        """
        # Get sentence embeddings
        emb1 = self.embedding_model.encode(response1, convert_to_tensor=True)
        emb2 = self.embedding_model.encode(response2, convert_to_tensor=True)
        
        # Calculate cosine similarity directly
        agreement = torch.nn.functional.cosine_similarity(
            emb1.unsqueeze(0),
            emb2.unsqueeze(0),
            dim=1
        ).mean().item()
        
        return agreement
    
    def _compute_ablation_statistics(self, results: Dict) -> Dict:
        """Compute summary statistics for ablation results."""
        stats = {}
        
        # Convert to numpy arrays for easier computation
        moral_sims = np.array(results['moral_similarities'])
        immoral_sims = np.array(results['immoral_similarities'])
        response_changes = np.array(results['response_changes'])
        
        # Moral statistics
        stats.update({
            # Averages
            'avg_moral_prompt_to_orig': np.mean(moral_sims[:, 0]),
            'avg_moral_prompt_to_ablated': np.mean(moral_sims[:, 1]),
            'avg_moral_orig_to_ablated': np.mean(moral_sims[:, 2]),
            # Standard deviations
            'std_moral_prompt_to_orig': np.std(moral_sims[:, 0]),
            'std_moral_prompt_to_ablated': np.std(moral_sims[:, 1]),
            'std_moral_orig_to_ablated': np.std(moral_sims[:, 2]),
            # Min/Max
            'min_moral_prompt_to_orig': np.min(moral_sims[:, 0]),
            'max_moral_prompt_to_orig': np.max(moral_sims[:, 0]),
            'min_moral_prompt_to_ablated': np.min(moral_sims[:, 1]),
            'max_moral_prompt_to_ablated': np.max(moral_sims[:, 1]),
        })
        
        # Immoral statistics
        stats.update({
            # Averages
            'avg_immoral_prompt_to_orig': np.mean(immoral_sims[:, 0]),
            'avg_immoral_prompt_to_ablated': np.mean(immoral_sims[:, 1]),
            'avg_immoral_orig_to_ablated': np.mean(immoral_sims[:, 2]),
            # Standard deviations
            'std_immoral_prompt_to_orig': np.std(immoral_sims[:, 0]),
            'std_immoral_prompt_to_ablated': np.std(immoral_sims[:, 1]),
            'std_immoral_orig_to_ablated': np.std(immoral_sims[:, 2]),
            # Min/Max
            'min_immoral_prompt_to_orig': np.min(immoral_sims[:, 0]),
            'max_immoral_prompt_to_orig': np.max(immoral_sims[:, 0]),
            'min_immoral_prompt_to_ablated': np.min(immoral_sims[:, 1]),
            'max_immoral_prompt_to_ablated': np.max(immoral_sims[:, 1]),
        })
        
        # Backward compatibility statistics
        stats.update({
            'avg_moral_change': np.mean(response_changes[:, 0]),
            'std_moral_change': np.std(response_changes[:, 0]),
            'avg_immoral_change': np.mean(response_changes[:, 1]),
            'std_immoral_change': np.std(response_changes[:, 1]),
        })
        
        # Overall agreement changes
        stats.update({
            'original_agreement': np.mean([moral_sims[:, 0], immoral_sims[:, 0]]),
            'ablated_agreement': np.mean([moral_sims[:, 1], immoral_sims[:, 1]]),
            'agreement_change': np.mean([moral_sims[:, 1], immoral_sims[:, 1]]) - 
                              np.mean([moral_sims[:, 0], immoral_sims[:, 0]])
        })
        
        return stats
    
    def generate_text(self, text: str, max_new_tokens: int = 50, temperature: float = 1.0) -> str:
        """Generate text continuation without including the prompt."""
        tokens = self.model.to_tokens(text)
        with torch.no_grad():
            output = self.model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
        # Return only the newly generated tokens
        original_text = self.model.to_string(tokens[0])
        full_output = self.model.to_string(output[0])
        return full_output[len(original_text):] 