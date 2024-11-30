import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict

class TransformerAnalyzer:
    def __init__(self, model_name: str = 'gpt2'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # Get model dimensions
        self.n_layers = self.model.config.n_layer
        self.n_heads = self.model.config.n_head
        self.n_neurons = self.model.config.n_inner
        self.hidden_size = self.model.config.hidden_size
        
        print(f"Model architecture:")
        print(f"- {self.n_layers} layers")
        print(f"- {self.n_heads} attention heads per layer")
        print(f"- {self.n_neurons} neurons in MLP")
        print(f"- {self.hidden_size} hidden size")

    def get_all_component_activations(self, text: str) -> Dict:
        """
        Get activations for all major components:
        - Residual stream
        - Attention (Q, K, V matrices and attention patterns)
        - MLP activations
        """
        tokens = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = tokens.input_ids[0]
        token_texts = [self.tokenizer.decode(token_id) for token_id in input_ids]
        
        activations = {
            'residual': [],
            'attention': [],
            'mlp': [],
            'tokens': token_texts
        }
        
        # Store intermediate activations
        def residual_hook(layer_idx):
            def hook(module, input, output):
                # output[0] contains the layer output tensor
                residual_output = output[0]  # Get first element of tuple
                activations['residual'].append({
                    'layer': layer_idx,
                    'values': residual_output.detach().cpu().numpy()
                })
            return hook
        
        def attention_hook(layer_idx):
            def hook(module, input, output):
                # For GPT-2, output is a tuple where:
                # output[0] is the attention output
                # output[1] is a tuple containing attention weights
                attn_weights = output[1][0]  # Get attention weights from tuple
                activations['attention'].append({
                    'layer': layer_idx,
                    'attention_weights': attn_weights.detach().cpu().numpy()
                })
            return hook
        
        def mlp_hook(layer_idx):
            def hook(module, input, output):
                # Get MLP intermediate activations
                intermediate = module.act(module.c_fc(input[0]))
                activations['mlp'].append({
                    'layer': layer_idx,
                    'values': intermediate.detach().cpu().numpy()
                })
            return hook
        
        # Register hooks for all components
        hooks = []
        for layer_idx in range(self.n_layers):
            # Residual stream hooks
            hooks.append(
                self.model.transformer.h[layer_idx].register_forward_hook(
                    residual_hook(layer_idx)
                )
            )
            
            # Attention hooks
            hooks.append(
                self.model.transformer.h[layer_idx].attn.register_forward_hook(
                    attention_hook(layer_idx)
                )
            )
            
            # MLP hooks
            hooks.append(
                self.model.transformer.h[layer_idx].mlp.register_forward_hook(
                    mlp_hook(layer_idx)
                )
            )
        
        # Forward pass
        with torch.no_grad():
            self.model(**tokens)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return activations

    def analyze_attention_patterns(self, activations: Dict, layer_idx: int) -> Dict:
        """
        Analyze attention patterns in a specific layer.
        """
        attention_data = activations['attention'][layer_idx]
        attention_weights = attention_data['attention_weights']
        
        # Analyze attention patterns per head
        head_patterns = []
        for head_idx in range(self.n_heads):
            head_attention = attention_weights[0, head_idx, :, :]  # Added index for batch dimension
            pattern = {
                'head_idx': head_idx,
                'attention_scores': head_attention
            }
            head_patterns.append(pattern)
        
        return head_patterns

    def analyze_residual_stream(self, activations: Dict, layer_idx: int) -> Dict:
        """
        Analyze the residual stream at a specific layer.
        """
        residual_data = activations['residual'][layer_idx]
        values = residual_data['values']
        
        # Analyze how information flows through the residual stream
        analysis = {
            'mean_activation': np.mean(values),
            'std_activation': np.std(values),
            'max_activation': np.max(values),
            'min_activation': np.min(values)
        }
        
        return analysis

    def find_component_interactions(self, text: str, layer_idx: int) -> Dict:
        """
        Analyze interactions between different components at a specific layer.
        """
        activations = self.get_all_component_activations(text)
        
        # Get component activations for the specified layer
        residual = self.analyze_residual_stream(activations, layer_idx)
        attention = self.analyze_attention_patterns(activations, layer_idx)
        mlp = activations['mlp'][layer_idx]
        
        # Analyze interactions
        interactions = {
            'layer': layer_idx,
            'residual_stats': residual,
            'attention_patterns': attention,
            'mlp_stats': {
                'mean_activation': np.mean(mlp['values']),
                'max_activation': np.max(mlp['values'])
            },
            'tokens': activations['tokens']
        }
        
        return interactions

# Example usage
if __name__ == "__main__":
    analyzer = TransformerAnalyzer('gpt2')
    
    # Example text
    text = "The quick brown fox jumps over the lazy dog."
    
    # Analyze specific layer
    layer_idx = 5
    interactions = analyzer.find_component_interactions(text, layer_idx)
    
    print(f"\nAnalysis for layer {layer_idx}:")
    print("\nResidual Stream Statistics:")
    print(interactions['residual_stats'])
    
    print("\nAttention Patterns:")
    print(f"Number of heads analyzed: {len(interactions['attention_patterns'])}")
    
    print("\nMLP Statistics:")
    print(interactions['mlp_stats'])