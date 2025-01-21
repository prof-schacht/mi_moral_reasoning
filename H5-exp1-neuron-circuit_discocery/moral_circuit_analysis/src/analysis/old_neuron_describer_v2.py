import torch
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from tqdm.auto import tqdm
from collections import defaultdict
import openai
from transformer_lens import HookedTransformer

# Usage
# # Initialize the model
# from transformer_lens import HookedTransformer
# model = HookedTransformer.from_pretrained("gpt2-small")

# # Initialize the evaluator
# evaluator = NeuronEvaluator(model=model)

# # Your input texts to analyze
# texts = [
#     "Example text 1",
#     "Example text 2",
#     # ... more texts ...
# ]

# # Evaluate a specific neuron
# result = evaluator.evaluate_neuron(
#     layer=5,           # layer to analyze
#     neuron_idx=123,    # neuron index to analyze
#     texts=texts        # texts to analyze
# )

# # Print results
# print(f"Neuron Description: {result['description']}")
# print(f"\nTop activating tokens: {result['analysis']['top_tokens']}")
# print(f"Average activation: {result['analysis']['avg_activation']:.3f}")

@dataclass 
class NeuronActivation:
    text: str
    token: str
    token_index: int
    activation: float
    context_before: str
    context_after: str

class NeuronEvaluator:
    def __init__(
        self,
        model: HookedTransformer,
        llm_name: str = "gpt-4o",
        top_k: int = 50,
        activation_threshold: float = 0.5,
        batch_size: int = 32,
        api_key: str = None
    ):
        self.model = model
        self.llm_name = llm_name
        self.top_k = top_k
        self.activation_threshold = activation_threshold
        self.batch_size = batch_size
        self.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)

    def get_neuron_activations(
        self, 
        layer: int,
        neuron_idx: int,
        texts: List[str],
        context_window: int = 5
    ) -> List[NeuronActivation]:
        """Get detailed activation records for a specific neuron."""
        all_activations = []
        
        for i in tqdm(range(0, len(texts), self.batch_size)):
            batch_texts = texts[i:i + self.batch_size]
            batch_tokens = [self.model.to_tokens(text) for text in batch_texts]
            
            # Process each text in batch
            for tokens, text in zip(batch_tokens, batch_texts):
                with torch.no_grad():
                    _, cache = self.model.run_with_cache(tokens)
                    activations = cache['post', layer, 'mlp'][0, :, neuron_idx]
                    
                    # Find significant activations
                    significant_indices = torch.where(activations > self.activation_threshold)[0]
                    
                    for idx in significant_indices:
                        # Get context around activation
                        start_idx = max(0, idx - context_window)
                        end_idx = min(len(tokens[0]), idx + context_window + 1)
                        
                        context_tokens = tokens[0][start_idx:end_idx]
                        target_token = tokens[0][idx]
                        
                        # Convert tokens to strings
                        context_before = self.model.to_string(context_tokens[:context_window])
                        target_str = self.model.to_string(target_token)
                        context_after = self.model.to_string(context_tokens[context_window+1:])
                        
                        all_activations.append(NeuronActivation(
                            text=text,
                            token=target_str,
                            token_index=idx.item(),
                            activation=activations[idx].item(),
                            context_before=context_before,
                            context_after=context_after
                        ))

        # Sort by activation strength and return top-k
        all_activations.sort(key=lambda x: x.activation, reverse=True)
        return all_activations[:self.top_k]

    def analyze_activation_patterns(
        self,
        activations: List[NeuronActivation]
    ) -> Dict:
        """Analyze patterns in neuron activations."""
        analysis = {
            'total_activations': len(activations),
            'avg_activation': sum(a.activation for a in activations) / len(activations),
            'max_activation': max(a.activation for a in activations),
            'token_frequency': defaultdict(int),
            'context_patterns': [],
            'position_distribution': defaultdict(int)
        }
        
        # Analyze token patterns
        for act in activations:
            analysis['token_frequency'][act.token] += 1
            analysis['position_distribution'][act.token_index] += 1
        
        # Sort and get top tokens
        analysis['top_tokens'] = sorted(
            analysis['token_frequency'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        return analysis

    def generate_neuron_description(
        self,
        layer: int,
        neuron_idx: int,
        activations: List[NeuronActivation],
        analysis: Dict
    ) -> str:
        """Generate detailed description of neuron behavior."""
        prompt = (
            f"Analyze the following neuron activation patterns:\n"
            f"- Maximum activation: {analysis['max_activation']:.3f}\n"
            f"Top activating tokens and their frequencies:\n"
            f"{chr(10).join(f'- {token}: {freq} times' for token, freq in analysis['top_tokens'])}\n"
            f"Example activating contexts (top 5):\n"
            f"{chr(10).join(f'Text: {act.text}{chr(10)}Token: {act.token} (activation: {act.activation:.3f}){chr(10)}Context before: {act.context_before}{chr(10)}Context after: {act.context_after}{chr(10)}' for act in activations[:5])}\n"
            f"Based on these patterns, provide:\n"
            f"1. A concise description of what concept or pattern this neuron detects\n"
            f"2. Any notable contextual patterns\n"
            f"3. Possible semantic or syntactic role of this neuron"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.llm_name,
                messages=[
                    {"role": "system", "content": "You are an expert at interpreting neural network behavior."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            import traceback
            error_msg = f"Error generating description:\nType: {type(e).__name__}\nDetails: {str(e)}\nTraceback:\n{traceback.format_exc()}"
            print(error_msg)
            return f"Description generation failed: {type(e).__name__} - {str(e)}"

    def evaluate_neuron(
        self,
        layer: int,
        neuron_idx: int,
        texts: List[str]
    ) -> Dict:
        """Complete evaluation of a single neuron."""
        # Get activations
        activations = self.get_neuron_activations(layer, neuron_idx, texts)
        
        # Analyze patterns
        analysis = self.analyze_activation_patterns(activations)
        
        # Generate description
        description = self.generate_neuron_description(layer, neuron_idx, activations, analysis)
        
        return {
            'neuron_id': (layer, neuron_idx),
            'activations': activations,
            'analysis': analysis,
            'description': description
        }