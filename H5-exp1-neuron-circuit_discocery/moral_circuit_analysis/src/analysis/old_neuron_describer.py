import openai
from typing import Dict, List, Tuple
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
from .moral_analyzer import MoralBehaviorAnalyzer

class MoralNeuronDescriber(MoralBehaviorAnalyzer):
    def __init__(self, model: HookedTransformer, llm_name: str = "gpt-4"):
        super().__init__(model)
        self.llm_name = llm_name
        self.client = openai.OpenAI()
        
    def describe_moral_neurons(self, results: Dict) -> Dict[Tuple[int, int], str]:
        """Generate descriptions for moral/immoral neurons using LLM."""
        descriptions = {}
        
        all_neurons = results['moral_neurons'] + results['immoral_neurons']
        
        for layer, neuron in tqdm(all_neurons, desc="Describing neurons"):
            exemplars = self._get_neuron_exemplars(layer, neuron)
            
            description = self._generate_neuron_description(
                layer, 
                neuron,
                exemplars,
                is_moral=(layer, neuron) in results['moral_neurons']
            )
            
            descriptions[(layer, neuron)] = description
            
        return descriptions
    
    def _get_neuron_exemplars(self, layer: int, neuron: int, num_exemplars: int = 5) -> List[Dict]:
        """Get top activating examples for a neuron."""
        exemplars = []
        
        tokens = self.model.to_tokens(self.validation_texts)
        with torch.no_grad():
            _, cache = self.model.run_with_cache(tokens)
            activations = cache['post', layer, 'mlp'][..., neuron]
            
            top_indices = torch.topk(activations.flatten(), num_exemplars).indices
            batch_indices = top_indices // activations.size(1)
            token_indices = top_indices % activations.size(1)
            
            for batch_idx, token_idx in zip(batch_indices, token_indices):
                text = self.validation_texts[batch_idx]
                activation = activations[batch_idx, token_idx].item()
                token = self.model.to_string(tokens[batch_idx, token_idx])
                
                exemplars.append({
                    'text': text,
                    'token': token,
                    'activation': activation
                })
                
        return exemplars
    
    def _generate_neuron_description(self, layer: int, neuron: int, 
                                   exemplars: List[Dict], is_moral: bool) -> str:
        """Generate natural language description of neuron behavior using LLM."""
        prompt = f"""Describe the behavior of a neuron in layer {layer} of a language model that responds to moral/ethical concepts.

The neuron appears to be {'positively' if is_moral else 'negatively'} associated with moral behavior.

Here are some example texts and tokens where this neuron strongly activates:

"""
        for ex in exemplars:
            prompt += f"Text: {ex['text']}\n"
            prompt += f"Token: {ex['token']}\n"
            prompt += f"Activation: {ex['activation']:.3f}\n\n"
            
        prompt += "\nBased on these examples, provide a concise description of what concept or pattern this neuron appears to detect:"
        
        try:
            response = self.client.chat.completions.create(
                model=self.llm_name,
                messages=[
                    {"role": "system", "content": "You are an expert at interpreting neural network behavior."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            import traceback
            error_msg = f"Error generating description:\nType: {type(e).__name__}\nDetails: {str(e)}\nTraceback:\n{traceback.format_exc()}"
            print(error_msg)
            return f"Description generation failed: {type(e).__name__} - {str(e)}" 