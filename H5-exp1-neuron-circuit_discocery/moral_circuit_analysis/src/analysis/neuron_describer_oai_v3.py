import torch
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from tqdm.auto import tqdm
import numpy as np
from collections import defaultdict
import openai
from transformer_lens import HookedTransformer

@dataclass
class NeuronActivation:
    """Stores information about a specific neuron activation."""
    text: str
    token: str
    token_index: int
    activation: float
    context_before: str
    context_after: str

@dataclass
class ExplanationResult:
    """Stores the results of neuron explanation generation and evaluation."""
    neuron_id: Tuple[int, int]  # (layer, neuron_idx)
    explanation: str
    score: float
    top_activations: List[NeuronActivation]
    analysis: Dict
    revision: Optional[str] = None
    revision_score: Optional[float] = None

class ImprovedNeuronEvaluator:
    def __init__(
        self,
        model: HookedTransformer,
        llm_name: str = "gpt-4",
        num_top_sequences: int = 5,
        batch_size: int = 32,
        activation_quantile: float = 0.9996,
        api_key: Optional[str] = None
    ):
        """
        Initialize the neuron evaluator.
        
        Args:
            model: The transformer model to analyze
            llm_name: Name of the LLM to use for explanation generation
            num_top_sequences: Number of top activating sequences to use (OpenAI found 5 optimal)
            batch_size: Batch size for processing
            activation_quantile: Quantile threshold for top activations (OpenAI used 0.9996)
            api_key: OpenAI API key
        """
        self.model = model
        self.llm_name = llm_name
        self.num_top_sequences = num_top_sequences
        self.batch_size = batch_size
        self.activation_quantile = activation_quantile
        self.client = openai.OpenAI(api_key=api_key)
        
    def get_top_activating_sequences(
        self, 
        layer: int,
        neuron_idx: int,
        texts: List[str],
        context_window: int = 5
    ) -> List[NeuronActivation]:
        """Get top activating sequences for a neuron."""
        all_activations = []
        
        for i in tqdm(range(0, len(texts), self.batch_size)):
            batch_texts = texts[i:i + self.batch_size]
            batch_tokens = [self.model.to_tokens(text) for text in batch_texts]
            
            for tokens, text in zip(batch_tokens, batch_texts):
                with torch.no_grad():
                    _, cache = self.model.run_with_cache(tokens)
                    activations = cache['post', layer, 'mlp'][0, :, neuron_idx]
                    
                    # Find activation above threshold
                    threshold = torch.quantile(activations, self.activation_quantile)
                    significant_indices = torch.where(activations > threshold)[0]
                    
                    for idx in significant_indices:
                        # Get context around activation
                        start_idx = max(0, idx - context_window)
                        end_idx = min(len(tokens[0]), idx + context_window + 1)
                        
                        context_tokens = tokens[0][start_idx:end_idx]
                        target_token = tokens[0][idx]
                        
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
        return all_activations[:self.num_top_sequences]

    def generate_explanation(
        self,
        layer: int,
        neuron_idx: int,
        activations: List[NeuronActivation]
    ) -> str:
        """Generate explanation using GPT-4."""
        prompt = self._create_explanation_prompt(layer, neuron_idx, activations)
        
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
            print(f"Error generating explanation: {str(e)}")
            return ""

    def simulate_activations(
        self,
        explanation: str,
        texts: List[str],
        method: str = "all_at_once"
    ) -> torch.Tensor:
        """Simulate neuron activations based on explanation."""
        if method == "all_at_once":
            return self._parallel_simulation(explanation, texts)
        return self._sequential_simulation(explanation, texts)

    def _parallel_simulation(
        self,
        explanation: str,
        texts: List[str]
    ) -> torch.Tensor:
        """Simulate all activations in parallel."""
        prompt = self._create_parallel_simulation_prompt(explanation, texts)
        
        try:
            response = self.client.chat.completions.create(
                model=self.llm_name,
                messages=[
                    {"role": "system", "content": "You are simulating neuron activations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=len(texts) * 10
            )
            
            # Parse response into activations
            return self._parse_parallel_simulation(response.choices[0].message.content)
        except Exception as e:
            print(f"Error in parallel simulation: {str(e)}")
            return torch.zeros(len(texts))

    def compute_correlation_score(
        self,
        real_activations: torch.Tensor,
        simulated_activations: torch.Tensor
    ) -> float:
        """Compute correlation score between real and simulated activations."""
        # Normalize activations
        real_mean = real_activations.mean()
        real_std = real_activations.std()
        sim_mean = simulated_activations.mean()
        sim_std = simulated_activations.std()
        
        real_norm = (real_activations - real_mean) / real_std
        sim_norm = (simulated_activations - sim_mean) / sim_std
        
        # Compute correlation
        correlation = torch.corrcoef(
            torch.stack([real_norm, sim_norm])
        )[0, 1].item()
        
        return correlation

    def generate_test_cases(self, explanation: str, n: int = 10) -> List[str]:
        """Generate test cases to challenge the explanation."""
        prompt = f"""Based on this explanation of a neuron's behavior:
        "{explanation}"
        
        Generate {n} diverse text examples that should activate this neuron according to the explanation.
        Make sure to include edge cases and variations."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.llm_name,
                messages=[
                    {"role": "system", "content": "Generate diverse test cases for neuron behavior."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            # Parse response into list of test cases
            return self._parse_test_cases(response.choices[0].message.content)
        except Exception as e:
            print(f"Error generating test cases: {str(e)}")
            return []

    def revise_explanation(
        self,
        original_explanation: str,
        test_cases: List[str],
        activations: List[NeuronActivation]
    ) -> str:
        """Revise explanation based on test cases and their activations."""
        prompt = self._create_revision_prompt(
            original_explanation,
            test_cases,
            activations
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.llm_name,
                messages=[
                    {"role": "system", "content": "Revise neuron behavior explanation based on new evidence."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error revising explanation: {str(e)}")
            return original_explanation

    def evaluate_neuron(
        self,
        layer: int,
        neuron_idx: int,
        texts: List[str],
        random_texts: Optional[List[str]] = None,
        revise: bool = True
    ) -> ExplanationResult:
        """Complete evaluation of a single neuron with optional revision."""
        # Get top activating sequences
        top_activations = self.get_top_activating_sequences(layer, neuron_idx, texts)
        
        # Generate initial explanation
        explanation = self.generate_explanation(layer, neuron_idx, top_activations)
        
        # Simulate activations and compute score
        real_activations = self._get_real_activations(layer, neuron_idx, texts)
        simulated_activations = self.simulate_activations(explanation, texts)
        score = self.compute_correlation_score(real_activations, simulated_activations)
        
        # Generate and evaluate test cases if revision is requested
        revision = None
        revision_score = None
        
        if revise:
            test_cases = self.generate_test_cases(explanation)
            test_activations = self._get_real_activations(layer, neuron_idx, test_cases)
            revision = self.revise_explanation(explanation, test_cases, test_activations)
            
            # Score revision
            sim_revised = self.simulate_activations(revision, texts)
            revision_score = self.compute_correlation_score(real_activations, sim_revised)
        
        # Analyze patterns
        analysis = self._analyze_activation_patterns(top_activations)
        
        return ExplanationResult(
            neuron_id=(layer, neuron_idx),
            explanation=explanation,
            score=score,
            top_activations=top_activations,
            analysis=analysis,
            revision=revision,
            revision_score=revision_score
        )

    def _analyze_activation_patterns(self, activations: List[NeuronActivation]) -> Dict:
        """Analyze patterns in neuron activations."""
        analysis = {
            'total_activations': len(activations),
            'avg_activation': np.mean([a.activation for a in activations]),
            'max_activation': max(a.activation for a in activations),
            'token_frequency': defaultdict(int),
            'position_distribution': defaultdict(int)
        }
        
        # Analyze token patterns
        for act in activations:
            analysis['token_frequency'][act.token] += 1
            analysis['position_distribution'][act.token_index] += 1
        
        # Get top tokens
        analysis['top_tokens'] = sorted(
            analysis['token_frequency'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return analysis

    def _get_real_activations(self, layer: int, neuron_idx: int, texts: List[str]) -> torch.Tensor:
        """Get actual neuron activations for a set of texts."""
        all_activations = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_tokens = [self.model.to_tokens(text) for text in batch_texts]
            
            with torch.no_grad():
                for tokens in batch_tokens:
                    _, cache = self.model.run_with_cache(tokens)
                    # Get max activation for each sequence
                    activations = cache['post', layer, 'mlp'][0, :, neuron_idx]
                    all_activations.append(activations.max().item())
        
        return torch.tensor(all_activations)

    def _create_explanation_prompt(self, layer: int, neuron_idx: int, activations: List[NeuronActivation]) -> str:
        """Create prompt for explanation generation."""
        prompt = f"""Analyze the activation pattern of a neuron in layer {layer} (index {neuron_idx}).

Here are the top activating sequences:

"""
        for act in activations:
            prompt += f"Text: {act.text}\n"
            prompt += f"Token: {act.token} (activation: {act.activation:.3f})\n"
            prompt += f"Context before: {act.context_before}\n"
            prompt += f"Context after: {act.context_after}\n\n"
            
        prompt += "Based on these patterns, provide a concise description of what concept or pattern this neuron detects."
        return prompt

    def _create_parallel_simulation_prompt(self, explanation: str, texts: List[str]) -> str:
        """Create prompt for parallel activation simulation."""
        prompt = f"""Based on this explanation of a neuron's behavior:
        "{explanation}"
        
        For each of the following texts, predict the neuron's activation on a scale of 0-10:
        
        """
        for text in texts:
            prompt += f"Text: {text}\nActivation: "
        
        return prompt

    def _create_revision_prompt(
        self,
        original_explanation: str,
        test_cases: List[str],
        activations: List[NeuronActivation]
    ) -> str:
        """Create prompt for explanation revision."""
        prompt = f"""Original explanation of neuron behavior:
        "{original_explanation}"
        
        New evidence from test cases:
        """
        
        for test, act in zip(test_cases, activations):
            prompt += f"\nText: {test}\n"
            prompt += f"Actual activation: {act.activation:.3f}\n"
        
        prompt += "\nBased on this new evidence, please provide a revised explanation of the neuron's behavior."
        return prompt

    def _parse_parallel_simulation(self, response: str) -> torch.Tensor:
        """Parse the response from parallel simulation into activation values."""
        try:
            # Extract numbers after "Activation:" from each line
            activations = []
            for line in response.split('\n'):
                if 'Activation:' in line:
                    value_str = line.split('Activation:')[1].strip()
                    # Convert the 0-10 scale to a float
                    try:
                        value = float(value_str)
                        # Normalize to match the scale of real activations
                        activations.append(value)
                    except ValueError:
                        activations.append(0.0)  # Default value if parsing fails
            
            return torch.tensor(activations)
        except Exception as e:
            print(f"Error parsing simulation response: {str(e)}")
            return torch.zeros(1)  # Return zero tensor as fallback

    def _parse_test_cases(self, response: str) -> List[str]:
        """Parse the response containing generated test cases."""
        test_cases = []
        current_case = []
        
        # Split response into lines and process
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line:
                # Remove common prefixes that might be in the response
                line = line.lstrip('0123456789.- )"')
                if line.lower().startswith(('example', 'test case', 'case')):
                    if current_case:
                        test_cases.append(' '.join(current_case))
                        current_case = []
                    line = ':'.join(line.split(':')[1:]) if ':' in line else line
                current_case.append(line.strip())
        
        # Add the last case if exists
        if current_case:
            test_cases.append(' '.join(current_case))
        
        # Clean up and filter empty cases
        test_cases = [case.strip() for case in test_cases if case.strip()]
        return test_cases