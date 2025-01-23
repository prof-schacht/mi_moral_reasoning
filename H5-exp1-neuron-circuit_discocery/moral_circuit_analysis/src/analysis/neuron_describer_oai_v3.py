import torch
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from tqdm.auto import tqdm
import numpy as np
from collections import defaultdict
import openai
from transformer_lens import HookedTransformer
import os
from datetime import datetime

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

# Key improvements to add in ImprovedNeuronEvaluator class:

class ImprovedNeuronEvaluator:
    def __init__(
        self,
        model: HookedTransformer,
        llm_name: str = "gpt-4",
        num_top_sequences: int = 5,
        batch_size: int = 32,
        activation_quantile: float = 0.9996,
        api_key: Optional[str] = None,
        log_dir: str = "logs/prompts",
        dimension: str = None
    ):
        """Initialize the neuron evaluator.
        
        Args:
            model: The transformer model to analyze
            llm_name: Name of the LLM to use for explanation generation
            num_top_sequences: Number of top activating sequences to use
            batch_size: Batch size for processing
            activation_quantile: Quantile threshold for top activations
            api_key: OpenAI API key
            log_dir: Directory to store prompt logs
        """
        self.model = model
        self.llm_name = llm_name
        self.num_top_sequences = num_top_sequences
        self.batch_size = batch_size
        self.activation_quantile = activation_quantile
        self.client = openai.OpenAI(api_key=api_key)
        self.log_dir = log_dir
        self.dimension = dimension
        # Create log directory if it doesn't exist
        
        os.makedirs(os.path.join(self.log_dir, self.dimension), exist_ok=True)
        
        # Add tracking of max activation per neuron
        self.neuron_max_activations = {}
        
        # Add API call and token tracking
        self.api_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0

    def _log_prompt(self, prompt: str, layer: int, neuron_idx: int, kind: str) -> None:
        """Log a prompt to a file.
        
        Args:
            prompt: The prompt to log
            layer: Layer number
            neuron_idx: Neuron index
            kind: Type of prompt (e.g., 'explanation', 'simulation', 'test-cases', 'revision')
        """
        folder_model = self.model.cfg.model_name.replace('/', '-').replace('.', '-')
        folder_dimension = self.dimension
        folder_neuron = f"L{layer}-N{neuron_idx}"
        os.makedirs(os.path.join(self.log_dir, folder_model, folder_dimension, folder_neuron), exist_ok=True)
        
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d")
            filename = f"{timestamp}_{folder_model}_L{layer}-N{neuron_idx}_{kind}_Prompt.txt"
            filepath = os.path.join(self.log_dir, folder_model, folder_dimension, folder_neuron, filename)
            
            with open(filepath, 'w') as f:
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Model: {self.model.cfg.model_name}\n")
                f.write(f"Layer: {layer}, Neuron: {neuron_idx}\n")
                f.write(f"Prompt Type: {kind}\n")
                f.write("\n=== PROMPT ===\n\n")
                f.write(prompt)
        except Exception as e:
            print(f"Error logging prompt: {str(e)}")
            
    def _log_response(self, response: str, layer: int, neuron_idx: int, kind: str) -> None:
        """Log a response to a file."""
        folder_model = self.model.cfg.model_name.replace('/', '-').replace('.', '-')
        folder_dimension = self.dimension
        folder_neuron = f"L{layer}-N{neuron_idx}"
        os.makedirs(os.path.join(self.log_dir, folder_model, folder_dimension, folder_neuron), exist_ok=True)
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d")
            filename = f"{timestamp}_{folder_model}_L{layer}-N{neuron_idx}_{kind}_Response.txt"
            filepath = os.path.join(self.log_dir, folder_model, folder_dimension, folder_neuron, filename)
            with open(filepath, "w") as f:
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Model: {self.model.cfg.model_name}\n")
                f.write(f"Layer: {layer}, Neuron: {neuron_idx}\n")
                f.write(f"Prompt Type: {kind}\n")
                f.write("\n=== RESPONSE ===\n\n")
                f.write(response)
        except Exception as e:
            print(f"Error logging response: {str(e)}")

    def _track_usage(self, response) -> None:
        """Track API usage from a response."""
        self.api_calls += 1
        usage = response.usage
        if usage:
            self.total_prompt_tokens += usage.prompt_tokens
            self.total_completion_tokens += usage.completion_tokens
            
            
    def get_usage_stats(self) -> Dict:
        """Get current API usage statistics."""
        return {
            'api_calls': self.api_calls,
            'prompt_tokens': self.total_prompt_tokens,
            'completion_tokens': self.total_completion_tokens,
            'total_tokens': self.total_prompt_tokens + self.total_completion_tokens,
        }

    def reset_usage_stats(self) -> None:
        """Reset all usage statistics."""
        self.api_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0

    def _normalize_activations(self, activations: torch.Tensor, neuron_id: Tuple[int, int]) -> torch.Tensor:
        """Normalize activations to 0-10 scale as per paper methodology."""
        # Get or calculate max activation for this neuron
        if neuron_id not in self.neuron_max_activations:
            self.neuron_max_activations[neuron_id] = torch.max(activations).item()
        max_activation = self.neuron_max_activations[neuron_id]
        
        # Normalize to 0-10 scale
        normalized = torch.clamp(activations / max_activation * 10, min=0, max=10)
        # Discretize to integers as per paper
        return torch.round(normalized)

    def get_top_activating_sequences(
        self, 
        layer: int,
        neuron_idx: int,
        texts: List[str],
        context_window: int = 5
    ) -> List[NeuronActivation]:
        """Modified to handle sparse activations as per paper."""
        all_activations = []
        neuron_id = (layer, neuron_idx)
        
        for i in tqdm(range(0, len(texts), self.batch_size)):
            batch_texts = texts[i:i + self.batch_size]
            batch_tokens = [self.model.to_tokens(text) for text in batch_texts]
            
            for tokens, text in zip(batch_tokens, batch_texts):
                with torch.no_grad():
                    _, cache = self.model.run_with_cache(tokens)
                    activations = cache['post', layer, 'mlp'][0, :, neuron_idx]
                    
                    # Normalize activations as per paper
                    normalized_activations = self._normalize_activations(activations, neuron_id)
                    
                    # Check for sparsity (<20% non-zero as per paper)
                    non_zero_ratio = (normalized_activations != 0).float().mean()
                    is_sparse = non_zero_ratio < 0.2
                    
                    # Find activation above threshold
                    threshold = torch.quantile(activations, self.activation_quantile)  # Use original activations
                    significant_indices = torch.where(activations > threshold)[0]  # Use original activations
                    
                    for idx in significant_indices:
                        activation_info = self._create_activation_info(
                            tokens, text, idx, activations[idx],  # Use original activations here
                            context_window
                        )
                        
                        # Check for duplicates before adding
                        if activation_info not in all_activations:
                            all_activations.append(activation_info)
                        
                        # For sparse activations, repeat non-zero activations as per paper
                        if is_sparse and normalized_activations[idx] != 0:
                            if activation_info not in all_activations:
                                all_activations.append(activation_info)
        
        # Sort by original activation strength and return top-k
        all_activations.sort(key=lambda x: x.activation, reverse=True)  # Sort by original activations
        return all_activations[:self.num_top_sequences]

    def _create_activation_info(
        self, 
        tokens, 
        text: str, 
        idx: int, 
        activation: float,
        context_window: int
    ) -> NeuronActivation:
        """Helper to create activation info with proper context."""
        start_idx = max(0, idx - context_window)
        end_idx = min(len(tokens[0]), idx + context_window + 1)
        
        context_tokens = tokens[0][start_idx:end_idx]
        target_token = tokens[0][idx]
        
        context_before = self.model.to_string(context_tokens[:context_window])
        target_str = self.model.to_string(target_token)
        context_after = self.model.to_string(context_tokens[context_window+1:])
        
        return NeuronActivation(
            text=text,
            token=target_str,
            token_index=idx.item(),
            activation=activation.item(),
            context_before=context_before,
            context_after=context_after
        )

    def simulate_activations(
        self,
        explanation: str,
        texts: List[str],
        layer: int = None,
        neuron_idx: int = None,
        method: str = "all_at_once"
    ) -> torch.Tensor:
        """Modified to ensure proper scaling and handling of sparse activations."""
        simulated = self._parallel_simulation(explanation, texts, layer, neuron_idx) if method == "all_at_once" \
                   else self._sequential_simulation(explanation, texts, layer, neuron_idx)
        
        # Ensure simulated activations are properly scaled
        simulated = torch.clamp(simulated, min=0, max=10)
        simulated = torch.round(simulated)  # Discretize to integers
        
        # Check for sparsity
        non_zero_ratio = (simulated != 0).float().mean()
        if non_zero_ratio < 0.2:
            # Handle sparse case - could adjust simulation strategy
            print(f"Warning: Sparse activations detected ({non_zero_ratio:.3f})")
        
        return simulated

    def evaluate_neuron(
        self,
        layer: int,
        neuron_idx: int,
        texts: List[str],
        random_texts: Optional[List[str]] = None,
        revise: bool = True,
        dimension: str = None
    ) -> ExplanationResult:
        """Enhanced evaluation with both top-and-random and random-only scoring."""
        neuron_id = (layer, neuron_idx)
        
        # Create progress bar for overall process
        pbar = tqdm(total=6 if revise else 4, desc="Neuron Analysis", position=0)
        
        # Get top activating sequences
        pbar.set_description("Finding top activating sequences")
        top_activations = self.get_top_activating_sequences(layer, neuron_idx, texts)
        pbar.update(1)
        
        # Generate initial explanation
        pbar.set_description("Generating initial explanation")
        explanation = self.generate_explanation(layer, neuron_idx, top_activations)
        pbar.update(1)
        
        # Get both random and top activations for comprehensive scoring
        pbar.set_description("Computing real activations")
        real_top = self._get_real_activations(layer, neuron_idx, texts)
        real_random = self._get_real_activations(layer, neuron_idx, random_texts) if random_texts else None
        pbar.update(1)
        
        # Simulate activations for both sets
        pbar.set_description("Simulating activations")
        sim_top = self.simulate_activations(explanation, texts, layer, neuron_idx)
        sim_random = self.simulate_activations(explanation, random_texts, layer, neuron_idx) if random_texts else None
        pbar.update(1)
        
        # Compute both scoring types as per paper
        top_and_random_score = self._compute_combined_score(real_top, sim_top, real_random, sim_random)
        random_only_score = self.compute_correlation_score(real_random, sim_random) if random_texts else None
        
        # Perform revision if requested
        revision = None
        revision_scores = None
        if revise:
            pbar.set_description("Generating test cases")
            test_cases = self.generate_test_cases(explanation, n=10, layer=layer, neuron_idx=neuron_idx)
            pbar.update(1)
            
            pbar.set_description("Revising explanation")
            revision, revision_scores = self._perform_revision(
                explanation, test_cases, texts, random_texts, layer, neuron_idx
            )
            pbar.update(1)
        
        # Analyze patterns
        analysis = self._analyze_activation_patterns(top_activations)
        
        pbar.close()
        
        result = ExplanationResult(
            neuron_id=neuron_id,
            explanation=explanation,
            score=top_and_random_score,
            top_activations=top_activations,
            analysis={
                **analysis,
                'random_only_score': random_only_score,
                'is_sparse': (real_top != 0).float().mean() < 0.2
            },
            revision=revision,
            revision_score=revision_scores
        )
        
        model_name = self.model.cfg.model_name.replace('/', '-').replace('.', '-')
        report = NeuronReport(result, self.dimension)
        
        report.save_report(os.path.join(self.log_dir, model_name, self.dimension), layer, neuron_idx)
        
        return result

    def _perform_revision(
        self,
        explanation: str,
        test_cases: List[str],
        texts: List[str],
        random_texts: Optional[List[str]],
        layer: int,
        neuron_idx: int
    ) -> Tuple[str, Dict[str, float]]:
        """Perform explanation revision as per paper methodology."""
        # Split test cases into revision and scoring sets
        revision_cases = test_cases[:5]
        scoring_cases = test_cases[5:]
        
        # Get activations for revision cases
        revision_activations = self._get_real_activations(layer, neuron_idx, revision_cases)
        
        # Generate revised explanation
        revised = self.revise_explanation(explanation, revision_cases, revision_activations, layer, neuron_idx)
        
        # Score revised explanation
        scores = {}
        scores['original_top'] = self.compute_correlation_score(
            self._get_real_activations(layer, neuron_idx, texts),
            self.simulate_activations(explanation, texts, layer, neuron_idx)
        )
        scores['revised_top'] = self.compute_correlation_score(
            self._get_real_activations(layer, neuron_idx, texts),
            self.simulate_activations(revised, texts, layer, neuron_idx)
        )
        
        if random_texts:
            scores['original_random'] = self.compute_correlation_score(
                self._get_real_activations(layer, neuron_idx, random_texts),
                self.simulate_activations(explanation, random_texts, layer, neuron_idx)
            )
            scores['revised_random'] = self.compute_correlation_score(
                self._get_real_activations(layer, neuron_idx, random_texts),
                self.simulate_activations(revised, random_texts, layer, neuron_idx)
            )
        
        return revised, scores

    def _compute_combined_score(
        self,
        real_top: torch.Tensor,
        sim_top: torch.Tensor,
        real_random: Optional[torch.Tensor],
        sim_random: Optional[torch.Tensor]
    ) -> float:
        """Compute combined top-and-random score as per paper."""
        if real_random is None or sim_random is None:
            return self.compute_correlation_score(real_top, sim_top)
            
        # Combine top and random activations
        real_combined = torch.cat([real_top, real_random])
        sim_combined = torch.cat([sim_top, sim_random])
        
        return self.compute_correlation_score(real_combined, sim_combined)
    
    def generate_explanation(
        self,
        layer: int,
        neuron_idx: int,
        activations: List[NeuronActivation]
    ) -> str:
        """Generate explanation using GPT-4 following paper methodology."""
        prompt = self._create_explanation_prompt(layer, neuron_idx, activations)
        
        # Log the explanation prompt
        self._log_prompt(prompt, layer, neuron_idx, "explanation")
        
        try:
            response = self.client.chat.completions.create(
                model=self.llm_name,
                messages=[
                    {"role": "system", "content": "You are an expert at interpreting neural network behavior. "
                    "Provide clear, specific explanations of what patterns or concepts a neuron responds to."},
                    {"role": "user", "content": prompt}
                ],
                temperature=1.0,
                max_tokens=200
            )
            self._track_usage(response)
            
            self._log_response(response.choices[0].message.content.strip(), layer, neuron_idx, "explanation")
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating explanation: {str(e)}")
            return ""
    
    def _create_explanation_prompt(
        self,
        layer: int,
        neuron_idx: int,
        activations: List[NeuronActivation]
    ) -> str:
        """Create prompt for explanation generation following paper methodology."""
        # Format the activation data into token/activation pairs
        activation_data = []
        for act in activations:
            # Get normalized activation value (0-10 scale)
            norm_activation = self._normalize_activations(
                torch.tensor([act.activation]), 
                (layer, neuron_idx)
            )[0].item()
            
            activation_data.append(
                f"Token: {act.token}\t"
                f"Activation: {norm_activation}\t"
                f"Context: {act.context_before}{act.token}{act.context_after}"
            )

        # Create the prompt following paper's format
        prompt = ("Analyze this neuron (layer " + str(layer) + ", index " + str(neuron_idx) + ") based on its highest activations.\n\n"
                 "For each example below, you'll see:\n"
                 "- The token where the neuron activates\n" 
                 "- The activation strength (0-10 scale)\n"
                 "- The surrounding context\n\n"
                 "Examples:\n"
                 + "\n\n".join(activation_data) + "\n\n"
                 "Based on these examples, what pattern or concept does this neuron detect?\n"
                 "Give a clear, specific explanation focusing on when and why the neuron activates.\n"
                 "Keep in mind that neurons often respond to specific linguistic patterns rather than broad topics.\n\n"
                 "Your explanation should:\n"
                 "1. Be precise about the exact pattern that triggers activation\n"
                 "2. Account for the context around the activating tokens\n"
                 "3. Note any consistent themes or variations in the activation pattern")

        return prompt

    def _normalize_single_activation(
            self,
            activation: float,
            neuron_id: Tuple[int, int]
        ) -> int:
            """Normalize a single activation value to 0-10 scale."""
            if neuron_id not in self.neuron_max_activations:
                return 0  # Return 0 if we don't have max activation reference
                
            max_activation = self.neuron_max_activations[neuron_id]
            normalized = min(10, max(0, (activation / max_activation) * 10))
            return round(normalized)  #
    def _get_real_activations(
        self,
        layer: int,
        neuron_idx: int,
        texts: List[str]
    ) -> torch.Tensor:
        """Get actual neuron activations for a set of texts.
        
        Args:
            layer: The layer index
            neuron_idx: The neuron index
            texts: List of text sequences to analyze
            
        Returns:
            torch.Tensor: Normalized activations on 0-10 scale
        """
        if texts is None:
            return None
            
        all_activations = []
        neuron_id = (layer, neuron_idx)
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_tokens = [self.model.to_tokens(text) for text in batch_texts]
            
            with torch.no_grad():
                for tokens in batch_tokens:
                    _, cache = self.model.run_with_cache(tokens)
                    # Get activations for the neuron
                    activations = cache['post', layer, 'mlp'][0, :, neuron_idx]
                    
                    # Update max activation tracking if needed
                    current_max = torch.max(activations).item()
                    if neuron_id not in self.neuron_max_activations or current_max > self.neuron_max_activations[neuron_id]:
                        self.neuron_max_activations[neuron_id] = current_max
                    
                    # For each sequence, get max activation
                    all_activations.append(activations.max().item())
        
        # Convert to tensor
        activation_tensor = torch.tensor(all_activations)
        
        # Normalize to 0-10 scale and discretize as per paper
        normalized = self._normalize_activations(activation_tensor, neuron_id)
        
        # Check for sparsity
        if (normalized != 0).float().mean() < 0.2:
            print(f"Warning: Sparse activations detected for neuron {neuron_id}")
        
        return normalized
    def _parallel_simulation(
        self,
        explanation: str,
        texts: List[str],
        layer: int = None,
        neuron_idx: int = None
    ) -> torch.Tensor:
        """Simulate all activations in parallel."""
        prompt = self._create_parallel_simulation_prompt(explanation, texts)
        
        # Log the simulation prompt with the provided layer and neuron_idx
        if layer is not None and neuron_idx is not None:
            self._log_prompt(prompt, layer, neuron_idx, "simulation")
        
        try:
            response = self.client.chat.completions.create(
                model=self.llm_name,
                messages=[
                    {"role": "system", "content": "You are simulating neuron activations. "
                    "For each text, predict the neuron's activation on a scale of 0-10, "
                    "where 0 means no activation and 10 means maximum activation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=len(texts) * 20
            )
            self._track_usage(response)
            
            self._log_response(response.choices[0].message.content.strip(), layer, neuron_idx, "simulation")
            return self._parse_parallel_simulation(response.choices[0].message.content)
        except Exception as e:
            print(f"Error in parallel simulation: {str(e)}")
            return torch.zeros(len(texts))

    def _sequential_simulation(
        self,
        explanation: str,
        texts: List[str],
        layer: int = None,
        neuron_idx: int = None
    ) -> torch.Tensor:
        """Simulate activations one at a time using the one-at-a-time method from the paper."""
        simulated_activations = []
        
        for text in texts:
            prompt = self._create_sequential_simulation_prompt(explanation, text)
            
            try:
                response = self.client.chat.completions.create(
                    model=self.llm_name,
                    messages=[
                        {"role": "system", "content": "You are simulating a neuron's activation. "
                        "Predict the activation on a scale of 0-10, where 0 means no activation "
                        "and 10 means maximum activation."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=20
                )
                
                # Parse single activation value
                try:
                    value = float(response.choices[0].message.content.strip())
                    value = max(0, min(10, round(value)))  # Ensure valid range and integer
                    simulated_activations.append(value)
                except ValueError:
                    print(f"Error parsing activation value: {response.choices[0].message.content}")
                    simulated_activations.append(0.0)
                    
            except Exception as e:
                print(f"Error in sequential simulation: {str(e)}")
                simulated_activations.append(0.0)
        
        return torch.tensor(simulated_activations)

    def _create_parallel_simulation_prompt(
        self,
        explanation: str,
        texts: List[str]
    ) -> str:
        """Create prompt for parallel activation simulation following paper format."""
        prompt = f"""Based on this explanation of a neuron's behavior:
    "{explanation}"

    For each of the following texts, predict the neuron's activation on a scale of 0-10,
    where 0 means no activation and 10 means maximum activation.
    Only provide numerical values prefixed with "Activation:".

    """
        for i, text in enumerate(texts, 1):
            prompt += f"\nText {i}: {text}\nActivation: "
        
        return prompt

    def _create_sequential_simulation_prompt(
        self,
        explanation: str,
        text: str
    ) -> str:
        """Create prompt for single text activation simulation."""
        return f"""Based on this explanation of a neuron's behavior:
    "{explanation}"

    For the following text, predict the neuron's activation on a scale of 0-10,
    where 0 means no activation and 10 means maximum activation.
    Provide only the numerical value.

    Text: {text}
    Activation: """

    def _parse_parallel_simulation(
        self,
        response: str
    ) -> torch.Tensor:
        """Parse the response from parallel simulation into activation values.
        
        Follows paper's methodology for activation value handling."""
        try:
            # Extract numbers after "Activation:" from each line
            activations = []
            for line in response.split('\n'):
                if 'Activation:' in line:
                    value_str = line.split('Activation:')[1].strip()
                    try:
                        # Convert to float and ensure integer in 0-10 range
                        value = float(value_str)
                        value = max(0, min(10, round(value)))
                        activations.append(value)
                    except ValueError:
                        print(f"Error parsing activation value: {value_str}")
                        activations.append(0.0)
            
            return torch.tensor(activations, dtype=torch.float32)
            
        except Exception as e:
            print(f"Error parsing simulation response: {str(e)}")
            return torch.zeros(1)
    
    def compute_correlation_score(
        self,
        real_activations: torch.Tensor,
        simulated_activations: torch.Tensor
    ) -> float:
        """Fixed version of correlation score computation."""
        if real_activations is None or simulated_activations is None:
            return 0.0
            
        if len(real_activations) != len(simulated_activations):
            print(f"Warning: Activation length mismatch: {len(real_activations)} vs {len(simulated_activations)}")
            min_len = min(len(real_activations), len(simulated_activations))
            real_activations = real_activations[:min_len]
            simulated_activations = simulated_activations[:min_len]
        
        try:
            # Ensure we have tensors
            real_activations = torch.as_tensor(real_activations, dtype=torch.float32)
            simulated_activations = torch.as_tensor(simulated_activations, dtype=torch.float32)
            
            # Handle constant activation case
            if torch.all(real_activations == real_activations[0]) or \
            torch.all(simulated_activations == simulated_activations[0]):
                return 0.0
                
            # Normalize activations
            real_mean = real_activations.mean()
            real_std = real_activations.std()
            sim_mean = simulated_activations.mean()
            sim_std = simulated_activations.std()
            
            # Avoid division by zero
            if real_std == 0 or sim_std == 0:
                return 0.0
                
            real_norm = (real_activations - real_mean) / real_std
            sim_norm = (simulated_activations - sim_mean) / sim_std
            
            # Compute correlation coefficient
            correlation = torch.corrcoef(torch.stack([real_norm, sim_norm]))[0, 1]
            
            # Convert to Python float and handle NaN
            correlation_value = correlation.item()
            if isinstance(correlation_value, float) and torch.tensor(correlation_value).isnan():
                return 0.0
                
            return correlation_value
            
        except Exception as e:
            print(f"Error computing correlation score: {str(e)}")
            return 0.0
        
    def generate_test_cases(
        self,
        explanation: str,
        n: int = 10,
        layer: int = None,
        neuron_idx: int = None
    ) -> List[str]:
        """Generate test cases to evaluate the explanation."""
        prompt = self._create_test_case_prompt(explanation, n)
        
        # Log the test cases prompt with the provided layer and neuron_idx
        if layer is not None and neuron_idx is not None:
            self._log_prompt(prompt, layer, neuron_idx, "test-cases")
        
        try:
            response = self.client.chat.completions.create(
                model=self.llm_name,
                messages=[
                    {"role": "system", "content": "Generate diverse test cases to evaluate "
                    "a neuron's behavior explanation. Include both typical cases and edge cases."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            self._track_usage(response)
            test_cases = self._parse_test_cases(response.choices[0].message.content)
            
            self._log_response(response.choices[0].message.content.strip(), layer, neuron_idx, "test-cases")
            
            if len(test_cases) < n:
                while len(test_cases) < n:
                    test_cases.append(self._generate_fallback_test_case(explanation))
            elif len(test_cases) > n:
                test_cases = test_cases[:n]
                
            return test_cases
            
        except Exception as e:
            print(f"Error generating test cases: {str(e)}")
            return [self._generate_fallback_test_case(explanation) for _ in range(n)]

    def _create_test_case_prompt(
        self,
        explanation: str,
        n: int
    ) -> str:
        """Create prompt for test case generation."""
        return f"""Based on this explanation of a neuron's behavior:
    "{explanation}"

    Generate {n} diverse text examples that should activate this neuron according to the explanation.

    Guidelines:
    1. Include typical cases that clearly match the explanation
    2. Include edge cases that test the boundaries of the explanation
    3. Include variations that might reveal ambiguities in the explanation
    4. Make examples diverse in content and structure
    5. Each example should be 1-3 sentences long

    Format your response as:
    Test case 1: [your example]
    Test case 2: [your example]
    ...and so on.
    """

    def _parse_test_cases(
        self,
        response: str
    ) -> List[str]:
        """Parse generated test cases from response."""
        test_cases = []
        current_case = []
        
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Clean up common prefixes
            line = line.lstrip('0123456789.- )"')
            
            # Check for new test case
            if line.lower().startswith(('test case', 'case', 'example')):
                if current_case:
                    test_cases.append(' '.join(current_case))
                    current_case = []
                # Remove the prefix if it exists
                line = ':'.join(line.split(':')[1:]) if ':' in line else line
                
            current_case.append(line.strip())
        
        # Add the last case
        if current_case:
            test_cases.append(' '.join(current_case))
        
        # Clean up and validate
        test_cases = [case.strip() for case in test_cases if case.strip()]
        return test_cases

    def _generate_fallback_test_case(
        self,
        explanation: str
    ) -> str:
        """Generate a single fallback test case if main generation fails."""
        try:
            response = self.client.chat.completions.create(
                model=self.llm_name,
                messages=[
                    {"role": "system", "content": "Generate a single test case example."},
                    {"role": "user", "content": f'Generate one example text that should trigger this neuron: "{explanation}"'}
                ],
                temperature=0.7,
                max_tokens=100
            )
            self._track_usage(response)  # Track usage
            return response.choices[0].message.content.strip()
        except Exception:
            return f"Fallback test case for: {explanation}"
        
    def revise_explanation(
        self,
        original_explanation: str,
        test_cases: List[str],
        activations: torch.Tensor,
        layer: int = None,
        neuron_idx: int = None
    ) -> str:
        """Revise explanation based on test cases and their activations."""
        prompt = self._create_revision_prompt(
            original_explanation,
            test_cases,
            activations
        )
        
        # Log the revision prompt with the provided layer and neuron_idx
        if layer is not None and neuron_idx is not None:
            self._log_prompt(prompt, layer, neuron_idx, "revision")
        
        try:
            response = self.client.chat.completions.create(
                model=self.llm_name,
                messages=[
                    {"role": "system", "content": "Revise neuron behavior explanation based on new evidence. "
                    "Be specific about what patterns do and don't trigger the neuron."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=300
            )
            self._track_usage(response)
            
            self._log_response(response.choices[0].message.content.strip(), layer, neuron_idx, "revision")
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error revising explanation: {str(e)}")
            return original_explanation

    def _create_revision_prompt(
        self,
        original_explanation: str,
        test_cases: List[str],
        activations: torch.Tensor
    ) -> str:
        """Create prompt for explanation revision."""
        # Format test cases and their activations
        evidence = []
        for test_case, activation in zip(test_cases, activations):
            # Convert tensor to float if needed
            act_value = activation.item() if isinstance(activation, torch.Tensor) else float(activation)
            evidence.append(f"Text: {test_case}\nActivation: {act_value:.3f}")
        
        prompt = f"""Original explanation of neuron behavior:
    "{original_explanation}"

    New evidence from test cases:
    {chr(10).join(evidence)}

    Based on this new evidence, please provide a revised explanation of the neuron's behavior.
    Consider:
    1. Which aspects of the original explanation are supported by the new evidence?
    2. Which aspects need to be modified or removed?
    3. What new patterns or nuances are revealed by the test cases?

    Provide a clear, specific explanation that accounts for both the original and new evidence."""

        return prompt

    def _analyze_activation_patterns(
    self,
    activations: List[NeuronActivation]
) -> Dict:
        """Analyze patterns in neuron activations.
        
        Following paper methodology for analyzing activation patterns and contexts.
        
        Args:
            activations: List of NeuronActivation objects to analyze
        
        Returns:
            Dict: Analysis results including statistics and patterns
        """
        if not activations:
            return {
                'total_activations': 0,
                'avg_activation': 0.0,
                'max_activation': 0.0,
                'token_frequency': {},
                'position_distribution': {},
                'context_patterns': {},
                'top_tokens': [],
                'sparsity': 0.0
            }
        
        try:
            analysis = {
                'total_activations': len(activations),
                'avg_activation': np.mean([a.activation for a in activations]),
                'max_activation': max(a.activation for a in activations),
                'token_frequency': defaultdict(int),
                'position_distribution': defaultdict(int),
                'context_patterns': defaultdict(list),
                'non_zero_ratio': sum(1 for a in activations if a.activation > 0) / len(activations)
            }
            
            # Analyze token patterns
            for act in activations:
                # Track token frequencies
                analysis['token_frequency'][act.token] += 1
                # Track position distribution
                analysis['position_distribution'][act.token_index] += 1
                
                # Analyze context patterns
                context = f"{act.context_before}{act.token}{act.context_after}"
                analysis['context_patterns'][act.token].append({
                    'context': context,
                    'activation': act.activation
                })
            
            # Get top tokens by frequency
            analysis['top_tokens'] = sorted(
                analysis['token_frequency'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            # Calculate sparsity metrics
            analysis['sparsity'] = 1.0 - analysis['non_zero_ratio']
            
            # Identify most common positional patterns
            analysis['common_positions'] = sorted(
                analysis['position_distribution'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            # Analyze activation value distribution
            activation_values = [a.activation for a in activations]
            analysis['activation_stats'] = {
                'mean': float(np.mean(activation_values)),
                'std': float(np.std(activation_values)),
                'median': float(np.median(activation_values)),
                'q1': float(np.percentile(activation_values, 25)),
                'q3': float(np.percentile(activation_values, 75))
            }
            
            # Context analysis
            analysis['context_stats'] = {
                'avg_context_length': np.mean([
                    len(act.context_before) + len(act.context_after) 
                    for act in activations
                ]),
                'unique_contexts': len(set(
                    f"{act.context_before}{act.token}{act.context_after}" 
                    for act in activations
                ))
            }
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing activation patterns: {str(e)}")
            return {
                'error': str(e),
                'total_activations': len(activations),
                'avg_activation': np.mean([a.activation for a in activations]) if activations else 0.0,
                'max_activation': max(a.activation for a in activations) if activations else 0.0
            }
            
class NeuronReport:
    def __init__(self, result: ExplanationResult, dimension: str):
        self.result = result
        self.dimension = dimension

    def generate_report(self):
        print(f"Neuron {self.result.neuron_id} Analysis:")
        print(f"\nInitial Explanation: {self.result.explanation}")
        print(f"Correlation Score: {self.result.score:.3f}")
        if self.result.revision:
            print(f"\nRevised Explanation: {self.result.revision}")
            print(f"Revised Score: {self.result.revision_score}")
        # 6. Examine top activations
        print("\nTop Activating Sequences:")
        for activation in self.result.top_activations[:3]:  # Show top 3
            print(f"\nText: {activation.text}")
            print(f"Token: {activation.token}")
            print(f"Activation: {activation.activation:.3f}")
            print(f"Context: {activation.context_before}[{activation.token}]{activation.context_after}")
    
    def save_report(self, folder: str, layer: int, neuron_idx: int):
        layer_folder = f"L{layer}-N{neuron_idx}"
        os.makedirs(os.path.join(folder, layer_folder), exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d")
        filename = f"{timestamp}_neuron_report_L{layer}-N{neuron_idx}.txt"
        with open(os.path.join(folder, layer_folder, filename), "w") as f:
            f.write(f"Neuron {self.result.neuron_id} Analysis:\n")
            f.write(f"\nInitial Explanation: {self.result.explanation}\n")
            f.write(f"Correlation Score: {self.result.score:.3f}\n")
            if self.result.revision:
                f.write(f"\nRevised Explanation: {self.result.revision}\n")
                f.write(f"Revised Score: {self.result.revision_score}\n")
            # 6. Examine top activations
            f.write("\nTop Activating Sequences:\n")
            for activation in self.result.top_activations[:3]:  # Show top 3
                f.write(f"\nText: {activation.text}\n")
                f.write(f"Token: {activation.token}\n")
                f.write(f"Activation: {activation.activation:.3f}\n")
                f.write(f"Context: {activation.context_before}[{activation.token}]{activation.context_after}\n")
