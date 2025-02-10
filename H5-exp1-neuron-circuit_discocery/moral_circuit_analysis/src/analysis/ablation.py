import torch
from typing import List, Tuple, Dict, Optional
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
import numpy as np
from sentence_transformers import SentenceTransformer

class MoralProbe:
    def __init__(self, model: HookedTransformer):
        self.model = model
        self.device = model.cfg.device
        # Initialize classifier with correct input dimension (d_model from last layer)
        self.classifier = torch.nn.Linear(model.cfg.d_model, 1).to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.classifier.parameters())
        
    def get_last_layer_logits(self, text: str) -> torch.Tensor:
        """Get hidden states from the last layer for a given text."""
        tokens = self.model.to_tokens(text)
        with torch.no_grad():
            # Get hidden states from the last layer using TransformerLens
            _, cache = self.model.run_with_cache(
                tokens,
                names_filter=lambda name: name == f"blocks.{self.model.cfg.n_layers-1}.hook_resid_post"
            )
            # Get the last layer's hidden states for the last token
            last_token_hidden = cache[f"blocks.{self.model.cfg.n_layers-1}.hook_resid_post"][:, -1, :]
            return last_token_hidden
    
    def train(self, moral_texts: List[str], neutral_texts: List[str], epochs: int = 5):
        """Train the probe on moral vs neutral texts."""
        print("Training moral probe...")
        
        # Prepare training data
        moral_logits = []
        neutral_logits = []
        
        # Process moral texts
        print("Processing moral texts...")
        for text in moral_texts:
            logits = self.get_last_layer_logits(text)
            moral_logits.append(logits)
            
        # Process neutral texts
        print("Processing neutral texts...")
        for text in neutral_texts:
            logits = self.get_last_layer_logits(text)
            neutral_logits.append(logits)
        
        # Stack all logits
        moral_logits = torch.cat(moral_logits, dim=0)
        neutral_logits = torch.cat(neutral_logits, dim=0)
        
        X = torch.cat([moral_logits, neutral_logits], dim=0)
        y = torch.cat([
            torch.ones(len(moral_texts)), 
            torch.zeros(len(neutral_texts))
        ]).to(self.device)
        
        print(f"Training probe with input shape: {X.shape}, output shape: {y.shape}")
        
        # Training loop
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            pred = self.classifier(X).squeeze()
            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step()
            
            if (epoch + 1) % 1 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    def predict(self, text: str) -> float:
        """Predict moral probability for a given text."""
        with torch.no_grad():
            logits = self.get_last_layer_logits(text)
            pred = torch.sigmoid(self.classifier(logits)).item()
        return pred

class AblationAnalyzer:
    def __init__(self, model: HookedTransformer):
        self.model = model
        self.device = model.cfg.device
        self.n_layers = model.cfg.n_layers
        self.n_neurons = model.cfg.d_mlp
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.probe = None  # Will be initialized when needed
        
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
    
    def train_moral_probe(self, moral_texts: List[str], neutral_texts: List[str]):
        """Initialize and train the moral probe."""
        self.probe = MoralProbe(self.model)
        self.probe.train(moral_texts, neutral_texts)
    
    def analyze_ablation_impact_with_probe(self,
                                         moral_pairs: List[Tuple[str, str]],
                                         neurons: List[Tuple[int, int]],
                                         ablation_value: Optional[float] = 0.0,
                                         max_new_tokens: Optional[int] = 50,
                                         temperature: Optional[float] = 1.0) -> Dict:
        """
        Analyze ablation impact using the moral probe.
        
        Args:
            moral_pairs: List of (moral_text, immoral_text) pairs
            neurons: List of (layer, neuron) pairs to ablate
            ablation_value: Value to set neurons to
            
        Returns:
            Dictionary with probe-based analysis results
        """
        if self.probe is None:
            raise ValueError("Moral probe not initialized. Call train_moral_probe first.")
            
        results = {
            'moral_responses': [],      # (prompt, orig_resp, ablated_resp)
            'moral_predictions': [],    # (orig_pred, ablated_pred)
            'immoral_responses': [],
            'immoral_predictions': []
        }
        
        for moral_text, immoral_text in tqdm(moral_pairs, desc="Analyzing with probe"):
            # Get original and ablated responses / logit predictions
            orig_moral = self.generate_text(moral_text, max_new_tokens, temperature)
            orig_immoral = self.generate_text(immoral_text, max_new_tokens, temperature)
            ablated_moral = self.ablate_neurons(moral_text, neurons, ablation_value, max_new_tokens, temperature)
            ablated_immoral = self.ablate_neurons(immoral_text, neurons, ablation_value, max_new_tokens, temperature)
            
            # Get probe predictions
            orig_moral_pred = self.probe.predict(orig_moral)
            ablated_moral_pred = self.probe.predict(ablated_moral)
            orig_immoral_pred = self.probe.predict(orig_immoral)
            ablated_immoral_pred = self.probe.predict(ablated_immoral)
            
            # Store results
            results['moral_responses'].append((moral_text, orig_moral, ablated_moral))
            results['moral_predictions'].append((orig_moral_pred, ablated_moral_pred))
            results['immoral_responses'].append((immoral_text, orig_immoral, ablated_immoral))
            results['immoral_predictions'].append((orig_immoral_pred, ablated_immoral_pred))
        
        # Add summary statistics
        moral_pred_changes = np.array([(abl - orig) for orig, abl in results['moral_predictions']])
        immoral_pred_changes = np.array([(abl - orig) for orig, abl in results['immoral_predictions']])
        
        results['summary'] = {
            'avg_moral_pred_change': float(np.mean(moral_pred_changes)),
            'std_moral_pred_change': float(np.std(moral_pred_changes)),
            'avg_immoral_pred_change': float(np.mean(immoral_pred_changes)),
            'std_immoral_pred_change': float(np.std(immoral_pred_changes)),
            'moral_effect_size': float(np.mean(moral_pred_changes) / np.std(moral_pred_changes)),
            'immoral_effect_size': float(np.mean(immoral_pred_changes) / np.std(immoral_pred_changes))
        }
        
        return results 