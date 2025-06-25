"""
Utility analysis module implementing Thurstonian models for preference coherence.
Based on the Utility Engineering paper methodology.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize
from tqdm import tqdm
import json
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

from ..data.preference_data import PreferenceScenario
from .preference_elicitor import AggregatedPreference


@dataclass
class ThurstoneUtility:
    """Thurstonian utility model for an outcome."""
    outcome: str
    mean: float  # μ(o)
    variance: float  # σ²(o)
    
    @property
    def std(self) -> float:
        """Standard deviation."""
        return np.sqrt(self.variance)


@dataclass
class UtilityModel:
    """Complete utility model for a set of outcomes."""
    utilities: Dict[str, ThurstoneUtility]
    accuracy: float  # Model fit accuracy
    transitivity_score: float  # Measure of preference transitivity
    completeness_score: float  # Measure of preference completeness
    
    def get_preference_probability(self, option_a: str, option_b: str) -> float:
        """
        Calculate P(option_a > option_b) under the Thurstonian model.
        
        Returns:
            Probability that option_a is preferred to option_b
        """
        if option_a not in self.utilities or option_b not in self.utilities:
            return 0.5  # Default to indifference if not found
        
        util_a = self.utilities[option_a]
        util_b = self.utilities[option_b]
        
        # Difference is Gaussian with mean μ_a - μ_b and variance σ²_a + σ²_b
        mean_diff = util_a.mean - util_b.mean
        var_diff = util_a.variance + util_b.variance
        std_diff = np.sqrt(var_diff)
        
        # P(a > b) = P(U(a) - U(b) > 0) = Φ(mean_diff / std_diff)
        if std_diff > 0:
            return stats.norm.cdf(mean_diff / std_diff)
        else:
            return 1.0 if mean_diff > 0 else 0.0


class UtilityAnalyzer:
    """Analyze and compute utility functions from preference data."""
    
    def __init__(self, regularization: float = 0.1):
        """
        Initialize the utility analyzer.
        
        Args:
            regularization: Regularization parameter for utility fitting
        """
        self.regularization = regularization
    
    def fit_thurstonian_model(
        self,
        preferences: List[AggregatedPreference],
        max_iterations: int = 1000,
        convergence_threshold: float = 1e-6
    ) -> UtilityModel:
        """
        Fit a Thurstonian utility model to preference data.
        
        Args:
            preferences: List of aggregated preferences
            max_iterations: Maximum optimization iterations
            convergence_threshold: Convergence criterion
            
        Returns:
            Fitted UtilityModel
        """
        # Extract unique outcomes
        outcomes = set()
        for pref in preferences:
            outcomes.add(pref.scenario.option_a)
            outcomes.add(pref.scenario.option_b)
        outcomes = sorted(list(outcomes))
        
        # Initialize parameters (means and log variances)
        n_outcomes = len(outcomes)
        params = np.zeros(2 * n_outcomes)  # [means, log_variances]
        params[n_outcomes:] = np.log(0.5)  # Initialize variances to 0.5
        
        # Create outcome index mapping
        outcome_to_idx = {outcome: i for i, outcome in enumerate(outcomes)}
        
        # Optimize parameters
        result = minimize(
            self._negative_log_likelihood,
            params,
            args=(preferences, outcome_to_idx),
            method='L-BFGS-B',
            options={'maxiter': max_iterations, 'ftol': convergence_threshold}
        )
        
        # Extract fitted parameters
        means = result.x[:n_outcomes]
        variances = np.exp(result.x[n_outcomes:])
        
        # Create utility objects
        utilities = {}
        for i, outcome in enumerate(outcomes):
            utilities[outcome] = ThurstoneUtility(
                outcome=outcome,
                mean=means[i],
                variance=variances[i]
            )
        
        # Calculate model accuracy
        accuracy = self._calculate_model_accuracy(utilities, preferences)
        
        # Calculate transitivity and completeness scores
        transitivity = self._calculate_transitivity_score(utilities, preferences)
        completeness = self._calculate_completeness_score(preferences)
        
        return UtilityModel(
            utilities=utilities,
            accuracy=accuracy,
            transitivity_score=transitivity,
            completeness_score=completeness
        )
    
    def _negative_log_likelihood(
        self,
        params: np.ndarray,
        preferences: List[AggregatedPreference],
        outcome_to_idx: Dict[str, int]
    ) -> float:
        """
        Compute negative log-likelihood for Thurstonian model.
        
        Args:
            params: Model parameters [means, log_variances]
            preferences: Preference data
            outcome_to_idx: Mapping from outcomes to indices
            
        Returns:
            Negative log-likelihood value
        """
        n_outcomes = len(outcome_to_idx)
        means = params[:n_outcomes]
        variances = np.exp(params[n_outcomes:])
        
        nll = 0.0
        
        for pref in preferences:
            # Get indices
            idx_a = outcome_to_idx.get(pref.scenario.option_a)
            idx_b = outcome_to_idx.get(pref.scenario.option_b)
            
            if idx_a is None or idx_b is None:
                continue
            
            # Calculate preference probability under model
            mean_diff = means[idx_a] - means[idx_b]
            var_diff = variances[idx_a] + variances[idx_b]
            std_diff = np.sqrt(var_diff)
            
            if std_diff > 0:
                prob_a = stats.norm.cdf(mean_diff / std_diff)
            else:
                prob_a = 1.0 if mean_diff > 0 else 0.0
            
            # Use empirical probabilities from aggregated preferences
            empirical_prob_a = pref.prefer_a_probability
            
            # Add to negative log-likelihood
            if empirical_prob_a > 0 and empirical_prob_a < 1:
                nll -= (empirical_prob_a * np.log(prob_a + 1e-10) + 
                       (1 - empirical_prob_a) * np.log(1 - prob_a + 1e-10))
        
        # Add regularization
        nll += self.regularization * np.sum(variances)
        
        return nll
    
    def _calculate_model_accuracy(
        self,
        utilities: Dict[str, ThurstoneUtility],
        preferences: List[AggregatedPreference]
    ) -> float:
        """Calculate accuracy of utility model predictions."""
        correct = 0
        total = 0
        
        for pref in preferences:
            if pref.scenario.option_a not in utilities or pref.scenario.option_b not in utilities:
                continue
            
            util_a = utilities[pref.scenario.option_a]
            util_b = utilities[pref.scenario.option_b]
            
            # Model prediction
            model_prefers_a = util_a.mean > util_b.mean
            
            # Empirical preference
            empirical_prefers_a = pref.prefer_a_probability > 0.5
            
            if model_prefers_a == empirical_prefers_a:
                correct += 1
            
            total += 1
        
        return correct / total if total > 0 else 0.0
    
    def _calculate_transitivity_score(
        self,
        utilities: Dict[str, ThurstoneUtility],
        preferences: List[AggregatedPreference],
        n_samples: int = 1000
    ) -> float:
        """
        Calculate transitivity score by sampling preference triads.
        
        Returns:
            Score between 0 and 1 (1 = perfectly transitive)
        """
        # Get unique outcomes
        outcomes = list(utilities.keys())
        if len(outcomes) < 3:
            return 1.0
        
        violations = 0
        
        for _ in range(n_samples):
            # Sample three distinct outcomes
            if len(outcomes) >= 3:
                a, b, c = np.random.choice(outcomes, size=3, replace=False)
            else:
                continue
            
            # Check transitivity: if a > b and b > c, then a > c
            util_a = utilities[a].mean
            util_b = utilities[b].mean
            util_c = utilities[c].mean
            
            if util_a > util_b and util_b > util_c:
                if util_a <= util_c:
                    violations += 1
            elif util_b > util_a and util_a > util_c:
                if util_b <= util_c:
                    violations += 1
            elif util_c > util_b and util_b > util_a:
                if util_c <= util_a:
                    violations += 1
        
        return 1.0 - (violations / n_samples)
    
    def _calculate_completeness_score(
        self,
        preferences: List[AggregatedPreference]
    ) -> float:
        """
        Calculate completeness score based on preference decisiveness.
        
        Returns:
            Average preference strength across all comparisons
        """
        if not preferences:
            return 0.0
        
        strengths = [pref.preference_strength for pref in preferences]
        return np.mean(strengths)
    
    def compute_expected_utility_property(
        self,
        utility_model: UtilityModel,
        lottery_preferences: List[AggregatedPreference]
    ) -> float:
        """
        Test if the model satisfies the expected utility property.
        
        Args:
            utility_model: Fitted utility model
            lottery_preferences: Preferences over lotteries (probabilistic outcomes)
            
        Returns:
            Score indicating how well expected utility property holds
        """
        # This would require lottery scenarios in the preference data
        # For now, return a placeholder
        # In full implementation, would test if U(lottery) = E[U(outcome)]
        return 0.0  # Placeholder
    
    def train_utility_probe(
        self,
        hidden_states: torch.Tensor,
        utilities: Dict[str, float],
        outcome_indices: Dict[str, int]
    ) -> nn.Module:
        """
        Train a linear probe to predict utilities from hidden states.
        
        Args:
            hidden_states: Model hidden states for each outcome
            utilities: Utility values for each outcome
            outcome_indices: Mapping from outcomes to indices
            
        Returns:
            Trained probe model
        """
        # Prepare data
        X = []
        y = []
        
        for outcome, idx in outcome_indices.items():
            if outcome in utilities:
                X.append(hidden_states[idx].cpu().numpy())
                y.append(utilities[outcome])
        
        X = np.array(X)
        y = np.array(y)
        
        # Train linear regression
        probe = LogisticRegression()
        probe.fit(X, y)
        
        return probe
    
    def save_utility_model(self, model: UtilityModel, output_path: Path):
        """Save utility model to JSON file."""
        data = {
            'utilities': {
                outcome: {
                    'mean': util.mean,
                    'variance': util.variance,
                    'std': util.std
                }
                for outcome, util in model.utilities.items()
            },
            'accuracy': model.accuracy,
            'transitivity_score': model.transitivity_score,
            'completeness_score': model.completeness_score
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved utility model to {output_path}")
    
    def load_utility_model(self, input_path: Path) -> UtilityModel:
        """Load utility model from JSON file."""
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        utilities = {}
        for outcome, util_data in data['utilities'].items():
            utilities[outcome] = ThurstoneUtility(
                outcome=outcome,
                mean=util_data['mean'],
                variance=util_data['variance']
            )
        
        return UtilityModel(
            utilities=utilities,
            accuracy=data['accuracy'],
            transitivity_score=data['transitivity_score'],
            completeness_score=data['completeness_score']
        )