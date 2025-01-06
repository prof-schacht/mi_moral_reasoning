"""
Trains logistic probes on the extracted activations.
"""

import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class LogisticProbe(nn.Module):
    """Simple logistic probe for binary classification."""
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Remove sigmoid since it's included in BCEWithLogitsLoss
        return self.linear(x)

class ProbeTrainer:
    """Handles training and evaluation of logistic probes for each moral foundation."""
    
    def __init__(
        self,
        activation_dir: str = "data/activations",
        probe_dir: str = "data/probes",
        num_classes: int = 8,
        batch_size: int = 256,
        learning_rate: float = 0.001,  # Reduced learning rate
        num_epochs: int = 50,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.activation_dir = Path(activation_dir)
        self.probe_dir = Path(probe_dir)
        self.probe_dir.mkdir(exist_ok=True)
        
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using device: {self.device}")
    
    def _normalize_features(self, train_activations: torch.Tensor, val_activations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize features using StandardScaler."""
        # Convert to numpy for sklearn
        train_np = train_activations.detach().cpu().numpy()
        val_np = val_activations.detach().cpu().numpy()
        
        # Fit scaler on training data
        scaler = StandardScaler()
        train_normalized = scaler.fit_transform(train_np)
        val_normalized = scaler.transform(val_np)
        
        # Convert back to torch tensors
        return (
            torch.FloatTensor(train_normalized),
            torch.FloatTensor(val_normalized)
        )
    
    def _create_binary_labels(self, labels: torch.Tensor, target_class: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create balanced binary dataset with equal numbers of positive and negative samples."""
        # Identify positive and negative samples
        positive_mask = (labels == target_class)
        negative_mask = (labels != target_class)
        
        # Get number of positive samples
        n_positive = positive_mask.sum().item()
        
        # Randomly select equal number of negative samples
        negative_indices = torch.where(negative_mask)[0]
        selected_negative_indices = negative_indices[torch.randperm(len(negative_indices))[:n_positive]]
        
        # Create final mask combining positive samples and selected negative samples
        final_mask = torch.zeros_like(labels, dtype=torch.bool)
        final_mask[positive_mask] = True
        final_mask[selected_negative_indices] = True
        
        # Create binary labels (1 for positive, 0 for negative)
        binary_labels = positive_mask[final_mask].float()
        
        return binary_labels, final_mask, positive_mask
    
    def _create_dataloader(
        self,
        activations: torch.Tensor,
        labels: torch.Tensor,
        target_class: int
    ) -> Tuple[DataLoader, Dict[str, int]]:
        """Create a balanced dataloader with equal positive and negative samples."""
        binary_labels, final_mask, positive_mask = self._create_binary_labels(labels, target_class)
        
        # Select the balanced subset of activations
        filtered_activations = activations[final_mask]
        
        # Move data to device
        filtered_activations = filtered_activations.to(self.device)
        binary_labels = binary_labels.to(self.device)
        
        # Create dataset
        dataset = TensorDataset(filtered_activations, binary_labels)
        
        # Count samples
        sample_counts = {
            'total': len(binary_labels),
            'positive': int(binary_labels.sum().item()),
            'negative': int((~binary_labels.bool()).sum().item())
        }
        
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True), sample_counts
    
    def _train_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer
    ) -> Tuple[float, float, float]:
        """Train for one epoch and return loss and basic metrics."""
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        total_positive = 0
        
        for batch_activations, batch_labels in dataloader:
            batch_activations = batch_activations.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            optimizer.zero_grad()
            logits = model(batch_activations).squeeze()
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            
            # Calculate basic metrics (apply sigmoid for predictions)
            predictions = (torch.sigmoid(logits) >= 0.5).float()
            total_correct += (predictions == batch_labels).sum().item()
            total_positive += predictions.sum().item()
            total_samples += len(batch_labels)
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples
        positive_rate = total_positive / total_samples
        
        return avg_loss, accuracy, positive_rate
    
    def _evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """Evaluate binary classifier using sklearn metrics."""
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch_activations, batch_labels in dataloader:
                batch_activations = batch_activations.to(self.device)
                logits = model(batch_activations).squeeze()
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs >= threshold).astype(int)
                
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(batch_labels.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics using sklearn
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        accuracy = accuracy_score(all_labels, all_preds)
        
        # Count support for each class
        support_positive = int(np.sum(all_labels == 1))
        support_negative = int(np.sum(all_labels == 0))
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'accuracy': float(accuracy),
            'support_positive': support_positive,
            'support_negative': support_negative,
            'avg_prob': float(np.mean(all_probs)),
            'std_prob': float(np.std(all_probs))
        }
    
    def train_probe(self, layer_name: str) -> None:
        """Train logistic probes for each moral foundation category."""
        # Load data
        train_path = self.activation_dir / "train" / f"{layer_name}.pt"
        val_path = self.activation_dir / "val" / f"{layer_name}.pt"
        
        self.logger.info(f"Loading training data from {train_path}")
        train_data = torch.load(train_path, weights_only=True)
        self.logger.info(f"Loading validation data from {val_path}")
        val_data = torch.load(val_path, weights_only=True)
        
        # Normalize features
        train_activations, val_activations = self._normalize_features(
            train_data['activations'],
            val_data['activations']
        )
        
        input_dim = train_activations.shape[1]
        layer_results = {}
        
        # Train a logistic classifier for each class
        for target_class in range(self.num_classes):
            self.logger.info(f"\nTraining probe for class {target_class} on layer {layer_name}")
            
            # Create balanced dataloaders
            train_loader, train_counts = self._create_dataloader(
                train_activations,
                train_data['labels'],
                target_class
            )
            val_loader, val_counts = self._create_dataloader(
                val_activations,
                val_data['labels'],
                target_class
            )
            
            # Log sample distribution
            self.logger.info(
                f"Training samples for class {target_class}:"
                f"\n  Total: {train_counts['total']}"
                f"\n  Positive: {train_counts['positive']}"
                f"\n  Negative: {train_counts['negative']}"
            )
            
            # Initialize model and training components
            model = LogisticProbe(input_dim).to(self.device)
            criterion = nn.BCEWithLogitsLoss()  # No class weights needed as data is balanced
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            
            # Training loop
            best_val_f1 = 0
            best_model_state = model.state_dict()
            patience = 5
            no_improve = 0
            
            for epoch in range(self.num_epochs):
                train_loss, train_acc, train_pos_rate = self._train_epoch(
                    model, train_loader, criterion, optimizer
                )
                val_metrics = self._evaluate(model, val_loader)
                
                self.logger.info(
                    f"Epoch {epoch+1}/{self.num_epochs} - "
                    f"Train Loss: {train_loss:.4f} - "
                    f"Train Acc: {train_acc:.4f} - "
                    f"Train Pos Rate: {train_pos_rate:.4f} - "
                    f"Val F1: {val_metrics['f1']:.4f} - "
                    f"Val Precision: {val_metrics['precision']:.4f} - "
                    f"Val Recall: {val_metrics['recall']:.4f} - "
                    f"Val Prob: {val_metrics['avg_prob']:.4f} ± {val_metrics['std_prob']:.4f}"
                )
                
                if val_metrics['f1'] > best_val_f1 + 0.001:
                    best_val_f1 = val_metrics['f1']
                    best_model_state = model.state_dict()
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        self.logger.info("Early stopping triggered")
                        break
            
            # Load best model and get final metrics
            model.load_state_dict(best_model_state)
            final_metrics = self._evaluate(model, val_loader)
            
            # Save results for this class
            layer_results[target_class] = {
                'model_state': best_model_state,
                'metrics': final_metrics,
                'config': {
                    'input_dim': input_dim,
                    'train_counts': train_counts,
                    'val_counts': val_counts
                }
            }
        
        # Save all probes for this layer
        save_path = self.probe_dir / f"{layer_name}_logistic_probes.pt"
        torch.save(layer_results, save_path)
        
        self.logger.info(f"\nSaved logistic probes to {save_path}")
        self.logger.info("\nFinal metrics for each class:")
        for class_idx, results in layer_results.items():
            metrics = results['metrics']
            self.logger.info(
                f"\nClass {class_idx}:"
                f"\n  F1: {metrics['f1']:.4f}"
                f"\n  Precision: {metrics['precision']:.4f}"
                f"\n  Recall: {metrics['recall']:.4f}"
                f"\n  Accuracy: {metrics['accuracy']:.4f}"
                f"\n  Positive samples: {metrics['support_positive']}"
                f"\n  Negative samples: {metrics['support_negative']}"
                f"\n  Average probability: {metrics['avg_prob']:.4f} ± {metrics['std_prob']:.4f}"
            )
    
    def train_all_probes(self) -> None:
        """Train logistic probes for all available layers."""
        layer_files = list(self.activation_dir.glob("train/*.pt"))
        
        for layer_file in sorted(layer_files):
            layer_name = layer_file.stem
            self.logger.info(f"\nTraining probes for layer: {layer_name}")
            self.train_probe(layer_name) 