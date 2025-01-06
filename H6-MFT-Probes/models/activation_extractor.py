"""
Handles the extraction of activations from language models using NNSight.
"""

import logging
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader
from nnsight import LanguageModel
import os

logger = logging.getLogger(__name__)

class ActivationExtractor:
    """Extracts and manages model activations using NNSight."""
    
    def __init__(self, model, layer_stride: int = 5):
        """Initialize the activation extractor.
        
        Args:
            model: The NNSight model to extract activations from
            layer_stride: Extract activations from every nth layer (default: 5)
        """
        self.model = model
        self.layer_stride = layer_stride
        self.logger = logging.getLogger(__name__)

    def _get_layer_names(self):
        """Get names of layers to extract activations from.
        Always includes every nth layer (based on stride) plus the last layer.
        """
        if hasattr(self.model, 'model'):  # Gemma
            num_layers = len(self.model.model.layers)
            # Get every nth layer
            layer_indices = list(range(0, num_layers, self.layer_stride))
            # Add last layer if not already included
            if (num_layers - 1) not in layer_indices:
                layer_indices.append(num_layers - 1)
            # Sort to maintain order
            layer_indices.sort()
            
            self.logger.info(f"Extracting layers: {layer_indices}")
            return [f"model.layers[{i}].output" for i in layer_indices]
        else:
            raise ValueError("Unsupported model type")

    def extract_batch_activations(self, batch_inputs):
        """Extract activations from the last token of each sequence in the batch.
        
        Args:
            batch_inputs: Batch of input sequences or tokenized inputs
            
        Returns:
            dict: Layer name -> tensor of shape (batch_size, hidden_dim)
        """
        # Handle both raw text and pre-tokenized inputs
        if isinstance(batch_inputs, dict) and 'input_ids' in batch_inputs:
            # Already tokenized
            tokenized = {
                'input_ids': batch_inputs['input_ids'].to('cuda:0'),
                'attention_mask': batch_inputs['attention_mask'].to('cuda:0')
            }
        else:
            # Raw text needs tokenization
            tokenized = self.model.tokenizer(
                batch_inputs,
                padding='max_length',
                truncation=True,
                max_length=256,
                return_tensors="pt",
                padding_side='right'
            ).to("cuda:0")
        
        # Find last token positions using attention mask
        attention_mask = tokenized['attention_mask']
        last_token_positions = attention_mask.sum(dim=1) - 1
        
        # Store saved items
        saved_items = {}
        layer_names = self._get_layer_names()
        
        self.logger.info(f"Extracting activations from {len(layer_names)} layers")
        
        with self.model.trace() as model_trace:
            with model_trace.invoke(tokenized['input_ids']) as invoker:
                for layer_name in layer_names:
                    saved_items[layer_name] = eval(f"self.model.{layer_name}.save()")
        
        # Process saved values
        activations = {}
        for layer_name, saved in saved_items.items():
            layer_value = saved.value
            if isinstance(layer_value, tuple):
                layer_value = layer_value[0]
            
            # Extract only last token activations
            last_token_activations = torch.stack([
                layer_value[i, pos.item(), :] 
                for i, pos in enumerate(last_token_positions)
            ])
            
            # Move to CPU to save memory
            activations[layer_name] = last_token_activations.cpu()
            
            # Clear CUDA cache after processing each layer
            torch.cuda.empty_cache()
        
        return activations

    def process_dataset(self, dataloader, save_dir: str):
        """Process entire dataset and save activations.
        
        Args:
            dataloader: DataLoader containing text inputs or tokenized inputs
            save_dir: Directory to save activations to
        """
        os.makedirs(save_dir, exist_ok=True)
        temp_dir = os.path.join(save_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Track total samples and which layers we've seen
        total_samples = 0
        layer_names = None
        
        # Update metadata tracking - remove texts, keep only tokens and labels
        all_input_tokens = []
        all_labels = []
        
        try:
            # Process each batch
            for batch_idx, batch in enumerate(dataloader):
                self.logger.info(f"Processing batch {batch_idx}")
                
                try:
                    # Handle pre-tokenized inputs and extract labels
                    if isinstance(batch, dict):
                        if 'input_ids' in batch:
                            inputs = batch
                            input_tokens = batch['input_ids']
                        else:
                            raise KeyError(f"No input_ids field found in batch. Available keys: {list(batch.keys())}")
                        
                        # Get labels
                        if 'labels' in batch:
                            labels = batch['labels']
                        else:
                            raise KeyError("No labels found in batch")
                            
                    elif isinstance(batch, (list, tuple)):
                        inputs = batch[0]
                        # Tokenize if not already tokenized
                        if not isinstance(inputs, torch.Tensor):
                            tokenized = self.model.tokenizer(
                                inputs,
                                padding='max_length',
                                truncation=True,
                                max_length=256,
                                return_tensors="pt"
                            )
                            input_tokens = tokenized['input_ids']
                        else:
                            input_tokens = inputs
                        labels = batch[1]
                    else:
                        raise TypeError(f"Unsupported batch type: {type(batch)}")
                    
                    # Ensure labels are tensors
                    if not isinstance(labels, torch.Tensor):
                        labels = torch.tensor(labels)
                    
                    # Store metadata
                    all_input_tokens.append(input_tokens.cpu())
                    all_labels.append(labels.cpu())
                    
                    # Extract activations for current batch
                    batch_activations = self.extract_batch_activations(inputs)
                    
                    # Initialize layer_names if not done yet
                    if layer_names is None:
                        layer_names = list(batch_activations.keys())
                    
                    # Save each layer's activations to a temporary file
                    for layer_name, activations in batch_activations.items():
                        temp_path = os.path.join(temp_dir, f"{layer_name}_batch_{batch_idx}.pt")
                        torch.save({
                            'activations': activations,
                            'batch_size': len(input_tokens)
                        }, temp_path)
                    
                    # Update total samples count
                    total_samples += len(input_tokens)
                    
                    # Clear batch memory
                    del batch_activations
                    torch.cuda.empty_cache()
                    
                    # Log progress
                    if (batch_idx + 1) % 10 == 0:
                        self.logger.info(f"Processed {total_samples} samples so far...")
                    
                except Exception as e:
                    self.logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                    self.logger.error(f"Batch structure: {batch}")
                    raise
            
            # After processing all batches, combine temporary files for each layer
            self.logger.info("Combining activations for each layer...")
            
            # Combine all metadata tensors
            all_input_tokens = torch.cat(all_input_tokens, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            for layer_name in layer_names:
                # Get all temporary files for this layer
                temp_files = sorted([
                    f for f in os.listdir(temp_dir)
                    if f.startswith(f"{layer_name}_batch_") and f.endswith(".pt")
                ])
                
                # Combine activations memory-efficiently
                combined_activations = []
                for temp_file in temp_files:
                    temp_path = os.path.join(temp_dir, temp_file)
                    data = torch.load(temp_path)
                    combined_activations.append(data['activations'])
                    os.remove(temp_path)  # Remove temporary file after loading
                
                # Concatenate and save final layer activations with metadata
                final_activations = torch.cat(combined_activations, dim=0)
                layer_save_path = os.path.join(save_dir, f"{layer_name}.pt")
                torch.save({
                    'activations': final_activations,
                    'input_tokens': all_input_tokens,
                    'labels': all_labels,
                    'total_samples': total_samples
                }, layer_save_path)
                self.logger.info(f"Saved {final_activations.shape} activations to {layer_save_path}")
                
                # Clear memory
                del combined_activations
                del final_activations
                torch.cuda.empty_cache()
            
            self.logger.info(f"Finished processing {total_samples} samples total")
            
        finally:
            # Clean up temporary directory
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                self.logger.info("Cleaned up temporary files") 