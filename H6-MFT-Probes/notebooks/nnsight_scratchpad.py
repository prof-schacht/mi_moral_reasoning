# %%
"""
Scratchpad for exploring NNSight model structure and hookpoints.
"""

import logging
from pathlib import Path
from nnsight import LanguageModel
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# %%
def explore_last_token_extraction(model_name: str = "google/gemma-2b-it"):
    """Explore extraction of last token activations."""
    
    logger.info(f"Loading model: {model_name}")
    model = LanguageModel(model_name, device_map="cuda:0")
    
    # Example batch with different lengths
    example_inputs = [
        "This is a short input.",
        "This is a longer input with more tokens to process.",
    ]
    
    # First tokenize the inputs to get input_ids and attention_mask
    tokenized = model.tokenizer(
        example_inputs,
        padding='max_length',  # Explicit padding strategy
        truncation=True,
        max_length=256,
        return_tensors="pt",
        padding_side='right'  # Explicitly set padding to right side
    ).to("cuda:0")
    
    # Print raw token IDs for debugging
    logger.info("\nRaw token IDs:")
    for i, ids in enumerate(tokenized['input_ids']):
        logger.info(f"Input {i}: {ids.tolist()}")
    
    # Get attention mask and find real token positions
    attention_mask = tokenized['attention_mask']
    
    # Find the last real token position (before padding starts)
    last_token_positions = attention_mask.sum(dim=1) - 1
    
    logger.info("\nInput shapes and positions:")
    logger.info(f"Input IDs shape: {tokenized['input_ids'].shape}")
    logger.info(f"Attention mask shape: {attention_mask.shape}")
    logger.info(f"Last token positions: {last_token_positions}")
    
    # Store saved items
    saved_items = {}
    
    logger.info("\nExploring last token extraction...")
    with model.trace() as model_trace:
        with model_trace.invoke(tokenized['input_ids']) as invoker:
            if hasattr(model, 'model'):  # Gemma
                # Extract activations only for last tokens
                layer_idx = 0  # Example with first layer
                saved_items['residual'] = model.model.layers[layer_idx].output.save()
    
    # After trace completes, process the saved values
    if 'residual' in saved_items:
        residual_value = saved_items['residual'].value
        if isinstance(residual_value, tuple):
            residual_value = residual_value[0]
            
        logger.info(f"\nResidual value type: {type(residual_value)}")
        logger.info(f"Residual value shape: {residual_value.shape}")
        
        # Select last token for each sequence in batch
        batch_size = len(last_token_positions)
        last_token_activations = torch.stack([
            residual_value[i, pos.item(), :] 
            for i, pos in enumerate(last_token_positions)
        ])
        
        logger.info("\nActivation shapes:")
        logger.info(f"Full residual stream: {residual_value.shape}")
        logger.info(f"Last token only: {last_token_activations.shape}")
        
        # Memory size comparison
        full_size = residual_value.element_size() * residual_value.nelement()
        last_token_size = last_token_activations.element_size() * last_token_activations.nelement()
        
        logger.info("\nMemory usage:")
        logger.info(f"Full activation size: {full_size / 1024 / 1024:.2f} MB")
        logger.info(f"Last token only size: {last_token_size / 1024 / 1024:.2f} MB")
        
        # Print detailed token information for verification
        logger.info("\nDetailed token information:")
        for i, (text, pos) in enumerate(zip(example_inputs, last_token_positions)):
            tokens = model.tokenizer.convert_ids_to_tokens(tokenized['input_ids'][i])
            non_pad_tokens = [t for t in tokens if t not in [model.tokenizer.pad_token]]
            logger.info(f"\nInput {i}: {text}")
            logger.info(f"All tokens: {tokens}")
            logger.info(f"Non-padding tokens: {non_pad_tokens}")
            logger.info(f"Attention mask: {attention_mask[i].tolist()}")
            logger.info(f"Last token position: {pos.item()}")
            logger.info(f"Last token: {tokens[pos.item()]}")
            logger.info(f"Last 3 tokens: {tokens[max(0, pos.item()-2):pos.item()+1]}")

    # Memory cleanup
    del model
    torch.cuda.empty_cache()

# %%
if __name__ == "__main__":
    explore_last_token_extraction()
# %%
