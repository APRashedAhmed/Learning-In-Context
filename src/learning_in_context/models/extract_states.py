"""
Extract hidden and cell states from trained models.

This module loads trained SequenceModel checkpoints and extracts their hidden states, 
cell states, and predictions for analysis using self-contained implementations.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import self-contained implementations
from .sequence_model import SequenceModel, SequenceModelBase
from ..datamodules.bouncing_ball import BouncingBallDataModule, create_eval_datamodule
from ..config.model_config import (
    ModelConfig, 
    ExtractionConfig, 
    get_model_config_for_id,
    override_model_config,
    parse_cli_overrides
)


def load_model_from_checkpoint(
    checkpoint_path: str, 
    model_config: Optional[ModelConfig] = None,
    model_id: Optional[str] = None,
    device: str = 'cpu'
):
    """
    Load SequenceModel from PyTorch Lightning checkpoint with configuration override support.
    
    Parameters
    ----------
    checkpoint_path : str
        Path to the checkpoint file
    model_config : ModelConfig, optional
        Model configuration to use. If None, will try to extract from checkpoint or use default
    model_id : str, optional
        Model ID for getting default configuration (e.g., "SAN-4378")
    device : str
        Device to load the model on
        
    Returns
    -------
    tuple
        (SequenceModelBase, ModelConfig) - loaded model and configuration used
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Determine model configuration
    if model_config is None:
        # Try to extract from checkpoint hyperparameters
        hparams = checkpoint.get('hyper_parameters', {})
        checkpoint_model_config = hparams.get('model_config', {})
        
        if checkpoint_model_config:
            logger.info("Using model config from checkpoint hyperparameters")
            model_config = ModelConfig.from_dict(checkpoint_model_config)
        elif model_id:
            logger.info(f"Using default model config for {model_id}")
            model_config = get_model_config_for_id(model_id)
        else:
            logger.warning("No model config found, using default configuration")
            model_config = ModelConfig()
    
    logger.info(f"Model configuration: recurrent_size={model_config.recurrent_size}, "
               f"recurrent_num_layers={model_config.recurrent_num_layers}")
    
    # Create the model base (computational component)
    sequence_model_base = SequenceModelBase(**model_config.to_dict())
    
    # Load state dict - extract only sequence_model weights
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Extract sequence_model weights
        sequence_model_dict = {
            k.replace('sequence_model.', ''): v 
            for k, v in state_dict.items() 
            if k.startswith('sequence_model.')
        }
    else:
        sequence_model_dict = checkpoint
    
    # Load weights
    try:
        sequence_model_base.load_state_dict(sequence_model_dict, strict=True)
        logger.info("Successfully loaded model weights")
    except RuntimeError as e:
        logger.warning(f"Strict loading failed: {e}")
        logger.info("Attempting non-strict loading...")
        sequence_model_base.load_state_dict(sequence_model_dict, strict=False)
    
    sequence_model_base.eval()
    sequence_model_base.to(device)
    
    return sequence_model_base, model_config


def create_datamodule(dataset_path: Optional[str] = None, batch_size: int = 32):
    """
    Create a datamodule for evaluation.
    
    Parameters
    ----------
    dataset_path : str, optional
        Path to evaluation dataset. If None, will use default dataset path
    batch_size : int
        Batch size for evaluation
        
    Returns
    -------
    BouncingBallDataModule
        Configured datamodule for evaluation
    """
    logger.info("Creating evaluation datamodule")
    
    # Use default dataset path if none provided
    if dataset_path is None:
        # This should be configured via environment or config
        default_path = Path.home() / "work/data/raw/hbb_v3_2_2/hbb_dataset_v3_2_2"
        if default_path.exists():
            dataset_path = str(default_path)
        else:
            logger.warning("No dataset path provided and default path not found")
            dataset_path = None
    
    if dataset_path:
        logger.info(f"Using dataset: {dataset_path}")
        datamodule = create_eval_datamodule(dataset_path, batch_size=batch_size)
    else:
        logger.warning("Creating datamodule without preset dataset")
        datamodule = BouncingBallDataModule(
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False
        )
        datamodule.setup(stage='test')
    
    return datamodule


def extract_states(model, dataloader, device='cpu', desc="Extracting states", num_batches=None):
    """Extract all hidden and cell states from model using forward_all_states."""
    model.eval()
    
    all_hiddens = []
    all_cells = []
    all_predictions = []
    
    # Determine total batches for progress bar
    total_batches = len(dataloader) if num_batches is None else min(num_batches, len(dataloader))
    
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=desc, total=total_batches)):
                # Handle different batch formats
                if isinstance(batch, (list, tuple)):
                    if len(batch) >= 3:
                        # Format: (samples, targets, metadata)
                        samples = batch[0].to(device)
                    else:
                        # Format: (samples, targets)
                        samples = batch[0].to(device)
                else:
                    samples = batch.to(device)
                
                # Get predictions and all states using forward_all_states
                predictions, (hiddens, cells) = model.forward_all_states(samples)
                
                # hiddens and cells have shape: (num_layers, batch, timesteps, hidden_size)
                # We want to reshape to (batch, timesteps, hidden_size) for the last layer
                # Take only the last layer (index -1)
                hidden_last_layer = hiddens[-1]  # (batch, timesteps, hidden_size)
                cell_last_layer = cells[-1]      # (batch, timesteps, hidden_size)
                
                # Store results (move to CPU to save GPU memory)
                all_hiddens.append(hidden_last_layer.cpu().numpy())
                all_cells.append(cell_last_layer.cpu().numpy())
                all_predictions.append(predictions.cpu().numpy())
                
                # Check if we've processed enough batches
                if num_batches is not None and batch_idx >= num_batches - 1:
                    logger.info(f"Processed {num_batches} batches as requested")
                    break
    
    except torch.cuda.OutOfMemoryError as e:
        logger.error("\n" + "="*70)
        logger.error("CUDA out of memory error encountered!")
        logger.error(f"Attempted to process dataset with batch_size={len(batch)} (loading entire dataset)")
        logger.error("\nThe dataset is too large to process in a single batch.")
        logger.error("Please run with a smaller batch_size, e.g.:")
        logger.error("  doit extract_states_group batch_size=32")
        logger.error("  doit extract batch_size=64")
        logger.error("="*70 + "\n")
        raise RuntimeError(
            f"CUDA OOM: Dataset too large for single batch. "
            f"Use 'batch_size=32' or similar to process in smaller batches."
        ) from e
    
    # Concatenate all batches
    return {
        'hiddens': np.concatenate(all_hiddens, axis=0),
        'cells': np.concatenate(all_cells, axis=0),
        'predictions': np.concatenate(all_predictions, axis=0),
    }


def save_states(states, output_path, model_id, dataset_name='participant', metadata=None, dataset_info=None):
    """Save extracted states with metadata."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving states to {output_path}")
    
    # Import dataset configs
    from ..core.constants import DATASET_CONFIGS
    
    # Add metadata
    save_dict = {
        **states,
        'model_id': model_id,
        'dataset_name': dataset_name,
        'dataset_config': DATASET_CONFIGS.get(dataset_name, {}),
        'extraction_time': time.time(),
        'shapes': {
            'hiddens': states['hiddens'].shape,
            'cells': states['cells'].shape,
            'predictions': states['predictions'].shape,
        }
    }
    
    if metadata:
        save_dict['metadata'] = metadata
    
    # Add dataset information if available
    if dataset_info:
        if 'df_data' in dataset_info:
            # Convert DataFrame to dict for serialization
            save_dict['df_data'] = dataset_info['df_data'].to_dict() if hasattr(dataset_info['df_data'], 'to_dict') else dataset_info['df_data']
        if 'dict_metadata' in dataset_info:
            save_dict['dict_metadata'] = dataset_info['dict_metadata']
        if 'samples' in dataset_info:
            save_dict['samples'] = dataset_info['samples']
        if 'targets' in dataset_info:
            save_dict['targets'] = dataset_info['targets']
    
    # Save with compression
    np.savez_compressed(output_path, **save_dict)
    
    # Log summary
    logger.info(f"Saved states for {states['hiddens'].shape[0]} samples")
    logger.info(f"Hidden states shape: {states['hiddens'].shape}")
    logger.info(f"Cell states shape: {states['cells'].shape}")
    logger.info(f"Predictions shape: {states['predictions'].shape}")
    if dataset_info:
        logger.info("Dataset metadata included in output")


def main():
    parser = argparse.ArgumentParser(description="Extract states from trained model")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for states')
    parser.add_argument('--model-id', type=str, required=True,
                        help='Model identifier')
    parser.add_argument('--dataset-path', type=str, default=None,
                        help='Path to evaluation dataset')
    parser.add_argument('--dataset-name', type=str, default='participant',
                        choices=['participant', 'extended', 'controlled', 'velocity'],
                        help='Dataset identifier for metadata tracking')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for processing')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--num-batches', type=int, default=None,
                        help='Number of batches to process (None for all)')
    
    # Model configuration overrides
    parser.add_argument('--recurrent-size', type=int, default=None,
                        help='Override recurrent hidden size')
    parser.add_argument('--recurrent-layers', type=int, default=None,
                        help='Override number of recurrent layers')
    parser.add_argument('--config-overrides', nargs='*', default=[],
                        help='Additional config overrides in format key=value')
    
    args = parser.parse_args()
    
    try:
        # Get base model configuration
        base_config = get_model_config_for_id(args.model_id)
        
        # Apply command-line overrides
        overrides = {}
        if args.recurrent_size is not None:
            overrides['recurrent_size'] = args.recurrent_size
        if args.recurrent_layers is not None:
            overrides['recurrent_num_layers'] = args.recurrent_layers
        
        # Parse additional overrides
        if args.config_overrides:
            additional_overrides = parse_cli_overrides(args.config_overrides)
            overrides.update(additional_overrides)
        
        # Apply overrides to base configuration
        if overrides:
            logger.info(f"Applying configuration overrides: {overrides}")
            model_config = override_model_config(base_config, overrides)
        else:
            model_config = base_config
        
        # Load model
        model, final_config = load_model_from_checkpoint(
            args.checkpoint, 
            model_config=model_config,
            model_id=args.model_id,
            device=args.device
        )
        logger.info(f"Loaded model with final config: recurrent_size={final_config.recurrent_size}, "
                   f"recurrent_num_layers={final_config.recurrent_num_layers}")
        
        # Create datamodule and get test dataloader
        datamodule = create_datamodule(dataset_path=args.dataset_path, batch_size=args.batch_size)
        dataloader = datamodule.test_dataloader()
        logger.info(f"Created dataloader with {len(dataloader)} batches")
        
        # Extract states
        states = extract_states(model, dataloader, device=args.device, num_batches=args.num_batches)
        
        # Add model configuration to metadata
        metadata = {
            'model_config': final_config.to_dict(),
            'extraction_params': {
                'batch_size': args.batch_size,
                'device': args.device,
                'checkpoint': str(args.checkpoint),
                'dataset_path': args.dataset_path,
                'num_batches': args.num_batches,
            },
            'overrides_applied': overrides,
        }
        
        # Extract dataset info if available
        dataset_info = {}
        if hasattr(datamodule, 'test_dataset') and datamodule.test_dataset:
            test_dataset = datamodule.test_dataset
            if hasattr(test_dataset, 'df_data'):
                dataset_info['df_data'] = test_dataset.df_data
            if hasattr(test_dataset, 'dict_metadata'):
                dataset_info['dict_metadata'] = test_dataset.dict_metadata
            if hasattr(test_dataset, 'task'):
                dataset_info['samples'] = test_dataset.task.samples
                dataset_info['targets'] = test_dataset.task.targets
        
        # Save results
        save_states(states, args.output, args.model_id, dataset_name=args.dataset_name, 
                   metadata=metadata, dataset_info=dataset_info)
        
        logger.info("State extraction complete!")
        
    except Exception as e:
        logger.error(f"Error during state extraction: {e}")
        raise


def extract_states_with_config(
    checkpoint_path: str,
    output_path: str,
    model_id: str,
    config_overrides: Optional[Dict[str, Any]] = None,
    dataset_path: Optional[str] = None,
    dataset_name: str = 'participant',
    batch_size: int = 32,
    device: str = 'auto',
    num_batches: Optional[int] = None
) -> Dict[str, Any]:
    """
    High-level function for extracting states with configuration support.
    
    This function provides a programmatic interface to state extraction,
    making it easy to use from other parts of the pipeline.
    
    Parameters
    ----------
    checkpoint_path : str
        Path to model checkpoint
    output_path : str
        Path to save extracted states
    model_id : str
        Model identifier
    config_overrides : Dict[str, Any], optional
        Configuration overrides to apply
    dataset_path : str, optional
        Path to evaluation dataset
    batch_size : int
        Batch size for processing
    device : str
        Device to use ('auto', 'cuda', 'cpu')
    num_batches : int, optional
        Number of batches to process
        
    Returns
    -------
    Dict[str, Any]
        Metadata about the extraction process
    """
    # Handle device selection
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Get and configure model
    base_config = get_model_config_for_id(model_id)
    if config_overrides:
        model_config = override_model_config(base_config, config_overrides)
    else:
        model_config = base_config
    
    # Load model
    model, final_config = load_model_from_checkpoint(
        checkpoint_path,
        model_config=model_config,
        model_id=model_id,
        device=device
    )
    
    # Create datamodule
    datamodule = create_datamodule(dataset_path=dataset_path, batch_size=batch_size)
    dataloader = datamodule.test_dataloader()
    
    # Extract states
    states = extract_states(model, dataloader, device=device, num_batches=num_batches)
    
    # Prepare metadata
    metadata = {
        'model_config': final_config.to_dict(),
        'extraction_params': {
            'batch_size': batch_size,
            'device': device,
            'checkpoint': str(checkpoint_path),
            'dataset_path': dataset_path,
            'num_batches': num_batches,
        },
        'overrides_applied': config_overrides or {},
    }
    
    # Extract dataset info if available
    dataset_info = {}
    if hasattr(datamodule, 'test_dataset') and datamodule.test_dataset:
        test_dataset = datamodule.test_dataset
        if hasattr(test_dataset, 'df_data'):
            dataset_info['df_data'] = test_dataset.df_data
        if hasattr(test_dataset, 'dict_metadata'):
            dataset_info['dict_metadata'] = test_dataset.dict_metadata
        if hasattr(test_dataset, 'task'):
            dataset_info['samples'] = test_dataset.task.samples
            dataset_info['targets'] = test_dataset.task.targets
    
    # Save results
    save_states(states, output_path, model_id, dataset_name=dataset_name, 
               metadata=metadata, dataset_info=dataset_info)
    
    return metadata


if __name__ == '__main__':
    main()