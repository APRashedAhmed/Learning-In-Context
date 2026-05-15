"""State normalization utilities for extracted neural network states."""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


def compute_zscore_stats(
    states: np.ndarray,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and standard deviation for z-score normalization.
    
    Args:
        states: Array of states to compute statistics for
        axis: Axis or axes along which to compute statistics
        keepdims: Whether to keep dimensions for broadcasting
        
    Returns:
        Tuple of (mean, std) arrays
    """
    mean = np.mean(states, axis=axis, keepdims=keepdims)
    std = np.std(states, axis=axis, keepdims=keepdims)
    # Avoid division by zero
    std = np.where(std == 0, 1.0, std)
    return mean, std


def zscore_normalize(
    states: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Apply z-score normalization to states.
    
    Args:
        states: Array of states to normalize
        mean: Pre-computed mean (if None, will compute)
        std: Pre-computed standard deviation (if None, will compute)
        axis: Axis or axes along which to normalize
        mask: Boolean mask indicating valid timesteps (True = valid, False = padded)
        
    Returns:
        Tuple of (normalized_states, stats_dict)
    """
    if mean is None or std is None:
        if mask is not None:
            # Compute stats only on valid timesteps
            masked_states = np.where(mask[..., np.newaxis], states, np.nan)
            mean = np.nanmean(masked_states, axis=axis, keepdims=True)
            std = np.nanstd(masked_states, axis=axis, keepdims=True)
            # Replace NaN with 1.0 to avoid division by zero
            std = np.where(np.isnan(std) | (std == 0), 1.0, std)
        else:
            mean, std = compute_zscore_stats(states, axis=axis, keepdims=True)
    
    normalized = (states - mean) / std
    
    stats = {
        'mean': mean,
        'std': std,
        'normalization_axis': axis,
        'normalization_method': 'zscore'
    }
    
    return normalized, stats


def normalize_states(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    method: str = 'zscore',
    axis: Optional[Union[int, Tuple[int, ...]]] = 1,  # Default: normalize across time
    per_unit: bool = False,
    padding_value: float = -100.0
) -> None:
    """Normalize extracted states and save to file.
    
    Args:
        input_path: Path to raw states file (.npz)
        output_path: Path to save normalized states (.npz)
        method: Normalization method ('zscore', 'minmax', or 'none')
        axis: Axis or axes along which to normalize (1 = time dimension)
        per_unit: If True, normalize each unit separately
        padding_value: Value to set for padded timesteps after normalization
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    logger.info(f"Loading states from {input_path}")
    data = np.load(input_path, allow_pickle=True)
    
    # Extract states
    hiddens = data['hiddens']
    cells = data['cells']
    
    logger.info(f"State shapes - Hidden: {hiddens.shape}, Cell: {cells.shape}")
    
    # Extract trial lengths for masking
    mask = None
    if 'df_data' in data:
        try:
            import pandas as pd
            df_data = pd.DataFrame(data['df_data'].item())
            if 'length' in df_data.columns:
                trial_lengths = df_data['length'].values
                n_trials, max_timesteps, _ = hiddens.shape
                
                # Create mask: True for valid timesteps, False for padding
                mask = np.zeros((n_trials, max_timesteps), dtype=bool)
                for i, length in enumerate(trial_lengths):
                    mask[i, :length] = True
                
                logger.info(f"Created mask from trial lengths: {trial_lengths.shape[0]} trials, "
                           f"lengths range {trial_lengths.min()}-{trial_lengths.max()}")
        except Exception as e:
            logger.warning(f"Could not extract trial lengths from df_data: {e}")
            mask = None
    
    if method == 'none':
        logger.info("Skipping normalization (method='none')")
        normalized_hiddens = hiddens
        normalized_cells = cells
        hidden_stats = {'normalization_method': 'none'}
        cell_stats = {'normalization_method': 'none'}
    
    elif method == 'zscore':
        # For per-unit normalization, normalize across trials and time
        norm_axis = (0, 1)  # Average over trials and time
        
        logger.info(f"Applying z-score normalization per unit (axis={norm_axis})")
        
        # Normalize hidden states
        normalized_hiddens, hidden_stats = zscore_normalize(hiddens, axis=norm_axis, mask=mask)
        
        # Normalize cell states
        normalized_cells, cell_stats = zscore_normalize(cells, axis=norm_axis, mask=mask)
        
        # Set padded values to padding_value
        if mask is not None:
            # Expand mask to match state dimensions
            mask_expanded = mask[..., np.newaxis]
            normalized_hiddens = np.where(mask_expanded, normalized_hiddens, padding_value)
            normalized_cells = np.where(mask_expanded, normalized_cells, padding_value)
            logger.info(f"Set padded timesteps to {padding_value}")
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Prepare output data
    output_data = {
        'hiddens': normalized_hiddens,
        'cells': normalized_cells,
        'predictions': data['predictions'],  # Keep predictions unchanged
        'model_id': data['model_id'],
        'metadata': data.get('metadata', {}),
        'normalization_info': {
            'method': method,
            'axis': norm_axis if method != 'none' else axis,
            'per_unit': per_unit,
            'padding_value': padding_value,
            'padding_handled': mask is not None,
            'hidden_stats': hidden_stats,
            'cell_stats': cell_stats,
            'source_file': str(input_path)
        }
    }
    
    # Copy over other metadata if present
    for key in ['df_data', 'dict_metadata', 'samples', 'targets']:
        if key in data:
            output_data[key] = data[key]
    
    # Save normalized states
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **output_data)
    logger.info(f"Saved normalized states to {output_path}")


def main():
    """CLI interface for state normalization."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Normalize extracted neural network states")
    parser.add_argument("input_path", type=str, help="Path to raw states file (.npz)")
    parser.add_argument("output_path", type=str, help="Path to save normalized states (.npz)")
    parser.add_argument("--method", type=str, default="zscore", 
                       choices=["zscore", "minmax", "none"],
                       help="Normalization method")
    parser.add_argument("--axis", type=int, default=1,
                       help="Axis along which to normalize (1 = time dimension)")
    parser.add_argument("--per-unit", action="store_true", default=True,
                       help="Normalize each unit separately (default: True)")
    parser.add_argument("--no-per-unit", dest="per_unit", action="store_false",
                       help="Disable per-unit normalization")
    parser.add_argument("--padding-value", type=float, default=-100.0,
                       help="Value to set for padded timesteps after normalization (default: -100)")
    
    args = parser.parse_args()
    
    normalize_states(
        args.input_path,
        args.output_path,
        method=args.method,
        axis=args.axis,
        per_unit=args.per_unit,
        padding_value=args.padding_value
    )


if __name__ == "__main__":
    main()