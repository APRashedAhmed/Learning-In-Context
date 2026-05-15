"""Helper functions for neural tuning analysis."""

import numpy as np
from scipy import stats
from typing import List, Dict, Tuple, Optional
import pandas as pd


def window_samples(
    samples: np.ndarray,
    endpoints: np.ndarray,
    N: int,
) -> np.ndarray:
    """Extract last N timesteps from each trial.
    
    Args:
        samples: Array of shape (n_trials, timesteps, features)
        endpoints: Array of trial lengths
        N: Window size
        
    Returns:
        Windowed samples of shape (n_trials, N, features)
    """
    b, T, f = samples.shape
    # create a (b, N) array of time‑indices for each batch
    t_idx = endpoints[:, None] + np.arange(-N, 0)  # shape (b, N)
    # defensive clip if you might hit negative
    # t_idx = np.clip(t_idx, 0, T-1)
    
    # create matching batch indices
    b_idx = np.arange(b)[:, None]                 # shape (b, 1)
    
    # fancy‑index into samples
    return samples[b_idx, t_idx, :]               # -> (b, N, f)


def concatenate_states(
    hiddens: np.ndarray, 
    cells: np.ndarray, 
    unit_indices: List[int],
    num_hidden_units: int
) -> np.ndarray:
    """Concatenate selected hidden and cell states.
    
    Args:
        hiddens: Hidden states array (n_trials, timesteps, n_hidden_units)
        cells: Cell states array (n_trials, timesteps, n_cell_units)
        unit_indices: Global indices of units to extract
        num_hidden_units: Number of hidden units (for indexing)
        
    Returns:
        Concatenated states for selected units (n_trials, timesteps, n_selected_units)
    """
    selected_states = []
    
    for idx in unit_indices:
        if idx < num_hidden_units:
            # This is a hidden unit
            selected_states.append(hiddens[:, :, idx])
        else:
            # This is a cell unit
            cell_idx = idx - num_hidden_units
            selected_states.append(cells[:, :, cell_idx])
    
    # Stack along the last dimension
    return np.stack(selected_states, axis=-1)


def zscore_normalize(states: np.ndarray, ddof: int = 1) -> np.ndarray:
    """Apply z-score normalization matching hmdcpd implementation.
    
    Args:
        states: States array (n_trials, timesteps, n_units)
        ddof: Degrees of freedom for standard deviation calculation
        
    Returns:
        Z-scored states with mean=0, std=1 across trials and timesteps
    """
    # Normalize across both trials (axis=0) and timesteps (axis=1)
    return stats.zscore(states, axis=(0, 1), ddof=ddof)


def extract_unit_states(
    hiddens: np.ndarray,
    cells: np.ndarray,
    unit_indices: List[int],
    num_hidden_units: int,
    trial_lengths: Optional[np.ndarray] = None,
    window_size: Optional[int] = None,
    normalize: bool = True
) -> Dict[str, np.ndarray]:
    """Extract and process states for critical units.
    
    Args:
        hiddens: Hidden states (n_trials, timesteps, n_hidden_units)
        cells: Cell states (n_trials, timesteps, n_cell_units)
        unit_indices: Global indices of units to extract
        num_hidden_units: Number of hidden units
        trial_lengths: Array of trial lengths for windowing
        window_size: Size of window (if None, use full trials)
        normalize: Whether to apply z-score normalization
        
    Returns:
        Dictionary with processed states and metadata
    """
    # First concatenate the selected units
    states = concatenate_states(hiddens, cells, unit_indices, num_hidden_units)
    
    result = {
        'full_states': states.copy(),
        'unit_indices': unit_indices,
        'num_hidden_units': num_hidden_units
    }
    
    # Window if requested
    if window_size is not None and trial_lengths is not None:
        windowed_states = window_samples(states, trial_lengths, window_size)
        result['windowed_states'] = windowed_states
        
        # Normalize windowed states if requested
        if normalize:
            result['windowed_states_normalized'] = zscore_normalize(windowed_states)
    
    # Normalize full states if requested
    if normalize:
        result['full_states_normalized'] = zscore_normalize(states)
    
    return result


def create_unit_mapping(unit_indices: List[int], num_hidden_units: int) -> Dict[int, Dict[str, str]]:
    """Create mapping of unit indices to names and types.
    
    Args:
        unit_indices: Global indices of units
        num_hidden_units: Number of hidden units
        
    Returns:
        Dictionary mapping index to {name, type}
    """
    mapping = {}
    
    for idx in unit_indices:
        if idx < num_hidden_units:
            mapping[idx] = {
                'name': f'Hidden {idx}',
                'type': 'hidden',
                'local_idx': idx
            }
        else:
            local_idx = idx - num_hidden_units
            mapping[idx] = {
                'name': f'Cell {local_idx}',
                'type': 'cell',
                'local_idx': local_idx
            }
    
    return mapping


def prepare_visualization_vectors(
    states: np.ndarray,
    metadata: pd.DataFrame,
    window_size: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """Prepare data vectors for visualization.
    
    Args:
        states: States array (n_trials, timesteps, n_units)
        metadata: Trial metadata DataFrame
        window_size: Window size (for creating appropriate indices)
        
    Returns:
        Dictionary with various data vectors for plotting
    """
    n_trials, timesteps, n_units = states.shape
    
    # If windowed, timesteps is the window size
    if window_size is not None:
        timesteps = window_size
    
    # Create trial indices
    trial_indices = np.repeat(np.arange(n_trials), timesteps)
    
    # Create timestep indices  
    timestep_indices = np.tile(np.arange(timesteps), n_trials)
    
    # Create hazard rate binary (0 for Low, 1 for High)
    hazard_binary = metadata['Hazard Rate'].apply(
        lambda x: 0 if x == 'Low' else 1
    ).values
    hazard_binary_repeated = np.repeat(hazard_binary, timesteps)
    
    # Get effective hazard rate if available
    vectors = {
        'trial_indices': trial_indices,
        'timestep_indices': timestep_indices,
        'hazard_binary': hazard_binary_repeated,
    }
    
    if 'PCCNVC_effective' in metadata.columns:
        hazard_continuous = metadata['PCCNVC_effective'].values
        vectors['hazard_continuous'] = np.repeat(hazard_continuous, timesteps)
    
    # Add trial type information
    if 'trial' in metadata.columns:
        trial_types = metadata['trial'].values
        vectors['trial_types'] = np.repeat(trial_types, timesteps)
    
    return vectors


def create_trial_groupings(
    metadata: pd.DataFrame,
    group_by: str = 'hazard_rate'
) -> Dict[str, List[int]]:
    """Create trial groupings for visualization.
    
    Args:
        metadata: Trial metadata
        group_by: Grouping strategy ('hazard_rate', 'trial_type', etc.)
        
    Returns:
        Dictionary mapping group names to trial indices
    """
    groupings = {}
    
    if group_by == 'hazard_rate':
        for hz in ['Low', 'High']:
            mask = metadata['Hazard Rate'] == hz
            groupings[hz] = metadata[mask].index.tolist()
            
    elif group_by == 'trial_type':
        for trial_type in metadata['trial'].unique():
            mask = metadata['trial'] == trial_type
            groupings[trial_type] = metadata[mask].index.tolist()
            
    elif group_by == 'hazard_and_length':
        # Group by hazard rate and sort by length within each group
        for hz in ['Low', 'High']:
            mask = metadata['Hazard Rate'] == hz
            subset = metadata[mask].sort_values('length', ascending=False)
            groupings[f'{hz}_sorted'] = subset.index.tolist()
    
    return groupings