"""Enhanced tuning profiles computation with windowing and normalization."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

from . import tuning_utils
from ..core import StateData
from ..core.constants import TRIAL_TYPES, HAZARD_RATES, CONTINGENCIES


def compute_tuning_profiles(
    states: StateData,
    critical_units: Dict[str, Any],
    metadata: Dict[str, Any],
    df_data: Optional[pd.DataFrame] = None,
    window_size: int = 200,
    normalize: bool = True,
    compute_full_trial: bool = True,
    compute_windowed: bool = True
) -> Dict[str, Any]:
    """Compute tuning profiles with windowing and normalization.
    
    Args:
        states: StateData object containing neural states
        critical_units: Results from critical units analysis
        metadata: Trial metadata dictionary
        df_data: DataFrame with trial metadata (if available)
        window_size: Size of window for last-N analysis
        normalize: Whether to apply z-score normalization
        compute_full_trial: Include full trial analysis
        compute_windowed: Include windowed analysis
        
    Returns:
        Dictionary containing tuning profiles and statistics
    """
    # Get critical unit indices
    unit_indices = critical_units.get("unit_indices", [])
    
    if len(unit_indices) == 0:
        return {
            "message": "No critical units found",
            "unit_indices": [],
            "profiles": {}
        }
    
    # Get model dimensions
    n_trials, n_timesteps, n_hidden = states.hiddens.shape
    n_cell = states.cells.shape[2] if hasattr(states, 'cells') and states.cells is not None else 0
    
    # Extract trial lengths from metadata or df_data
    trial_lengths = None
    if df_data is not None and 'length' in df_data.columns:
        trial_lengths = df_data['length'].values
    elif 'trial_lengths' in metadata:
        trial_lengths = np.array(metadata['trial_lengths'])
    
    # Extract and process states
    extracted_states = tuning_utils.extract_unit_states(
        hiddens=states.hiddens,
        cells=states.cells if hasattr(states, 'cells') and states.cells is not None else np.zeros((n_trials, n_timesteps, 0)),
        unit_indices=unit_indices,
        num_hidden_units=n_hidden,
        trial_lengths=trial_lengths,
        window_size=window_size if compute_windowed else None,
        normalize=normalize
    )
    
    # Create unit mapping
    unit_mapping = tuning_utils.create_unit_mapping(unit_indices, n_hidden)
    
    # Initialize results
    results = {
        "model_id": critical_units.get("model_id", "unknown"),
        "metadata": {
            "window_size": window_size,
            "normalization_applied": normalize,
            "n_units_analyzed": len(unit_indices),
            "analysis_timestamp": datetime.now().isoformat(),
            "n_trials": n_trials,
            "n_timesteps": n_timesteps,
            "n_hidden_units": n_hidden,
            "n_cell_units": n_cell
        },
        "units_analyzed": {
            "indices": unit_indices,
            "mapping": unit_mapping
        },
        "normalized_states": {},
        "profiles": {},
        "condition_statistics": {}
    }
    
    # Store normalized states if requested
    if normalize:
        if compute_full_trial and 'full_states_normalized' in extracted_states:
            results["normalized_states"]["full_trial"] = {
                "shape": extracted_states['full_states_normalized'].shape,
                "mean": float(np.mean(extracted_states['full_states_normalized'])),
                "std": float(np.std(extracted_states['full_states_normalized']))
            }
        
        if compute_windowed and 'windowed_states_normalized' in extracted_states:
            results["normalized_states"]["windowed"] = {
                "shape": extracted_states['windowed_states_normalized'].shape,
                "mean": float(np.mean(extracted_states['windowed_states_normalized'])),
                "std": float(np.std(extracted_states['windowed_states_normalized']))
            }
    
    # Compute condition statistics
    if df_data is not None:
        results["condition_statistics"] = compute_condition_statistics(
            extracted_states, df_data, compute_windowed, window_size
        )
        
        # Prepare visualization data
        results["visualization_data"] = prepare_visualization_data(
            extracted_states, df_data, compute_windowed, window_size
        )
    
    # Compute traditional profiles for compatibility
    states_for_profiles = extracted_states.get('windowed_states_normalized', 
                                             extracted_states.get('full_states_normalized',
                                                                extracted_states['full_states']))
    
    results["profiles"] = compute_traditional_profiles(
        states_for_profiles, metadata, df_data
    )
    
    return results


def compute_condition_statistics(
    extracted_states: Dict[str, np.ndarray],
    df_data: pd.DataFrame,
    compute_windowed: bool,
    window_size: int
) -> Dict[str, Any]:
    """Compute statistics by experimental conditions."""
    stats = {}
    
    # Choose which states to analyze
    if compute_windowed and 'windowed_states_normalized' in extracted_states:
        states = extracted_states['windowed_states_normalized']
    else:
        states = extracted_states.get('full_states_normalized', extracted_states['full_states'])
    
    # Hazard rate statistics
    if 'Hazard Rate' in df_data.columns:
        stats["hazard_rate"] = {}
        for hz in df_data['Hazard Rate'].unique():
            mask = df_data['Hazard Rate'] == hz
            if np.any(mask):
                hz_states = states[mask]
                stats["hazard_rate"][hz.lower()] = {
                    "mean": np.mean(hz_states, axis=(0, 1)).tolist(),
                    "std": np.std(hz_states, axis=(0, 1)).tolist(),
                    "n_trials": int(np.sum(mask))
                }
    
    # Trial type statistics
    if 'trial' in df_data.columns:
        stats["trial_type"] = {}
        for trial_type in df_data['trial'].unique():
            mask = df_data['trial'] == trial_type
            if np.any(mask):
                trial_states = states[mask]
                stats["trial_type"][trial_type] = {
                    "mean": np.mean(trial_states, axis=(0, 1)).tolist(),
                    "std": np.std(trial_states, axis=(0, 1)).tolist(),
                    "n_trials": int(np.sum(mask))
                }
    
    return stats


def prepare_visualization_data(
    extracted_states: Dict[str, np.ndarray],
    df_data: pd.DataFrame,
    compute_windowed: bool,
    window_size: int
) -> Dict[str, Any]:
    """Prepare data for visualization."""
    viz_data = {}
    
    # Choose which states to visualize
    if compute_windowed and 'windowed_states_normalized' in extracted_states:
        states = extracted_states['windowed_states_normalized']
        current_window_size = window_size
    else:
        states = extracted_states.get('full_states_normalized', extracted_states['full_states'])
        current_window_size = None
    
    # Create visualization vectors
    vectors = tuning_utils.prepare_visualization_vectors(
        states, df_data, current_window_size
    )
    
    # Trial activity data
    viz_data["trial_activity"] = {
        "by_hazard_rate": tuning_utils.create_trial_groupings(df_data, 'hazard_rate'),
        "by_trial_type": tuning_utils.create_trial_groupings(df_data, 'trial_type'),
        "by_hazard_and_length": tuning_utils.create_trial_groupings(df_data, 'hazard_and_length')
    }
    
    # Scatter plot data
    n_units = states.shape[2]
    if n_units >= 2:
        # Create unit pairs for scatter plots
        unit_pairs = []
        for i in range(0, n_units-1, 2):
            if i+1 < n_units:
                unit_pairs.append([i, i+1])
        
        viz_data["scatter_plots"] = {
            "unit_pairs": unit_pairs,
            "vectors": vectors
        }
    
    return viz_data


def compute_traditional_profiles(
    states: np.ndarray,
    metadata: Dict[str, Any],
    df_data: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """Compute traditional tuning profiles for backward compatibility."""
    profiles = {
        "by_trial_type": {},
        "by_hazard_rate": {},
        "temporal_dynamics": {}
    }
    
    # Use df_data if available, otherwise fall back to metadata
    if df_data is not None:
        # By trial type
        if 'trial' in df_data.columns:
            for trial_type in df_data['trial'].unique():
                mask = df_data['trial'] == trial_type
                if np.any(mask):
                    profiles["by_trial_type"][trial_type] = {
                        "mean": np.mean(states[mask], axis=(0, 1)).tolist(),
                        "std": np.std(states[mask], axis=(0, 1)).tolist(),
                        "n_trials": int(np.sum(mask))
                    }
        
        # By hazard rate
        if 'Hazard Rate' in df_data.columns:
            for hazard in df_data['Hazard Rate'].unique():
                mask = df_data['Hazard Rate'] == hazard
                if np.any(mask):
                    profiles["by_hazard_rate"][hazard] = {
                        "mean": np.mean(states[mask], axis=(0, 1)).tolist(),
                        "std": np.std(states[mask], axis=(0, 1)).tolist(),
                        "n_trials": int(np.sum(mask))
                    }
    else:
        # Fall back to metadata arrays
        trial_types = metadata.get("trial_types", [])
        hazard_rates = metadata.get("hazard_rates", [])
        
        if len(trial_types) > 0:
            for trial_type in TRIAL_TYPES:
                mask = np.array(trial_types) == trial_type
                if np.any(mask):
                    profiles["by_trial_type"][trial_type] = {
                        "mean": np.mean(states[mask], axis=(0, 1)).tolist(),
                        "std": np.std(states[mask], axis=(0, 1)).tolist(),
                        "n_trials": int(np.sum(mask))
                    }
        
        if len(hazard_rates) > 0:
            for hazard in HAZARD_RATES:
                mask = np.array(hazard_rates) == hazard
                if np.any(mask):
                    profiles["by_hazard_rate"][hazard] = {
                        "mean": np.mean(states[mask], axis=(0, 1)).tolist(),
                        "std": np.std(states[mask], axis=(0, 1)).tolist(),
                        "n_trials": int(np.sum(mask))
                    }
    
    # Temporal dynamics
    profiles["temporal_dynamics"] = {
        "mean": np.mean(states, axis=0).tolist(),
        "std": np.std(states, axis=0).tolist(),
        "timesteps": states.shape[1]
    }
    
    return profiles


def main():
    """Command-line interface for enhanced tuning profiles computation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--states', type=Path, required=True, 
                       help='Path to states file (.npz)')
    parser.add_argument('--units', type=Path, required=True, 
                       help='Path to critical units file')
    parser.add_argument('--output', type=Path, required=True, 
                       help='Output path')
    parser.add_argument('--model-id', type=str, required=True, 
                       help='Model identifier')
    parser.add_argument('--window-size', type=int, default=200,
                       help='Window size for last-N analysis')
    parser.add_argument('--normalize', action='store_true',
                       help='Apply z-score normalization')
    parser.add_argument('--no-windowed', action='store_true',
                       help='Skip windowed analysis')
    parser.add_argument('--no-full-trial', action='store_true',
                       help='Skip full trial analysis')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading states from {args.states}")
    
    # Try to load as npz first (from extract_states output)
    data = np.load(args.states)
    
    # Check if this is from extract_states (has hiddens/cells)
    if 'hiddens' in data and 'cells' in data:
        # Create StateData from arrays
        hiddens = data['hiddens']
        cells = data['cells']
        
        # Only use first layer as per hmdcpd analysis
        if len(hiddens.shape) == 4:  # (layers, trials, timesteps, units)
            hiddens = hiddens[0]
            cells = cells[0] if cells.shape[0] > 0 else cells
        
        states = StateData(hiddens=hiddens, cells=cells)
        
        # Load metadata if available
        df_data = None
        if 'df_data' in data:
            # If df_data was saved as structured array, convert to DataFrame
            df_data_array = data['df_data']
            if hasattr(df_data_array, 'dtype') and df_data_array.dtype.names:
                df_data = pd.DataFrame(df_data_array)
            else:
                # Try to reconstruct from dict_metadata
                if 'dict_metadata' in data:
                    dict_metadata = data['dict_metadata'].item() if data['dict_metadata'].shape == () else data['dict_metadata']
                    # Create minimal df_data
                    df_data = pd.DataFrame()
                    if 'hazard_rates' in dict_metadata:
                        df_data['Hazard Rate'] = dict_metadata['hazard_rates']
                    if 'trial_types' in dict_metadata:
                        df_data['trial'] = dict_metadata['trial_types']
                    if 'trial_lengths' in dict_metadata:
                        df_data['length'] = dict_metadata['trial_lengths']
        
        metadata = {}
        if 'dict_metadata' in data:
            metadata = data['dict_metadata'].item() if data['dict_metadata'].shape == () else data['dict_metadata']
    else:
        # Fall back to loading as StateData
        states = StateData.load(args.states)
        df_data = None
        metadata = {}
    
    print(f"Loading critical units from {args.units}")
    with open(args.units, 'r') as f:
        critical_units = json.load(f)
    
    # Compute tuning profiles
    print(f"Computing tuning profiles for {args.model_id}")
    results = compute_tuning_profiles(
        states=states,
        critical_units=critical_units,
        metadata=metadata,
        df_data=df_data,
        window_size=args.window_size,
        normalize=args.normalize,
        compute_full_trial=not args.no_full_trial,
        compute_windowed=not args.no_windowed
    )
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {args.output}")
    print(f"Analyzed {results['metadata']['n_units_analyzed']} critical units")
    if args.normalize:
        print(f"Applied z-score normalization")
    if not args.no_windowed:
        print(f"Computed windowed analysis (window_size={args.window_size})")


if __name__ == '__main__':
    main()