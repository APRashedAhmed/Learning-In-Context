"""Compute activity matrices for whole dataset characterization.

This module implements the compute_activity_matrix DAG node from the dissertation 
specification (§3.3.2 Neural Tuning Profiles - Whole Dataset Characterization).
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.constants import TRIAL_TYPES, HAZARD_RATES, CONTINGENCIES


def compute_activity_matrix(
    unit_activities: Dict[str, Any],
    use_windowed: bool = True,
    use_normalized: bool = True
) -> Dict[str, Any]:
    """Compute trial × time activity matrices for visualization and analysis.
    
    Args:
        unit_activities: Results from extract_unit_activities
        use_windowed: Whether to use windowed activities (if available)
        use_normalized: Whether to use normalized activities (if available)
        
    Returns:
        Dictionary containing activity matrices and metadata
    """
    # Determine which activities to use
    activity_key = None
    activities = unit_activities.get('activities', {})
    
    # Priority order for selecting activities
    if use_windowed and use_normalized and 'windowed_normalized' in activities:
        activity_key = 'windowed_normalized'
    elif use_windowed and 'windowed_raw' in activities:
        activity_key = 'windowed_raw'
    elif use_normalized and 'full_normalized' in activities:
        activity_key = 'full_normalized'
    elif 'full_raw' in activities:
        activity_key = 'full_raw'
    else:
        return {
            "error": "No suitable activity data found",
            "available_keys": list(activities.keys())
        }
    
    # Load the selected activity matrix
    activity_matrix = activities[activity_key]  # Shape: (n_trials, timesteps, n_units)
    n_trials, n_timesteps, n_units = activity_matrix.shape
    
    # Get trial metadata
    trial_metadata = unit_activities.get('trial_metadata', {})
    units_analyzed = unit_activities.get('units_analyzed', {})
    
    # Create results structure
    results = {
        "model_id": unit_activities.get('model_id', 'unknown'),
        "metadata": {
            "activity_source": activity_key,
            "matrix_shape": [n_trials, n_timesteps, n_units],
            "n_trials": n_trials,
            "n_timesteps": n_timesteps,
            "n_units": n_units,
            "use_windowed": use_windowed,
            "use_normalized": use_normalized
        },
        "units_analyzed": units_analyzed,
        "activity_matrices": {},
        "summary_statistics": {}
    }
    
    # Store the main activity matrix
    results["activity_matrices"]["full"] = activity_matrix
    
    # Compute per-unit activity matrices (trials × time for each unit)
    unit_matrices = {}
    for unit_idx in range(n_units):
        unit_matrices[f"unit_{unit_idx}"] = activity_matrix[:, :, unit_idx]  # (n_trials, n_timesteps)
    
    results["activity_matrices"]["per_unit"] = unit_matrices
    
    # Compute summary statistics across the full matrix
    results["summary_statistics"]["overall"] = {
        "mean": float(np.mean(activity_matrix)),
        "std": float(np.std(activity_matrix)),
        "min": float(np.min(activity_matrix)),
        "max": float(np.max(activity_matrix))
    }
    
    # Compute per-unit statistics (across trials and time)
    unit_stats = {}
    for unit_idx in range(n_units):
        unit_data = activity_matrix[:, :, unit_idx]
        unit_stats[f"unit_{unit_idx}"] = {
            "mean": float(np.mean(unit_data)),
            "std": float(np.std(unit_data)),
            "min": float(np.min(unit_data)),
            "max": float(np.max(unit_data))
        }
    
    results["summary_statistics"]["per_unit"] = unit_stats
    
    # Compute per-trial statistics (across time and units)
    trial_stats = {}
    for trial_idx in range(n_trials):
        trial_data = activity_matrix[trial_idx, :, :]
        trial_stats[f"trial_{trial_idx}"] = {
            "mean": float(np.mean(trial_data)),
            "std": float(np.std(trial_data)),
            "min": float(np.min(trial_data)),
            "max": float(np.max(trial_data))
        }
    
    results["summary_statistics"]["per_trial"] = trial_stats
    
    # Compute temporal statistics (across trials and units for each timestep)
    temporal_stats = {}
    for t in range(n_timesteps):
        timestep_data = activity_matrix[:, t, :]
        temporal_stats[f"timestep_{t}"] = {
            "mean": float(np.mean(timestep_data)),
            "std": float(np.std(timestep_data)),
            "min": float(np.min(timestep_data)),
            "max": float(np.max(timestep_data))
        }
    
    results["summary_statistics"]["temporal"] = temporal_stats
    
    # Add trial metadata for downstream sorting
    if trial_metadata:
        results["trial_metadata"] = trial_metadata
        
        # Compute condition-specific matrix sizes for validation
        condition_counts = {}
        if trial_metadata.get('hazard_rates'):
            for hz in set(trial_metadata['hazard_rates']):
                condition_counts[f"hazard_{hz}"] = trial_metadata['hazard_rates'].count(hz)
        
        if trial_metadata.get('trial_types'):
            for tt in set(trial_metadata['trial_types']):
                condition_counts[f"trial_type_{tt}"] = trial_metadata['trial_types'].count(tt)
        
        results["metadata"]["condition_counts"] = condition_counts
    
    return results


def main():
    """Command-line interface for compute_activity_matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--activities', type=Path, required=True,
                       help='Path to unit activities file (.npz)')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output path for activity matrices (.npz)')
    parser.add_argument('--model-id', type=str, required=True,
                       help='Model identifier')
    parser.add_argument('--use-windowed', action='store_true', default=True,
                       help='Prefer windowed activities if available')
    parser.add_argument('--use-normalized', action='store_true', default=True,
                       help='Prefer normalized activities if available')
    parser.add_argument('--no-windowed', action='store_true',
                       help='Do not use windowed activities')
    parser.add_argument('--no-normalized', action='store_true',
                       help='Do not use normalized activities')
    
    args = parser.parse_args()
    
    # Handle argument conflicts
    use_windowed = args.use_windowed and not args.no_windowed
    use_normalized = args.use_normalized and not args.no_normalized
    
    # Load unit activities
    print(f"Loading unit activities from {args.activities}")
    data = np.load(args.activities, allow_pickle=True)
    
    # Reconstruct the unit_activities dictionary
    unit_activities = {
        'model_id': args.model_id,
        'metadata': data['metadata'].item() if 'metadata' in data else {},
        'units_analyzed': data['units_analyzed'].item() if 'units_analyzed' in data else {},
        'trial_metadata': data['trial_metadata'].item() if 'trial_metadata' in data else {},
        'activities': {}
    }
    
    # Load activity arrays
    for key in data.files:
        if key.startswith('activities_'):
            activity_type = key[11:]  # Remove 'activities_' prefix
            unit_activities['activities'][activity_type] = data[key]
    
    # Compute activity matrices
    print(f"Computing activity matrices for {args.model_id}")
    results = compute_activity_matrix(
        unit_activities=unit_activities,
        use_windowed=use_windowed,
        use_normalized=use_normalized
    )
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for saving
    save_data = {
        'metadata': results['metadata'],
        'units_analyzed': results['units_analyzed'],
        'summary_statistics': results['summary_statistics']
    }
    
    # Add trial metadata if available
    if 'trial_metadata' in results:
        save_data['trial_metadata'] = results['trial_metadata']
    
    # Add activity matrices
    for key, matrix in results['activity_matrices'].items():
        if key == 'per_unit':
            # Save per-unit matrices as separate arrays
            for unit_key, unit_matrix in matrix.items():
                save_data[f'matrix_{unit_key}'] = unit_matrix
        else:
            save_data[f'matrix_{key}'] = matrix
    
    np.savez_compressed(args.output, **save_data)
    
    print(f"Results saved to {args.output}")
    print(f"Activity matrix shape: {results['metadata']['matrix_shape']}")
    print(f"Activity source: {results['metadata']['activity_source']}")


if __name__ == '__main__':
    main()