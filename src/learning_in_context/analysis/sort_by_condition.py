"""Sort trials by experimental conditions.

This module implements the sort_by_condition DAG node from the dissertation 
specification (§3.3.2 Neural Tuning Profiles - Whole Dataset Characterization).
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..core.constants import TRIAL_TYPES, HAZARD_RATES, CONTINGENCIES


def sort_by_condition(
    activity_matrices: Dict[str, Any],
    sort_conditions: List[str] = None
) -> Dict[str, Any]:
    """Sort trials by experimental conditions for organized analysis.
    
    Args:
        activity_matrices: Results from compute_activity_matrix
        sort_conditions: List of conditions to sort by ('hazard_rate', 'trial_type', 'contingency')
                        If None, sorts by all available conditions
        
    Returns:
        Dictionary containing sorted activity matrices and indices
    """
    if sort_conditions is None:
        sort_conditions = ['hazard_rate', 'trial_type', 'contingency']
    
    # Get trial metadata
    trial_metadata = activity_matrices.get('trial_metadata', {})
    if not trial_metadata:
        return {
            "error": "No trial metadata available for sorting",
            "metadata": activity_matrices.get('metadata', {})
        }
    
    # Load the main activity matrix
    main_matrix = None
    matrix_key = None
    
    # Find the main activity matrix
    activity_matrices_data = activity_matrices.get('activity_matrices', {})
    if isinstance(activity_matrices_data, dict):
        if 'full' in activity_matrices_data:
            main_matrix = activity_matrices_data['full']
            matrix_key = 'full'
    else:
        # If loaded from npz, look for matrix files
        for key in activity_matrices.keys():
            if key.startswith('matrix_full'):
                main_matrix = activity_matrices[key]
                matrix_key = key
                break
    
    if main_matrix is None:
        return {
            "error": "No main activity matrix found",
            "available_keys": list(activity_matrices.keys())
        }
    
    n_trials, n_timesteps, n_units = main_matrix.shape
    
    # Initialize results
    results = {
        "model_id": activity_matrices.get('model_id', 'unknown'),
        "metadata": {
            "matrix_source": matrix_key,
            "matrix_shape": [n_trials, n_timesteps, n_units],
            "sort_conditions": sort_conditions,
            "n_trials": n_trials,
            "n_timesteps": n_timesteps,
            "n_units": n_units
        },
        "sorted_matrices": {},
        "sort_indices": {},
        "condition_statistics": {}
    }
    
    # Process each sort condition
    for condition in sort_conditions:
        if condition == 'hazard_rate' and trial_metadata.get('hazard_rates'):
            hazard_rates = trial_metadata['hazard_rates']
            unique_hazards = sorted(set(hazard_rates))
            
            condition_matrices = {}
            condition_indices = {}
            condition_stats = {}
            
            for hazard in unique_hazards:
                # Get indices for this condition
                indices = [i for i, hr in enumerate(hazard_rates) if hr == hazard]
                condition_indices[hazard] = indices
                
                # Extract matrix for this condition
                condition_matrix = main_matrix[indices, :, :]  # (n_condition_trials, timesteps, n_units)
                condition_matrices[hazard] = condition_matrix
                
                # Compute statistics
                condition_stats[hazard] = {
                    "n_trials": len(indices),
                    "mean_activity": float(np.mean(condition_matrix)),
                    "std_activity": float(np.std(condition_matrix)),
                    "min_activity": float(np.min(condition_matrix)),
                    "max_activity": float(np.max(condition_matrix))
                }
            
            results["sorted_matrices"][condition] = condition_matrices
            results["sort_indices"][condition] = condition_indices
            results["condition_statistics"][condition] = condition_stats
        
        elif condition == 'trial_type' and trial_metadata.get('trial_types'):
            trial_types = trial_metadata['trial_types']
            unique_types = sorted(set(trial_types))
            
            condition_matrices = {}
            condition_indices = {}
            condition_stats = {}
            
            for trial_type in unique_types:
                # Get indices for this condition
                indices = [i for i, tt in enumerate(trial_types) if tt == trial_type]
                condition_indices[trial_type] = indices
                
                # Extract matrix for this condition
                condition_matrix = main_matrix[indices, :, :]
                condition_matrices[trial_type] = condition_matrix
                
                # Compute statistics
                condition_stats[trial_type] = {
                    "n_trials": len(indices),
                    "mean_activity": float(np.mean(condition_matrix)),
                    "std_activity": float(np.std(condition_matrix)),
                    "min_activity": float(np.min(condition_matrix)),
                    "max_activity": float(np.max(condition_matrix))
                }
            
            results["sorted_matrices"][condition] = condition_matrices
            results["sort_indices"][condition] = condition_indices
            results["condition_statistics"][condition] = condition_stats
        
        elif condition == 'contingency' and trial_metadata.get('contingencies'):
            contingencies = trial_metadata['contingencies']
            unique_contingencies = sorted(set(contingencies))
            
            condition_matrices = {}
            condition_indices = {}
            condition_stats = {}
            
            for contingency in unique_contingencies:
                # Get indices for this condition
                indices = [i for i, cont in enumerate(contingencies) if cont == contingency]
                condition_indices[contingency] = indices
                
                # Extract matrix for this condition
                condition_matrix = main_matrix[indices, :, :]
                condition_matrices[contingency] = condition_matrix
                
                # Compute statistics
                condition_stats[contingency] = {
                    "n_trials": len(indices),
                    "mean_activity": float(np.mean(condition_matrix)),
                    "std_activity": float(np.std(condition_matrix)),
                    "min_activity": float(np.min(condition_matrix)),
                    "max_activity": float(np.max(condition_matrix))
                }
            
            results["sorted_matrices"][condition] = condition_matrices
            results["sort_indices"][condition] = condition_indices
            results["condition_statistics"][condition] = condition_stats
    
    # Add cross-condition analysis
    if 'hazard_rate' in results["sorted_matrices"] and 'trial_type' in results["sorted_matrices"]:
        results["cross_condition_analysis"] = compute_cross_condition_analysis(
            trial_metadata, main_matrix
        )
    
    # Copy over other metadata
    if 'units_analyzed' in activity_matrices:
        results['units_analyzed'] = activity_matrices['units_analyzed']
    
    return results


def compute_cross_condition_analysis(
    trial_metadata: Dict[str, List],
    main_matrix: np.ndarray
) -> Dict[str, Any]:
    """Compute cross-condition analysis for hazard rate × trial type."""
    hazard_rates = trial_metadata.get('hazard_rates', [])
    trial_types = trial_metadata.get('trial_types', [])
    
    if not hazard_rates or not trial_types:
        return {"error": "Missing hazard rates or trial types"}
    
    cross_analysis = {}
    
    # Get unique values
    unique_hazards = sorted(set(hazard_rates))
    unique_types = sorted(set(trial_types))
    
    # Create cross-condition matrices
    for hazard in unique_hazards:
        for trial_type in unique_types:
            # Find trials matching both conditions
            indices = [
                i for i, (hr, tt) in enumerate(zip(hazard_rates, trial_types))
                if hr == hazard and tt == trial_type
            ]
            
            if indices:
                condition_key = f"{hazard}_{trial_type}"
                condition_matrix = main_matrix[indices, :, :]
                
                cross_analysis[condition_key] = {
                    "indices": indices,
                    "n_trials": len(indices),
                    "matrix_shape": list(condition_matrix.shape),
                    "mean_activity": float(np.mean(condition_matrix)),
                    "std_activity": float(np.std(condition_matrix))
                }
    
    return cross_analysis


def main():
    """Command-line interface for sort_by_condition."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrices', type=Path, required=True,
                       help='Path to activity matrices file (.npz)')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output path for sorted matrices (.npz)')
    parser.add_argument('--model-id', type=str, required=True,
                       help='Model identifier')
    parser.add_argument('--conditions', type=str, nargs='*',
                       choices=['hazard_rate', 'trial_type', 'contingency'],
                       default=['hazard_rate', 'trial_type'],
                       help='Conditions to sort by')
    
    args = parser.parse_args()
    
    # Load activity matrices
    print(f"Loading activity matrices from {args.matrices}")
    data = np.load(args.matrices, allow_pickle=True)
    
    # Reconstruct the activity_matrices dictionary
    activity_matrices = {
        'model_id': args.model_id,
        'metadata': data['metadata'].item() if 'metadata' in data else {},
        'units_analyzed': data['units_analyzed'].item() if 'units_analyzed' in data else {},
        'trial_metadata': data['trial_metadata'].item() if 'trial_metadata' in data else {},
        'activity_matrices': {},
        'summary_statistics': data['summary_statistics'].item() if 'summary_statistics' in data else {}
    }
    
    # Load matrix arrays
    for key in data.files:
        if key.startswith('matrix_'):
            matrix_type = key[7:]  # Remove 'matrix_' prefix
            activity_matrices[key] = data[key]
            if matrix_type == 'full':
                activity_matrices['activity_matrices'][matrix_type] = data[key]
    
    # Sort by conditions
    print(f"Sorting by conditions: {args.conditions}")
    results = sort_by_condition(
        activity_matrices=activity_matrices,
        sort_conditions=args.conditions
    )
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for saving
    save_data = {
        'metadata': results['metadata'],
        'sort_indices': results['sort_indices'],
        'condition_statistics': results['condition_statistics']
    }
    
    # Add units_analyzed if available
    if 'units_analyzed' in results:
        save_data['units_analyzed'] = results['units_analyzed']
    
    # Add cross-condition analysis if available
    if 'cross_condition_analysis' in results:
        save_data['cross_condition_analysis'] = results['cross_condition_analysis']
    
    # Add sorted matrices (flattened structure for npz)
    for condition, condition_matrices in results['sorted_matrices'].items():
        for condition_value, matrix in condition_matrices.items():
            key = f"sorted_{condition}_{condition_value}"
            save_data[key] = matrix
    
    np.savez_compressed(args.output, **save_data)
    
    print(f"Results saved to {args.output}")
    print(f"Sorted by conditions: {results['metadata']['sort_conditions']}")
    for condition in results['sorted_matrices']:
        n_conditions = len(results['sorted_matrices'][condition])
        print(f"  {condition}: {n_conditions} subconditions")


if __name__ == '__main__':
    main()