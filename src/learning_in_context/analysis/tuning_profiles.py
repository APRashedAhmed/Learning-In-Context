"""Compute tuning profiles for critical units."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..core import StateData
from ..core.constants import TRIAL_TYPES, HAZARD_RATES, CONTINGENCIES


def compute_tuning_profiles(
    states: StateData,
    critical_units: Dict[str, Any],
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Compute tuning profiles for critical units.
    
    Args:
        states: StateData object containing neural states
        critical_units: Results from critical units analysis
        metadata: Trial metadata
        
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
    
    # Extract states for critical units only
    critical_states = states.hiddens[:, :, unit_indices]
    
    # Initialize results
    profiles = {
        "by_trial_type": {},
        "by_hazard_rate": {},
        "by_contingency": {},
        "temporal_dynamics": {}
    }
    
    # Compute mean activation by trial type
    for trial_type in TRIAL_TYPES:
        trial_mask = metadata.get("trial_types") == trial_type
        if np.any(trial_mask):
            profiles["by_trial_type"][trial_type] = {
                "mean": np.mean(critical_states[trial_mask], axis=(0, 1)),
                "std": np.std(critical_states[trial_mask], axis=(0, 1)),
                "n_trials": np.sum(trial_mask)
            }
    
    # Compute mean activation by hazard rate
    for hazard in HAZARD_RATES:
        hazard_mask = metadata.get("hazard_rates") == hazard
        if np.any(hazard_mask):
            profiles["by_hazard_rate"][hazard] = {
                "mean": np.mean(critical_states[hazard_mask], axis=(0, 1)),
                "std": np.std(critical_states[hazard_mask], axis=(0, 1)),
                "n_trials": np.sum(hazard_mask)
            }
    
    # Compute mean activation by contingency
    for contingency in CONTINGENCIES:
        cont_mask = metadata.get("contingencies") == contingency
        if np.any(cont_mask):
            profiles["by_contingency"][contingency] = {
                "mean": np.mean(critical_states[cont_mask], axis=(0, 1)),
                "std": np.std(critical_states[cont_mask], axis=(0, 1)),
                "n_trials": np.sum(cont_mask)
            }
    
    # Compute temporal dynamics (average across trials)
    temporal_mean = np.mean(critical_states, axis=0)  # (timesteps, n_units)
    temporal_std = np.std(critical_states, axis=0)
    
    profiles["temporal_dynamics"] = {
        "mean": temporal_mean.tolist(),
        "std": temporal_std.tolist(),
        "timesteps": temporal_mean.shape[0]
    }
    
    # Compute selectivity indices
    selectivity = compute_selectivity_metrics(critical_states, metadata)
    
    return {
        "unit_indices": unit_indices.tolist() if hasattr(unit_indices, 'tolist') else list(unit_indices),
        "n_units": len(unit_indices),
        "profiles": profiles,
        "selectivity": selectivity,
        "metadata": {
            "n_trials": states.hiddens.shape[0],
            "n_timesteps": states.hiddens.shape[1]
        }
    }


def compute_selectivity_metrics(
    states: np.ndarray,
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Compute selectivity metrics for units.
    
    Args:
        states: Neural states for critical units (n_trials, timesteps, n_units)
        metadata: Trial metadata
        
    Returns:
        Dictionary of selectivity metrics
    """
    selectivity = {}
    
    # Trial type selectivity
    trial_types = metadata.get("trial_types", [])
    if len(trial_types) > 0:
        unique_types = np.unique(trial_types)
        if len(unique_types) > 1:
            # Compute selectivity index for each unit
            selectivity["trial_type"] = []
            for unit_idx in range(states.shape[2]):
                unit_states = states[:, :, unit_idx]
                
                # Calculate mean response for each trial type
                means = []
                for trial_type in unique_types:
                    mask = trial_types == trial_type
                    if np.any(mask):
                        means.append(np.mean(unit_states[mask]))
                
                # Selectivity index: (max - min) / (max + min)
                if len(means) > 1 and max(means) > 0:
                    si = (max(means) - min(means)) / (max(means) + min(means))
                else:
                    si = 0.0
                
                selectivity["trial_type"].append(si)
    
    return selectivity


def main():
    """Command-line interface for tuning profiles computation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--states', type=Path, required=True, 
                       help='Path to states file')
    parser.add_argument('--units', type=Path, required=True, 
                       help='Path to critical units file')
    parser.add_argument('--metadata', type=Path, required=True,
                       help='Path to trial metadata')
    parser.add_argument('--output', type=Path, required=True, 
                       help='Output path')
    parser.add_argument('--model-id', type=str, required=True, 
                       help='Model identifier')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading states from {args.states}")
    states = StateData.load(args.states)
    
    print(f"Loading critical units from {args.units}")
    with open(args.units, 'r') as f:
        critical_units = json.load(f)
    
    print(f"Loading metadata from {args.metadata}")
    with open(args.metadata, 'r') as f:
        metadata = json.load(f)
    
    # Compute tuning profiles
    print(f"Computing tuning profiles for {args.model_id}")
    results = compute_tuning_profiles(states, critical_units, metadata)
    
    # Add model info
    results["model_id"] = args.model_id
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {args.output}")
    print(f"Analyzed {results['n_units']} critical units")


if __name__ == '__main__':
    main()