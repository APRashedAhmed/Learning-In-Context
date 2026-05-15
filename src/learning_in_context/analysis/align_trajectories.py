"""Align trajectories across variants for controlled stimulus analysis.

This module implements the align_trajectories DAG node from the dissertation 
specification (§3.3.2 Neural Tuning Profiles - Controlled Stimulus Analysis).
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.constants import TRIAL_TYPES, HAZARD_RATES, CONTINGENCIES


def align_trajectories(
    unit_activities: Dict[str, Any],
    controlled_dataset_metadata: Optional[Dict[str, Any]] = None,
    alignment_method: str = "time",
    reference_events: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Align trajectories across variants for controlled stimulus analysis.
    
    This implements the controlled stimulus analysis from §3.3.2, where each base 
    trajectory has 7 variants (no change, random low/high, contingent low/med/high)
    that differ only in color change dynamics.
    
    Args:
        unit_activities: Results from extract_unit_activities
        controlled_dataset_metadata: Metadata about controlled dataset structure
        alignment_method: Method for alignment ('time', 'event', 'phase')
        reference_events: List of events to align on (if event-based alignment)
        
    Returns:
        Dictionary containing aligned trajectories and analysis
    """
    # Get activity data
    activities = unit_activities.get('activities', {})
    trial_metadata = unit_activities.get('trial_metadata', {})
    
    # Determine which activities to use (prefer normalized)
    activity_key = None
    if 'windowed_normalized' in activities:
        activity_key = 'windowed_normalized'
        activity_matrix = activities[activity_key]
    elif 'full_normalized' in activities:
        activity_key = 'full_normalized'
        activity_matrix = activities[activity_key]
    elif 'windowed_raw' in activities:
        activity_key = 'windowed_raw'
        activity_matrix = activities[activity_key]
    elif 'full_raw' in activities:
        activity_key = 'full_raw'
        activity_matrix = activities[activity_key]
    else:
        return {
            "error": "No suitable activity data found",
            "available_keys": list(activities.keys())
        }
    
    n_trials, n_timesteps, n_units = activity_matrix.shape
    
    # Initialize results
    results = {
        "model_id": unit_activities.get('model_id', 'unknown'),
        "metadata": {
            "activity_source": activity_key,
            "alignment_method": alignment_method,
            "reference_events": reference_events,
            "matrix_shape": [n_trials, n_timesteps, n_units],
            "n_trials": n_trials,
            "n_timesteps": n_timesteps,
            "n_units": n_units
        },
        "aligned_trajectories": {},
        "trajectory_variants": {},
        "alignment_statistics": {}
    }
    
    # If controlled dataset metadata is available, use it for trajectory grouping
    if controlled_dataset_metadata:
        results = _align_controlled_trajectories(
            activity_matrix, controlled_dataset_metadata, results, alignment_method
        )
    else:
        # Fall back to general trajectory alignment based on trial metadata
        results = _align_general_trajectories(
            activity_matrix, trial_metadata, results, alignment_method
        )
    
    # Add trajectory comparison analysis
    results["trajectory_analysis"] = compute_trajectory_analysis(
        results["aligned_trajectories"]
    )
    
    # Copy over units analyzed
    if 'units_analyzed' in unit_activities:
        results['units_analyzed'] = unit_activities['units_analyzed']
    
    return results


def _align_controlled_trajectories(
    activity_matrix: np.ndarray,
    controlled_metadata: Dict[str, Any],
    results: Dict[str, Any],
    alignment_method: str
) -> Dict[str, Any]:
    """Align trajectories for controlled color change dataset."""
    
    # Expected controlled dataset structure from §3.2.3:
    # 7 variants per base trajectory:
    # - No Change Condition
    # - Random Change Conditions [Low|High] 
    # - Contingent Change Conditions [Low|Medium|High]
    # Each variant generated 3 times (once per starting color)
    
    base_trajectories = controlled_metadata.get('base_trajectories', {})
    variant_mapping = controlled_metadata.get('variant_mapping', {})
    
    aligned_trajectories = {}
    trajectory_variants = {}
    
    for base_id, base_info in base_trajectories.items():
        variant_indices = variant_mapping.get(base_id, {})
        
        # Extract trajectories for each variant of this base
        base_variants = {}
        
        for variant_name, trial_indices in variant_indices.items():
            if trial_indices:
                # Extract activity for this variant
                variant_activity = activity_matrix[trial_indices, :, :]  # (n_variant_trials, timesteps, n_units)
                
                # Apply alignment
                if alignment_method == "time":
                    # Simple time alignment (already aligned)
                    aligned_variant = variant_activity
                elif alignment_method == "event":
                    # Align based on color change events (if metadata available)
                    aligned_variant = _align_by_events(variant_activity, base_info.get('events', []))
                elif alignment_method == "phase":
                    # Align based on task phases
                    aligned_variant = _align_by_phases(variant_activity, base_info.get('phases', []))
                else:
                    aligned_variant = variant_activity
                
                base_variants[variant_name] = aligned_variant
        
        aligned_trajectories[base_id] = base_variants
        
        # Compute variant statistics
        variant_stats = {}
        for variant_name, variant_data in base_variants.items():
            variant_stats[variant_name] = {
                "n_trials": variant_data.shape[0],
                "mean_activity": float(np.mean(variant_data)),
                "std_activity": float(np.std(variant_data)),
                "shape": list(variant_data.shape)
            }
        
        trajectory_variants[base_id] = variant_stats
    
    results["aligned_trajectories"] = aligned_trajectories
    results["trajectory_variants"] = trajectory_variants
    
    return results


def _align_general_trajectories(
    activity_matrix: np.ndarray,
    trial_metadata: Dict[str, List],
    results: Dict[str, Any],
    alignment_method: str
) -> Dict[str, Any]:
    """General trajectory alignment based on trial conditions."""
    
    # Group trials by similar conditions for trajectory analysis
    hazard_rates = trial_metadata.get('hazard_rates', [])
    trial_types = trial_metadata.get('trial_types', [])
    
    aligned_trajectories = {}
    
    if hazard_rates and trial_types:
        # Group by hazard rate and trial type combinations
        unique_hazards = sorted(set(hazard_rates))
        unique_types = sorted(set(trial_types))
        
        for hazard in unique_hazards:
            for trial_type in unique_types:
                # Find trials matching this condition
                indices = [
                    i for i, (hr, tt) in enumerate(zip(hazard_rates, trial_types))
                    if hr == hazard and tt == trial_type
                ]
                
                if indices:
                    condition_key = f"{hazard}_{trial_type}"
                    condition_activity = activity_matrix[indices, :, :]
                    
                    # Apply alignment
                    if alignment_method == "time":
                        aligned_activity = condition_activity
                    else:
                        # For general case, fall back to time alignment
                        aligned_activity = condition_activity
                    
                    aligned_trajectories[condition_key] = {
                        "all_trials": aligned_activity,
                        "mean_trajectory": np.mean(aligned_activity, axis=0),
                        "std_trajectory": np.std(aligned_activity, axis=0)
                    }
    
    results["aligned_trajectories"] = aligned_trajectories
    
    return results


def _align_by_events(
    activity_matrix: np.ndarray,
    event_times: List[int]
) -> np.ndarray:
    """Align trajectories based on specific events."""
    # For controlled dataset, align based on color change events
    # This is a simplified implementation - could be enhanced based on specific needs
    
    if not event_times:
        return activity_matrix
    
    # Use first event as reference point
    reference_time = event_times[0] if event_times else activity_matrix.shape[1] // 2
    
    # For now, just return original (could implement more sophisticated alignment)
    return activity_matrix


def _align_by_phases(
    activity_matrix: np.ndarray,
    phase_boundaries: List[int]
) -> np.ndarray:
    """Align trajectories based on task phases."""
    # Could implement phase-based alignment if phase information is available
    return activity_matrix


def compute_trajectory_analysis(
    aligned_trajectories: Dict[str, Any]
) -> Dict[str, Any]:
    """Compute analysis comparing different trajectory variants."""
    analysis = {
        "variant_comparisons": {},
        "similarity_metrics": {},
        "difference_maps": {}
    }
    
    # For controlled dataset analysis
    for base_id, variants in aligned_trajectories.items():
        if isinstance(variants, dict) and len(variants) > 1:
            # Compare variants within each base trajectory
            variant_names = list(variants.keys())
            comparisons = {}
            
            for i, variant1 in enumerate(variant_names):
                for j, variant2 in enumerate(variant_names):
                    if i < j:  # Avoid duplicate comparisons
                        comparison_key = f"{variant1}_vs_{variant2}"
                        
                        # Get mean trajectories for comparison
                        traj1 = variants[variant1]
                        traj2 = variants[variant2]
                        
                        # Handle different data structures
                        if isinstance(traj1, np.ndarray) and len(traj1.shape) == 3:
                            mean1 = np.mean(traj1, axis=0)  # Average over trials
                        else:
                            mean1 = traj1
                        
                        if isinstance(traj2, np.ndarray) and len(traj2.shape) == 3:
                            mean2 = np.mean(traj2, axis=0)  # Average over trials
                        else:
                            mean2 = traj2
                        
                        # Compute similarity metrics
                        if isinstance(mean1, np.ndarray) and isinstance(mean2, np.ndarray) and mean1.shape == mean2.shape:
                            # Correlation across time and units
                            correlation = np.corrcoef(mean1.flatten(), mean2.flatten())[0, 1]
                            
                            # Mean squared difference
                            mse = np.mean((mean1 - mean2) ** 2)
                            
                            # Cosine similarity
                            norm1 = np.linalg.norm(mean1.flatten())
                            norm2 = np.linalg.norm(mean2.flatten())
                            if norm1 > 0 and norm2 > 0:
                                cosine_sim = np.dot(mean1.flatten(), mean2.flatten()) / (norm1 * norm2)
                            else:
                                cosine_sim = 0.0
                            
                            comparisons[comparison_key] = {
                                "correlation": float(correlation) if not np.isnan(correlation) else 0.0,
                                "mse": float(mse),
                                "cosine_similarity": float(cosine_sim),
                                "shapes": [list(mean1.shape), list(mean2.shape)]
                            }
            
            analysis["variant_comparisons"][base_id] = comparisons
    
    return analysis


def main():
    """Command-line interface for align_trajectories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--activities', type=Path, required=True,
                       help='Path to unit activities file (.npz)')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output path for aligned trajectories (.npz)')
    parser.add_argument('--model-id', type=str, required=True,
                       help='Model identifier')
    parser.add_argument('--controlled-metadata', type=Path,
                       help='Path to controlled dataset metadata (.json)')
    parser.add_argument('--alignment-method', choices=['time', 'event', 'phase'],
                       default='time',
                       help='Method for trajectory alignment')
    parser.add_argument('--reference-events', type=str, nargs='*',
                       help='Reference events for alignment')
    
    args = parser.parse_args()
    
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
    
    # Load controlled dataset metadata if provided
    controlled_metadata = None
    if args.controlled_metadata and args.controlled_metadata.exists():
        print(f"Loading controlled dataset metadata from {args.controlled_metadata}")
        with open(args.controlled_metadata, 'r') as f:
            controlled_metadata = json.load(f)
    
    # Align trajectories
    print(f"Aligning trajectories for {args.model_id}")
    results = align_trajectories(
        unit_activities=unit_activities,
        controlled_dataset_metadata=controlled_metadata,
        alignment_method=args.alignment_method,
        reference_events=args.reference_events
    )
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for saving (using JSON for complex nested structure)
    with open(args.output.with_suffix('.json'), 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        def convert_arrays(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_arrays(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_arrays(item) for item in obj]
            else:
                return obj
        
        json_results = convert_arrays(results)
        json.dump(json_results, f, indent=2)
    
    print(f"Results saved to {args.output.with_suffix('.json')}")
    print(f"Alignment method: {args.alignment_method}")
    if results.get('aligned_trajectories'):
        print(f"Aligned {len(results['aligned_trajectories'])} trajectory groups")


if __name__ == '__main__':
    main()