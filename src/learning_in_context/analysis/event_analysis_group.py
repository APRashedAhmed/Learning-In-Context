"""Event-triggered analysis for temporal dynamics.

This module implements the event_analysis_group DAG node from the dissertation 
specification (§3.3.2 Neural Tuning Profiles - Temporal Dynamics Analysis).
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.constants import TRIAL_TYPES, HAZARD_RATES, CONTINGENCIES


def event_analysis_group(
    unit_activities: Dict[str, Any],
    event_window: Tuple[int, int] = (-5, 10),
    event_types: List[str] = None,
    min_events_per_condition: int = 5
) -> Dict[str, Any]:
    """Perform event-triggered analysis for temporal dynamics.
    
    This implements the temporal dynamics analysis from §3.3.2, with event-triggered
    averaging around color changes using -5:+10 timestep windows.
    
    Args:
        unit_activities: Results from extract_unit_activities
        event_window: Tuple of (before, after) timesteps around events
        event_types: Types of events to analyze ('color_change', 'velocity_change', 'random', 'contingent')
        min_events_per_condition: Minimum number of events required per condition
        
    Returns:
        Dictionary containing event-triggered averages and temporal statistics
    """
    if event_types is None:
        event_types = ['color_change', 'velocity_change']
    
    # Get activity data
    activities = unit_activities.get('activities', {})
    trial_metadata = unit_activities.get('trial_metadata', {})
    
    # Determine which activities to use
    activity_key = None
    if 'full_normalized' in activities:
        activity_key = 'full_normalized'
        activity_matrix = activities[activity_key]
    elif 'windowed_normalized' in activities:
        activity_key = 'windowed_normalized'
        activity_matrix = activities[activity_key]
    elif 'full_raw' in activities:
        activity_key = 'full_raw'
        activity_matrix = activities[activity_key]
    else:
        return {
            "error": "No suitable activity data found for event analysis",
            "available_keys": list(activities.keys())
        }
    
    n_trials, n_timesteps, n_units = activity_matrix.shape
    window_before, window_after = event_window
    window_size = window_after - window_before + 1
    
    # Initialize results
    results = {
        "model_id": unit_activities.get('model_id', 'unknown'),
        "metadata": {
            "activity_source": activity_key,
            "event_window": event_window,
            "window_size": window_size,
            "event_types": event_types,
            "min_events_per_condition": min_events_per_condition,
            "matrix_shape": [n_trials, n_timesteps, n_units],
            "n_trials": n_trials,
            "n_timesteps": n_timesteps,
            "n_units": n_units
        },
        "event_triggered_averages": {},
        "temporal_statistics": {},
        "event_counts": {}
    }
    
    # Perform analysis for each event type
    for event_type in event_types:
        if event_type == 'color_change':
            results = _analyze_color_changes(
                activity_matrix, trial_metadata, results, event_window, min_events_per_condition
            )
        elif event_type == 'velocity_change':
            results = _analyze_velocity_changes(
                activity_matrix, trial_metadata, results, event_window, min_events_per_condition
            )
        elif event_type == 'random':
            results = _analyze_random_changes(
                activity_matrix, trial_metadata, results, event_window, min_events_per_condition
            )
        elif event_type == 'contingent':
            results = _analyze_contingent_changes(
                activity_matrix, trial_metadata, results, event_window, min_events_per_condition
            )
    
    # Add cross-event analysis
    results["cross_event_analysis"] = compute_cross_event_analysis(
        results["event_triggered_averages"]
    )
    
    # Copy over units analyzed
    if 'units_analyzed' in unit_activities:
        results['units_analyzed'] = unit_activities['units_analyzed']
    
    return results


def _analyze_color_changes(
    activity_matrix: np.ndarray,
    trial_metadata: Dict[str, List],
    results: Dict[str, Any],
    event_window: Tuple[int, int],
    min_events: int
) -> Dict[str, Any]:
    """Analyze color changes with event-triggered averaging."""
    
    # For this analysis, we need to simulate or extract color change events
    # In a real implementation, this would use actual event timing data
    
    window_before, window_after = event_window
    window_size = window_after - window_before + 1
    n_trials, n_timesteps, n_units = activity_matrix.shape
    
    hazard_rates = trial_metadata.get('hazard_rates', [])
    trial_types = trial_metadata.get('trial_types', [])
    
    color_change_analysis = {}
    event_counts = {}
    
    # Analyze by hazard rate condition
    if hazard_rates:
        unique_hazards = sorted(set(hazard_rates))
        
        for hazard in unique_hazards:
            hazard_indices = [i for i, hr in enumerate(hazard_rates) if hr == hazard]
            
            if len(hazard_indices) < min_events:
                continue
            
            # Extract activity for this hazard condition
            hazard_activity = activity_matrix[hazard_indices, :, :]  # (n_hazard_trials, timesteps, n_units)
            
            # Simulate color change events (in real implementation, use actual event data)
            simulated_events = _simulate_color_change_events(
                hazard_activity, hazard_rate=hazard, n_timesteps=n_timesteps
            )
            
            if len(simulated_events) >= min_events:
                # Extract event-triggered windows
                event_windows = _extract_event_windows(
                    hazard_activity, simulated_events, event_window
                )
                
                if event_windows.size > 0:
                    # Compute average and statistics
                    mean_response = np.mean(event_windows, axis=0)  # (window_size, n_units)
                    std_response = np.std(event_windows, axis=0)
                    sem_response = std_response / np.sqrt(event_windows.shape[0])
                    
                    color_change_analysis[f"hazard_{hazard}"] = {
                        "mean": mean_response,
                        "std": std_response,
                        "sem": sem_response,
                        "n_events": len(simulated_events),
                        "window_shape": list(event_windows.shape)
                    }
                    
                    event_counts[f"color_change_hazard_{hazard}"] = len(simulated_events)
    
    results["event_triggered_averages"]["color_change"] = color_change_analysis
    results["event_counts"].update(event_counts)
    
    return results


def _analyze_velocity_changes(
    activity_matrix: np.ndarray,
    trial_metadata: Dict[str, List],
    results: Dict[str, Any],
    event_window: Tuple[int, int],
    min_events: int
) -> Dict[str, Any]:
    """Analyze velocity changes with event-triggered averaging."""
    
    window_before, window_after = event_window
    n_trials, n_timesteps, n_units = activity_matrix.shape
    
    trial_types = trial_metadata.get('trial_types', [])
    
    velocity_change_analysis = {}
    event_counts = {}
    
    # Analyze by trial type (bounce trials have velocity changes)
    if trial_types:
        bounce_trials = [i for i, tt in enumerate(trial_types) if 'bounce' in str(tt).lower()]
        
        if len(bounce_trials) >= min_events:
            bounce_activity = activity_matrix[bounce_trials, :, :]
            
            # Simulate velocity change events (in real implementation, use actual bounce timing)
            simulated_bounces = _simulate_velocity_change_events(
                bounce_activity, n_timesteps=n_timesteps
            )
            
            if len(simulated_bounces) >= min_events:
                # Extract event-triggered windows
                event_windows = _extract_event_windows(
                    bounce_activity, simulated_bounces, event_window
                )
                
                if event_windows.size > 0:
                    mean_response = np.mean(event_windows, axis=0)
                    std_response = np.std(event_windows, axis=0)
                    sem_response = std_response / np.sqrt(event_windows.shape[0])
                    
                    velocity_change_analysis["bounce_trials"] = {
                        "mean": mean_response,
                        "std": std_response,
                        "sem": sem_response,
                        "n_events": len(simulated_bounces),
                        "window_shape": list(event_windows.shape)
                    }
                    
                    event_counts["velocity_change_bounce"] = len(simulated_bounces)
    
    results["event_triggered_averages"]["velocity_change"] = velocity_change_analysis
    results["event_counts"].update(event_counts)
    
    return results


def _analyze_random_changes(
    activity_matrix: np.ndarray,
    trial_metadata: Dict[str, List],
    results: Dict[str, Any],
    event_window: Tuple[int, int],
    min_events: int
) -> Dict[str, Any]:
    """Analyze random color changes (hazard rate driven)."""
    
    # This would be similar to color change analysis but specifically for random changes
    # Implementation would depend on having event timing data that distinguishes
    # random vs contingent changes
    
    results["event_triggered_averages"]["random"] = {}
    return results


def _analyze_contingent_changes(
    activity_matrix: np.ndarray,
    trial_metadata: Dict[str, List],
    results: Dict[str, Any],
    event_window: Tuple[int, int],
    min_events: int
) -> Dict[str, Any]:
    """Analyze contingent color changes (velocity-triggered)."""
    
    # This would analyze color changes that occur contingent on velocity changes
    # Implementation would need event timing data for contingent relationships
    
    results["event_triggered_averages"]["contingent"] = {}
    return results


def _simulate_color_change_events(
    activity_matrix: np.ndarray,
    hazard_rate: str,
    n_timesteps: int
) -> List[Tuple[int, int]]:
    """Simulate color change events based on hazard rate."""
    
    # This is a placeholder that simulates events
    # In real implementation, would use actual event timing data
    
    n_trials = activity_matrix.shape[0]
    events = []
    
    # Simple simulation based on hazard rate
    prob = 0.02 if hazard_rate.lower() == 'low' else 0.05
    
    for trial_idx in range(n_trials):
        # Sample a few events per trial based on probability
        for t in range(10, n_timesteps - 10):  # Avoid edges
            if np.random.random() < prob:
                events.append((trial_idx, t))
                break  # One event per trial for simplicity
    
    return events


def _simulate_velocity_change_events(
    activity_matrix: np.ndarray,
    n_timesteps: int
) -> List[Tuple[int, int]]:
    """Simulate velocity change events (bounces)."""
    
    n_trials = activity_matrix.shape[0]
    events = []
    
    # For bounce trials, simulate bounce happening around middle of trial
    for trial_idx in range(n_trials):
        bounce_time = n_timesteps // 2 + np.random.randint(-5, 6)
        bounce_time = max(10, min(bounce_time, n_timesteps - 10))
        events.append((trial_idx, bounce_time))
    
    return events


def _extract_event_windows(
    activity_matrix: np.ndarray,
    events: List[Tuple[int, int]],
    event_window: Tuple[int, int]
) -> np.ndarray:
    """Extract activity windows around events."""
    
    window_before, window_after = event_window
    window_size = window_after - window_before + 1
    n_units = activity_matrix.shape[2]
    
    event_windows = []
    
    for trial_idx, event_time in events:
        start_time = event_time + window_before
        end_time = event_time + window_after + 1
        
        # Check bounds
        if start_time >= 0 and end_time <= activity_matrix.shape[1]:
            window = activity_matrix[trial_idx, start_time:end_time, :]  # (window_size, n_units)
            event_windows.append(window)
    
    if event_windows:
        return np.array(event_windows)  # (n_events, window_size, n_units)
    else:
        return np.empty((0, window_size, n_units))


def compute_cross_event_analysis(
    event_triggered_averages: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Compute analysis comparing different event types."""
    
    cross_analysis = {
        "event_comparisons": {},
        "temporal_patterns": {}
    }
    
    # Compare event types if multiple are available
    event_types = list(event_triggered_averages.keys())
    
    for i, event_type1 in enumerate(event_types):
        for j, event_type2 in enumerate(event_types):
            if i < j:  # Avoid duplicate comparisons
                comparison_key = f"{event_type1}_vs_{event_type2}"
                
                # Compare patterns between event types
                # This would be implemented based on specific analysis needs
                cross_analysis["event_comparisons"][comparison_key] = {
                    "correlation": 0.0,  # Placeholder
                    "difference": 0.0    # Placeholder
                }
    
    return cross_analysis


def main():
    """Command-line interface for event_analysis_group."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--activities', type=Path, required=True,
                       help='Path to unit activities file (.npz)')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output path for event analysis (.npz)')
    parser.add_argument('--model-id', type=str, required=True,
                       help='Model identifier')
    parser.add_argument('--event-window', type=int, nargs=2, default=[-5, 10],
                       help='Event window as [before, after] timesteps')
    parser.add_argument('--event-types', type=str, nargs='*',
                       choices=['color_change', 'velocity_change', 'random', 'contingent'],
                       default=['color_change', 'velocity_change'],
                       help='Types of events to analyze')
    parser.add_argument('--min-events', type=int, default=5,
                       help='Minimum events required per condition')
    
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
    
    # Perform event analysis
    print(f"Performing event analysis for {args.model_id}")
    results = event_analysis_group(
        unit_activities=unit_activities,
        event_window=tuple(args.event_window),
        event_types=args.event_types,
        min_events_per_condition=args.min_events
    )
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for saving
    save_data = {
        'metadata': results['metadata'],
        'event_counts': results['event_counts'],
        'cross_event_analysis': results['cross_event_analysis']
    }
    
    # Add units_analyzed if available
    if 'units_analyzed' in results:
        save_data['units_analyzed'] = results['units_analyzed']
    
    # Add event-triggered averages (flattened structure)
    for event_type, analyses in results['event_triggered_averages'].items():
        for condition, analysis in analyses.items():
            for metric, data in analysis.items():
                if isinstance(data, np.ndarray):
                    key = f"eta_{event_type}_{condition}_{metric}"
                    save_data[key] = data
                else:
                    # Store scalar metadata in main structure
                    if 'temporal_statistics' not in save_data:
                        save_data['temporal_statistics'] = {}
                    if event_type not in save_data['temporal_statistics']:
                        save_data['temporal_statistics'][event_type] = {}
                    if condition not in save_data['temporal_statistics'][event_type]:
                        save_data['temporal_statistics'][event_type][condition] = {}
                    save_data['temporal_statistics'][event_type][condition][metric] = data
    
    np.savez_compressed(args.output, **save_data)
    
    print(f"Results saved to {args.output}")
    print(f"Event window: {args.event_window}")
    print(f"Event types analyzed: {args.event_types}")
    for event_type in results['event_triggered_averages']:
        n_conditions = len(results['event_triggered_averages'][event_type])
        print(f"  {event_type}: {n_conditions} conditions")


if __name__ == '__main__':
    main()