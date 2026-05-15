#!/usr/bin/env python3
"""
Aggregate tuning profile results from multiple analysis components.

This script combines results from the decomposed tuning profile DAG structure
(unit_activities, activity_matrix, sorted_conditions, aligned_trajectories, 
event_analysis) into a unified file for backward compatibility with downstream tasks.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np


def validate_npz_structure(data: Dict[str, Any], expected_keys: List[str], component_name: str) -> Optional[str]:
    """Validate that an NPZ file contains expected keys.
    
    Args:
        data: Dictionary loaded from NPZ file
        expected_keys: List of keys that should be present
        component_name: Name of the component for error messages
        
    Returns:
        Error message if validation fails, None if successful
    """
    if "error" in data:
        return data["error"]
    
    missing_keys = []
    for key in expected_keys:
        if key not in data and not any(key in nested for nested in data.values() if isinstance(nested, dict)):
            missing_keys.append(key)
    
    if missing_keys:
        return f"Missing required keys in {component_name}: {', '.join(missing_keys)}"
    
    return None


def load_npz_metadata(file_path: Path) -> Dict[str, Any]:
    """Load metadata from an npz file safely.
    
    Args:
        file_path: Path to npz file
        
    Returns:
        Dictionary with metadata or error information
    """
    try:
        data = np.load(file_path, allow_pickle=True)
        result = {}
        
        # Extract metadata if available
        if 'metadata' in data:
            result['metadata'] = data['metadata'].item()
        
        # Extract other useful summary information
        if 'units_analyzed' in data:
            result['units_analyzed'] = data['units_analyzed'].item()
            
        if 'summary_statistics' in data:
            result['summary_statistics'] = data['summary_statistics'].item()
            
        # Get array shapes without loading full arrays
        result['array_info'] = {}
        for key in data.files:
            if key.startswith('activities_') or key.startswith('matrix_'):
                result['array_info'][key] = data[key].shape
                
        return result
    except Exception as e:
        return {"error": f"Failed to load {file_path.name}: {str(e)}"}


def load_json_file(file_path: Path) -> Dict[str, Any]:
    """Load JSON file safely.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary with contents or error information
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        return {"error": f"Failed to load {file_path.name}: {str(e)}"}


def aggregate_tuning_components(
    cache_dir: Path, 
    model_id: str, 
    dataset_suffix: str = "participant"
) -> Dict[str, Any]:
    """Aggregate all tuning profile components.
    
    Args:
        cache_dir: Path to cache directory
        model_id: Model identifier
        dataset_suffix: Dataset suffix for file naming
        
    Returns:
        Aggregated results dictionary
    """
    tuning_dir = cache_dir / 'tuning_profiles'
    
    # Define component files
    components = {
        'unit_activities': tuning_dir / f'{model_id}_{dataset_suffix}_unit_activities.npz',
        'activity_matrix': tuning_dir / f'{model_id}_{dataset_suffix}_activity_matrix.npz',
        'sorted_conditions': tuning_dir / f'{model_id}_{dataset_suffix}_sorted_conditions.npz',
        'aligned_trajectories': tuning_dir / f'{model_id}_{dataset_suffix}_aligned_trajectories.json',
        'event_analysis': tuning_dir / f'{model_id}_{dataset_suffix}_event_analysis.npz'
    }
    
    # Load all components
    results = {
        "model_id": model_id,
        "dataset_suffix": dataset_suffix,
        "components": {},
        "metadata": {
            "aggregation_method": "tuning_profile_dag",
            "n_components": len(components),
            "n_components_loaded": 0,
            "component_status": {}
        }
    }
    
    # Process each component
    for component_name, file_path in components.items():
        if file_path.exists():
            if file_path.suffix == '.npz':
                component_data = load_npz_metadata(file_path)
            else:  # .json
                component_data = load_json_file(file_path)
                
            if "error" not in component_data:
                results["components"][component_name] = component_data
                results["metadata"]["n_components_loaded"] += 1
                results["metadata"]["component_status"][component_name] = "loaded"
                print(f"✓ Loaded {component_name}: {file_path}")
            else:
                results["components"][component_name] = component_data
                results["metadata"]["component_status"][component_name] = "error"
                print(f"✗ Error loading {component_name}: {component_data['error']}")
        else:
            results["components"][component_name] = {"error": f"File not found: {file_path}"}
            results["metadata"]["component_status"][component_name] = "missing"
            print(f"✗ Missing {component_name}: {file_path}")
    
    # Extract summary statistics
    summary_stats = extract_summary_statistics(results)
    results["summary"] = summary_stats
    
    return results


def extract_summary_statistics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract summary statistics from aggregated results.
    
    Args:
        results: Aggregated results dictionary
        
    Returns:
        Summary statistics dictionary
    """
    summary = {
        "n_components_available": results["metadata"]["n_components_loaded"],
        "n_components_total": results["metadata"]["n_components"],
    }
    
    # Extract unit information
    if "unit_activities" in results["components"]:
        unit_data = results["components"]["unit_activities"]
        if "units_analyzed" in unit_data:
            units_info = unit_data["units_analyzed"]
            summary["n_critical_units"] = len(units_info.get("indices", []))
            if "mapping" in units_info:
                summary["unit_types"] = {
                    "hidden": len([u for u in units_info["mapping"] if u.startswith("h")]),
                    "cell": len([u for u in units_info["mapping"] if u.startswith("c")])
                }
        
        if "metadata" in unit_data:
            meta = unit_data["metadata"]
            summary["n_trials"] = meta.get("n_trials", 0)
            summary["n_timesteps"] = meta.get("n_timesteps", 0)
            summary["window_size"] = meta.get("window_size", 0)
    
    # Extract condition information
    if "sorted_conditions" in results["components"]:
        sorted_data = results["components"]["sorted_conditions"]
        if "metadata" in sorted_data and "condition_counts" in sorted_data["metadata"]:
            summary["condition_counts"] = sorted_data["metadata"]["condition_counts"]
    
    # Extract event analysis information
    if "event_analysis" in results["components"]:
        event_data = results["components"]["event_analysis"]
        if "metadata" in event_data:
            meta = event_data["metadata"]
            summary["event_window"] = meta.get("event_window", None)
            summary["event_types_analyzed"] = meta.get("event_types", [])
    
    return summary


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True,
                       help="Path to cache directory")
    parser.add_argument("--model-id", type=str, required=True,
                       help="Model identifier")
    parser.add_argument("--output", type=Path, required=True,
                       help="Output path for aggregated results")
    parser.add_argument("--dataset-suffix", type=str, default="participant",
                       help="Dataset suffix for file naming (e.g., 'participant', 'extended')")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.cache_dir.exists():
        print(f"Error: Cache directory does not exist: {args.cache_dir}")
        sys.exit(1)
    
    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Aggregating tuning profile components for model: {args.model_id}")
    print(f"Dataset: {args.dataset_suffix}")
    print(f"Cache directory: {args.cache_dir}")
    print(f"Output file: {args.output}")
    print("-" * 60)
    
    # Aggregate components
    aggregated_result = aggregate_tuning_components(
        args.cache_dir, 
        args.model_id, 
        args.dataset_suffix
    )
    
    # Save aggregated result
    try:
        with open(args.output, "w") as f:
            json.dump(aggregated_result, f, indent=2)
        print(f"\n✓ Successfully saved aggregated results to: {args.output}")
    except Exception as e:
        print(f"\n✗ Error saving results: {e}")
        sys.exit(1)
    
    # Print summary
    print(f"\nAggregation Summary:")
    print(f"  Model: {args.model_id}")
    print(f"  Dataset: {args.dataset_suffix}")
    print(f"  Components processed: {aggregated_result['metadata']['n_components_loaded']}/{aggregated_result['metadata']['n_components']}")
    
    summary = aggregated_result.get("summary", {})
    if summary:
        print(f"\nModel Statistics:")
        if "n_critical_units" in summary:
            print(f"  Critical units: {summary['n_critical_units']}")
        if "unit_types" in summary:
            print(f"  Unit types: {summary['unit_types']['hidden']} hidden, {summary['unit_types']['cell']} cell")
        if "n_trials" in summary:
            print(f"  Trials analyzed: {summary['n_trials']}")
        if "condition_counts" in summary:
            print(f"  Conditions: {summary['condition_counts']}")
    
    # Report any errors
    error_components = [
        name for name, status in aggregated_result["metadata"]["component_status"].items()
        if status in ["error", "missing"]
    ]
    if error_components:
        print(f"\n⚠ Warning: Failed to load components: {', '.join(error_components)}")
    
    # Exit with appropriate code
    if aggregated_result['metadata']['n_components_loaded'] == 0:
        print("\n✗ Error: No components were successfully loaded!")
        sys.exit(1)
    else:
        # Always exit with 0 if at least some components loaded
        # DoIt interprets any non-zero exit code as failure
        sys.exit(0)


if __name__ == "__main__":
    main()