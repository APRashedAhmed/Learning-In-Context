#!/usr/bin/env python3
"""
Aggregate critical units results from multiple decoder types.

This script combines results from individual decoder analyses (hazard, contingency, 
color, velocity_x, velocity_y) into a unified critical units file that maintains
backward compatibility with downstream tasks.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional


def validate_decoder_result(result: Dict[str, Any], decoder_type: str) -> Optional[str]:
    """Validate that a decoder result has required fields.
    
    Args:
        result: Decoder result dictionary
        decoder_type: Type of decoder for error messages
        
    Returns:
        Error message if validation fails, None if successful
    """
    required_fields = ["unit_indices", "coefficients", "best_alpha", "r2_scores", "metadata"]
    missing_fields = []
    
    for field in required_fields:
        if field not in result:
            missing_fields.append(field)
    
    if missing_fields:
        return f"Missing required fields in {decoder_type}: {', '.join(missing_fields)}"
    
    # Validate metadata
    if "metadata" in result:
        required_metadata = ["n_units_total", "n_units_critical", "decoder_type"]
        missing_metadata = []
        for field in required_metadata:
            if field not in result["metadata"]:
                missing_metadata.append(field)
        
        if missing_metadata:
            return f"Missing metadata fields in {decoder_type}: {', '.join(missing_metadata)}"
    
    return None


def load_decoder_results(cache_dir: Path, model_id: str, dataset_suffix: str = "", decoder_types: List[str] = None) -> Dict[str, Any]:
    """Load results from specified decoder types.
    
    Args:
        cache_dir: Path to cache directory
        model_id: Model identifier
        dataset_suffix: Dataset suffix (e.g., "participant", "extended")
        decoder_types: List of decoder types to load (default: all)
        
    Returns:
        Dictionary mapping decoder_type to results
    """
    decoder_results = {}
    if decoder_types is None:
        decoder_types = ["hazard", "contingency", "color", "velocity_x", "velocity_y"]
    
    # Build file path pattern based on dataset suffix
    if dataset_suffix:
        pattern = f"{model_id}_{dataset_suffix}_{{}}_units.json"
    else:
        pattern = f"{model_id}_{{}}_units.json"
    
    for decoder_type in decoder_types:
        file_path = cache_dir / "critical_units" / pattern.format(decoder_type)
        if file_path.exists():
            try:
                with open(file_path, "r") as f:
                    result = json.load(f)
                
                # Validate the result structure
                validation_error = validate_decoder_result(result, decoder_type)
                if validation_error:
                    print(f"Warning: {validation_error}")
                    continue
                
                decoder_results[decoder_type] = result
                print(f"Loaded {decoder_type} decoder results: {file_path}")
            except Exception as e:
                print(f"Warning: Failed to load {decoder_type} results: {e}")
        else:
            print(f"Warning: {decoder_type} decoder results not found: {file_path}")
    
    return decoder_results


def aggregate_results(decoder_results: Dict[str, Any], model_id: str) -> Dict[str, Any]:
    """Aggregate results from multiple decoders.
    
    Args:
        decoder_results: Dictionary mapping decoder_type to results
        model_id: Model identifier
        
    Returns:
        Aggregated results dictionary
    """
    # Collect all critical units and coefficients
    all_units = []
    all_coeffs = []
    total_units = 0
    
    for decoder_type, result in decoder_results.items():
        if "unit_indices" in result and result["unit_indices"]:
            units = result["unit_indices"]
            coeffs = result.get("coefficients", [1.0] * len(units))
            all_units.extend(units)
            all_coeffs.extend([abs(c) for c in coeffs])
        
        # Track maximum number of units
        if "metadata" in result and "n_units_total" in result["metadata"]:
            total_units = max(total_units, result["metadata"]["n_units_total"])
    
    # Find union of all critical units (remove duplicates)
    unique_units = list(set(all_units))
    
    # Calculate aggregation statistics
    n_decoders_with_results = len([r for r in decoder_results.values() 
                                  if "unit_indices" in r and r["unit_indices"]])
    
    # Create aggregated result maintaining backward compatibility
    aggregated_result = {
        "model_id": model_id,
        "unit_indices": unique_units,
        "coefficients": [1.0] * len(unique_units),  # Placeholder coefficients
        "r2_scores": [0.0],  # Placeholder scores
        "best_alpha": 0.001,  # Default alpha
        "cv_scores": {"mean": 0.0, "std": 0.0, "all": [0.0]},
        "metadata": {
            "n_units_total": total_units,
            "n_units_critical": len(unique_units),
            "decoder_results": decoder_results,
            "aggregation_method": "union_of_decoders",
            "n_decoders": len(decoder_results),
            "n_decoders_with_results": n_decoders_with_results,
            "total_critical_units_before_union": len(all_units),
            "overlap_ratio": 1 - len(unique_units) / len(all_units) if all_units else 0.0
        }
    }
    
    return aggregated_result


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True,
                       help="Path to cache directory")
    parser.add_argument("--model-id", type=str, required=True,
                       help="Model identifier")
    parser.add_argument("--output", type=Path, required=True,
                       help="Output path for aggregated results")
    parser.add_argument("--dataset-suffix", type=str, default="",
                       help="Dataset suffix for file naming (e.g., 'participant', 'extended')")
    parser.add_argument("--decoders", type=str, default="hazard,contingency,color,velocity_x,velocity_y",
                       help="Comma-separated list of decoders to aggregate (default: all)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.cache_dir.exists():
        print(f"Error: Cache directory does not exist: {args.cache_dir}")
        sys.exit(1)
    
    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Parse decoder list
    decoder_types = [d.strip() for d in args.decoders.split(',')]
    
    print(f"Aggregating critical units for model: {args.model_id}")
    print(f"Cache directory: {args.cache_dir}")
    print(f"Output file: {args.output}")
    print(f"Decoders: {', '.join(decoder_types)}")
    
    # Load decoder results
    decoder_results = load_decoder_results(args.cache_dir, args.model_id, args.dataset_suffix, decoder_types)
    
    if not decoder_results:
        print("Error: No decoder results found!")
        sys.exit(1)
    
    # Aggregate results
    aggregated_result = aggregate_results(decoder_results, args.model_id)
    
    # Save aggregated result
    try:
        with open(args.output, "w") as f:
            json.dump(aggregated_result, f, indent=2)
        print(f"Successfully saved aggregated results to: {args.output}")
    except Exception as e:
        print(f"Error saving results: {e}")
        sys.exit(1)
    
    # Print summary
    metadata = aggregated_result["metadata"]
    print(f"\nAggregation Summary:")
    print(f"  Model: {args.model_id}")
    print(f"  Decoders processed: {metadata['n_decoders_with_results']}/{metadata['n_decoders']}")
    print(f"  Total critical units (before union): {metadata['total_critical_units_before_union']}")
    print(f"  Unique critical units (after union): {metadata['n_units_critical']}")
    print(f"  Overlap ratio: {metadata['overlap_ratio']:.3f}")
    print(f"  Total model units: {metadata['n_units_total']}")


if __name__ == "__main__":
    main()