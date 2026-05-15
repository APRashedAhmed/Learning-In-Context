"""Extract activities for specific critical units.

This module implements the extract_unit_activities DAG node from the dissertation 
specification (§3.3.2 Neural Tuning Profiles).
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import tuning_utils
from ..core import StateData
from ..core.constants import TRIAL_TYPES, HAZARD_RATES, CONTINGENCIES


def extract_unit_activities(
    states: StateData,
    critical_units: Dict[str, Any],
    metadata: Dict[str, Any],
    df_data: Optional[pd.DataFrame] = None,
    window_size: int = 200,
    normalize: bool = True,
    output_format: str = "both"  # "full", "windowed", or "both"
) -> Dict[str, Any]:
    """Extract activities for specific critical units.
    
    Args:
        states: StateData object containing neural states
        critical_units: Results from critical units analysis
        metadata: Trial metadata dictionary
        df_data: DataFrame with trial metadata (if available)
        window_size: Size of window for last-N analysis
        normalize: Whether to apply z-score normalization
        output_format: Which activities to extract ("full", "windowed", or "both")
        
    Returns:
        Dictionary containing extracted unit activities and metadata
    """
    # Get critical unit indices
    unit_indices = critical_units.get("unit_indices", [])
    
    if len(unit_indices) == 0:
        return {
            "message": "No critical units found",
            "unit_indices": [],
            "activities": {}
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
    
    # Extract and process states using existing utility
    compute_full = output_format in ["full", "both"]
    compute_windowed = output_format in ["windowed", "both"]
    
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
    
    # Build results
    results = {
        "model_id": critical_units.get("model_id", "unknown"),
        "metadata": {
            "window_size": window_size,
            "normalization_applied": normalize,
            "output_format": output_format,
            "n_units_extracted": len(unit_indices),
            "n_trials": n_trials,
            "n_timesteps": n_timesteps,
            "n_hidden_units": n_hidden,
            "n_cell_units": n_cell
        },
        "units_analyzed": {
            "indices": unit_indices,
            "mapping": unit_mapping
        },
        "activities": {}
    }
    
    # Store requested activity formats
    if compute_full:
        if normalize and 'full_states_normalized' in extracted_states:
            results["activities"]["full_normalized"] = extracted_states['full_states_normalized']
        if 'full_states' in extracted_states:
            results["activities"]["full_raw"] = extracted_states['full_states']
    
    if compute_windowed:
        if normalize and 'windowed_states_normalized' in extracted_states:
            results["activities"]["windowed_normalized"] = extracted_states['windowed_states_normalized']
        if 'windowed_states' in extracted_states:
            results["activities"]["windowed_raw"] = extracted_states['windowed_states']
    
    # Add trial metadata if available
    if df_data is not None:
        results["trial_metadata"] = {
            "hazard_rates": df_data.get('Hazard Rate', []).tolist() if 'Hazard Rate' in df_data.columns else [],
            "trial_types": df_data.get('trial', []).tolist() if 'trial' in df_data.columns else [],
            "contingencies": df_data.get('Contingency', []).tolist() if 'Contingency' in df_data.columns else [],
            "trial_lengths": df_data.get('length', []).tolist() if 'length' in df_data.columns else []
        }
    elif metadata:
        results["trial_metadata"] = {
            "hazard_rates": metadata.get('hazard_rates', []),
            "trial_types": metadata.get('trial_types', []),
            "contingencies": metadata.get('contingencies', []),
            "trial_lengths": metadata.get('trial_lengths', [])
        }
    
    return results


def main():
    """Command-line interface for extract_unit_activities."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--states', type=Path, required=True,
                       help='Path to states file (.npz)')
    parser.add_argument('--units', type=Path, required=True,
                       help='Path to critical units file (.json)')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output path for extracted activities (.npz)')
    parser.add_argument('--model-id', type=str, required=True,
                       help='Model identifier')
    parser.add_argument('--window-size', type=int, default=200,
                       help='Window size for last-N analysis')
    parser.add_argument('--normalize', action='store_true',
                       help='Apply z-score normalization')
    parser.add_argument('--output-format', choices=['full', 'windowed', 'both'], default='both',
                       help='Which activity formats to extract')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading states from {args.states}")
    
    # Load states data
    data = np.load(args.states)
    
    # Create StateData from arrays
    if 'hiddens' in data and 'cells' in data:
        hiddens = data['hiddens']
        cells = data['cells']
        
        # Use first layer only as per hmdcpd analysis
        if len(hiddens.shape) == 4:  # (layers, trials, timesteps, units)
            hiddens = hiddens[0]
            cells = cells[0] if cells.shape[0] > 0 else cells
        
        states = StateData(hiddens=hiddens, cells=cells)
        
        # Load metadata if available
        df_data = None
        if 'df_data' in data:
            df_data_array = data['df_data']
            if hasattr(df_data_array, 'dtype') and df_data_array.dtype.names:
                df_data = pd.DataFrame(df_data_array)
            else:
                # Reconstruct from dict_metadata
                if 'dict_metadata' in data:
                    dict_metadata = data['dict_metadata'].item() if data['dict_metadata'].shape == () else data['dict_metadata']
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
        # Fall back to StateData loading
        states = StateData.load(args.states)
        df_data = None
        metadata = {}
    
    print(f"Loading critical units from {args.units}")
    with open(args.units, 'r') as f:
        critical_units = json.load(f)
    
    # Extract unit activities
    print(f"Extracting unit activities for {args.model_id}")
    results = extract_unit_activities(
        states=states,
        critical_units=critical_units,
        metadata=metadata,
        df_data=df_data,
        window_size=args.window_size,
        normalize=args.normalize,
        output_format=args.output_format
    )
    
    # Save results as npz for efficiency with large arrays
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for saving
    save_data = {
        'metadata': results['metadata'],
        'units_analyzed': results['units_analyzed'],
        'trial_metadata': results.get('trial_metadata', {})
    }
    
    # Add activity arrays
    for key, array in results.get('activities', {}).items():
        save_data[f'activities_{key}'] = array
    
    np.savez_compressed(args.output, **save_data)
    
    print(f"Results saved to {args.output}")
    print(f"Extracted activities for {results['metadata']['n_units_extracted']} critical units")
    if args.normalize:
        print(f"Applied z-score normalization")
    print(f"Output format: {args.output_format}")


if __name__ == '__main__':
    main()