"""Visualization functions for neural tuning profiles."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import seaborn as sns


def plot_unit_trial_activity(
    states: np.ndarray,
    units: Dict[int, Dict[str, str]],
    df_data: pd.DataFrame,
    cmap: str = 'seismic',
    title: str = "",
    figsize: Tuple[int, int] = (12, 8),
    order_func: Optional[Callable] = None,
    output_path: Optional[Path] = None
) -> Tuple[plt.Figure, np.ndarray]:
    """Create trial activity heatmap for critical units.
    
    Args:
        states: Normalized states array (n_trials, timesteps, n_units)
        units: Unit mapping from tuning analysis
        df_data: Trial metadata DataFrame
        cmap: Colormap for heatmap
        title: Figure title
        figsize: Figure size
        order_func: Function to order trials within groups
        output_path: Path to save figure
        
    Returns:
        Figure and axes objects
    """
    num_units = len(units)
    trial_types = df_data['trial'].unique()
    
    # Create subplot grid
    fig, axes = plt.subplots(
        num_units, len(trial_types), 
        figsize=figsize, 
        sharex=True, 
        layout='compressed'
    )
    
    # Ensure axes is 2D
    if num_units == 1:
        axes = axes.reshape(1, -1)
    if len(trial_types) == 1:
        axes = axes.reshape(-1, 1)
    
    num_trials, timesteps, _ = states.shape
    
    # Create meshgrid for plotting
    X, Y = np.meshgrid(range(timesteps), np.arange(len(df_data) // len(trial_types)))
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    # Plot each unit and trial type combination
    for idx_unit, (unit_idx, unit_info) in enumerate(units.items()):
        # Get the local index within the states array
        unit_states_idx = list(units.keys()).index(unit_idx)
        
        for idx_trial, trial_type in enumerate(trial_types):
            ax = axes[idx_unit, idx_trial]
            
            # Apply ordering function if provided
            if order_func:
                df_ordered = order_func(df_data[df_data['trial'] == trial_type])
            else:
                df_ordered = df_data[df_data['trial'] == trial_type]
            
            # Extract states for this unit and trial type
            Z = states[df_ordered.index, :, unit_states_idx]
            
            # Create heatmap
            im = ax.pcolormesh(
                X[:len(df_ordered), :], 
                Y[:len(df_ordered), :], 
                Z, 
                cmap=cmap, 
                shading='nearest', 
                rasterized=True
            )
            
            # Set limits and labels
            ax.set_ylim(0, len(df_ordered))
            ax.set_aspect('auto')
            
            # Add labels
            if idx_unit == 0:
                ax.set_title(trial_type.title())
            if idx_trial == 0:
                ax.set_ylabel(f"{unit_info['name']}\\nTrial")
            if idx_unit == num_units - 1:
                ax.set_xlabel("Timestep")
            
            # Add colorbar
            plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    
    plt.tight_layout()
    
    # Save if requested
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {output_path}")
    
    return fig, axes


def plot_unit_scatter(
    states: np.ndarray,
    unit_pairs: List[List[int]],
    unit_names: List[str],
    color_values: np.ndarray,
    color_label: str = "Hazard Rate",
    cmap: str = 'viridis',
    figsize: Tuple[int, int] = (12, 8),
    output_path: Optional[Path] = None,
    alpha: float = 0.6
) -> Tuple[plt.Figure, np.ndarray]:
    """Create unit-unit scatter plots.
    
    Args:
        states: States array (n_trials, timesteps, n_units) 
        unit_pairs: List of [unit1_idx, unit2_idx] pairs to plot
        unit_names: Names of units
        color_values: Values for coloring points
        color_label: Label for color axis
        cmap: Colormap
        figsize: Figure size
        output_path: Path to save figure
        alpha: Point transparency
        
    Returns:
        Figure and axes objects
    """
    n_pairs = len(unit_pairs)
    
    # Create subplot grid
    n_cols = min(3, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(
        n_rows, n_cols, 
        figsize=figsize,
        squeeze=False
    )
    axes = axes.flatten()
    
    # Flatten states for scatter plots
    n_trials, n_timesteps, n_units = states.shape
    states_flat = states.reshape(n_trials * n_timesteps, n_units)
    
    # Plot each unit pair
    for idx, (unit1, unit2) in enumerate(unit_pairs):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Create scatter plot
        scatter = ax.scatter(
            states_flat[:, unit1],
            states_flat[:, unit2],
            c=color_values,
            cmap=cmap,
            alpha=alpha,
            s=1,
            rasterized=True
        )
        
        # Labels
        ax.set_xlabel(unit_names[unit1])
        ax.set_ylabel(unit_names[unit2])
        ax.set_title(f"{unit_names[unit1]} vs {unit_names[unit2]}")
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(color_label)
    
    # Remove extra subplots
    for idx in range(n_pairs, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    
    # Save if requested
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {output_path}")
    
    return fig, axes


def create_ordering_functions() -> Dict[str, Callable]:
    """Create common ordering functions for trial grouping."""
    return {
        'hazard_rate': lambda trials: pd.concat([
            trials[trials['Hazard Rate'] == hz] 
            for hz in ['Low', 'High']
        ]),
        
        'hazard_rate_length': lambda trials: pd.concat([
            trials[trials['Hazard Rate'] == hz].sort_values('length', ascending=False)
            for hz in ['Low', 'High']
        ]),
        
        'length': lambda trials: trials.sort_values('length', ascending=False),
        
        'trial_type': lambda trials: pd.concat([
            trials[trials['trial'] == tt]
            for tt in ['straight', 'bounce', 'catch']
            if tt in trials['trial'].values
        ])
    }


def plot_temporal_dynamics(
    temporal_mean: np.ndarray,
    temporal_std: np.ndarray,
    unit_names: List[str],
    figsize: Tuple[int, int] = (10, 6),
    output_path: Optional[Path] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot temporal dynamics of units.
    
    Args:
        temporal_mean: Mean activation over time (timesteps, n_units)
        temporal_std: Std activation over time
        unit_names: Names of units
        figsize: Figure size
        output_path: Path to save figure
        
    Returns:
        Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    timesteps = np.arange(temporal_mean.shape[0])
    
    # Plot each unit
    for unit_idx, unit_name in enumerate(unit_names):
        mean = temporal_mean[:, unit_idx]
        std = temporal_std[:, unit_idx]
        
        # Plot mean with shaded std
        ax.plot(timesteps, mean, label=unit_name, linewidth=2)
        ax.fill_between(
            timesteps,
            mean - std,
            mean + std,
            alpha=0.2
        )
    
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Activation (z-scored)")
    ax.set_title("Temporal Dynamics of Critical Units")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if requested
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {output_path}")
    
    return fig, ax


def plot_condition_comparison(
    condition_stats: Dict[str, Dict],
    unit_names: List[str],
    condition_type: str = "hazard_rate",
    figsize: Tuple[int, int] = (10, 6),
    output_path: Optional[Path] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot comparison of unit activations across conditions.
    
    Args:
        condition_stats: Statistics by condition from tuning analysis
        unit_names: Names of units
        condition_type: Type of condition to plot
        figsize: Figure size
        output_path: Path to save figure
        
    Returns:
        Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get conditions and units
    conditions = list(condition_stats[condition_type].keys())
    n_units = len(unit_names)
    
    # Set up bar positions
    x = np.arange(n_units)
    width = 0.8 / len(conditions)
    
    # Plot bars for each condition
    for idx, condition in enumerate(conditions):
        means = condition_stats[condition_type][condition]['mean']
        stds = condition_stats[condition_type][condition]['std']
        
        offset = (idx - len(conditions)/2 + 0.5) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            label=condition.title(),
            alpha=0.8
        )
    
    # Formatting
    ax.set_xlabel("Unit")
    ax.set_ylabel("Mean Activation (z-scored)")
    ax.set_title(f"Unit Activations by {condition_type.replace('_', ' ').title()}")
    ax.set_xticks(x)
    ax.set_xticklabels(unit_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save if requested
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {output_path}")
    
    return fig, ax


def create_tuning_profile_report(
    tuning_results: Dict[str, Any],
    output_dir: Path,
    states_data: Optional[np.ndarray] = None,
    df_data: Optional[pd.DataFrame] = None
):
    """Create a comprehensive visualization report from tuning results.
    
    Args:
        tuning_results: Results from compute_tuning_profiles
        output_dir: Directory to save plots
        states_data: Normalized states array if available
        df_data: Trial metadata if available
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract key information
    unit_mapping = tuning_results['units_analyzed']['mapping']
    unit_names = [info['name'] for info in unit_mapping.values()]
    
    # 1. Condition comparison plots
    if 'condition_statistics' in tuning_results:
        for condition_type in ['hazard_rate', 'trial_type']:
            if condition_type in tuning_results['condition_statistics']:
                plot_condition_comparison(
                    tuning_results['condition_statistics'],
                    unit_names,
                    condition_type,
                    output_path=output_dir / f"condition_comparison_{condition_type}.pdf"
                )
    
    # 2. Temporal dynamics
    if 'temporal_dynamics' in tuning_results.get('profiles', {}):
        dynamics = tuning_results['profiles']['temporal_dynamics']
        plot_temporal_dynamics(
            np.array(dynamics['mean']),
            np.array(dynamics['std']),
            unit_names,
            output_path=output_dir / "temporal_dynamics.pdf"
        )
    
    # 3. Trial activity heatmaps (if states and metadata available)
    if states_data is not None and df_data is not None:
        ordering_funcs = create_ordering_functions()
        
        # Plot with hazard rate ordering
        plot_unit_trial_activity(
            states_data,
            unit_mapping,
            df_data,
            title="Neural Activity by Trial Type (Grouped by Hazard Rate)",
            order_func=ordering_funcs['hazard_rate'],
            output_path=output_dir / "trial_activity_by_hazard.pdf"
        )
        
        # Plot with length ordering
        plot_unit_trial_activity(
            states_data,
            unit_mapping,
            df_data,
            title="Neural Activity by Trial Type (Sorted by Length)",
            order_func=ordering_funcs['hazard_rate_length'],
            output_path=output_dir / "trial_activity_by_length.pdf"
        )
    
    print(f"Visualization report saved to {output_dir}")