"""Visualization functions for critical units analysis."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plot_coefficient_analysis(
    coefficient_paths: np.ndarray,
    intercept_paths: np.ndarray,
    performance_curves: Dict[str, np.ndarray],
    alpha_values: np.ndarray,
    unit_indices: np.ndarray,
    critical_indices: np.ndarray,
    output_path: Path,
    task_variable: str = "unknown",
    cmap: str = 'seismic',
    figsize: tuple = (12, 10),
    ylabel_show_every: int = 2,
    hline_chance: Optional[float] = None,
    vline_performance: Optional[tuple] = None
):
    """Create coefficient analysis visualization from cached data.
    
    This function creates a 3-panel visualization showing:
    1. Performance metrics vs regularization strength
    2. Intercept values vs regularization strength  
    3. Coefficient heatmap across regularization strengths
    
    Args:
        coefficient_paths: Coefficient values across alphas (n_units, n_alphas)
        intercept_paths: Intercept values across alphas (n_classes_or_1, n_alphas)
        performance_curves: Dict of performance metrics {'accuracy': [...], 'r2': [...]}
        alpha_values: Regularization strength values used
        unit_indices: All unit indices used in analysis
        critical_indices: Indices of critical units identified
        output_path: Path to save the figure
        task_variable: Name of task variable for titles
        cmap: Colormap for coefficient heatmap
        figsize: Figure size tuple
        ylabel_show_every: Show y-axis labels every N units
        hline_chance: Horizontal line for chance performance
        vline_performance: Tuple of (index, metric_name) for performance threshold
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    coefs = coefficient_paths.T  # Transpose to match original format (n_alphas, n_units)
    intercepts = intercept_paths.T if intercept_paths.size > 0 else np.array([[]])
    C_logspace = 1.0 / alpha_values  # Convert back to C values for x-axis
    
    # Create figure titles
    task_title = task_variable.replace('_', ' ').title()
    heatmap_title = f"{task_title} Regression Coefficients with Decreasing ElasticNet Alpha"
    metrics_title = f"{task_title} Metrics with Decreasing ElasticNet Alpha"
    intercept_title = f"{task_title} Intercept Value with Decreasing ElasticNet Alpha"
    xlabel = "ElasticNet Alpha"
    heatmap_ylabel = "(H)idden / (C)ell Unit Number"
    
    # Create unit labels
    n_units = coefs.shape[1]
    unit_number = [f"{state}{number:0>2}" for state in ["H", "C"] for number in range(n_units//2)]
    if len(unit_number) < n_units:
        # Handle case where we don't have exactly hidden+cell units
        unit_number = [f"U{i:0>2}" for i in range(n_units)]
    
    # Set up subplot layout
    height_ratios = [1, 1, max(2, len(unit_number) / 10)]
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1,
        figsize=figsize,
        gridspec_kw={'height_ratios': height_ratios}
    )
    
    # Top panel: Performance metrics
    for label, values in performance_curves.items():
        values = np.array(values)
        if len(values.shape) > 1:
            for i, sub in enumerate(values.T):
                ax1.plot(alpha_values, sub, label=f"{label.title()} - Label {i}")
        else:
            ax1.plot(alpha_values, values, label=label.title())
    
    if hline_chance:
        ax1.axhline(hline_chance, color="grey", ls="--", label=f"Chance: {int(hline_chance*100)}%")
    
    if vline_performance:
        idx, metric_name = vline_performance
        metric_value = performance_curves[metric_name.lower()][idx]
        ax1.axvline(
            alpha_values[idx],
            color="grey",
            ls="-.",
            label=f"{metric_name.title()}: {int(metric_value*100)}%",
        )
    
    ax1.set_xscale('log')
    ax1.set_xlim(alpha_values[0], alpha_values[-1])
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Score')
    ax1.legend(loc="lower left")
    ax1.set_title(metrics_title)
    
    # Middle panel: Intercepts
    if intercepts.size > 0:
        for i, row in enumerate(intercepts.T):
            if intercepts.shape[1] == 1:
                label = None
            else:
                label = f"Intercept - Class {i}"
            ax2.plot(alpha_values, row, label=label)
        
        if intercepts.shape[1] > 1:
            ax2.legend()
    
    ax2.set_xscale('log')
    ax2.set_xlim(alpha_values[0], alpha_values[-1])
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('Intercept Value')
    ax2.set_title(intercept_title)
    
    if vline_performance:
        ax2.axvline(
            alpha_values[vline_performance[0]],
            color="grey",
            ls="-.",
        )
    
    # Bottom panel: Coefficient heatmap
    if coefs.size > 0:
        # Compute symmetric bounds for colormap
        abs_max = max(abs(coefs.min()), abs(coefs.max()))
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
        
        # Create meshgrid
        X, Y = np.meshgrid(alpha_values, np.arange(n_units))
        im1 = ax3.pcolor(X, Y, coefs.T, cmap=cmap, norm=norm)
        
        # Set y-axis ticks and labels
        ax3.set_yticks(np.arange(0, n_units, ylabel_show_every))
        ax3.set_yticklabels([unit_number[i] for i in range(0, n_units, ylabel_show_every)])
        ax3.set_ylabel(heatmap_ylabel)
        
        ax3.set_xscale('log')
        ax3.set_xlim(alpha_values[0], alpha_values[-1])
        ax3.set_xlabel(xlabel)
        ax3.set_title(heatmap_title)
        
        if vline_performance:
            ax3.axvline(
                alpha_values[vline_performance[0]],
                color="grey",
                ls="-.",
            )
        
        # Add colorbar
        axins = inset_axes(
            ax3,
            height="97%",
            width="2.5%",
            loc="right",
        )
        cb = plt.colorbar(
            im1,
            cax=axins,
            orientation="vertical",
        )
        axins.yaxis.set_ticks_position("left")
        cb.set_label("Coefficient Value", labelpad=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved coefficient analysis plot to {output_path}")


def generate_critical_units_visualizations(
    model_id: str,
    cache_dir: Path,
    output_dir: Path,
    decoders: Optional[List[str]] = None,
    dataset_suffix: str = "participant"
):
    """Generate all critical units visualizations from cached results.
    
    Args:
        model_id: Model identifier
        cache_dir: Directory containing cached critical units results
        output_dir: Directory to save visualizations
        decoders: List of decoder types to process
        dataset_suffix: Dataset suffix for cache files
    """
    if decoders is None:
        decoders = ['hazard', 'contingency', 'color', 'velocity_x', 'velocity_y']
    
    cache_dir = Path(cache_dir) 
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for decoder in decoders:
        cache_file = cache_dir / 'critical_units' / f'{model_id}_{dataset_suffix}_{decoder}_units.json'
        
        if not cache_file.exists():
            print(f"Warning: Cache file not found: {cache_file}")
            continue
        
        # Load cached result
        try:
            with open(cache_file, 'r') as f:
                result_data = json.load(f)
            
            # Check if visualization data is available
            required_keys = ['coefficient_paths', 'intercept_paths', 'performance_curves', 'alpha_values']
            if not all(key in result_data for key in required_keys):
                print(f"Warning: Visualization data not available in {cache_file}")
                continue
            
            # Generate visualization
            plot_coefficient_analysis(
                coefficient_paths=np.array(result_data['coefficient_paths']),
                intercept_paths=np.array(result_data['intercept_paths']),
                performance_curves={k: np.array(v) for k, v in result_data['performance_curves'].items()},
                alpha_values=np.array(result_data['alpha_values']),
                unit_indices=np.arange(len(result_data['coefficient_paths'])),  # Reconstruct unit indices
                critical_indices=np.array(result_data['unit_indices']),
                output_path=output_dir / f'{model_id}_{decoder}_coefficient_analysis.png',
                task_variable=decoder
            )
            
        except Exception as e:
            print(f"Error processing {cache_file}: {e}")
            continue


def main():
    """Main entry point for critical units visualization generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-id', type=str, required=True, 
                        help='Model identifier')
    parser.add_argument('--cache-dir', type=Path, required=True,
                        help='Directory containing cached results')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Output directory for figures')
    parser.add_argument('--decoders', type=str, nargs='+',
                        default=['hazard', 'contingency', 'color', 'velocity_x', 'velocity_y'],
                        help='Decoder types to process')
    parser.add_argument('--dataset-suffix', type=str, default='participant',
                        help='Dataset suffix for cache files')
    
    args = parser.parse_args()
    
    generate_critical_units_visualizations(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        decoders=args.decoders,
        dataset_suffix=args.dataset_suffix
    )


if __name__ == '__main__':
    main()