"""Generate figures for model analysis results."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from .critical_units_plots import generate_critical_units_visualizations


def generate_all_model_figures(
    model_id: str,
    cache_dir: Path,
    output_dir: Path,
    decoders: Optional[List[str]] = None,
    dataset_suffix: str = "participant",
    create_summary: bool = True
) -> Dict[str, str]:
    """Generate all available figures for a model.
    
    Args:
        model_id: Model identifier
        cache_dir: Directory containing cached results
        output_dir: Directory to save figures
        decoders: List of decoder types to process
        dataset_suffix: Dataset suffix for cache files
        create_summary: Whether to create a summary of generated figures
        
    Returns:
        Dictionary mapping figure type to status/path
    """
    if decoders is None:
        decoders = ['hazard', 'contingency', 'color', 'velocity_x', 'velocity_y']
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir)
    
    generated_figures = {}
    
    # Generate critical units visualizations
    print(f"Generating critical units visualizations for {model_id}...")
    critical_units_generated = []
    
    for decoder in decoders:
        cache_file = cache_dir / 'critical_units' / f'{model_id}_{dataset_suffix}_{decoder}_units.json'
        output_file = output_dir / f'{decoder}_coefficient_analysis.png'
        
        if cache_file.exists():
            try:
                # Check if visualization data is available
                with open(cache_file, 'r') as f:
                    result_data = json.load(f)
                
                required_keys = ['coefficient_paths', 'intercept_paths', 'performance_curves', 'alpha_values']
                if all(key in result_data for key in required_keys):
                    # Use existing function to generate individual plot
                    generate_critical_units_visualizations(
                        model_id=model_id,
                        cache_dir=cache_dir,
                        output_dir=output_dir,
                        decoders=[decoder],
                        dataset_suffix=dataset_suffix
                    )
                    critical_units_generated.append(decoder)
                    generated_figures[f'critical_units_{decoder}'] = str(output_file)
                else:
                    print(f"  Warning: Visualization data not available for {decoder}")
                    generated_figures[f'critical_units_{decoder}'] = "missing_visualization_data"
            except Exception as e:
                print(f"  Error generating {decoder} visualization: {e}")
                generated_figures[f'critical_units_{decoder}'] = f"error: {str(e)}"
        else:
            generated_figures[f'critical_units_{decoder}'] = "cache_not_found"
    
    if critical_units_generated:
        print(f"  Generated critical units plots for: {', '.join(critical_units_generated)}")
    
    # Create completion flag
    completion_flag = output_dir / 'figures_complete.flag'
    with open(completion_flag, 'w') as f:
        f.write(f"Figures generated for {model_id}\n")
        f.write(f"Critical units: {', '.join(critical_units_generated)}\n")
        f.write(f"Total figures: {len([v for v in generated_figures.values() if 'png' in str(v)])}\n")
    
    # Create summary JSON if requested
    if create_summary:
        summary_file = output_dir / 'figure_summary.json'
        with open(summary_file, 'w') as f:
            json.dump({
                'model_id': model_id,
                'dataset_suffix': dataset_suffix,
                'figures': generated_figures,
                'decoders_processed': critical_units_generated
            }, f, indent=2)
    
    return generated_figures


def main():
    """Main entry point for model figure generation."""
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
    
    # Generate all figures
    results = generate_all_model_figures(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        decoders=args.decoders,
        dataset_suffix=args.dataset_suffix
    )
    
    # Print summary
    successful = [k for k, v in results.items() if 'png' in str(v)]
    if successful:
        print(f"\nSuccessfully generated {len(successful)} figures for {args.model_id}")
    else:
        print(f"\nNo figures generated for {args.model_id}")


if __name__ == '__main__':
    main()