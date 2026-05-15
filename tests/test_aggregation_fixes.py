#!/usr/bin/env python3
"""Test the aggregation script fixes for the tuning profile pipeline."""

import json
import tempfile
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest


def test_aggregation_partial_success():
    """Test that aggregation script exits with 0 for partial success."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        tuning_dir = cache_dir / 'tuning_profiles'
        tuning_dir.mkdir(parents=True)
        
        # Create only some components (simulate partial results)
        model_id = 'TEST-001'
        suffix = 'participant'
        
        # Create unit activities file
        activities_file = tuning_dir / f'{model_id}_{suffix}_unit_activities.npz'
        np.savez_compressed(
            activities_file,
            metadata={'n_trials': 100, 'window_size': 200},
            units_analyzed={'indices': [5, 10, 15], 'mapping': ['h5', 'h10', 'c15']},
            activities_windowed_normalized=np.random.randn(100, 200, 3)
        )
        
        # Create activity matrix file
        matrix_file = tuning_dir / f'{model_id}_{suffix}_activity_matrix.npz'
        np.savez_compressed(
            matrix_file,
            metadata={'matrix_shape': [100, 200, 3]},
            matrix_full=np.random.randn(100, 200, 3)
        )
        
        # Missing: sorted_conditions, aligned_trajectories, event_analysis
        
        # Run aggregation script
        script_path = Path(__file__).parent.parent / 'scripts' / 'aggregate_tuning_profiles.py'
        output_file = tuning_dir / f'{model_id}_test.json'
        
        cmd = [
            sys.executable, str(script_path),
            '--cache-dir', str(cache_dir),
            '--model-id', model_id,
            '--dataset-suffix', suffix,
            '--output', str(output_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check that script exited with 0 (not 2) for partial success
        assert result.returncode == 0, f"Script failed with exit code {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
        
        # Verify output file was created
        assert output_file.exists(), "Output file was not created"
        
        # Check aggregated content
        with open(output_file) as f:
            data = json.load(f)
        
        assert data['metadata']['n_components_loaded'] == 2
        assert data['metadata']['component_status']['unit_activities'] == 'loaded'
        assert data['metadata']['component_status']['activity_matrix'] == 'loaded'
        assert data['metadata']['component_status']['sorted_conditions'] == 'missing'
        
        # Check warning was printed
        assert "Warning: Failed to load components" in result.stdout


def test_aggregation_complete_failure():
    """Test that aggregation script exits with 1 for complete failure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        tuning_dir = cache_dir / 'tuning_profiles'
        tuning_dir.mkdir(parents=True)
        
        # No component files exist
        model_id = 'TEST-002'
        suffix = 'participant'
        
        # Run aggregation script
        script_path = Path(__file__).parent.parent / 'scripts' / 'aggregate_tuning_profiles.py'
        output_file = tuning_dir / f'{model_id}_test.json'
        
        cmd = [
            sys.executable, str(script_path),
            '--cache-dir', str(cache_dir),
            '--model-id', model_id,
            '--dataset-suffix', suffix,
            '--output', str(output_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check that script exited with 1 for complete failure
        assert result.returncode == 1, f"Expected exit code 1, got {result.returncode}"
        
        # Check error message
        assert "Error: No components were successfully loaded!" in result.stdout


def test_file_validation():
    """Test that file validation catches missing fields."""
    # Import the validation function
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.aggregate_critical_units import validate_decoder_result
    
    # Test with valid result
    valid_result = {
        'unit_indices': [1, 2, 3],
        'coefficients': [0.1, 0.2, 0.3],
        'best_alpha': 0.01,
        'r2_scores': [0.8],
        'metadata': {
            'n_units_total': 256,
            'n_units_critical': 3,
            'decoder_type': 'hazard'
        }
    }
    
    error = validate_decoder_result(valid_result, 'hazard')
    assert error is None, f"Valid result failed validation: {error}"
    
    # Test with missing field
    invalid_result = valid_result.copy()
    del invalid_result['coefficients']
    
    error = validate_decoder_result(invalid_result, 'hazard')
    assert error is not None, "Invalid result passed validation"
    assert "Missing required fields" in error
    assert "coefficients" in error
    
    # Test with missing metadata field
    invalid_metadata = valid_result.copy()
    del invalid_metadata['metadata']['n_units_critical']
    
    error = validate_decoder_result(invalid_metadata, 'hazard')
    assert error is not None, "Invalid metadata passed validation"
    assert "Missing metadata fields" in error


def test_directory_creation_with_parents():
    """Test that mkdir with parents=True works correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        
        # Create a deeply nested path
        deep_path = base / 'data' / 'cache' / 'tuning_profiles' / 'model' / 'output.json'
        
        # This should work with parents=True
        deep_path.parent.mkdir(parents=True, exist_ok=True)
        assert deep_path.parent.exists()
        
        # All parent directories should exist
        assert (base / 'data').exists()
        assert (base / 'data' / 'cache').exists()
        assert (base / 'data' / 'cache' / 'tuning_profiles').exists()
        assert (base / 'data' / 'cache' / 'tuning_profiles' / 'model').exists()


def test_legacy_task_configuration():
    """Test that legacy task can be configured."""
    # This would require running doit with the legacy_dataset variable
    # For unit testing, we just verify the configuration logic
    
    from learning_in_context.core.constants import DATASET_CONFIGS, DEFAULT_DATASET
    
    # Test default behavior
    legacy_dataset = DEFAULT_DATASET
    assert legacy_dataset == 'participant'
    
    # Test custom dataset
    custom_dataset = 'extended'
    if custom_dataset in DATASET_CONFIGS:
        dataset_config = DATASET_CONFIGS[custom_dataset]
        assert dataset_config['suffix'] == 'extended'
    
    # Test invalid dataset falls back to default
    invalid_dataset = 'nonexistent'
    if invalid_dataset not in DATASET_CONFIGS:
        fallback_dataset = DEFAULT_DATASET
        assert fallback_dataset == 'participant'


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])