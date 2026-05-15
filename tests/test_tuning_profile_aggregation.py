#!/usr/bin/env python3
"""Test tuning profile aggregation functionality."""

import json
import tempfile
from pathlib import Path
import numpy as np
import pytest
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def create_mock_unit_activities(output_path: Path, n_units: int = 10, n_trials: int = 100):
    """Create mock unit activities npz file."""
    metadata = {
        "window_size": 200,
        "normalization_applied": True,
        "output_format": "both",
        "n_units_extracted": n_units,
        "n_trials": n_trials,
        "n_timesteps": 600,
        "n_hidden_units": 128,
        "n_cell_units": 128
    }
    
    units_analyzed = {
        "indices": list(range(n_units)),
        "mapping": [f"h{i}" for i in range(n_units // 2)] + [f"c{i}" for i in range(n_units // 2)]
    }
    
    # Create mock activity arrays
    activities_windowed_normalized = np.random.randn(n_trials, 200, n_units)
    
    np.savez_compressed(
        output_path,
        metadata=metadata,
        units_analyzed=units_analyzed,
        activities_windowed_normalized=activities_windowed_normalized
    )


def create_mock_activity_matrix(output_path: Path, n_units: int = 10, n_trials: int = 100):
    """Create mock activity matrix npz file."""
    metadata = {
        "matrix_shape": [n_trials, 200, n_units],
        "activity_source": "windowed_normalized",
        "use_windowed": True,
        "use_normalized": True
    }
    
    summary_statistics = {
        "per_unit_mean": np.random.randn(n_units).tolist(),
        "per_unit_std": np.random.rand(n_units).tolist(),
        "per_trial_mean": np.random.randn(n_trials).tolist()
    }
    
    np.savez_compressed(
        output_path,
        metadata=metadata,
        summary_statistics=summary_statistics,
        matrix_full=np.random.randn(n_trials, 200, n_units)
    )


def create_mock_sorted_conditions(output_path: Path, n_trials: int = 100):
    """Create mock sorted conditions npz file."""
    metadata = {
        "sort_conditions": ["hazard_rate", "trial_type"],
        "condition_counts": {
            "Low_Straight": 20,
            "Low_Bounce": 20,
            "High_Straight": 30,
            "High_Bounce": 30
        }
    }
    
    np.savez_compressed(
        output_path,
        metadata=metadata,
        sorted_indices=np.arange(n_trials)
    )


def create_mock_aligned_trajectories(output_path: Path):
    """Create mock aligned trajectories JSON file."""
    data = {
        "metadata": {
            "alignment_method": "time",
            "n_trajectories": 21,
            "trajectory_structure": "7 variants × 3 colors"
        },
        "trajectory_data": {
            "base_trajectory_0": {
                "variants": ["no_change", "random_low", "random_high"],
                "colors": ["red", "green", "blue"]
            }
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def create_mock_event_analysis(output_path: Path, n_units: int = 10):
    """Create mock event analysis npz file."""
    metadata = {
        "event_window": [-5, 10],
        "event_types": ["color_change", "velocity_change"],
        "min_events": 5,
        "n_events_found": {
            "color_change": 45,
            "velocity_change": 38
        }
    }
    
    np.savez_compressed(
        output_path,
        metadata=metadata,
        event_triggered_average_color=np.random.randn(15, n_units),
        event_triggered_average_velocity=np.random.randn(15, n_units)
    )


def test_aggregation_script_import():
    """Test that aggregation script can be imported."""
    try:
        from aggregate_tuning_profiles import aggregate_tuning_components
        assert callable(aggregate_tuning_components)
    except ImportError:
        pytest.fail("Failed to import aggregate_tuning_profiles module")


def test_aggregate_all_components():
    """Test aggregation with all components present."""
    from aggregate_tuning_profiles import aggregate_tuning_components
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        tuning_dir = cache_dir / "tuning_profiles"
        tuning_dir.mkdir(parents=True)
        
        model_id = "TEST-001"
        suffix = "participant"
        
        # Create all mock files
        create_mock_unit_activities(tuning_dir / f"{model_id}_{suffix}_unit_activities.npz")
        create_mock_activity_matrix(tuning_dir / f"{model_id}_{suffix}_activity_matrix.npz")
        create_mock_sorted_conditions(tuning_dir / f"{model_id}_{suffix}_sorted_conditions.npz")
        create_mock_aligned_trajectories(tuning_dir / f"{model_id}_{suffix}_aligned_trajectories.json")
        create_mock_event_analysis(tuning_dir / f"{model_id}_{suffix}_event_analysis.npz")
        
        # Run aggregation
        result = aggregate_tuning_components(cache_dir, model_id, suffix)
        
        # Verify result structure
        assert result["model_id"] == model_id
        assert result["dataset_suffix"] == suffix
        assert "components" in result
        assert "metadata" in result
        assert "summary" in result
        
        # Check all components loaded
        assert result["metadata"]["n_components_loaded"] == 5
        assert all(status == "loaded" for status in result["metadata"]["component_status"].values())
        
        # Check summary statistics
        assert "n_critical_units" in result["summary"]
        assert "n_trials" in result["summary"]
        assert "event_window" in result["summary"]


def test_aggregate_missing_components():
    """Test aggregation with some components missing."""
    from aggregate_tuning_profiles import aggregate_tuning_components
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        tuning_dir = cache_dir / "tuning_profiles"
        tuning_dir.mkdir(parents=True)
        
        model_id = "TEST-002"
        suffix = "extended"
        
        # Create only some mock files
        create_mock_unit_activities(tuning_dir / f"{model_id}_{suffix}_unit_activities.npz")
        create_mock_activity_matrix(tuning_dir / f"{model_id}_{suffix}_activity_matrix.npz")
        # Skip sorted_conditions, aligned_trajectories, event_analysis
        
        # Run aggregation
        result = aggregate_tuning_components(cache_dir, model_id, suffix)
        
        # Check partial loading
        assert result["metadata"]["n_components_loaded"] == 2
        assert result["metadata"]["component_status"]["unit_activities"] == "loaded"
        assert result["metadata"]["component_status"]["activity_matrix"] == "loaded"
        assert result["metadata"]["component_status"]["sorted_conditions"] == "missing"
        assert result["metadata"]["component_status"]["aligned_trajectories"] == "missing"
        assert result["metadata"]["component_status"]["event_analysis"] == "missing"


def test_aggregate_corrupted_file():
    """Test aggregation with corrupted file."""
    from aggregate_tuning_profiles import aggregate_tuning_components
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        tuning_dir = cache_dir / "tuning_profiles"
        tuning_dir.mkdir(parents=True)
        
        model_id = "TEST-003"
        suffix = "participant"
        
        # Create valid file
        create_mock_unit_activities(tuning_dir / f"{model_id}_{suffix}_unit_activities.npz")
        
        # Create corrupted npz file
        bad_file = tuning_dir / f"{model_id}_{suffix}_activity_matrix.npz"
        with open(bad_file, 'wb') as f:
            f.write(b"This is not a valid npz file")
        
        # Run aggregation
        result = aggregate_tuning_components(cache_dir, model_id, suffix)
        
        # Check error handling
        assert result["metadata"]["n_components_loaded"] == 1
        assert result["metadata"]["component_status"]["unit_activities"] == "loaded"
        assert result["metadata"]["component_status"]["activity_matrix"] == "error"
        assert "error" in result["components"]["activity_matrix"]


def test_extract_summary_statistics():
    """Test summary statistics extraction."""
    from aggregate_tuning_profiles import extract_summary_statistics
    
    # Create mock result structure
    result = {
        "components": {
            "unit_activities": {
                "units_analyzed": {
                    "indices": [1, 5, 10, 15, 20],
                    "mapping": ["h1", "h5", "c10", "c15", "c20"]
                },
                "metadata": {
                    "n_trials": 168,
                    "n_timesteps": 600,
                    "window_size": 200
                }
            },
            "sorted_conditions": {
                "metadata": {
                    "condition_counts": {
                        "Low_Straight": 40,
                        "Low_Bounce": 41,
                        "High_Straight": 40,
                        "High_Bounce": 36
                    }
                }
            },
            "event_analysis": {
                "metadata": {
                    "event_window": [-5, 10],
                    "event_types": ["color_change", "velocity_change"]
                }
            }
        }
    }
    
    summary = extract_summary_statistics(result)
    
    # Verify summary contents
    assert summary["n_critical_units"] == 5
    assert summary["n_trials"] == 168
    assert summary["window_size"] == 200
    assert summary["unit_types"]["hidden"] == 2
    assert summary["unit_types"]["cell"] == 3
    assert summary["condition_counts"]["Low_Straight"] == 40
    assert summary["event_window"] == [-5, 10]
    assert summary["event_types_analyzed"] == ["color_change", "velocity_change"]


def test_command_line_interface():
    """Test CLI argument parsing."""
    from aggregate_tuning_profiles import main
    import sys
    
    # Mock sys.argv
    old_argv = sys.argv
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            tuning_dir = cache_dir / "tuning_profiles"
            tuning_dir.mkdir(parents=True)
            
            output_file = tuning_dir / "TEST-004_tuning.json"
            
            # Create minimal files
            create_mock_unit_activities(tuning_dir / "TEST-004_participant_unit_activities.npz")
            
            sys.argv = [
                "aggregate_tuning_profiles.py",
                "--cache-dir", str(cache_dir),
                "--model-id", "TEST-004",
                "--dataset-suffix", "participant",
                "--output", str(output_file)
            ]
            
            # Should exit with code 2 (partial success) since only 1 component exists
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2
            
            # Check output file was created
            assert output_file.exists()
            
            # Verify content
            with open(output_file) as f:
                data = json.load(f)
            assert data["model_id"] == "TEST-004"
            assert data["metadata"]["n_components_loaded"] == 1
            
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    pytest.main([__file__, "-v"])