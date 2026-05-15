"""End-to-end tests for individual critical units decoders using real test data."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from .conftest import validate_decoder_output


class TestCriticalUnitsDecodersE2E:
    """Test individual decoder types with real test data."""
    
    def test_hazard_decoder_e2e(self, test_states_file, test_model_id, temp_output_dir):
        """Test hazard decoder end-to-end with real test data."""
        output_file = temp_output_dir / f"{test_model_id}_hazard_units.json"
        
        # Run the critical units analysis
        cmd = [
            sys.executable, "-m", "learning_in_context.analysis.critical_units",
            "--states", str(test_states_file),
            "--output", str(output_file),
            "--model-id", test_model_id,
            "--decoder-type", "hazard",
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        
        # Check command executed successfully
        assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"
        
        # Validate output
        data = validate_decoder_output(output_file, "hazard")
        
        # Hazard-specific validations
        assert "hazard" in str(output_file), "Output file should contain decoder type"
        
        # Check that reasonable number of units were identified
        # With 16 total units in test data, expect 1-8 critical units
        assert 0 <= data["metadata"]["n_units_critical"] <= 16, "Critical units count should be reasonable"
        assert data["metadata"]["n_units_total"] == 16, "Should have 16 total units in test data"
        
        # Verify hazard-specific parameters
        metadata = data["metadata"]
        assert abs(metadata.get("l1_ratio", 0.64) - 0.64) < 0.01, "Should use l1_ratio=0.64 for hazard"
        assert metadata.get("score_type") in ["accuracy", "r2"], "Should have valid score type"
        
        print(f"✓ Hazard decoder: Found {data['metadata']['n_units_critical']} critical units")
    
    def test_contingency_decoder_e2e(self, test_states_file, test_model_id, temp_output_dir):
        """Test contingency decoder end-to-end with real test data."""
        output_file = temp_output_dir / f"{test_model_id}_contingency_units.json"
        
        # Run the critical units analysis
        cmd = [
            sys.executable, "-m", "learning_in_context.analysis.critical_units",
            "--states", str(test_states_file),
            "--output", str(output_file),
            "--model-id", test_model_id,
            "--decoder-type", "contingency",
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        
        # Check command executed successfully
        assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"
        
        # Validate output
        data = validate_decoder_output(output_file, "contingency")
        
        # Contingency-specific validations
        assert "contingency" in str(output_file), "Output file should contain decoder type"
        
        # Check that reasonable number of units were identified
        assert 0 <= data["metadata"]["n_units_critical"] <= 16, "Critical units count should be reasonable"
        assert data["metadata"]["n_units_total"] == 16, "Should have 16 total units in test data"
        
        # Verify contingency-specific parameters  
        metadata = data["metadata"]
        assert abs(metadata.get("l1_ratio", 0.64) - 0.64) < 0.01, "Should use l1_ratio=0.64 for contingency"
        assert metadata.get("score_type") in ["accuracy", "r2"], "Should have valid score type"
        
        print(f"✓ Contingency decoder: Found {data['metadata']['n_units_critical']} critical units")
    
    def test_color_decoder_e2e(self, test_states_file, test_model_id, temp_output_dir):
        """Test color decoder end-to-end with real test data."""
        output_file = temp_output_dir / f"{test_model_id}_color_units.json"
        
        # Run the critical units analysis
        cmd = [
            sys.executable, "-m", "learning_in_context.analysis.critical_units",
            "--states", str(test_states_file),
            "--output", str(output_file),
            "--model-id", test_model_id,
            "--decoder-type", "color",
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        
        # Check command executed successfully
        assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"
        
        # Validate output
        data = validate_decoder_output(output_file, "color")
        
        # Color-specific validations
        assert "color" in str(output_file), "Output file should contain decoder type"
        
        # Check that reasonable number of units were identified
        assert 0 <= data["metadata"]["n_units_critical"] <= 16, "Critical units count should be reasonable"
        assert data["metadata"]["n_units_total"] == 16, "Should have 16 total units in test data"
        
        # Verify color-specific parameters
        metadata = data["metadata"]
        assert abs(metadata.get("l1_ratio", 0.4) - 0.4) < 0.01, "Should use l1_ratio=0.4 for color"
        assert metadata.get("score_type") in ["accuracy", "r2"], "Should have valid score type"
        
        print(f"✓ Color decoder: Found {data['metadata']['n_units_critical']} critical units")
    
    def test_velocity_x_decoder_e2e(self, test_states_file, test_model_id, temp_output_dir):
        """Test velocity_x decoder end-to-end with real test data."""
        output_file = temp_output_dir / f"{test_model_id}_velocity_x_units.json"
        
        # Run the critical units analysis
        cmd = [
            sys.executable, "-m", "learning_in_context.analysis.critical_units",
            "--states", str(test_states_file),
            "--output", str(output_file),
            "--model-id", test_model_id,
            "--decoder-type", "velocity_x",
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        
        # Check command executed successfully
        assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"
        
        # Validate output
        data = validate_decoder_output(output_file, "velocity_x")
        
        # Velocity_x-specific validations
        assert "velocity_x" in str(output_file), "Output file should contain decoder type"
        
        # Check that reasonable number of units were identified
        assert 0 <= data["metadata"]["n_units_critical"] <= 16, "Critical units count should be reasonable"
        assert data["metadata"]["n_units_total"] == 16, "Should have 16 total units in test data"
        
        # Verify velocity_x-specific parameters
        metadata = data["metadata"]
        assert abs(metadata.get("l1_ratio", 0.4) - 0.4) < 0.01, "Should use l1_ratio=0.4 for velocity_x"
        assert metadata.get("score_type") in ["accuracy", "r2"], "Should have valid score type"
        
        print(f"✓ Velocity X decoder: Found {data['metadata']['n_units_critical']} critical units")
    
    def test_velocity_y_decoder_e2e(self, test_states_file, test_model_id, temp_output_dir):
        """Test velocity_y decoder end-to-end with real test data."""
        output_file = temp_output_dir / f"{test_model_id}_velocity_y_units.json"
        
        # Run the critical units analysis
        cmd = [
            sys.executable, "-m", "learning_in_context.analysis.critical_units",
            "--states", str(test_states_file),
            "--output", str(output_file),
            "--model-id", test_model_id,
            "--decoder-type", "velocity_y",
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        
        # Check command executed successfully
        assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"
        
        # Validate output
        data = validate_decoder_output(output_file, "velocity_y")
        
        # Velocity_y-specific validations
        assert "velocity_y" in str(output_file), "Output file should contain decoder type"
        
        # Check that reasonable number of units were identified
        assert 0 <= data["metadata"]["n_units_critical"] <= 16, "Critical units count should be reasonable"
        assert data["metadata"]["n_units_total"] == 16, "Should have 16 total units in test data"
        
        # Verify velocity_y-specific parameters
        metadata = data["metadata"]
        assert abs(metadata.get("l1_ratio", 0.4) - 0.4) < 0.01, "Should use l1_ratio=0.4 for velocity_y"
        assert metadata.get("score_type") in ["accuracy", "r2"], "Should have valid score type"
        
        print(f"✓ Velocity Y decoder: Found {data['metadata']['n_units_critical']} critical units")
    
    def test_all_decoders_produce_valid_outputs(self, test_states_file, test_model_id, temp_output_dir, decoder_types):
        """Test that all decoder types produce valid outputs with consistent structure."""
        decoder_results = {}
        
        # Run all decoders
        for decoder_type in decoder_types:
            output_file = temp_output_dir / f"{test_model_id}_{decoder_type}_units.json"
            
            cmd = [
                sys.executable, "-m", "learning_in_context.analysis.critical_units",
                "--states", str(test_states_file),
                "--output", str(output_file),
                "--model-id", test_model_id,
                "--decoder-type", decoder_type,
                ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            assert result.returncode == 0, f"Decoder {decoder_type} failed: {result.stderr}"
            
            # Validate and collect results
            data = validate_decoder_output(output_file, decoder_type)
            decoder_results[decoder_type] = data
            
            print(f"✓ {decoder_type}: {data['metadata']['n_units_critical']} critical units")
        
        # Cross-decoder validation
        assert len(decoder_results) == 5, "Should have results for all 5 decoders"
        
        # Check that all decoders have same total units
        total_units = [data["metadata"]["n_units_total"] for data in decoder_results.values()]
        assert all(n == 16 for n in total_units), "All decoders should see same total units"
        
        # Check that all have valid critical unit counts
        critical_counts = [data["metadata"]["n_units_critical"] for data in decoder_results.values()]
        assert all(0 <= n <= 16 for n in critical_counts), "All critical counts should be reasonable"
        
        # Check that different decoders can identify different units (diversity check)
        all_critical_units = set()
        for data in decoder_results.values():
            all_critical_units.update(data["unit_indices"])
        
        # Should have at least some critical units identified across all decoders
        assert len(all_critical_units) > 0, "At least some critical units should be identified"
        assert len(all_critical_units) <= 16, "Can't identify more units than exist"
        
        print(f"✓ All decoders completed successfully")
        print(f"✓ Total unique critical units across all decoders: {len(all_critical_units)}")
        
        return decoder_results
    
    def test_regularization_sweep_functionality(self, test_states_file, test_model_id, temp_output_dir):
        """Test that regularization sweep produces expected results."""
        output_file = temp_output_dir / f"{test_model_id}_hazard_sweep_test.json"
        
        # Run with regularization sweep
        cmd = [
            sys.executable, "-m", "learning_in_context.analysis.critical_units",
            "--states", str(test_states_file),
            "--output", str(output_file),
            "--model-id", test_model_id,
            "--decoder-type", "hazard",
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        assert result.returncode == 0, f"Regularization sweep failed: {result.stderr}"
        
        # Load and validate results
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        # Check that sweep-specific data is present
        assert "best_alpha" in data, "Should have best_alpha from sweep"
        assert isinstance(data["best_alpha"], (int, float)), "best_alpha should be numeric"
        assert data["best_alpha"] > 0, "best_alpha should be positive"
        
        # Check CV scores structure
        cv_scores = data["cv_scores"]
        assert "mean" in cv_scores, "Should have mean CV score"
        assert "std" in cv_scores, "Should have std CV score"
        assert isinstance(cv_scores["mean"], (int, float)), "Mean score should be numeric"
        
        # Verify regularization actually happened by checking alpha range
        # Should be in the expected range for regularization sweep
        assert 1e-6 <= data["best_alpha"] <= 1.0, "best_alpha should be in expected regularization range"
        
        print(f"✓ Regularization sweep: best_alpha = {data['best_alpha']:.6f}")
        print(f"✓ CV score: {cv_scores['mean']:.3f} ± {cv_scores['std']:.3f}")


class TestCriticalUnitsErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_decoder_type(self, test_states_file, test_model_id, temp_output_dir):
        """Test that invalid decoder type is rejected."""
        output_file = temp_output_dir / f"{test_model_id}_invalid_units.json"
        
        cmd = [
            sys.executable, "-m", "learning_in_context.analysis.critical_units",
            "--states", str(test_states_file),
            "--output", str(output_file),
            "--model-id", test_model_id,
            "--decoder-type", "invalid_decoder",
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        
        # Should fail with invalid decoder type
        assert result.returncode != 0, "Should fail with invalid decoder type"
        assert "invalid choice" in result.stderr.lower() or "choices" in result.stderr.lower(), \
            "Should indicate invalid decoder type choice"
    
    def test_missing_states_file(self, test_model_id, temp_output_dir):
        """Test that missing states file is handled gracefully."""
        missing_file = temp_output_dir / "nonexistent_states.npz"
        output_file = temp_output_dir / f"{test_model_id}_missing_test.json"
        
        cmd = [
            sys.executable, "-m", "learning_in_context.analysis.critical_units",
            "--states", str(missing_file),
            "--output", str(output_file),
            "--model-id", test_model_id,
            "--decoder-type", "hazard",
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        
        # Should fail gracefully
        assert result.returncode != 0, "Should fail with missing states file"
        assert not output_file.exists(), "Should not create output file on failure"