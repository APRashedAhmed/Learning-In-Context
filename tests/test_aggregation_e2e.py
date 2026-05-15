"""End-to-end tests for critical units aggregation using real test data."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from .conftest import validate_decoder_output, validate_aggregated_output


class TestCriticalUnitsAggregationE2E:
    """Test aggregation logic with real test data."""
    
    def test_full_aggregation_pipeline(self, test_states_file, test_model_id, temp_output_dir, decoder_types):
        """Test complete aggregation pipeline from individual decoders to unified output."""
        
        # Step 1: Run all individual decoder analyses
        print("Step 1: Running individual decoder analyses...")
        decoder_results = {}
        decoder_files = []
        
        for decoder_type in decoder_types:
            output_file = temp_output_dir / f"{test_model_id}_{decoder_type}_units.json"
            decoder_files.append(output_file)
            
            cmd = [
                sys.executable, "-m", "learning_in_context.analysis.critical_units",
                "--states", str(test_states_file),
                "--output", str(output_file),
                "--model-id", test_model_id,
                "--decoder-type", decoder_type,
                "--regularization-sweep",
                "--use-all-timesteps",
                "--concatenate-states"
                # Note: --use-normalized is omitted since test file contains raw states
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            assert result.returncode == 0, f"Decoder {decoder_type} failed: {result.stderr}"
            
            # Validate and store results
            data = validate_decoder_output(output_file, decoder_type)
            decoder_results[decoder_type] = data
            
            print(f"  ✓ {decoder_type}: {data['metadata']['n_units_critical']} critical units")
        
        # Step 2: Create cache directory structure for aggregation script
        cache_dir = temp_output_dir / "cache"
        critical_units_dir = cache_dir / "critical_units"
        critical_units_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy decoder files to expected location for aggregation script
        for decoder_type, source_file in zip(decoder_types, decoder_files):
            dest_file = critical_units_dir / f"{test_model_id}_{decoder_type}_units.json"
            dest_file.write_text(source_file.read_text())
        
        # Step 3: Run aggregation script
        print("Step 2: Running aggregation script...")
        aggregated_file = critical_units_dir / f"{test_model_id}_units.json"
        
        aggregation_script = Path("scripts/aggregate_critical_units.py")
        cmd = [
            sys.executable, str(aggregation_script),
            "--cache-dir", str(cache_dir),
            "--model-id", test_model_id,
            "--output", str(aggregated_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        assert result.returncode == 0, f"Aggregation failed: {result.stderr}"
        
        # Step 4: Validate aggregated output
        print("Step 3: Validating aggregated output...")
        aggregated_data = validate_aggregated_output(aggregated_file, decoder_results)
        
        # Additional aggregation-specific validations
        metadata = aggregated_data["metadata"]
        
        # Check that all decoders were processed
        assert metadata["n_decoders"] == 5, "Should have processed 5 decoders"
        assert metadata["n_decoders_with_results"] == 5, "All decoders should have produced results"
        
        # Verify union logic
        individual_units = set()
        for decoder_data in decoder_results.values():
            individual_units.update(decoder_data["unit_indices"])
        
        aggregated_units = set(aggregated_data["unit_indices"])
        assert aggregated_units == individual_units, "Aggregated units should be union of individual results"
        
        # Check overlap statistics
        total_individual = sum(len(d["unit_indices"]) for d in decoder_results.values())
        expected_overlap = 1 - len(aggregated_units) / total_individual if total_individual > 0 else 0
        assert abs(metadata["overlap_ratio"] - expected_overlap) < 0.01, "Overlap ratio should be calculated correctly"
        
        # Verify backward compatibility keys
        required_keys = ["model_id", "unit_indices", "coefficients", "r2_scores", "best_alpha", "cv_scores"]
        for key in required_keys:
            assert key in aggregated_data, f"Missing backward compatibility key: {key}"
        
        # Check that individual decoder results are preserved
        assert "decoder_results" in metadata, "Should preserve individual decoder results"
        assert len(metadata["decoder_results"]) == 5, "Should preserve all decoder results"
        
        print(f"  ✓ Total individual critical units: {total_individual}")
        print(f"  ✓ Unique critical units after union: {len(aggregated_units)}")
        print(f"  ✓ Overlap ratio: {metadata['overlap_ratio']:.3f}")
        print(f"  ✓ Aggregation method: {metadata['aggregation_method']}")
        
        return decoder_results, aggregated_data
    
    def test_aggregation_with_missing_decoders(self, test_states_file, test_model_id, temp_output_dir):
        """Test aggregation when some decoder results are missing."""
        
        # Step 1: Run only 3 out of 5 decoders
        selected_decoders = ["hazard", "color", "velocity_x"]
        decoder_results = {}
        
        # Create cache directory structure
        cache_dir = temp_output_dir / "cache_partial"
        critical_units_dir = cache_dir / "critical_units"
        critical_units_dir.mkdir(parents=True, exist_ok=True)
        
        for decoder_type in selected_decoders:
            output_file = temp_output_dir / f"{test_model_id}_{decoder_type}_units.json"
            
            cmd = [
                sys.executable, "-m", "learning_in_context.analysis.critical_units",
                "--states", str(test_states_file),
                "--output", str(output_file),
                "--model-id", test_model_id,
                "--decoder-type", decoder_type,
                "--regularization-sweep",
                "--use-all-timesteps",
                "--concatenate-states"
                # Note: --use-normalized is omitted since test file contains raw states
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            assert result.returncode == 0, f"Decoder {decoder_type} failed: {result.stderr}"
            
            # Validate and copy to cache location
            data = validate_decoder_output(output_file, decoder_type)
            decoder_results[decoder_type] = data
            
            dest_file = critical_units_dir / f"{test_model_id}_{decoder_type}_units.json"
            dest_file.write_text(output_file.read_text())
        
        # Step 2: Run aggregation script (should handle missing decoders gracefully)
        aggregated_file = critical_units_dir / f"{test_model_id}_units.json"
        
        aggregation_script = Path("scripts/aggregate_critical_units.py")
        cmd = [
            sys.executable, str(aggregation_script),
            "--cache-dir", str(cache_dir),
            "--model-id", test_model_id,
            "--output", str(aggregated_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        assert result.returncode == 0, f"Aggregation with missing decoders failed: {result.stderr}"
        
        # Step 3: Validate that aggregation handled missing decoders correctly
        with open(aggregated_file, 'r') as f:
            aggregated_data = json.load(f)
        
        metadata = aggregated_data["metadata"]
        
        # Should track that only 3 decoders had results
        assert metadata["n_decoders_with_results"] == 3, "Should detect 3 decoders with results"
        assert metadata["n_decoders"] >= 3, "Should track at least the attempted decoders"
        
        # Union should still work correctly
        individual_units = set()
        for decoder_data in decoder_results.values():
            individual_units.update(decoder_data["unit_indices"])
        
        aggregated_units = set(aggregated_data["unit_indices"])
        assert aggregated_units == individual_units, "Should correctly aggregate available results"
        
        print(f"  ✓ Handled missing decoders: {metadata['n_decoders_with_results']}/5 decoders")
        print(f"  ✓ Aggregated {len(aggregated_units)} unique critical units from available decoders")
    
    def test_aggregation_output_format_compatibility(self, test_states_file, test_model_id, temp_output_dir):
        """Test that aggregated output maintains backward compatibility format."""
        
        # Run minimal decoder set for faster testing
        decoder_types = ["hazard", "color"]
        decoder_results = {}
        
        # Create cache directory structure
        cache_dir = temp_output_dir / "cache_compat"
        critical_units_dir = cache_dir / "critical_units"
        critical_units_dir.mkdir(parents=True, exist_ok=True)
        
        # Run decoders
        for decoder_type in decoder_types:
            output_file = temp_output_dir / f"{test_model_id}_{decoder_type}_units.json"
            
            cmd = [
                sys.executable, "-m", "learning_in_context.analysis.critical_units",
                "--states", str(test_states_file),
                "--output", str(output_file),
                "--model-id", test_model_id,
                "--decoder-type", decoder_type,
                "--regularization-sweep",
                "--use-all-timesteps",
                "--concatenate-states"
                # Note: --use-normalized is omitted since test file contains raw states
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            assert result.returncode == 0, f"Decoder {decoder_type} failed: {result.stderr}"
            
            data = validate_decoder_output(output_file, decoder_type)
            decoder_results[decoder_type] = data
            
            dest_file = critical_units_dir / f"{test_model_id}_{decoder_type}_units.json"
            dest_file.write_text(output_file.read_text())
        
        # Run aggregation
        aggregated_file = critical_units_dir / f"{test_model_id}_units.json"
        
        aggregation_script = Path("scripts/aggregate_critical_units.py")
        cmd = [
            sys.executable, str(aggregation_script),
            "--cache-dir", str(cache_dir),
            "--model-id", test_model_id,
            "--output", str(aggregated_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        assert result.returncode == 0, f"Aggregation failed: {result.stderr}"
        
        # Load and check format compatibility
        with open(aggregated_file, 'r') as f:
            data = json.load(f)
        
        # Check all expected keys for backward compatibility
        required_top_level = ["model_id", "unit_indices", "coefficients", "r2_scores", "best_alpha", "cv_scores", "metadata"]
        for key in required_top_level:
            assert key in data, f"Missing backward compatibility key: {key}"
        
        # Check data types match expectations
        assert isinstance(data["model_id"], str), "model_id should be string"
        assert isinstance(data["unit_indices"], list), "unit_indices should be list"
        assert isinstance(data["coefficients"], list), "coefficients should be list"
        assert isinstance(data["r2_scores"], list), "r2_scores should be list"
        assert isinstance(data["best_alpha"], (int, float)), "best_alpha should be numeric"
        assert isinstance(data["cv_scores"], dict), "cv_scores should be dict"
        assert isinstance(data["metadata"], dict), "metadata should be dict"
        
        # Check that model_id matches expectation
        assert data["model_id"] == test_model_id, "model_id should match input"
        
        # Verify enhanced metadata (new features)
        metadata = data["metadata"]
        enhanced_keys = ["aggregation_method", "decoder_results", "n_decoders", "overlap_ratio"]
        for key in enhanced_keys:
            assert key in metadata, f"Missing enhanced metadata key: {key}"
        
        print("  ✓ Backward compatibility format verified")
        print("  ✓ Enhanced metadata features present")
        print(f"  ✓ Model ID: {data['model_id']}")
        print(f"  ✓ Aggregation method: {metadata['aggregation_method']}")


class TestAggregationEdgeCases:
    """Test edge cases and error handling in aggregation."""
    
    def test_aggregation_with_empty_decoder_results(self, test_model_id, temp_output_dir):
        """Test aggregation when decoders find no critical units."""
        
        # Create cache directory with empty decoder results
        cache_dir = temp_output_dir / "cache_empty"
        critical_units_dir = cache_dir / "critical_units"
        critical_units_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mock empty decoder results
        empty_result = {
            "unit_indices": [],
            "coefficients": [],
            "r2_scores": [0.0],
            "best_alpha": 0.001,
            "cv_scores": {"mean": 0.0, "std": 0.0, "all": [0.0]},
            "metadata": {
                "n_units_total": 16,
                "n_units_critical": 0,
                "l1_ratio": 0.64,
                "score_type": "accuracy"
            }
        }
        
        # Save empty results for all decoders
        decoder_types = ["hazard", "contingency", "color", "velocity_x", "velocity_y"]
        for decoder_type in decoder_types:
            decoder_file = critical_units_dir / f"{test_model_id}_{decoder_type}_units.json"
            with open(decoder_file, 'w') as f:
                json.dump(empty_result, f)
        
        # Run aggregation
        aggregated_file = critical_units_dir / f"{test_model_id}_units.json"
        
        aggregation_script = Path("scripts/aggregate_critical_units.py")
        cmd = [
            sys.executable, str(aggregation_script),
            "--cache-dir", str(cache_dir),
            "--model-id", test_model_id,
            "--output", str(aggregated_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        assert result.returncode == 0, f"Aggregation with empty results failed: {result.stderr}"
        
        # Validate that empty aggregation works correctly
        with open(aggregated_file, 'r') as f:
            data = json.load(f)
        
        assert data["unit_indices"] == [], "Should have empty unit indices"
        assert data["metadata"]["n_units_critical"] == 0, "Should have 0 critical units"
        assert data["metadata"]["n_decoders_with_results"] == 5, "Should still count all decoders"
        assert data["metadata"]["overlap_ratio"] == 0.0, "Overlap should be 0 for empty results"
        
        print("  ✓ Empty decoder results handled correctly")
    
    def test_aggregation_error_handling(self, test_model_id, temp_output_dir):
        """Test aggregation error handling for invalid inputs."""
        
        # Test with non-existent cache directory
        nonexistent_cache = temp_output_dir / "nonexistent_cache"
        output_file = temp_output_dir / "test_output.json"
        
        aggregation_script = Path("scripts/aggregate_critical_units.py")
        cmd = [
            sys.executable, str(aggregation_script),
            "--cache-dir", str(nonexistent_cache),
            "--model-id", test_model_id,
            "--output", str(output_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        
        # Should handle missing cache directory gracefully
        # The script might create empty results or fail gracefully
        if result.returncode != 0:
            assert "not found" in result.stderr or "No such file" in result.stderr, \
                "Should provide informative error for missing cache"
        else:
            # If it succeeds, should create valid empty output
            assert output_file.exists(), "Should create output file even with missing cache"
            with open(output_file, 'r') as f:
                data = json.load(f)
            assert data["metadata"]["n_decoders_with_results"] == 0, "Should indicate no results found"
        
        print("  ✓ Error handling for missing cache directory verified")