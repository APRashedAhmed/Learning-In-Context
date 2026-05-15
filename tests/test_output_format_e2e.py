"""End-to-end tests for output format validation using real test data."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


class TestOutputFormatValidationE2E:
    """Test that decoder and aggregated outputs have correct formats."""
    
    def test_decoder_output_format_comprehensive(self, test_states_file, test_model_id, temp_output_dir, 
                                                expected_output_keys, expected_metadata_keys):
        """Test comprehensive output format validation for individual decoders."""
        
        # Test with hazard decoder
        output_file = temp_output_dir / f"{test_model_id}_hazard_format_test.json"
        
        cmd = [
            sys.executable, "-m", "learning_in_context.analysis.critical_units",
            "--states", str(test_states_file),
            "--output", str(output_file),
            "--model-id", test_model_id,
            "--decoder-type", "hazard",
            "--regularization-sweep"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        
        # Load and validate output format
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        # Test top-level structure
        for key in expected_output_keys:
            assert key in data, f"Missing required top-level key: {key}"
        
        # Test data types
        assert isinstance(data["unit_indices"], list), "unit_indices should be list"
        assert isinstance(data["coefficients"], list), "coefficients should be list"
        assert isinstance(data["r2_scores"], list), "r2_scores should be list"
        assert isinstance(data["best_alpha"], (int, float)), "best_alpha should be numeric"
        assert isinstance(data["cv_scores"], dict), "cv_scores should be dict"
        assert isinstance(data["metadata"], dict), "metadata should be dict"
        
        # Test metadata structure
        metadata = data["metadata"]
        for key in expected_metadata_keys:
            assert key in metadata, f"Missing metadata key: {key}"
        
        # Test metadata data types
        assert isinstance(metadata["n_units_total"], int), "n_units_total should be int"
        assert isinstance(metadata["n_units_critical"], int), "n_units_critical should be int"
        assert isinstance(metadata["l1_ratio"], (int, float)), "l1_ratio should be numeric"
        assert isinstance(metadata["timestep"], int), "timestep should be int"
        assert isinstance(metadata["exclude_low_variance"], bool), "exclude_low_variance should be bool"
        assert isinstance(metadata["score_type"], str), "score_type should be string"
        
        # Test value constraints
        assert data["best_alpha"] > 0, "best_alpha should be positive"
        assert metadata["n_units_total"] > 0, "n_units_total should be positive"
        assert metadata["n_units_critical"] >= 0, "n_units_critical should be non-negative"
        assert metadata["n_units_critical"] <= metadata["n_units_total"], "critical <= total units"
        assert 0 <= metadata["l1_ratio"] <= 1, "l1_ratio should be in [0,1]"
        
        # Test list lengths consistency
        if len(data["unit_indices"]) > 0:
            # When we have critical units, should have corresponding coefficients
            assert len(data["coefficients"]) > 0, "Should have coefficients when units found"
        
        # Test CV scores structure
        cv_scores = data["cv_scores"]
        required_cv_keys = ["mean", "std", "all"]
        for key in required_cv_keys:
            assert key in cv_scores, f"Missing CV scores key: {key}"
        
        assert isinstance(cv_scores["mean"], (int, float)), "CV mean should be numeric"
        assert isinstance(cv_scores["std"], (int, float)), "CV std should be numeric"
        assert isinstance(cv_scores["all"], list), "CV all should be list"
        
        print(f"✓ Decoder output format validation passed")
        print(f"  - Model ID: {test_model_id}")
        print(f"  - Critical units: {metadata['n_units_critical']}")
        print(f"  - Best alpha: {data['best_alpha']:.1e}")
        print(f"  - Score type: {metadata['score_type']}")
        
        return data
    
    def test_aggregated_output_format_comprehensive(self, test_states_file, test_model_id, temp_output_dir,
                                                   expected_aggregated_keys, expected_aggregated_metadata_keys,
                                                   decoder_types):
        """Test comprehensive output format validation for aggregated results."""
        
        # Step 1: Generate individual decoder results
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
                "--regularization-sweep"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            assert result.returncode == 0, f"Decoder {decoder_type} failed: {result.stderr}"
        
        # Step 2: Set up aggregation
        cache_dir = temp_output_dir / "cache_format"
        critical_units_dir = cache_dir / "critical_units"
        critical_units_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files to cache location
        for decoder_type, source_file in zip(decoder_types, decoder_files):
            dest_file = critical_units_dir / f"{test_model_id}_{decoder_type}_units.json"
            dest_file.write_text(source_file.read_text())
        
        # Step 3: Run aggregation
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
        
        # Step 4: Validate aggregated output format
        with open(aggregated_file, 'r') as f:
            data = json.load(f)
        
        # Test top-level structure (backward compatibility)
        for key in expected_aggregated_keys:
            assert key in data, f"Missing required aggregated key: {key}"
        
        # Test top-level data types
        assert isinstance(data["model_id"], str), "model_id should be string"
        assert isinstance(data["unit_indices"], list), "unit_indices should be list"
        assert isinstance(data["coefficients"], list), "coefficients should be list"
        assert isinstance(data["r2_scores"], list), "r2_scores should be list"
        assert isinstance(data["best_alpha"], (int, float)), "best_alpha should be numeric"
        assert isinstance(data["cv_scores"], dict), "cv_scores should be dict"
        assert isinstance(data["metadata"], dict), "metadata should be dict"
        
        # Test aggregated metadata structure
        metadata = data["metadata"]
        for key in expected_aggregated_metadata_keys:
            assert key in metadata, f"Missing aggregated metadata key: {key}"
        
        # Test aggregated metadata data types
        assert isinstance(metadata["n_units_total"], int), "n_units_total should be int"
        assert isinstance(metadata["n_units_critical"], int), "n_units_critical should be int"
        assert isinstance(metadata["decoder_results"], dict), "decoder_results should be dict"
        assert isinstance(metadata["aggregation_method"], str), "aggregation_method should be string"
        assert isinstance(metadata["n_decoders"], int), "n_decoders should be int"
        assert isinstance(metadata["n_decoders_with_results"], int), "n_decoders_with_results should be int"
        assert isinstance(metadata["overlap_ratio"], (int, float)), "overlap_ratio should be numeric"
        
        # Test aggregated value constraints
        assert data["model_id"] == test_model_id, "model_id should match input"
        assert metadata["aggregation_method"] == "union_of_decoders", "Should use union aggregation"
        assert metadata["n_decoders"] == len(decoder_types), "Should track all decoders"
        assert metadata["n_decoders_with_results"] <= metadata["n_decoders"], "Results <= total decoders"
        assert 0 <= metadata["overlap_ratio"] <= 1, "overlap_ratio should be in [0,1]"
        assert metadata["n_units_critical"] >= 0, "critical units should be non-negative"
        assert metadata["n_units_total"] > 0, "total units should be positive"
        
        # Test decoder results preservation
        decoder_results = metadata["decoder_results"]
        assert len(decoder_results) == len(decoder_types), "Should preserve all decoder results"
        
        for decoder_type in decoder_types:
            assert decoder_type in decoder_results, f"Should preserve {decoder_type} results"
            decoder_data = decoder_results[decoder_type]
            
            # Each preserved decoder result should have key information
            assert "unit_indices" in decoder_data, f"{decoder_type} should preserve unit_indices"
            assert "metadata" in decoder_data, f"{decoder_type} should preserve metadata"
        
        # Test list consistency
        assert len(data["unit_indices"]) == metadata["n_units_critical"], "List length should match count"
        assert len(data["coefficients"]) == len(data["unit_indices"]), "Coefficients should match units"
        
        print(f"✓ Aggregated output format validation passed")
        print(f"  - Model ID: {data['model_id']}")
        print(f"  - Aggregation method: {metadata['aggregation_method']}")
        print(f"  - Decoders processed: {metadata['n_decoders_with_results']}/{metadata['n_decoders']}")
        print(f"  - Total critical units: {metadata['n_units_critical']}")
        print(f"  - Overlap ratio: {metadata['overlap_ratio']:.3f}")
        
        return data
    
    def test_json_serialization_compatibility(self, test_states_file, test_model_id, temp_output_dir):
        """Test that outputs can be properly serialized and deserialized as JSON."""
        
        output_file = temp_output_dir / f"{test_model_id}_json_test.json"
        
        # Generate a decoder output
        cmd = [
            sys.executable, "-m", "learning_in_context.analysis.critical_units",
            "--states", str(test_states_file),
            "--output", str(output_file),
            "--model-id", test_model_id,
            "--decoder-type", "color",
            "--regularization-sweep"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        
        # Test multiple serialization/deserialization cycles
        with open(output_file, 'r') as f:
            original_data = json.load(f)
        
        # Re-serialize and deserialize
        json_str = json.dumps(original_data, indent=2)
        reloaded_data = json.loads(json_str)
        
        # Should be identical after round-trip
        assert reloaded_data == original_data, "Data should be identical after JSON round-trip"
        
        # Test that all values are JSON-serializable types
        def check_json_types(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    check_json_types(value, f"{path}.{key}")
            elif isinstance(obj, list):
                for i, value in enumerate(obj):
                    check_json_types(value, f"{path}[{i}]")
            else:
                # Should be basic JSON type
                assert isinstance(obj, (str, int, float, bool, type(None))), \
                    f"Non-JSON type at {path}: {type(obj)}"
        
        check_json_types(original_data)
        
        # Test that numbers are in reasonable ranges (not NaN, inf, etc.)
        def check_numeric_values(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    check_numeric_values(value, f"{path}.{key}")
            elif isinstance(obj, list):
                for i, value in enumerate(obj):
                    check_numeric_values(value, f"{path}[{i}]")
            elif isinstance(obj, (int, float)):
                import math
                assert not math.isnan(obj), f"NaN value at {path}"
                assert not math.isinf(obj), f"Infinite value at {path}"
        
        check_numeric_values(original_data)
        
        print("✓ JSON serialization compatibility verified")
        print("✓ All values are valid JSON types")
        print("✓ No NaN or infinite values detected")
    
    def test_output_format_consistency_across_decoders(self, test_states_file, test_model_id, 
                                                      temp_output_dir, decoder_types):
        """Test that all decoder types produce outputs with consistent format."""
        
        decoder_outputs = {}
        
        # Generate outputs for all decoder types
        for decoder_type in decoder_types:
            output_file = temp_output_dir / f"{test_model_id}_{decoder_type}_consistency_test.json"
            
            cmd = [
                sys.executable, "-m", "learning_in_context.analysis.critical_units",
                "--states", str(test_states_file),
                "--output", str(output_file),
                "--model-id", test_model_id,
                "--decoder-type", decoder_type,
                "--regularization-sweep"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            assert result.returncode == 0, f"Decoder {decoder_type} failed: {result.stderr}"
            
            with open(output_file, 'r') as f:
                decoder_outputs[decoder_type] = json.load(f)
        
        # Check format consistency across all decoders
        reference_keys = set(decoder_outputs[decoder_types[0]].keys())
        reference_metadata_keys = set(decoder_outputs[decoder_types[0]]["metadata"].keys())
        
        for decoder_type in decoder_types[1:]:
            output = decoder_outputs[decoder_type]
            
            # Check top-level keys consistency
            assert set(output.keys()) == reference_keys, \
                f"{decoder_type} has different top-level keys than {decoder_types[0]}"
            
            # Check metadata keys consistency  
            assert set(output["metadata"].keys()) >= reference_metadata_keys, \
                f"{decoder_type} missing metadata keys compared to {decoder_types[0]}"
            
            # Check data type consistency
            for key in reference_keys:
                if key != "metadata":
                    assert type(output[key]) == type(decoder_outputs[decoder_types[0]][key]), \
                        f"{decoder_type}.{key} has different type than {decoder_types[0]}"
        
        # Check that all have same total units (should be using same test data)
        total_units = [data["metadata"]["n_units_total"] for data in decoder_outputs.values()]
        assert all(n == total_units[0] for n in total_units), "All decoders should see same total units"
        
        # Check decoder-specific parameter consistency
        for decoder_type, output in decoder_outputs.items():
            metadata = output["metadata"]
            
            if decoder_type in ["hazard", "contingency"]:
                assert abs(metadata.get("l1_ratio", 0.64) - 0.64) < 0.01, \
                    f"{decoder_type} should use l1_ratio=0.64"
            elif decoder_type in ["color", "velocity_x", "velocity_y"]:
                assert abs(metadata.get("l1_ratio", 0.4) - 0.4) < 0.01, \
                    f"{decoder_type} should use l1_ratio=0.4"
        
        print("✓ Output format consistency verified across all decoders")
        print(f"  - Common top-level keys: {len(reference_keys)}")
        print(f"  - Common metadata keys: {len(reference_metadata_keys)}")
        print(f"  - Total units consistent: {total_units[0]}")
        
        return decoder_outputs


class TestOutputFormatEdgeCases:
    """Test edge cases in output formatting."""
    
    def test_empty_critical_units_format(self, test_model_id, temp_output_dir):
        """Test output format when no critical units are found."""
        
        # Create mock empty result to test format
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
                "timestep": -1,
                "exclude_low_variance": True,
                "unit_variance_threshold": 1e-6,
                "score_type": "accuracy"
            }
        }
        
        # Test JSON serialization
        output_file = temp_output_dir / f"{test_model_id}_empty_test.json"
        with open(output_file, 'w') as f:
            json.dump(empty_result, f, indent=2)
        
        # Reload and validate
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        assert data["unit_indices"] == [], "Empty unit indices should be empty list"
        assert data["coefficients"] == [], "Empty coefficients should be empty list"
        assert data["metadata"]["n_units_critical"] == 0, "Should indicate 0 critical units"
        assert len(data["unit_indices"]) == data["metadata"]["n_units_critical"], "Counts should match"
        
        print("✓ Empty critical units format validated")
    
    def test_large_numbers_format(self, test_model_id, temp_output_dir):
        """Test output format with large numbers."""
        
        # Create result with large numbers
        large_result = {
            "unit_indices": list(range(100)),  # Large list
            "coefficients": [1e6] * 100,  # Large coefficients
            "r2_scores": [0.999999],
            "best_alpha": 1e-10,  # Very small alpha
            "cv_scores": {"mean": 0.95, "std": 0.001, "all": [0.95] * 100},
            "metadata": {
                "n_units_total": 1000,  # Large number of units
                "n_units_critical": 100,
                "l1_ratio": 0.64,
                "timestep": -1,
                "exclude_low_variance": True,
                "unit_variance_threshold": 1e-6,
                "score_type": "accuracy"
            }
        }
        
        # Test JSON serialization with large numbers
        output_file = temp_output_dir / f"{test_model_id}_large_test.json"
        with open(output_file, 'w') as f:
            json.dump(large_result, f, indent=2)
        
        # Reload and validate
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        assert len(data["unit_indices"]) == 100, "Should preserve large lists"
        assert data["best_alpha"] == 1e-10, "Should preserve very small numbers"
        assert data["metadata"]["n_units_total"] == 1000, "Should preserve large integers"
        
        print("✓ Large numbers format validated")