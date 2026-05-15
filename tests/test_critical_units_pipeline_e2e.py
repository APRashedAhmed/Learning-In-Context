"""End-to-end tests for critical units pipeline integration using real test data."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


class TestCriticalUnitsPipelineE2E:
    """Test the complete critical units pipeline integration."""
    
    def test_sequential_pipeline_execution(self, test_states_file, test_model_id, temp_output_dir, decoder_types):
        """Test that the pipeline can be executed sequentially without doit."""
        
        print("Testing sequential pipeline execution...")
        
        # Step 1: Run all individual decoder analyses
        print("  Step 1: Running individual decoder analyses...")
        decoder_results = {}
        decoder_files = []
        
        for i, decoder_type in enumerate(decoder_types):
            output_file = temp_output_dir / f"{test_model_id}_{decoder_type}_units.json"
            decoder_files.append(output_file)
            
            print(f"    Running {decoder_type} decoder ({i+1}/{len(decoder_types)})...")
            
            cmd = [
                sys.executable, "-m", "learning_in_context.analysis.critical_units",
                "--states", str(test_states_file),
                "--output", str(output_file),
                "--model-id", test_model_id,
                "--decoder-type", decoder_type,
                "--use-all-timesteps",
                "--concatenate-states",
                # Note: --use-normalized is omitted since test file contains raw states
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            assert result.returncode == 0, f"Decoder {decoder_type} failed: {result.stderr}"
            
            # Load and validate results
            with open(output_file, 'r') as f:
                data = json.load(f)
            
            decoder_results[decoder_type] = data
            
            # Basic validation
            assert "unit_indices" in data, f"{decoder_type} missing unit_indices"
            assert "metadata" in data, f"{decoder_type} missing metadata"
            assert data["metadata"]["n_units_critical"] >= 0, f"{decoder_type} invalid critical count"
            
            print(f"      ✓ {decoder_type}: {data['metadata']['n_units_critical']} critical units")
        
        # Step 2: Set up cache structure for aggregation
        print("  Step 2: Setting up cache structure...")
        cache_dir = temp_output_dir / "cache"
        critical_units_dir = cache_dir / "critical_units"
        critical_units_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy decoder files to expected locations
        for decoder_type, source_file in zip(decoder_types, decoder_files):
            dest_file = critical_units_dir / f"{test_model_id}_{decoder_type}_units.json"
            dest_file.write_text(source_file.read_text())
        
        # Step 3: Run aggregation
        print("  Step 3: Running aggregation...")
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
        
        # Validate aggregated output
        with open(aggregated_file, 'r') as f:
            aggregated_data = json.load(f)
        
        # Step 4: Validate complete pipeline results
        print("  Step 4: Validating pipeline results...")
        
        # Check aggregated structure
        required_keys = ["model_id", "unit_indices", "coefficients", "metadata"]
        for key in required_keys:
            assert key in aggregated_data, f"Aggregated output missing {key}"
        
        metadata = aggregated_data["metadata"]
        assert "decoder_results" in metadata, "Should preserve individual decoder results"
        assert metadata["aggregation_method"] == "union_of_decoders", "Should use union aggregation"
        assert metadata["n_decoders"] == len(decoder_types), "Should track all decoders"
        
        # Verify union logic
        all_individual_units = set()
        for decoder_data in decoder_results.values():
            all_individual_units.update(decoder_data["unit_indices"])
        
        aggregated_units = set(aggregated_data["unit_indices"])
        assert aggregated_units == all_individual_units, "Union logic should be correct"
        
        print(f"    ✓ Pipeline completed successfully")
        print(f"    ✓ Individual critical units total: {sum(len(d['unit_indices']) for d in decoder_results.values())}")
        print(f"    ✓ Unique critical units after union: {len(aggregated_units)}")
        print(f"    ✓ Overlap ratio: {metadata.get('overlap_ratio', 0):.3f}")
        
        return decoder_results, aggregated_data
    
    def test_pipeline_file_dependencies(self, test_states_file, test_model_id, temp_output_dir):
        """Test that pipeline respects file dependencies correctly."""
        
        print("Testing pipeline file dependencies...")
        
        # Test that decoder analysis depends on states file
        output_file = temp_output_dir / f"{test_model_id}_dependency_test.json"
        
        # Should work with existing states file
        cmd = [
            sys.executable, "-m", "learning_in_context.analysis.critical_units",
            "--states", str(test_states_file),
            "--output", str(output_file),
            "--model-id", test_model_id,
            "--decoder-type", "hazard",
            "--regularization-sweep"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        assert result.returncode == 0, "Should succeed with valid states file"
        assert output_file.exists(), "Should create output file"
        
        # Test that aggregation depends on decoder files
        cache_dir = temp_output_dir / "cache_deps"
        critical_units_dir = cache_dir / "critical_units"
        critical_units_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy the decoder result to cache
        decoder_file = critical_units_dir / f"{test_model_id}_hazard_units.json"
        decoder_file.write_text(output_file.read_text())
        
        # Should work with existing decoder file
        aggregated_file = critical_units_dir / f"{test_model_id}_units.json"
        
        aggregation_script = Path("scripts/aggregate_critical_units.py")
        cmd = [
            sys.executable, str(aggregation_script),
            "--cache-dir", str(cache_dir),
            "--model-id", test_model_id,
            "--output", str(aggregated_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        assert result.returncode == 0, "Aggregation should succeed with decoder files"
        assert aggregated_file.exists(), "Should create aggregated file"
        
        print("    ✓ File dependencies respected correctly")
    
    def test_pipeline_error_propagation(self, test_model_id, temp_output_dir):
        """Test that errors propagate correctly through the pipeline."""
        
        print("Testing pipeline error propagation...")
        
        # Test with invalid states file
        invalid_states = temp_output_dir / "invalid_states.npz"
        output_file = temp_output_dir / f"{test_model_id}_error_test.json"
        
        cmd = [
            sys.executable, "-m", "learning_in_context.analysis.critical_units",
            "--states", str(invalid_states),
            "--output", str(output_file),
            "--model-id", test_model_id,
            "--decoder-type", "hazard",
            "--regularization-sweep"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        assert result.returncode != 0, "Should fail with invalid states file"
        assert not output_file.exists(), "Should not create output on failure"
        
        # Test aggregation with missing decoder files
        cache_dir = temp_output_dir / "cache_missing"
        critical_units_dir = cache_dir / "critical_units"
        critical_units_dir.mkdir(parents=True, exist_ok=True)
        
        aggregated_file = critical_units_dir / f"{test_model_id}_units.json"
        
        aggregation_script = Path("scripts/aggregate_critical_units.py")
        cmd = [
            sys.executable, str(aggregation_script),
            "--cache-dir", str(cache_dir),
            "--model-id", test_model_id,
            "--output", str(aggregated_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        
        # Aggregation might succeed with empty results or fail gracefully
        if result.returncode == 0:
            # If it succeeds, should handle missing files gracefully
            if aggregated_file.exists():
                with open(aggregated_file, 'r') as f:
                    data = json.load(f)
                assert data["metadata"]["n_decoders_with_results"] == 0, "Should indicate no results found"
        else:
            # If it fails, should be informative
            assert "not found" in result.stderr or "No such file" in result.stderr, \
                "Should provide informative error message"
        
        print("    ✓ Error propagation working correctly")
    
    def test_pipeline_output_consistency(self, test_states_file, test_model_id, temp_output_dir):
        """Test that pipeline produces consistent outputs across runs."""
        
        print("Testing pipeline output consistency...")
        
        # Run the same analysis twice
        results = []
        
        for run in range(2):
            output_file = temp_output_dir / f"{test_model_id}_consistency_run_{run}.json"
            
            cmd = [
                sys.executable, "-m", "learning_in_context.analysis.critical_units",
                "--states", str(test_states_file),
                "--output", str(output_file),
                "--model-id", test_model_id,
                "--decoder-type", "color",  # Use color as it should be most deterministic
                "--regularization-sweep"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            assert result.returncode == 0, f"Run {run} failed: {result.stderr}"
            
            with open(output_file, 'r') as f:
                data = json.load(f)
            
            results.append(data)
        
        # Compare results for consistency
        result1, result2 = results
        
        # Core results should be identical (same data, same algorithm)
        assert result1["unit_indices"] == result2["unit_indices"], "Unit indices should be consistent"
        assert result1["metadata"]["n_units_critical"] == result2["metadata"]["n_units_critical"], \
            "Critical unit count should be consistent"
        assert result1["metadata"]["n_units_total"] == result2["metadata"]["n_units_total"], \
            "Total unit count should be consistent"
        
        # Alpha selection might vary slightly due to cross-validation, but should be close
        alpha_ratio = result1["best_alpha"] / result2["best_alpha"]
        assert 0.1 < alpha_ratio < 10, "Best alpha should be in same order of magnitude"
        
        print("    ✓ Pipeline produces consistent outputs")
        print(f"      Run 1: {result1['metadata']['n_units_critical']} critical units, alpha={result1['best_alpha']:.1e}")
        print(f"      Run 2: {result2['metadata']['n_units_critical']} critical units, alpha={result2['best_alpha']:.1e}")


class TestPipelinePerformance:
    """Test pipeline performance characteristics."""
    
    def test_pipeline_execution_time(self, test_states_file, test_model_id, temp_output_dir):
        """Test that pipeline executes in reasonable time."""
        
        import time
        
        print("Testing pipeline execution time...")
        
        start_time = time.time()
        
        # Run single decoder analysis
        output_file = temp_output_dir / f"{test_model_id}_timing_test.json"
        
        cmd = [
            sys.executable, "-m", "learning_in_context.analysis.critical_units",
            "--states", str(test_states_file),
            "--output", str(output_file),
            "--model-id", test_model_id,
            "--decoder-type", "hazard",
            "--regularization-sweep"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        assert result.returncode == 0, f"Timing test failed: {result.stderr}"
        
        execution_time = time.time() - start_time
        
        # Should complete within reasonable time for test data
        # Test data has 168 trials, 409 timesteps, 16 units - should be fast
        assert execution_time < 60, f"Execution took too long: {execution_time:.1f} seconds"
        
        print(f"    ✓ Single decoder analysis completed in {execution_time:.1f} seconds")
        
        # Test aggregation timing
        cache_dir = temp_output_dir / "cache_timing"
        critical_units_dir = cache_dir / "critical_units"
        critical_units_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy result to cache
        decoder_file = critical_units_dir / f"{test_model_id}_hazard_units.json"
        decoder_file.write_text(output_file.read_text())
        
        start_time = time.time()
        
        aggregated_file = critical_units_dir / f"{test_model_id}_units.json"
        aggregation_script = Path("scripts/aggregate_critical_units.py")
        
        cmd = [
            sys.executable, str(aggregation_script),
            "--cache-dir", str(cache_dir),
            "--model-id", test_model_id,
            "--output", str(aggregated_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        assert result.returncode == 0, f"Aggregation timing test failed: {result.stderr}"
        
        aggregation_time = time.time() - start_time
        
        # Aggregation should be very fast
        assert aggregation_time < 5, f"Aggregation took too long: {aggregation_time:.1f} seconds"
        
        print(f"    ✓ Aggregation completed in {aggregation_time:.1f} seconds")
    
    def test_memory_usage_reasonable(self, test_states_file, test_model_id, temp_output_dir):
        """Test that pipeline uses reasonable memory."""
        
        # This is a basic test - for production, you might want to use memory profiling
        print("Testing memory usage (basic validation)...")
        
        output_file = temp_output_dir / f"{test_model_id}_memory_test.json"
        
        cmd = [
            sys.executable, "-m", "learning_in_context.analysis.critical_units",
            "--states", str(test_states_file),
            "--output", str(output_file),
            "--model-id", test_model_id,
            "--decoder-type", "color",
            "--regularization-sweep"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        assert result.returncode == 0, f"Memory test failed: {result.stderr}"
        
        # Check that stderr doesn't contain memory errors
        assert "MemoryError" not in result.stderr, "Should not have memory errors"
        assert "out of memory" not in result.stderr.lower(), "Should not run out of memory"
        
        print("    ✓ No memory errors detected")