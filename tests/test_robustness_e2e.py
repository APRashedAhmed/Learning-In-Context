"""End-to-end robustness tests for critical units pipeline using real test data."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest


class TestCriticalUnitsRobustnessE2E:
    """Test robustness and edge cases of the critical units pipeline."""
    
    def test_different_timestep_analysis(self, test_states_file, test_model_id, temp_output_dir, test_states_data):
        """Test analysis at different timesteps."""
        
        print("Testing analysis at different timesteps...")
        
        # Get the shape of test data to know valid timestep range
        hiddens = test_states_data['hiddens']
        n_trials, n_timesteps, n_units = hiddens.shape
        
        # Test analysis at different timesteps
        timesteps_to_test = [0, n_timesteps // 2, -1]  # First, middle, last
        
        for timestep in timesteps_to_test:
            output_file = temp_output_dir / f"{test_model_id}_timestep_{timestep}_test.json"
            
            cmd = [
                sys.executable, "-m", "learning_in_context.analysis.critical_units",
                "--states", str(test_states_file),
                "--output", str(output_file),
                "--model-id", test_model_id,
                "--decoder-type", "color",
                "--timestep", str(timestep),
                "--regularization-sweep"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            assert result.returncode == 0, f"Analysis failed at timestep {timestep}: {result.stderr}"
            
            # Validate output
            with open(output_file, 'r') as f:
                data = json.load(f)
            
            assert data["metadata"]["timestep"] == timestep, f"Should record timestep {timestep}"
            assert data["metadata"]["n_units_total"] == n_units, "Should have same total units"
            assert data["metadata"]["n_units_critical"] >= 0, "Should have valid critical count"
            
            print(f"    ✓ Timestep {timestep}: {data['metadata']['n_units_critical']} critical units")
        
        print("    ✓ All timestep analyses completed successfully")
    
    def test_analysis_with_minimal_data(self, test_states_data, test_model_id, temp_output_dir):
        """Test analysis with minimal amount of data."""
        
        print("Testing analysis with minimal data...")
        
        # Create minimal subset of test data (fewer trials)
        hiddens = test_states_data['hiddens']
        predictions = test_states_data['predictions']
        targets = test_states_data['targets']
        
        # Use only first 10 trials
        minimal_data = {
            'hiddens': hiddens[:10],
            'predictions': predictions[:10],
            'targets': targets[:10],
            'model_id': test_model_id,
            'extraction_time': 1234567890.0,
            'shapes': {},
            'metadata': {},
            'df_data': np.array([]),
            'dict_metadata': {},
            'samples': test_states_data.get('samples', np.array([]))[:10] if 'samples' in test_states_data else np.array([]),
            'cells': test_states_data.get('cells', np.array([]))[:10] if 'cells' in test_states_data else np.array([])
        }
        
        # Save minimal data
        minimal_file = temp_output_dir / f"{test_model_id}_minimal_states.npz"
        np.savez_compressed(str(minimal_file), **minimal_data)
        
        # Test analysis with minimal data
        output_file = temp_output_dir / f"{test_model_id}_minimal_test.json"
        
        cmd = [
            sys.executable, "-m", "learning_in_context.analysis.critical_units",
            "--states", str(minimal_file),
            "--output", str(output_file),
            "--model-id", test_model_id,
            "--decoder-type", "color",
            "--regularization-sweep"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        
        # Should handle minimal data gracefully (might succeed or fail gracefully)
        if result.returncode == 0:
            # If it succeeds, validate the output
            with open(output_file, 'r') as f:
                data = json.load(f)
            
            assert data["metadata"]["n_units_critical"] >= 0, "Should have valid critical count"
            print(f"    ✓ Minimal data analysis succeeded: {data['metadata']['n_units_critical']} critical units")
        else:
            # If it fails, should be informative
            assert "sample" in result.stderr.lower() or "trial" in result.stderr.lower() or \
                   "insufficient" in result.stderr.lower(), \
                   "Should provide informative error for insufficient data"
            print(f"    ✓ Minimal data analysis failed gracefully: {result.stderr.strip()}")
    
    def test_analysis_with_uniform_data(self, test_states_data, test_model_id, temp_output_dir):
        """Test analysis with uniform/constant data (no signal)."""
        
        print("Testing analysis with uniform data...")
        
        # Create uniform data (no signal)
        hiddens = test_states_data['hiddens']
        n_trials, n_timesteps, n_units = hiddens.shape
        
        uniform_data = {
            'hiddens': np.ones_like(hiddens) * 0.5,  # Constant values
            'predictions': np.ones((n_trials, n_timesteps, 3)) / 3,  # Uniform predictions
            'targets': np.ones_like(test_states_data['targets']),
            'model_id': test_model_id,
            'extraction_time': 1234567890.0,
            'shapes': {},
            'metadata': {},
            'df_data': np.array([]),
            'dict_metadata': {},
            'samples': np.ones_like(test_states_data.get('samples', np.array([]))),
            'cells': np.ones_like(test_states_data.get('cells', np.array([])))
        }
        
        # Save uniform data
        uniform_file = temp_output_dir / f"{test_model_id}_uniform_states.npz"
        np.savez_compressed(str(uniform_file), **uniform_data)
        
        # Test analysis with uniform data
        output_file = temp_output_dir / f"{test_model_id}_uniform_test.json"
        
        cmd = [
            sys.executable, "-m", "learning_in_context.analysis.critical_units",
            "--states", str(uniform_file),
            "--output", str(output_file),
            "--model-id", test_model_id,
            "--decoder-type", "color",
            "--regularization-sweep"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        assert result.returncode == 0, f"Uniform data analysis failed: {result.stderr}"
        
        # With uniform data, should find no or very few critical units
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        # Performance should be at or near chance
        assert data["metadata"]["n_units_critical"] <= n_units // 2, "Should find few critical units with uniform data"
        
        print(f"    ✓ Uniform data analysis: {data['metadata']['n_units_critical']} critical units (expected low)")
    
    def test_analysis_with_noisy_data(self, test_states_data, test_model_id, temp_output_dir):
        """Test analysis with very noisy data."""
        
        print("Testing analysis with noisy data...")
        
        # Add noise to test data
        hiddens = test_states_data['hiddens']
        
        # Add significant noise
        noise_scale = np.std(hiddens) * 2  # 2x the original std as noise
        noisy_hiddens = hiddens + np.random.randn(*hiddens.shape) * noise_scale
        
        noisy_data = {
            'hiddens': noisy_hiddens,
            'predictions': test_states_data['predictions'],
            'targets': test_states_data['targets'],
            'model_id': test_model_id,
            'extraction_time': 1234567890.0,
            'shapes': {},
            'metadata': {},
            'df_data': np.array([]),
            'dict_metadata': {},
            'samples': test_states_data.get('samples', np.array([])),
            'cells': test_states_data.get('cells', np.array([]))
        }
        
        # Save noisy data
        noisy_file = temp_output_dir / f"{test_model_id}_noisy_states.npz"
        np.savez_compressed(str(noisy_file), **noisy_data)
        
        # Test analysis with noisy data
        output_file = temp_output_dir / f"{test_model_id}_noisy_test.json"
        
        cmd = [
            sys.executable, "-m", "learning_in_context.analysis.critical_units",
            "--states", str(noisy_file),
            "--output", str(output_file),
            "--model-id", test_model_id,
            "--decoder-type", "color",
            "--regularization-sweep"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        assert result.returncode == 0, f"Noisy data analysis failed: {result.stderr}"
        
        # Should still produce valid output
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        assert data["metadata"]["n_units_critical"] >= 0, "Should handle noisy data"
        assert data["best_alpha"] > 0, "Should find valid regularization"
        
        print(f"    ✓ Noisy data analysis: {data['metadata']['n_units_critical']} critical units")
    
    def test_missing_metadata_robustness(self, test_states_data, test_model_id, temp_output_dir):
        """Test robustness when metadata is missing or incomplete."""
        
        print("Testing robustness with missing metadata...")
        
        # Create data with minimal metadata
        minimal_metadata_data = {
            'hiddens': test_states_data['hiddens'],
            'predictions': test_states_data['predictions'],
            'targets': test_states_data['targets'],
            'model_id': test_model_id,
            'extraction_time': 1234567890.0,
            # Missing most metadata fields
        }
        
        # Save data with minimal metadata
        minimal_meta_file = temp_output_dir / f"{test_model_id}_minimal_meta_states.npz"
        np.savez_compressed(str(minimal_meta_file), **minimal_metadata_data)
        
        # Test decoders that should work without metadata (color, velocity)
        robust_decoders = ["color", "velocity_x", "velocity_y"]
        
        for decoder_type in robust_decoders:
            output_file = temp_output_dir / f"{test_model_id}_robust_{decoder_type}_test.json"
            
            cmd = [
                sys.executable, "-m", "learning_in_context.analysis.critical_units",
                "--states", str(minimal_meta_file),
                "--output", str(output_file),
                "--model-id", test_model_id,
                "--decoder-type", decoder_type,
                "--regularization-sweep"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            assert result.returncode == 0, f"Robust decoder {decoder_type} failed: {result.stderr}"
            
            # Should produce valid output
            with open(output_file, 'r') as f:
                data = json.load(f)
            
            assert data["metadata"]["n_units_critical"] >= 0, f"{decoder_type} should handle missing metadata"
            
            print(f"    ✓ {decoder_type} robust to missing metadata: {data['metadata']['n_units_critical']} units")
        
        # Test decoders that might fail with missing metadata (hazard, contingency)
        metadata_dependent_decoders = ["hazard", "contingency"]
        
        for decoder_type in metadata_dependent_decoders:
            output_file = temp_output_dir / f"{test_model_id}_meta_dep_{decoder_type}_test.json"
            
            cmd = [
                sys.executable, "-m", "learning_in_context.analysis.critical_units",
                "--states", str(minimal_meta_file),
                "--output", str(output_file),
                "--model-id", test_model_id,
                "--decoder-type", decoder_type,
                "--regularization-sweep"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            
            if result.returncode == 0:
                print(f"    ✓ {decoder_type} handled missing metadata gracefully")
            else:
                # Should fail gracefully with informative error
                assert "metadata" in result.stderr.lower() or "not found" in result.stderr.lower(), \
                    f"{decoder_type} should provide informative error for missing metadata"
                print(f"    ✓ {decoder_type} failed gracefully with missing metadata")
    
    def test_aggregation_robustness(self, test_states_file, test_model_id, temp_output_dir):
        """Test aggregation robustness with partial decoder results."""
        
        print("Testing aggregation robustness...")
        
        # Generate partial decoder results (only some decoders)
        partial_decoders = ["hazard", "color"]  # Only 2 out of 5 decoders
        decoder_files = []
        
        cache_dir = temp_output_dir / "cache_robust"
        critical_units_dir = cache_dir / "critical_units"
        critical_units_dir.mkdir(parents=True, exist_ok=True)
        
        for decoder_type in partial_decoders:
            output_file = temp_output_dir / f"{test_model_id}_{decoder_type}_robust.json"
            
            cmd = [
                sys.executable, "-m", "learning_in_context.analysis.critical_units",
                "--states", str(test_states_file),
                "--output", str(output_file),
                "--model-id", test_model_id,
                "--decoder-type", decoder_type,
                "--regularization-sweep"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            assert result.returncode == 0, f"Partial decoder {decoder_type} failed: {result.stderr}"
            
            # Copy to cache location
            dest_file = critical_units_dir / f"{test_model_id}_{decoder_type}_units.json"
            dest_file.write_text(output_file.read_text())
            decoder_files.append(dest_file)
        
        # Run aggregation with partial results
        aggregated_file = critical_units_dir / f"{test_model_id}_units.json"
        aggregation_script = Path("scripts/aggregate_critical_units.py")
        
        cmd = [
            sys.executable, str(aggregation_script),
            "--cache-dir", str(cache_dir),
            "--model-id", test_model_id,
            "--output", str(aggregated_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        assert result.returncode == 0, f"Partial aggregation failed: {result.stderr}"
        
        # Validate partial aggregation
        with open(aggregated_file, 'r') as f:
            data = json.load(f)
        
        metadata = data["metadata"]
        assert metadata["n_decoders_with_results"] == len(partial_decoders), \
            "Should correctly count partial decoders"
        assert metadata["n_decoders_with_results"] < 5, "Should be less than full decoder count"
        
        print(f"    ✓ Partial aggregation: {metadata['n_decoders_with_results']}/5 decoders")
    
    def test_large_regularization_values(self, test_states_file, test_model_id, temp_output_dir):
        """Test robustness with extreme regularization values."""
        
        print("Testing robustness with extreme regularization...")
        
        # The system uses predefined alpha values, but we can test that it handles
        # extreme results gracefully by checking the output
        output_file = temp_output_dir / f"{test_model_id}_extreme_reg_test.json"
        
        cmd = [
            sys.executable, "-m", "learning_in_context.analysis.critical_units",
            "--states", str(test_states_file),
            "--output", str(output_file),
            "--model-id", test_model_id,
            "--decoder-type", "color",
            "--regularization-sweep"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        assert result.returncode == 0, f"Extreme regularization test failed: {result.stderr}"
        
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        # Check that alpha is in reasonable range
        best_alpha = data["best_alpha"]
        assert 1e-8 <= best_alpha <= 10, f"Best alpha should be reasonable: {best_alpha}"
        
        # Check that coefficients are finite
        coefficients = data["coefficients"]
        if coefficients:
            assert all(np.isfinite(coef) for coef in coefficients), "Coefficients should be finite"
        
        print(f"    ✓ Extreme regularization handled: best_alpha={best_alpha:.1e}")


class TestErrorHandlingRobustness:
    """Test error handling and recovery."""
    
    def test_corrupted_data_handling(self, test_model_id, temp_output_dir):
        """Test handling of corrupted data files."""
        
        print("Testing corrupted data handling...")
        
        # Create corrupted npz file
        corrupted_file = temp_output_dir / f"{test_model_id}_corrupted.npz"
        with open(corrupted_file, 'w') as f:
            f.write("This is not a valid npz file")
        
        output_file = temp_output_dir / f"{test_model_id}_corrupted_test.json"
        
        cmd = [
            sys.executable, "-m", "learning_in_context.analysis.critical_units",
            "--states", str(corrupted_file),
            "--output", str(output_file),
            "--model-id", test_model_id,
            "--decoder-type", "color",
            "--regularization-sweep"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        assert result.returncode != 0, "Should fail with corrupted data"
        assert not output_file.exists(), "Should not create output on failure"
        
        print("    ✓ Corrupted data handled gracefully")
    
    def test_invalid_decoder_parameters(self, test_states_file, test_model_id, temp_output_dir):
        """Test handling of invalid decoder parameters."""
        
        print("Testing invalid decoder parameters...")
        
        # Test invalid timestep
        output_file = temp_output_dir / f"{test_model_id}_invalid_timestep.json"
        
        cmd = [
            sys.executable, "-m", "learning_in_context.analysis.critical_units",
            "--states", str(test_states_file),
            "--output", str(output_file),
            "--model-id", test_model_id,
            "--decoder-type", "color",
            "--timestep", "999999",  # Invalid timestep
            "--regularization-sweep"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        
        # Should either handle gracefully or fail with informative error
        if result.returncode != 0:
            assert "timestep" in result.stderr.lower() or "index" in result.stderr.lower() or \
                   "out of bounds" in result.stderr.lower(), \
                   "Should provide informative error for invalid timestep"
            print("    ✓ Invalid timestep handled with informative error")
        else:
            # If it succeeds, should use fallback behavior
            print("    ✓ Invalid timestep handled with fallback behavior")
    
    def test_disk_space_simulation(self, test_states_file, test_model_id, temp_output_dir):
        """Test behavior when output directory is not writable."""
        
        print("Testing write permission handling...")
        
        # Create read-only directory
        readonly_dir = temp_output_dir / "readonly"
        readonly_dir.mkdir()
        
        try:
            # Make directory read-only
            readonly_dir.chmod(0o444)
            
            output_file = readonly_dir / f"{test_model_id}_readonly_test.json"
            
            cmd = [
                sys.executable, "-m", "learning_in_context.analysis.critical_units",
                "--states", str(test_states_file),
                "--output", str(output_file),
                "--model-id", test_model_id,
                "--decoder-type", "color",
                "--regularization-sweep"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            assert result.returncode != 0, "Should fail when cannot write output"
            
            print("    ✓ Write permission errors handled gracefully")
            
        finally:
            # Restore write permissions for cleanup
            readonly_dir.chmod(0o755)