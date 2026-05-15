"""Test multi-dataset aggregation functionality."""

import json
import pytest
import tempfile
from pathlib import Path

import sys
# Add scripts to path for aggregation functions
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


@pytest.mark.multi_dataset
class TestMultiDatasetAggregation:
    """Test aggregation script functionality with multi-dataset support."""

    @pytest.fixture
    def decoder_types(self):
        """List of all decoder types to test."""
        return ["hazard", "contingency", "color", "velocity_x", "velocity_y"]

    @pytest.fixture
    def mock_decoder_result(self):
        """Create mock decoder result data."""
        def _create_result(decoder_type, model_id, dataset_suffix="participant", n_critical=8, n_total=128):
            import random
            random.seed(42 + hash(decoder_type))  # Different seed per decoder
            
            # Generate mock critical units
            critical_indices = random.sample(range(n_total), n_critical)
            coefficients = [random.uniform(-2, 2) for _ in range(n_critical)]
            
            return {
                "decoder_type": decoder_type,
                "model_id": model_id,
                "dataset_suffix": dataset_suffix,
                "unit_indices": critical_indices,
                "coefficients": coefficients,
                "best_alpha": random.uniform(0.001, 0.1),
                "r2_scores": [random.uniform(0.6, 0.9)],
                "metadata": {
                    "n_units_total": n_total,
                    "n_units_critical": n_critical,
                    "decoder_type": decoder_type,
                    "problem_type": "classification" if decoder_type in ["hazard", "contingency", "color"] else "regression",
                    "l1_ratio": 0.64 if decoder_type in ["hazard", "contingency"] else 0.4
                }
            }
        return _create_result

    def test_aggregation_with_dataset_suffix(self, tmp_path, decoder_types, mock_decoder_result):
        """Test aggregation script with dataset suffix parameter."""
        from aggregate_critical_units import load_decoder_results, aggregate_results
        
        model_id = "TEST-001"
        dataset_suffix = "extended"
        
        # Create mock files directory
        cache_dir = tmp_path
        critical_units_dir = cache_dir / "critical_units"
        critical_units_dir.mkdir(parents=True)
        
        # Create mock decoder results with dataset suffix
        expected_files = []
        for decoder_type in decoder_types:
            result = mock_decoder_result(decoder_type, model_id, dataset_suffix)
            
            # Save with dataset suffix naming pattern
            file_path = critical_units_dir / f"{model_id}_{dataset_suffix}_{decoder_type}_units.json"
            expected_files.append(file_path)
            
            with open(file_path, "w") as f:
                json.dump(result, f, indent=2)
        
        # Test loading with dataset suffix
        loaded_results = load_decoder_results(cache_dir, model_id, dataset_suffix)
        
        assert len(loaded_results) == 5, f"Expected 5 decoders, got {len(loaded_results)}"
        
        for decoder_type in decoder_types:
            assert decoder_type in loaded_results, f"Missing {decoder_type} decoder"
        
        # Test aggregation
        aggregated = aggregate_results(loaded_results, model_id)
        
        # Validate aggregated result structure
        required_keys = ["model_id", "unit_indices", "coefficients", "r2_scores", 
                       "best_alpha", "cv_scores", "metadata"]
        for key in required_keys:
            assert key in aggregated, f"Missing key in aggregated result: {key}"
        
        # Check metadata
        metadata = aggregated["metadata"]
        assert metadata["n_decoders"] == 5, f"Expected 5 decoders in metadata, got {metadata['n_decoders']}"
        assert metadata["n_decoders_with_results"] == 5, f"Expected 5 decoders with results, got {metadata['n_decoders_with_results']}"

    @pytest.mark.backward_compat  
    def test_aggregation_without_suffix(self, tmp_path, decoder_types, mock_decoder_result):
        """Test aggregation script without dataset suffix (backward compatibility)."""
        from aggregate_critical_units import load_decoder_results, aggregate_results
        
        model_id = "TEST-002"
        
        # Create mock files directory
        cache_dir = tmp_path
        critical_units_dir = cache_dir / "critical_units"
        critical_units_dir.mkdir(parents=True)
        
        # Create mock decoder results without dataset suffix (old format)
        for decoder_type in decoder_types:
            result = mock_decoder_result(decoder_type, model_id, "", n_critical=6)
            
            # Save with old naming pattern (no suffix)
            file_path = critical_units_dir / f"{model_id}_{decoder_type}_units.json"
            
            with open(file_path, "w") as f:
                json.dump(result, f, indent=2)
        
        # Test loading without dataset suffix (empty string)
        loaded_results = load_decoder_results(cache_dir, model_id, "")
        
        assert len(loaded_results) == 5, f"Expected 5 decoders, got {len(loaded_results)}"
        
        # Test aggregation
        aggregated = aggregate_results(loaded_results, model_id)
        
        # Basic validation
        assert "unit_indices" in aggregated, "Missing unit_indices in aggregated result"
        
        metadata = aggregated["metadata"]
        assert metadata["n_decoders_with_results"] == 5, "Should have results from all 5 decoders"

    def test_aggregation_with_missing_files(self, tmp_path, mock_decoder_result):
        """Test aggregation script behavior with missing decoder files."""
        from aggregate_critical_units import load_decoder_results, aggregate_results
        
        model_id = "TEST-003"
        
        # Create mock files directory
        cache_dir = tmp_path
        critical_units_dir = cache_dir / "critical_units"
        critical_units_dir.mkdir(parents=True)
        
        # Create only hazard and color decoders
        available_decoders = ["hazard", "color"]
        for decoder_type in available_decoders:
            result = mock_decoder_result(decoder_type, model_id, "participant")
            
            file_path = critical_units_dir / f"{model_id}_participant_{decoder_type}_units.json"
            
            with open(file_path, "w") as f:
                json.dump(result, f, indent=2)
        
        # Test loading with some missing files
        loaded_results = load_decoder_results(cache_dir, model_id, "participant")
        
        assert len(loaded_results) == 2, f"Expected 2 decoders, got {len(loaded_results)}"
        
        expected_decoders = set(available_decoders)
        actual_decoders = set(loaded_results.keys())
        assert expected_decoders == actual_decoders, f"Expected {expected_decoders}, got {actual_decoders}"
        
        # Test aggregation with partial results
        aggregated = aggregate_results(loaded_results, model_id)
        
        metadata = aggregated["metadata"]
        assert metadata["n_decoders_with_results"] == 2, f"Expected 2 decoders with results, got {metadata['n_decoders_with_results']}"
        assert metadata["n_decoders"] == 2, f"Expected 2 total decoders, got {metadata['n_decoders']}"

    @pytest.mark.parametrize("dataset_suffix", ["participant", "extended", "controlled", "velocity"])
    def test_aggregation_all_datasets(self, tmp_path, dataset_suffix, decoder_types, mock_decoder_result):
        """Test aggregation works for all dataset types."""
        from aggregate_critical_units import load_decoder_results, aggregate_results
        
        model_id = "TEST-PARAM"
        
        # Create mock files directory
        cache_dir = tmp_path
        critical_units_dir = cache_dir / "critical_units"
        critical_units_dir.mkdir(parents=True)
        
        # Create decoder results for this dataset
        for decoder_type in decoder_types:
            result = mock_decoder_result(decoder_type, model_id, dataset_suffix)
            
            # Save with dataset suffix naming pattern
            file_path = critical_units_dir / f"{model_id}_{dataset_suffix}_{decoder_type}_units.json"
            
            with open(file_path, "w") as f:
                json.dump(result, f, indent=2)
        
        # Test loading and aggregation
        loaded_results = load_decoder_results(cache_dir, model_id, dataset_suffix)
        aggregated = aggregate_results(loaded_results, model_id)
        
        # Validate basic structure
        assert "unit_indices" in aggregated
        assert "metadata" in aggregated
        assert aggregated["model_id"] == model_id

    def test_dataset_suffix_parameter_validation(self, tmp_path, mock_decoder_result):
        """Test that dataset suffix parameter is handled correctly."""
        from aggregate_critical_units import load_decoder_results
        
        model_id = "TEST-SUFFIX"
        cache_dir = tmp_path
        critical_units_dir = cache_dir / "critical_units"
        critical_units_dir.mkdir(parents=True)
        
        # Create files with different suffixes
        suffixes = ["participant", "extended"] 
        for suffix in suffixes:
            result = mock_decoder_result("hazard", model_id, suffix)
            file_path = critical_units_dir / f"{model_id}_{suffix}_hazard_units.json"
            
            with open(file_path, "w") as f:
                json.dump(result, f, indent=2)
        
        # Test loading with specific suffix only loads that suffix
        participant_results = load_decoder_results(cache_dir, model_id, "participant")
        extended_results = load_decoder_results(cache_dir, model_id, "extended")
        
        # Each should only load their own files
        assert len(participant_results) == 1, "Should only load participant files"
        assert len(extended_results) == 1, "Should only load extended files"
        
        # Files should not be mixed up
        assert "hazard" in participant_results
        assert "hazard" in extended_results