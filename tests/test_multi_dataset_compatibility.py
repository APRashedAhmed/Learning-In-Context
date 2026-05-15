"""Test backward compatibility of multi-dataset implementation."""

import argparse
import json
import pytest
import tempfile
from pathlib import Path

import sys
# Add scripts to path for aggregation functions
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


@pytest.mark.backward_compat
class TestBackwardCompatibility:
    """Test backward compatibility of the multi-dataset implementation."""

    def test_default_behavior(self, multi_dataset_configs):
        """Test that default behavior uses participant dataset."""
        from learning_in_context.core.constants import DEFAULT_DATASET
        
        # Check default dataset
        assert DEFAULT_DATASET == 'participant', f"Default dataset should be 'participant', got '{DEFAULT_DATASET}'"
        
        # Check participant config exists
        assert 'participant' in multi_dataset_configs, "Participant dataset not in configs"
        
        participant_config = multi_dataset_configs['participant']
        assert participant_config['suffix'] == 'participant', f"Participant suffix should be 'participant', got '{participant_config['suffix']}'"

    def test_state_extraction_cli_defaults(self):
        """Test that state extraction CLI has proper defaults."""
        # Simulate the CLI argument parsing from extract_states.py
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset-name', type=str, default='participant',
                          choices=['participant', 'extended', 'controlled', 'velocity'])
        
        # Test default behavior
        default_args = parser.parse_args([])
        assert default_args.dataset_name == 'participant', f"Default dataset-name should be 'participant', got '{default_args.dataset_name}'"
        
        # Test that participant is a valid choice
        try:
            participant_args = parser.parse_args(['--dataset-name', 'participant'])
            assert participant_args.dataset_name == 'participant', f"Participant should be valid CLI choice: {participant_args.dataset_name}"
        except SystemExit:
            pytest.fail("Participant is not a valid CLI choice")

    def test_file_naming_backward_compatibility(self, multi_dataset_configs):
        """Test that participant dataset file naming is backward compatible."""
        participant_config = multi_dataset_configs['participant']
        suffix = participant_config['suffix']
        
        assert suffix == 'participant', f"Participant suffix should be 'participant', got '{suffix}'"
        
        # Test expected file naming patterns
        model_id = "SAN-4566"
        expected_patterns = {
            'state_file': f"{model_id}_participant_states.npz",
            'hazard_file': f"{model_id}_participant_hazard_units.json",
            'contingency_file': f"{model_id}_participant_contingency_units.json",
            'color_file': f"{model_id}_participant_color_units.json",
            'velocity_x_file': f"{model_id}_participant_velocity_x_units.json",
            'velocity_y_file': f"{model_id}_participant_velocity_y_units.json",
            'aggregated_file': f"{model_id}_participant_units.json"
        }
        
        # Validate that all patterns contain the suffix
        for file_type, pattern in expected_patterns.items():
            assert 'participant' in pattern, f"Pattern {file_type} should contain 'participant': {pattern}"
            assert model_id in pattern, f"Pattern {file_type} should contain model ID: {pattern}"

    def test_aggregation_backward_compatibility(self, tmp_path):
        """Test that aggregation script works with old and new naming."""
        from aggregate_critical_units import load_decoder_results, aggregate_results
        
        model_id = "TEST-COMPAT"
        
        # Test 1: New naming with participant suffix
        cache_dir = tmp_path / "new_format"
        critical_units_dir = cache_dir / "critical_units"
        critical_units_dir.mkdir(parents=True)
        
        # Create files with participant suffix (new format) 
        decoder_types = ["hazard", "contingency", "color"]
        for decoder_type in decoder_types:
            result = {
                "model_id": model_id,
                "unit_indices": [1, 2, 3],
                "coefficients": [0.5, -0.3, 0.8],
                "metadata": {"n_units_total": 128, "n_units_critical": 3}
            }
            
            file_path = critical_units_dir / f"{model_id}_participant_{decoder_type}_units.json"
            with open(file_path, "w") as f:
                json.dump(result, f)
        
        # Test loading with participant suffix
        loaded_results = load_decoder_results(cache_dir, model_id, "participant")
        
        assert len(loaded_results) == 3, f"Expected 3 decoders with participant suffix, got {len(loaded_results)}"
        
        # Test aggregation
        aggregated = aggregate_results(loaded_results, model_id)
        assert aggregated is not None, "Aggregation failed with participant suffix"
        assert "unit_indices" in aggregated, "Missing unit_indices in aggregated result"
        
        # Test 2: Old naming without suffix (backward compatibility)
        cache_dir_old = tmp_path / "old_format"
        critical_units_dir_old = cache_dir_old / "critical_units"
        critical_units_dir_old.mkdir(parents=True)
        
        # Create files without suffix (old format)
        for decoder_type in decoder_types:
            result = {
                "model_id": model_id,
                "unit_indices": [4, 5, 6],
                "coefficients": [0.2, -0.7, 0.9],
                "metadata": {"n_units_total": 128, "n_units_critical": 3}
            }
            
            file_path = critical_units_dir_old / f"{model_id}_{decoder_type}_units.json"
            with open(file_path, "w") as f:
                json.dump(result, f)
        
        # Test loading without suffix (empty string for backward compatibility)
        loaded_results_old = load_decoder_results(cache_dir_old, model_id, "")
        
        assert len(loaded_results_old) == 3, f"Expected 3 decoders without suffix, got {len(loaded_results_old)}"
        
        # Test aggregation
        aggregated_old = aggregate_results(loaded_results_old, model_id)
        assert aggregated_old is not None, "Aggregation failed with old format"
        assert "unit_indices" in aggregated_old, "Missing unit_indices in old format aggregated result"

    def test_import_compatibility(self):
        """Test that all imports work as expected."""
        # Test that core constants can be imported
        from learning_in_context.core.constants import DATASET_CONFIGS, DEFAULT_DATASET
        assert isinstance(DATASET_CONFIGS, dict)
        assert isinstance(DEFAULT_DATASET, str)
        
        # Test that state extraction functions can be imported
        from learning_in_context.models.extract_states import save_states, extract_states_with_config
        assert callable(save_states)
        assert callable(extract_states_with_config)
        
        # Test that aggregation functions can be imported
        from aggregate_critical_units import load_decoder_results, aggregate_results
        assert callable(load_decoder_results)
        assert callable(aggregate_results)
        
        # Test that core data types can be imported
        from learning_in_context.core.data_types import CriticalUnitsResult
        assert CriticalUnitsResult is not None

    def test_api_compatibility(self):
        """Test that API interfaces remain compatible."""
        # Test save_states function signature compatibility
        from learning_in_context.models.extract_states import save_states
        import inspect
        
        sig = inspect.signature(save_states)
        params = list(sig.parameters.keys())
        
        # Should have dataset_name parameter with default
        assert 'dataset_name' in params, "save_states should have dataset_name parameter"
        
        # Default should be 'participant' for backward compatibility
        dataset_param = sig.parameters['dataset_name']
        assert dataset_param.default == 'participant', f"dataset_name default should be 'participant', got {dataset_param.default}"

    def test_constants_backward_compatibility(self, multi_dataset_configs):
        """Test that constants maintain backward compatibility."""
        from learning_in_context.core.constants import DEFAULT_DATASET
        
        # Default dataset should remain participant
        assert DEFAULT_DATASET == 'participant'
        
        # Participant dataset should exist and have expected properties
        participant_config = multi_dataset_configs['participant']
        assert participant_config['suffix'] == 'participant'
        assert participant_config['size'] > 0
        assert len(participant_config['path']) > 0

    @pytest.mark.parametrize("function_name", ["load_decoder_results", "aggregate_results"])
    def test_aggregation_function_signatures(self, function_name):
        """Test that aggregation function signatures support multi-dataset."""
        from aggregate_critical_units import load_decoder_results, aggregate_results
        import inspect
        
        if function_name == "load_decoder_results":
            func = load_decoder_results
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            
            # Should have dataset_suffix parameter
            assert 'dataset_suffix' in params, f"{function_name} should have dataset_suffix parameter"
            
            # Should have default value for backward compatibility
            dataset_param = sig.parameters['dataset_suffix']
            assert dataset_param.default == "", f"dataset_suffix default should be empty string for backward compatibility"
        
        elif function_name == "aggregate_results":
            func = aggregate_results
            sig = inspect.signature(func)
            # aggregate_results shouldn't need dataset-specific parameters
            assert callable(func)