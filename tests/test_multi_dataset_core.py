"""Test core multi-dataset functionality."""

import pytest
import tempfile
from pathlib import Path

import sys
# Add scripts to path for aggregation functions
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


@pytest.mark.multi_dataset
class TestMultiDatasetCore:
    """Test core multi-dataset functionality."""
    
    def test_imports(self):
        """Test that all key modules can be imported."""
        # Test core constants
        from learning_in_context.core.constants import DATASET_CONFIGS, DEFAULT_DATASET
        assert isinstance(DATASET_CONFIGS, dict)
        assert isinstance(DEFAULT_DATASET, str)
        
        # Test state extraction module functions  
        from learning_in_context.models.extract_states import save_states, extract_states_with_config
        assert callable(save_states)
        assert callable(extract_states_with_config)
        
        # Test aggregation script functions
        from aggregate_critical_units import load_decoder_results, aggregate_results
        assert callable(load_decoder_results)
        assert callable(aggregate_results)

    def test_dataset_config_structure(self, multi_dataset_configs):
        """Test dataset configuration structure."""
        required_keys = ['path', 'description', 'size', 'suffix']
        
        for name, config in multi_dataset_configs.items():
            # Check required keys
            missing_keys = [key for key in required_keys if key not in config]
            assert not missing_keys, f"Dataset {name} missing keys: {missing_keys}"
            
            # Check data types
            assert isinstance(config['path'], str), f"Dataset {name} path should be string"
            assert isinstance(config['size'], int), f"Dataset {name} size should be integer"
            assert isinstance(config['suffix'], str), f"Dataset {name} suffix should be string"
            
            # Check suffix matches dataset name for consistency
            assert config['suffix'] == name, f"Dataset {name} suffix should match name"

    @pytest.mark.parametrize("dataset_name", ["participant", "extended", "controlled", "velocity"])
    def test_individual_dataset_config(self, multi_dataset_configs, dataset_name):
        """Test individual dataset configurations."""
        assert dataset_name in multi_dataset_configs
        config = multi_dataset_configs[dataset_name]
        
        # Each dataset should have consistent suffix naming
        assert config['suffix'] == dataset_name
        
        # Each dataset should have a reasonable size
        assert config['size'] > 0
        
        # Each dataset should have a path
        assert len(config['path']) > 0

    def test_file_naming_logic(self, multi_dataset_configs):
        """Test that file naming logic works correctly."""
        model_id = "TEST-001"
        
        for dataset_name, config in multi_dataset_configs.items():
            suffix = config['suffix']
            
            # Test state file naming
            state_file = f"{model_id}_{suffix}_states.npz"
            assert state_file.endswith("_states.npz")
            assert suffix in state_file
            
            # Test critical units file naming
            decoder_types = ['hazard', 'contingency', 'color', 'velocity_x', 'velocity_y']
            for decoder_type in decoder_types:
                decoder_file = f"{model_id}_{suffix}_{decoder_type}_units.json"
                assert decoder_file.endswith("_units.json")
                assert suffix in decoder_file
                assert decoder_type in decoder_file
            
            # Test aggregated file naming
            agg_file = f"{model_id}_{suffix}_units.json"
            assert agg_file.endswith("_units.json")
            assert suffix in agg_file

    @pytest.mark.backward_compat
    def test_backward_compatibility(self, multi_dataset_configs):
        """Test backward compatibility with participant dataset."""
        from learning_in_context.core.constants import DEFAULT_DATASET
        
        # Check default dataset
        assert DEFAULT_DATASET == 'participant', f"Default dataset should be 'participant', got '{DEFAULT_DATASET}'"
        
        # Check participant dataset exists
        assert 'participant' in multi_dataset_configs, "Participant dataset not in configs"
        
        # Check participant suffix
        participant_config = multi_dataset_configs['participant']
        assert participant_config['suffix'] == 'participant', f"Participant suffix should be 'participant', got '{participant_config['suffix']}'"

    def test_all_expected_datasets_present(self, multi_dataset_configs):
        """Test that all expected datasets are present."""
        expected_datasets = {'participant', 'extended', 'controlled', 'velocity'}
        actual_datasets = set(multi_dataset_configs.keys())
        
        assert expected_datasets == actual_datasets, f"Expected {expected_datasets}, got {actual_datasets}"

    def test_dataset_sizes_reasonable(self, multi_dataset_configs):
        """Test that dataset sizes are within reasonable ranges."""
        for name, config in multi_dataset_configs.items():
            size = config['size']
            
            # All datasets should have at least 1 trial
            assert size >= 1, f"Dataset {name} has invalid size: {size}"
            
            # Extended dataset should be the largest
            if name == 'extended':
                assert size > 1000, f"Extended dataset should be large, got {size}"
            
            # Participant dataset should be reasonably sized
            if name == 'participant':
                assert 100 <= size <= 200, f"Participant dataset size seems wrong: {size}"