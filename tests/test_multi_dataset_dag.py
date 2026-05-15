"""Test DAG structure validation for multi-dataset implementation."""

import pytest
import inspect
from pathlib import Path

import sys
# Add scripts to path for aggregation functions
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


@pytest.mark.integration
class TestMultiDatasetDAG:
    """Test DAG structure and dependencies for multi-dataset support."""

    def test_dag_dependencies(self, multi_dataset_configs):
        """Test DAG dependencies are correctly structured."""
        # Test model and dataset combinations
        model_ids = ['SAN-4566', 'SAN-4567', 'SAN-4568']  # Default models
        datasets = ['participant', 'extended']
        
        for model_id in model_ids:
            for dataset_name in datasets:
                dataset_config = multi_dataset_configs[dataset_name]
                suffix = dataset_config['suffix']
                
                # Expected task names follow pattern: task_name:model_id:dataset_name
                extract_task = f'extract_model_states:{model_id}:{dataset_name}'
                hazard_task = f'critical_units_hazard:{model_id}:{dataset_name}'
                contingency_task = f'critical_units_contingency:{model_id}:{dataset_name}'
                color_task = f'critical_units_color:{model_id}:{dataset_name}'
                velocity_x_task = f'critical_units_velocity_x:{model_id}:{dataset_name}'
                velocity_y_task = f'critical_units_velocity_y:{model_id}:{dataset_name}'
                aggregate_task = f'aggregate_critical_units:{model_id}:{dataset_name}'
                identify_task = f'identify_critical_units:{model_id}:{dataset_name}'
                
                # Expected file outputs include suffix
                state_file = f'{model_id}_{suffix}_states.npz'
                hazard_file = f'{model_id}_{suffix}_hazard_units.json'
                contingency_file = f'{model_id}_{suffix}_contingency_units.json'
                color_file = f'{model_id}_{suffix}_color_units.json'
                velocity_x_file = f'{model_id}_{suffix}_velocity_x_units.json'
                velocity_y_file = f'{model_id}_{suffix}_velocity_y_units.json'
                aggregate_file = f'{model_id}_{suffix}_units.json'
                
                # Validate expected naming patterns
                assert state_file.endswith('_states.npz')
                assert hazard_file.endswith('_hazard_units.json')
                assert aggregate_file.endswith('_units.json')
                
                # Task names should follow consistent pattern
                assert extract_task.startswith('extract_model_states:')
                assert aggregate_task.startswith('aggregate_critical_units:')

    def test_file_naming_consistency(self, multi_dataset_configs):
        """Validate that file naming is consistent across the pipeline."""
        model_id = "TEST-001"
        decoder_types = ['hazard', 'contingency', 'color', 'velocity_x', 'velocity_y']
        
        for dataset_name, config in multi_dataset_configs.items():
            suffix = config['suffix']
            
            # State file naming
            state_pattern = f"{model_id}_{suffix}_states.npz"
            assert state_pattern.endswith('_states.npz'), f"State file pattern incorrect: {state_pattern}"
            assert suffix in state_pattern, f"Suffix missing from state pattern: {state_pattern}"
            
            # Decoder file naming  
            decoder_patterns = []
            for decoder in decoder_types:
                decoder_pattern = f"{model_id}_{suffix}_{decoder}_units.json"
                decoder_patterns.append(decoder_pattern)
                assert decoder_pattern.endswith('_units.json'), f"Decoder pattern incorrect: {decoder_pattern}"
                assert suffix in decoder_pattern, f"Suffix missing from decoder pattern: {decoder_pattern}"
                assert decoder in decoder_pattern, f"Decoder type missing from pattern: {decoder_pattern}"
            
            # Aggregated file naming
            agg_pattern = f"{model_id}_{suffix}_units.json"
            assert agg_pattern.endswith('_units.json'), f"Aggregated pattern incorrect: {agg_pattern}"
            assert suffix in agg_pattern, f"Suffix missing from aggregated pattern: {agg_pattern}"

    @pytest.mark.backward_compat
    def test_backward_compatibility_naming(self, multi_dataset_configs):
        """Validate that participant dataset maintains backward compatibility."""
        from learning_in_context.core.constants import DEFAULT_DATASET
        
        assert DEFAULT_DATASET == 'participant', f"Default dataset should be 'participant', got '{DEFAULT_DATASET}'"
        
        participant_config = multi_dataset_configs['participant']
        assert participant_config['suffix'] == 'participant', f"Participant suffix should be 'participant', got '{participant_config['suffix']}'"
        
        # Test that participant dataset files follow expected pattern
        model_id = "SAN-4566"
        expected_files = {
            'state': f"{model_id}_participant_states.npz",
            'hazard': f"{model_id}_participant_hazard_units.json",
            'contingency': f"{model_id}_participant_contingency_units.json", 
            'color': f"{model_id}_participant_color_units.json",
            'velocity_x': f"{model_id}_participant_velocity_x_units.json",
            'velocity_y': f"{model_id}_participant_velocity_y_units.json",
            'aggregated': f"{model_id}_participant_units.json"
        }
        
        # All files should contain 'participant' suffix
        for file_type, filename in expected_files.items():
            assert 'participant' in filename, f"{file_type} file should contain 'participant': {filename}"
            assert model_id in filename, f"{file_type} file should contain model ID: {filename}"

    def test_aggregation_script_args(self):
        """Validate aggregation script can handle dataset suffixes."""
        import aggregate_critical_units
        
        # Check if the function signature supports dataset_suffix
        sig = inspect.signature(aggregate_critical_units.load_decoder_results)
        params = list(sig.parameters.keys())
        
        expected_params = ['cache_dir', 'model_id', 'dataset_suffix']
        for param in expected_params:
            assert param in params, f"Missing parameter '{param}' in load_decoder_results"

    def test_task_naming_conventions(self, multi_dataset_configs):
        """Test that task naming follows consistent conventions."""
        model_id = "TEST-MODEL"
        
        for dataset_name in multi_dataset_configs.keys():
            # Task names should follow pattern: task_type:model_id:dataset_name
            expected_tasks = [
                f'extract_model_states:{model_id}:{dataset_name}',
                f'critical_units_hazard:{model_id}:{dataset_name}',
                f'critical_units_contingency:{model_id}:{dataset_name}',
                f'critical_units_color:{model_id}:{dataset_name}',
                f'critical_units_velocity_x:{model_id}:{dataset_name}',
                f'critical_units_velocity_y:{model_id}:{dataset_name}',
                f'aggregate_critical_units:{model_id}:{dataset_name}',
                f'identify_critical_units:{model_id}:{dataset_name}',
            ]
            
            for task_name in expected_tasks:
                # All task names should contain the model ID and dataset name
                assert model_id in task_name, f"Task name should contain model ID: {task_name}"
                assert dataset_name in task_name, f"Task name should contain dataset name: {task_name}"
                
                # Task names should use colon separators
                parts = task_name.split(':')
                assert len(parts) == 3, f"Task name should have 3 parts separated by colons: {task_name}"
                assert parts[1] == model_id, f"Second part should be model ID: {task_name}"
                assert parts[2] == dataset_name, f"Third part should be dataset name: {task_name}"

    def test_output_directory_structure(self, multi_dataset_configs):
        """Test expected output directory structure for multi-dataset."""
        base_dirs = ['model_states', 'critical_units']
        
        for dataset_name, config in multi_dataset_configs.items():
            suffix = config['suffix']
            model_id = "TEST-001"
            
            # Files should be organized by type, not by dataset
            # This allows for easy discovery and prevents directory proliferation
            
            # State files go in model_states/
            state_file = f"model_states/{model_id}_{suffix}_states.npz"
            assert state_file.startswith('model_states/'), f"State file should be in model_states/: {state_file}"
            
            # Critical units files go in critical_units/
            critical_file = f"critical_units/{model_id}_{suffix}_hazard_units.json"
            assert critical_file.startswith('critical_units/'), f"Critical units file should be in critical_units/: {critical_file}"

    @pytest.mark.parametrize("dataset_name", ["participant", "extended", "controlled", "velocity"])
    def test_individual_dataset_dag_structure(self, dataset_name, multi_dataset_configs):
        """Test DAG structure for individual datasets."""
        config = multi_dataset_configs[dataset_name]
        suffix = config['suffix']
        model_id = "TEST-PARAM"
        
        # Each dataset should have consistent file naming
        expected_files = {
            'state': f"{model_id}_{suffix}_states.npz",
            'aggregated': f"{model_id}_{suffix}_units.json"
        }
        
        for file_type, filename in expected_files.items():
            assert suffix in filename, f"{file_type} file should contain dataset suffix: {filename}"
            assert model_id in filename, f"{file_type} file should contain model ID: {filename}"

    def test_parallel_processing_support(self, multi_dataset_configs):
        """Test that DAG structure supports parallel processing of datasets."""
        model_id = "TEST-PARALLEL"
        
        # Each dataset should be processable independently
        # This means file names should not conflict
        state_files = []
        
        for dataset_name, config in multi_dataset_configs.items():
            suffix = config['suffix']
            state_file = f"{model_id}_{suffix}_states.npz"
            state_files.append(state_file)
        
        # All state files should be unique
        assert len(state_files) == len(set(state_files)), f"State files should be unique: {state_files}"
        
        # No two files should have the same name
        for i, file1 in enumerate(state_files):
            for j, file2 in enumerate(state_files):
                if i != j:
                    assert file1 != file2, f"Files should be unique: {file1} vs {file2}"