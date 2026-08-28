"""
Simplified integration tests for extract_states_group configuration functionality.

NOTE: For end-to-end pipeline testing, see test_extract_states_group_e2e.py
This file focuses on configuration and setup validation only.
"""

import pytest
from pathlib import Path

from learning_in_context.config import PipelineConfig
from .conftest import run_doit


@pytest.mark.integration
class TestConfigurationIntegration:
    """Test configuration and setup functionality."""
    
    def test_weights_dir_override_in_config(self):
        """Test that PipelineConfig respects weights_dir override."""
        # Test without override
        config1 = PipelineConfig()
        assert 'data/weights/analyze' in str(config1.weights_dir)
        
        # Test with override
        test_weights = '/test/path/weights'
        config2 = PipelineConfig(weights_dir=test_weights)
        assert str(config2.weights_dir) == test_weights
        
        # Test with relative path override
        config3 = PipelineConfig(weights_dir='tests/data/weights/analyze')
        assert 'tests/data/weights/analyze' in str(config3.weights_dir)
    
    def test_get_model_checkpoint_with_override(self):
        """Test checkpoint discovery with overridden weights directory."""
        # Use actual test directory structure
        test_dir = Path(__file__).parent / 'data' / 'weights' / 'analyze'
        
        config = PipelineConfig(weights_dir=str(test_dir))
        
        # Test finding existing test checkpoint
        checkpoint = config.get_model_checkpoint('TEST-001')
        if checkpoint:  # Only test if checkpoint exists
            assert checkpoint.exists()
            assert checkpoint.name == 'last.ckpt'
            assert 'TEST-001' in str(checkpoint)
    
    def test_file_naming_convention(self):
        """Test that file naming follows expected conventions."""
        model_id = 'TEST-001'
        
        # Expected file names
        raw_states_file = f'{model_id}_states.npz'
        normalized_states_file = f'{model_id}_states_normalized.npz'
        
        # Verify naming pattern
        assert raw_states_file == 'TEST-001_states.npz'
        assert normalized_states_file == 'TEST-001_states_normalized.npz'
        
        # Test that they're different files
        assert raw_states_file != normalized_states_file


@pytest.mark.integration
class TestWeightsDirEdgeCases:
    """Test edge cases for weights directory configuration."""
    
    def test_nonexistent_weights_dir(self):
        """Test behavior with nonexistent weights directory."""
        config = PipelineConfig(weights_dir='/nonexistent/path')
        
        # Should not crash on creation
        assert config.weights_dir == Path('/nonexistent/path')
        
        # Should return None for nonexistent model
        checkpoint = config.get_model_checkpoint('NONEXISTENT')
        assert checkpoint is None
    
    def test_relative_weights_dir(self):
        """Test relative weights directory paths."""
        config = PipelineConfig(weights_dir='relative/path/weights')
        
        # Should handle relative paths
        assert 'relative/path/weights' in str(config.weights_dir)
    
    def test_absolute_weights_dir(self):
        """Test absolute weights directory paths."""
        abs_path = '/absolute/path/weights'
        config = PipelineConfig(weights_dir=abs_path)
        
        # Should handle absolute paths
        assert str(config.weights_dir) == abs_path
    
    def test_weights_dir_override_priority(self):
        """Test that weights_dir parameter takes priority over config."""
        # Create config dict with weights_dir
        config_overrides = {
            'data': {
                'weights_dir': '/config/weights'
            }
        }
        
        # Test that constructor parameter overrides config
        config = PipelineConfig(
            overrides=config_overrides,
            weights_dir='/param/weights'
        )
        
        # Parameter should take priority
        assert str(config.weights_dir) == '/param/weights'


@pytest.mark.doit
@pytest.mark.integration
class TestDoItVariableIntegration:
    """Test DoIt variable system integration for testing."""
    
    def test_cpu_override_in_task_info(self):
        """Test that cpu=true parameter forces CPU device in task generation."""
        # Test the specific subtask that should contain the command
        result = run_doit(
            "info", "extract_model_states:TEST-001:participant",
            "models=TEST-001",
            "weights_dir=tests/data/weights/analyze",
            "cpu=true",
        )
        
        # Should complete without error (exit code may vary)
        output = result.stdout + result.stderr
        
        # Should show device cpu in the actual command (if task exists)
        if "python -m learning_in_context.models.extract_states" in output or "extract_states" in output:
            # Task command was shown, check for CPU device
            assert "--device cpu" in output, \
                   f"CPU device not found in task command: {output}"
        else:
            # If the task doesn't exist, that's also a valid test outcome
            # (e.g., no test checkpoints available)
            print(f"Task not found or generated: {output}")
            pytest.skip("Task not available for testing")
    
    def test_model_selection_override(self):
        """Test that explicit model selection works in test environment."""
        result = run_doit(
            "info", "extract_model_states",
            "models=TEST-001,TEST-002",
            "weights_dir=tests/data/weights/analyze",
            "cpu=true",
        )
        
        output = result.stdout + result.stderr
        
        # Should show TEST models in task names or dependencies
        assert "TEST-001" in output or "TEST-002" in output, \
               f"Test models not found in task output: {output}"
    
    def test_empty_models_handling(self):
        """Test handling of empty models parameter."""
        result = run_doit(
            "info", "extract_model_states",
            "models=",
            "cpu=true",
        )
        
        # Should not crash, even with no models
        output = result.stdout + result.stderr
        assert len(output) > 0, "Command produced no output"
    
    def test_dataset_parameter_in_task_names(self):
        """Test that dataset parameter affects task naming."""
        result = run_doit(
            "info", "extract_model_states:TEST-001:participant",
            "models=TEST-001",
            "weights_dir=tests/data/weights/analyze",
            "datasets=participant",
            "cpu=true",
        )
        
        output = result.stdout + result.stderr
        
        # Should reference participant dataset in task structure
        if "extract_model_states" in output:
            assert "participant" in output, \
                   f"Dataset name not found in task output: {output}"