"""Test critical units identification with regularization sweep implementation."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from learning_in_context.analysis.critical_units import identify_critical_units
from learning_in_context.core import CriticalUnitsResult


class TestRegularizationSweep:
    """Test the new regularization sweep implementation."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        n_trials = 50
        n_timesteps = 20
        n_units = 15
        
        # Create states with some predictive units
        states = np.random.randn(n_trials, n_timesteps, n_units)
        
        # Make units 0, 5, 10 predictive of labels
        labels = np.zeros((n_trials, n_timesteps), dtype=int)
        for unit_idx in [0, 5, 10]:
            unit_activation = states[:, :, unit_idx]
            labels += (unit_activation > 0).astype(int)
        
        labels = np.clip(labels, 0, 1)  # Binary classification
        
        return states, labels
    
    def test_sweep_basic_functionality(self, sample_data):
        """Test basic regularization sweep functionality."""
        states, labels = sample_data
        
        result = identify_critical_units(
            states=states,
            labels=labels,
            lambda_min=1e-4,
            lambda_max=1e-1,
            n_lambdas=10,
            binary_l1_ratio=0.64,
            timestep=-1,
            zscore_states=True,
            use_all_timesteps=False
        )
        
        # Check required keys
        assert isinstance(result, dict)
        required_keys = [
            "unit_indices", "coefficients", "r2_scores", 
            "best_alpha", "cv_scores", "metadata",
            "alpha_values", "coefficient_paths", 
            "intercept_paths", "performance_curves"
        ]
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"
        
        # Check metadata
        metadata = result["metadata"]
        assert metadata["lambda_min"] == 1e-4
        assert metadata["lambda_max"] == 1e-1
        assert metadata["n_lambdas"] == 10
        assert metadata["threshold_method"] == "chance"
        
        # Check sweep results
        assert len(result["alpha_values"]) == 10
        assert result["alpha_values"][0] == pytest.approx(1e-1)
        assert result["alpha_values"][-1] == pytest.approx(1e-4)
        
        # Check coefficient paths shape
        n_units_total = states.shape[2]
        coef_paths = np.array(result["coefficient_paths"])
        assert coef_paths.shape == (n_units_total, 10)
        
        # Check performance curves
        assert "accuracy" in result["performance_curves"]
        assert len(result["performance_curves"]["accuracy"]) == 10
    
    def test_sweep_threshold_detection(self, sample_data):
        """Test threshold detection in regularization sweep."""
        states, labels = sample_data
        
        result = identify_critical_units(
            states=states,
            labels=labels,
            lambda_min=1e-6,
            lambda_max=1.0,
            n_lambdas=20,
            chance_margin=0.05,
            threshold_method="chance",
            use_all_timesteps=False
        )
        
        # Check threshold analysis
        assert "threshold_analysis" in result
        threshold_info = result["threshold_analysis"]
        
        assert "threshold_idx" in threshold_info
        assert "threshold_alpha" in threshold_info  # Uses alpha terminology
        assert "threshold_score" in threshold_info
        assert "chance_level" in threshold_info
        
        # Threshold alpha should be within sweep range
        assert 1e-6 <= threshold_info["threshold_alpha"] <= 1.0
        
        # Score should be above chance
        assert threshold_info["threshold_score"] > threshold_info["chance_level"]
    
    def test_sweep_with_multiclass(self, sample_data):
        """Test sweep with multiclass classification."""
        states, _ = sample_data
        
        # Create multiclass labels
        labels = np.random.randint(0, 3, size=(states.shape[0], states.shape[1]))
        
        result = identify_critical_units(
            states=states,
            labels=labels,
            lambda_min=1e-4,
            lambda_max=1e-1,
            n_lambdas=10,
            multiclass_l1_ratio=0.4,  # Should use multiclass ratio
            decoder_type="color",  # Explicitly set multiclass decoder
            use_all_timesteps=False
        )
        
        # Check that multiclass parameters were used
        metadata = result["metadata"]
        assert metadata["l1_ratio"] == 0.4
        assert metadata["problem_type"] == "multiclass_classification"
        
        # Check threshold detection for multiclass
        threshold_info = result["threshold_analysis"]
        assert threshold_info["chance_level"] == pytest.approx(1/3, abs=0.01)
    
    def test_sweep_with_regression(self, sample_data):
        """Test sweep with regression task."""
        states, _ = sample_data
        
        # Create continuous labels
        labels = np.random.randn(states.shape[0], states.shape[1])
        
        result = identify_critical_units(
            states=states,
            labels=labels,
            lambda_min=1e-4,
            lambda_max=1e-1,
            n_lambdas=10,
            multiclass_l1_ratio=0.4,
            decoder_type="velocity_x",  # Regression decoder
            use_all_timesteps=False
        )
        
        # Check regression-specific settings
        metadata = result["metadata"]
        assert metadata["problem_type"] == "regression"
        assert metadata["score_type"] == "r2"
        
        # For regression, chance level should be 0
        threshold_info = result["threshold_analysis"]
        assert threshold_info["chance_level"] == 0.0
    
    def test_sweep_parameter_validation(self, sample_data):
        """Test parameter validation for sweep."""
        states, labels = sample_data
        
        # Test invalid lambda range
        with pytest.raises(ValueError):
            identify_critical_units(
                states=states,
                labels=labels,
                lambda_min=1.0,  # Min > Max
                lambda_max=0.1,
                use_all_timesteps=False
            )
    
    def test_backward_compatibility(self, sample_data):
        """Test backward compatibility with C parameter."""
        states, labels = sample_data
        
        # Should still work with legacy C parameter
        result = identify_critical_units(
            states=states,
            labels=labels,
            C=0.001,  # Legacy parameter
            use_all_timesteps=False
        )
        
        # Should perform sweep but use C to determine range
        assert "alpha_values" in result
        assert len(result["alpha_values"]) > 1  # Should be a sweep
    
    def test_save_and_load_sweep_results(self, sample_data, tmp_path):
        """Test saving and loading sweep results."""
        states, labels = sample_data
        
        result = identify_critical_units(
            states=states,
            labels=labels,
            lambda_min=1e-4,
            lambda_max=1e-1,
            n_lambdas=5,
            use_all_timesteps=False
        )
        
        # Save to file
        output_file = tmp_path / "sweep_results.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Load and verify
        with open(output_file, 'r') as f:
            loaded = json.load(f)
        
        # Check key fields preserved
        assert loaded["metadata"]["lambda_min"] == result["metadata"]["lambda_min"]
        assert len(loaded["alpha_values"]) == len(result["alpha_values"])
        assert loaded["threshold_analysis"]["threshold_alpha"] == result["threshold_analysis"]["threshold_alpha"]
    
    def test_coefficient_path_sparsity(self, sample_data):
        """Test that regularization increases sparsity."""
        states, labels = sample_data
        
        result = identify_critical_units(
            states=states,
            labels=labels,
            lambda_min=1e-6,
            lambda_max=1.0,
            n_lambdas=20,
            use_all_timesteps=False
        )
        
        # Check coefficient paths
        coef_paths = np.array(result["coefficient_paths"])
        
        # Count non-zero coefficients at each lambda
        n_nonzero = np.sum(np.abs(coef_paths) > 1e-8, axis=0)
        
        # Should generally decrease with stronger regularization
        # (lambdas go from high to low)
        assert np.any(n_nonzero[:-1] <= n_nonzero[1:]), "Sparsity should generally increase with regularization"
        
        # At highest regularization, should have fewer non-zero than at lowest
        assert n_nonzero[0] <= n_nonzero[-1], "Highest regularization should have more sparsity"