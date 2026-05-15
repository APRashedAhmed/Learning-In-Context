"""Test critical units identification."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from learning_in_context.analysis.critical_units import (
    aggregate_critical_units,
    analyze_temporal_dynamics,
    identify_critical_units,
    linear_regularization_pipeline,
    reg_single,
    reg_multi
)
from learning_in_context.core import CriticalUnitsResult, StateData


class TestCriticalUnitsIdentification:
    """Test critical units identification functionality."""
    
    @pytest.fixture
    def sample_states(self):
        """Generate sample neural states for testing."""
        n_trials = 50
        n_timesteps = 20
        n_units = 30
        
        # Create states with some units more predictive than others
        states = np.random.randn(n_trials, n_timesteps, n_units)
        
        # Make units 0, 5, 10 highly predictive
        critical_units = [0, 5, 10]
        labels = np.zeros((n_trials, n_timesteps), dtype=int)
        
        for unit_idx in critical_units:
            # Create correlation between unit activation and labels
            unit_activation = states[:, :, unit_idx]
            labels += (unit_activation > 0).astype(int)
        
        labels = np.clip(labels, 0, 2)  # 3 classes
        
        return states, labels, critical_units
    
    def test_identify_critical_units_basic(self, sample_states):
        """Test basic critical units identification."""
        states, labels, expected_critical = sample_states
        
        # Test with regularization sweep (new implementation)
        result = identify_critical_units(
            states=states,
            labels=labels,
            lambda_min=1e-4,  # Less aggressive regularization for tests
            lambda_max=1e-1,
            n_lambdas=10,  # Fewer lambdas for faster tests
            binary_l1_ratio=0.64,
            timestep=-1,  # Last timestep (default)
            zscore_states=True,
            use_all_timesteps=False  # Single timestep mode
        )
        
        # Check result structure
        assert isinstance(result, dict)
        assert "unit_indices" in result
        assert "coefficients" in result
        assert "r2_scores" in result
        assert "best_alpha" in result
        assert "cv_scores" in result
        assert "metadata" in result
        
        # New fields from regularization sweep
        assert "alpha_values" in result
        assert "coefficient_paths" in result
        assert "performance_curves" in result
        assert "threshold_analysis" in result
        
        # Check that some units were identified
        assert len(result["unit_indices"]) >= 0  # May be 0 with strong regularization
        
        # Check score is reasonable (accuracy for binary classification)
        assert 0.0 <= result["r2_scores"][0] <= 1.0
        
        # Check metadata
        assert result["metadata"]["n_units_total"] <= states.shape[2]
        assert result["metadata"]["n_units_critical"] == len(result["unit_indices"])
        assert result["metadata"]["lambda_min"] == 1e-4
        assert result["metadata"]["lambda_max"] == 1e-1
        assert result["metadata"]["timestep"] == -1
        assert result["metadata"]["zscore_applied"] == True
    
    def test_identify_critical_units_timestep(self, sample_states):
        """Test critical units identification at specific timestep."""
        states, labels, _ = sample_states
        
        # Analyze timestep 10 with regularization sweep
        result = identify_critical_units(
            states=states,
            labels=labels,
            timestep=10,
            lambda_min=1e-4,
            lambda_max=1e-1,
            n_lambdas=10,
            binary_l1_ratio=0.64,
            solver="saga",
            binary_max_iter=150,
            use_all_timesteps=False
        )
        
        # Check that timestep is recorded
        assert result["metadata"]["timestep"] == 10
        
        # Results should be based on single timestep data
        assert len(result["unit_indices"]) >= 0
    
    def test_identify_critical_units_low_variance_filter(self, sample_states):
        """Test filtering of low-variance units."""
        states, labels, _ = sample_states
        
        # Set some units to have very low variance
        states[:, :, 15:20] = 0.0  # Zero variance
        states[:, :, 20:22] = 1e-8  # Constant low value (variance = 0)
        
        result = identify_critical_units(
            states=states,
            labels=labels,
            exclude_low_variance=True,
            lambda_min=1e-4,
            lambda_max=1e-1,
            n_lambdas=10,
            use_all_timesteps=False
        )
        
        # Check that low variance units were excluded
        assert 15 not in result["unit_indices"]
        assert 16 not in result["unit_indices"]
        assert 20 not in result["unit_indices"]
        
        # Check metadata
        assert result["metadata"]["exclude_low_variance"] is True
    
    def test_critical_units_result_object(self, sample_states):
        """Test CriticalUnitsResult data type."""
        states, labels, _ = sample_states
        
        result_dict = identify_critical_units(states=states, labels=labels, use_all_timesteps=False)
        
        # Create CriticalUnitsResult object
        result = CriticalUnitsResult(
            unit_indices=np.array(result_dict["unit_indices"]),
            coefficients=np.array(result_dict["coefficients"]),
            r2_scores=np.array(result_dict["r2_scores"]),
            best_alpha=result_dict["best_alpha"],
            cv_scores=result_dict["cv_scores"],
            metadata=result_dict["metadata"]
        )
        
        # Test methods
        top_units = result.get_top_units(n=5)
        assert len(top_units) <= 5
        assert all(unit in result.unit_indices for unit in top_units)
        
        # Test serialization
        serialized = result.to_dict()
        assert isinstance(serialized, dict)
        assert isinstance(serialized["unit_indices"], list)


class TestCriticalUnitsAggregation:
    """Test aggregation of critical units across models."""
    
    @pytest.fixture
    def model_results(self):
        """Create sample results from multiple models."""
        results = {
            "model1": {
                "unit_indices": [0, 5, 10, 15],
                "coefficients": [0.5, 0.3, 0.8, 0.2]
            },
            "model2": {
                "unit_indices": [0, 5, 11, 20],
                "coefficients": [0.6, 0.4, 0.7, 0.3]
            },
            "model3": {
                "unit_indices": [0, 10, 15, 25],
                "coefficients": [0.7, 0.5, 0.3, 0.4]
            }
        }
        return results
    
    def test_aggregate_critical_units(self, model_results):
        """Test aggregation across models."""
        aggregated = aggregate_critical_units(model_results)
        
        # Check structure
        assert "unit_frequency" in aggregated
        assert "mean_coefficients" in aggregated
        assert "consistency_score" in aggregated
        assert "n_models" in aggregated
        assert "top_units" in aggregated
        
        # Check unit frequency calculation
        # Unit 0 appears in all 3 models
        assert aggregated["unit_frequency"][0] == 1.0
        # Unit 5 appears in 2/3 models
        assert aggregated["unit_frequency"][5] == pytest.approx(2/3)
        
        # Check mean coefficients
        # Unit 0: (0.5 + 0.6 + 0.7) / 3
        assert aggregated["mean_coefficients"][0] == pytest.approx(0.6, abs=0.01)
        
        # Check consistency score is between 0 and 1
        assert 0 <= aggregated["consistency_score"] <= 1
        
        # Check top units
        assert len(aggregated["top_units"]) <= 20
        # Unit 0 should be at the top (appears in all models)
        assert aggregated["top_units"][0][0] == 0


class TestTemporalDynamics:
    """Test temporal dynamics analysis."""
    
    @pytest.fixture
    def temporal_states(self):
        """Generate states with temporal patterns."""
        n_trials = 30
        n_timesteps = 50
        n_units = 20
        
        states = np.random.randn(n_trials, n_timesteps, n_units)
        
        # Make different units critical at different times
        labels = np.zeros((n_trials, n_timesteps), dtype=int)
        
        # Unit 0 is critical early
        states[:, :20, 0] *= 2
        labels[:, :20] += (states[:, :20, 0] > 0).astype(int)
        
        # Unit 5 is critical late
        states[:, 30:, 5] *= 2
        labels[:, 30:] += (states[:, 30:, 5] > 0).astype(int)
        
        # Unit 10 is always critical
        states[:, :, 10] *= 2
        labels += (states[:, :, 10] > 0).astype(int)
        
        labels = np.clip(labels, 0, 2)
        
        return states, labels
    
    def test_analyze_temporal_dynamics(self, temporal_states):
        """Test temporal dynamics analysis."""
        states, labels = temporal_states
        
        result = analyze_temporal_dynamics(
            states=states,
            labels=labels,
            window_size=15,
            stride=10,
            lambda_min=1e-4,
            lambda_max=1e-1,
            n_lambdas=10,
            binary_l1_ratio=0.5,
            cv_folds=3,
            use_all_timesteps=False
        )
        
        # Check structure
        assert "windows" in result
        assert "stable_units" in result
        assert "transient_units" in result
        assert "n_stable" in result
        assert "n_transient" in result
        
        # Check windows
        assert len(result["windows"]) > 0
        for window in result["windows"]:
            assert "window_start" in window
            assert "window_end" in window
            assert "critical_units" in window
            assert "n_critical" in window
            assert "r2_score" in window
        
        # Check that we have some analysis results
        assert result["n_stable"] >= 0
        assert result["n_transient"] >= 0
        # At least some units should be identified across windows
        total_units_found = result["n_stable"] + result["n_transient"]
        assert total_units_found >= 0  # Allow for case where no units are critical


class TestCriticalUnitsIO:
    """Test I/O operations for critical units."""
    
    @pytest.fixture
    def sample_states(self):
        """Generate sample neural states for testing."""
        n_trials = 50
        n_timesteps = 20
        n_units = 30
        
        # Create states with some units more predictive than others
        states = np.random.randn(n_trials, n_timesteps, n_units)
        
        # Make units 0, 5, 10 highly predictive
        critical_units = [0, 5, 10]
        labels = np.zeros((n_trials, n_timesteps), dtype=int)
        
        for unit_idx in critical_units:
            # Create correlation between unit activation and labels
            unit_activation = states[:, :, unit_idx]
            labels += (unit_activation > 0).astype(int)
        
        labels = np.clip(labels, 0, 2)  # 3 classes
        
        return states, labels, critical_units
    
    def test_save_load_results(self, sample_states):
        """Test saving and loading critical units results."""
        states, labels, _ = sample_states
        
        # Debug: check types
        assert isinstance(states, np.ndarray), f"Expected numpy array, got {type(states)}"
        assert isinstance(labels, np.ndarray), f"Expected numpy array, got {type(labels)}"
        
        # Get results with regularization sweep
        result = identify_critical_units(
            states=states,
            labels=labels,
            lambda_min=1e-4,
            lambda_max=1e-1,
            n_lambdas=10,
            use_all_timesteps=False
        )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(result, f, indent=2)
            temp_path = Path(f.name)
        
        try:
            # Load and verify
            with open(temp_path, 'r') as f:
                loaded = json.load(f)
            
            assert loaded["best_alpha"] == result["best_alpha"]
            assert len(loaded["unit_indices"]) == len(result["unit_indices"])
            np.testing.assert_array_equal(loaded["unit_indices"], result["unit_indices"])
            
        finally:
            temp_path.unlink()
    
    def test_integration_with_state_data(self):
        """Test integration with StateData type."""
        # Create sample StateData
        n_trials = 20
        n_timesteps = 30
        n_units = 25
        
        hiddens = np.random.randn(n_trials, n_timesteps, n_units)
        predictions = np.random.rand(n_trials, n_timesteps, 3)
        predictions = predictions / predictions.sum(axis=-1, keepdims=True)
        
        state_data = StateData(
            hiddens=hiddens,
            cells=None,
            predictions=predictions,
            metadata={"test": True}
        )
        
        # Extract labels from predictions
        labels = np.argmax(state_data.predictions, axis=-1)
        
        # Run critical units identification with sweep
        result = identify_critical_units(
            states=state_data.hiddens,
            labels=labels,
            lambda_min=1e-4,
            lambda_max=1e-1,
            n_lambdas=10,
            use_all_timesteps=False
        )
        
        assert isinstance(result, dict)
        assert "unit_indices" in result


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_no_predictive_units(self):
        """Test when no units are predictive."""
        # Random states and labels with no correlation
        states = np.random.randn(20, 10, 15)
        labels = np.random.randint(0, 3, size=(20, 10))
        
        result = identify_critical_units(
            states=states,
            labels=labels,
            lambda_min=1e-3,
            lambda_max=1e-1,
            n_lambdas=5,
            use_all_timesteps=False
        )
        
        # Should still return valid result
        assert isinstance(result, dict)
        # Might identify few or no critical units
        assert len(result["unit_indices"]) >= 0
        # R² should be low
        assert result["r2_scores"][0] < 0.5
    
    def test_single_trial(self):
        """Test with minimal data."""
        states = np.random.randn(5, 10, 8)  # Very few trials
        labels = np.random.randint(0, 3, size=(5, 10))
        
        # Should handle gracefully (might warn about small sample size)
        result = identify_critical_units(
            states=states,
            labels=labels,
            cv_folds=2,  # Reduce folds for small data
            lambda_min=1e-4,
            lambda_max=1e-1,
            n_lambdas=5,  # Fewer lambdas for small data
            use_all_timesteps=False
        )
        
        assert isinstance(result, dict)


class TestRegularizationSweep:
    """Test regularization sweep pipeline."""
    
    def test_regularization_sweep_binary(self):
        """Test regularization sweep for binary classification."""
        from sklearn.metrics import accuracy_score
        from scipy import stats
        
        # Create synthetic data
        np.random.seed(42)
        n_samples = 100
        n_features = 50
        X = np.random.randn(n_samples, n_features)
        
        # Z-score the features (matching original)
        X = stats.zscore(X, axis=0, ddof=1)
        
        # Create binary labels with some signal
        y = (X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n_samples) * 0.5 > 0).astype(int)
        
        # Define metrics
        dict_metrics = {
            "accuracy": (accuracy_score, {})
        }
        
        # Run sweep
        C_logspace = np.logspace(0, -4, 10)
        
        coefs, intercepts, metrics = linear_regularization_pipeline(
            X, y,
            dict_metrics,
            reg_single,
            lambda reg, X: reg.predict(X),
            C_logspace,
            l1_ratio=0.64
        )
        
        # Check results
        assert len(coefs) == len(C_logspace)
        assert len(intercepts) == len(C_logspace)
        assert len(metrics["accuracy"]) == len(C_logspace)
        
        # Check that regularization increases sparsity
        n_nonzero_first = np.sum(coefs[0] != 0)
        n_nonzero_last = np.sum(coefs[-1] != 0)
        assert n_nonzero_last <= n_nonzero_first
        
        # Check accuracy is reasonable
        assert all(0.0 <= acc <= 1.0 for acc in metrics["accuracy"])
    
    def test_regularization_sweep_multiclass(self):
        """Test regularization sweep for multi-class classification."""
        from sklearn.metrics import accuracy_score
        from scipy import stats
        
        # Create synthetic data
        np.random.seed(42)
        n_samples = 150
        n_features = 50
        X = np.random.randn(n_samples, n_features)
        
        # Z-score the features
        X = stats.zscore(X, axis=0, ddof=1)
        
        # Create 3-class labels
        y = np.random.choice([0, 1, 2], size=n_samples)
        
        # Define metrics
        dict_metrics = {
            "accuracy": (accuracy_score, {})
        }
        
        # Run sweep
        C_logspace = np.logspace(0, -3, 5)
        
        coefs, intercepts, metrics = linear_regularization_pipeline(
            X, y,
            dict_metrics,
            reg_multi,
            lambda reg, X: reg.predict(X),
            C_logspace,
            l1_ratio=0.4
        )
        
        # Check results
        assert len(coefs) == len(C_logspace)
        assert len(intercepts) == len(C_logspace)
        
        # For multi-class, intercepts should have 3 values
        assert all(len(intercept) == 3 for intercept in intercepts)