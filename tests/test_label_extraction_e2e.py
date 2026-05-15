"""End-to-end tests for label extraction from real test data."""

import numpy as np
import pytest

from learning_in_context.analysis.critical_units import extract_labels_from_state_data


class TestLabelExtractionE2E:
    """Test label extraction logic with real test data."""
    
    def test_label_extraction_data_structure(self, test_states_data):
        """Validate the structure of test data for label extraction."""
        # Check that test data has expected structure
        required_keys = ['hiddens', 'predictions', 'targets']
        for key in required_keys:
            assert key in test_states_data, f"Missing required key: {key}"
        
        # Check shapes
        hiddens = test_states_data['hiddens']
        predictions = test_states_data['predictions']
        targets = test_states_data['targets']
        
        assert len(hiddens.shape) == 3, "Hiddens should be 3D (trials, timesteps, units)"
        assert len(predictions.shape) == 3, "Predictions should be 3D (trials, timesteps, features)"
        assert len(targets.shape) == 3, "Targets should be 3D (trials, timesteps, features)"
        
        # Check that dimensions align
        n_trials, n_timesteps, n_units = hiddens.shape
        assert predictions.shape[:2] == (n_trials, n_timesteps), "Predictions should match trials/timesteps"
        assert targets.shape[:2] == (n_trials, n_timesteps), "Targets should match trials/timesteps"
        
        print(f"✓ Test data structure: {n_trials} trials, {n_timesteps} timesteps, {n_units} units")
        print(f"✓ Predictions shape: {predictions.shape}")
        print(f"✓ Targets shape: {targets.shape}")
    
    def test_color_label_extraction(self, test_states_data):
        """Test color label extraction using model predictions."""
        labels = extract_labels_from_state_data(test_states_data, "color")
        
        # Check basic properties
        assert isinstance(labels, np.ndarray), "Labels should be numpy array"
        assert len(labels.shape) <= 2, "Labels should be 1D or 2D"
        
        # Should have appropriate number of trials
        expected_trials = test_states_data['hiddens'].shape[0]
        if len(labels.shape) == 1:
            assert len(labels) == expected_trials, f"Should have {expected_trials} labels"
        else:
            assert labels.shape[0] == expected_trials, f"Should have {expected_trials} trials"
        
        # Color labels should be integers (class indices)
        assert np.issubdtype(labels.dtype, np.integer) or np.issubdtype(labels.dtype, np.floating), \
            "Color labels should be numeric"
        
        # Check value range for color indices (should be 0, 1, 2 or similar)
        unique_labels = np.unique(labels)
        assert len(unique_labels) >= 1, "Should have at least one unique color"
        assert len(unique_labels) <= 5, "Should not have too many unique colors"
        
        print(f"✓ Color labels extracted: shape {labels.shape}")
        print(f"✓ Unique color values: {unique_labels}")
        print(f"✓ Label dtype: {labels.dtype}")
    
    def test_velocity_x_label_extraction(self, test_states_data):
        """Test velocity_x label extraction from targets."""
        labels = extract_labels_from_state_data(test_states_data, "velocity_x")
        
        # Check basic properties
        assert isinstance(labels, np.ndarray), "Labels should be numpy array"
        expected_trials = test_states_data['hiddens'].shape[0]
        
        if len(labels.shape) == 1:
            assert len(labels) == expected_trials, f"Should have {expected_trials} labels"
        else:
            assert labels.shape[0] == expected_trials, f"Should have {expected_trials} trials"
        
        # Velocity labels should be continuous (floating point)
        assert np.issubdtype(labels.dtype, np.floating), "Velocity labels should be floating point"
        
        # Check that we have reasonable velocity values (not all zeros)
        assert not np.allclose(labels, 0), "Velocity labels should not all be zero"
        
        # Check value range is reasonable for velocity
        assert np.abs(labels).max() < 1000, "Velocity values should be reasonable"
        
        print(f"✓ Velocity X labels extracted: shape {labels.shape}")
        print(f"✓ Velocity X range: [{labels.min():.3f}, {labels.max():.3f}]")
        print(f"✓ Velocity X mean: {labels.mean():.3f}")
    
    def test_velocity_y_label_extraction(self, test_states_data):
        """Test velocity_y label extraction from targets."""
        labels = extract_labels_from_state_data(test_states_data, "velocity_y")
        
        # Check basic properties
        assert isinstance(labels, np.ndarray), "Labels should be numpy array"
        expected_trials = test_states_data['hiddens'].shape[0]
        
        if len(labels.shape) == 1:
            assert len(labels) == expected_trials, f"Should have {expected_trials} labels"
        else:
            assert labels.shape[0] == expected_trials, f"Should have {expected_trials} trials"
        
        # Velocity labels should be continuous (floating point)
        assert np.issubdtype(labels.dtype, np.floating), "Velocity labels should be floating point"
        
        # Check that we have reasonable velocity values
        assert not np.allclose(labels, 0), "Velocity labels should not all be zero"
        
        # Check value range is reasonable for velocity
        assert np.abs(labels).max() < 1000, "Velocity values should be reasonable"
        
        print(f"✓ Velocity Y labels extracted: shape {labels.shape}")
        print(f"✓ Velocity Y range: [{labels.min():.3f}, {labels.max():.3f}]")
        print(f"✓ Velocity Y mean: {labels.mean():.3f}")
    
    def test_hazard_label_extraction(self, test_states_data):
        """Test hazard label extraction (binary classification)."""
        try:
            labels = extract_labels_from_state_data(test_states_data, "hazard")
            
            # Check basic properties
            assert isinstance(labels, np.ndarray), "Labels should be numpy array"
            expected_trials = test_states_data['hiddens'].shape[0]
            
            if len(labels.shape) == 1:
                assert len(labels) == expected_trials, f"Should have {expected_trials} labels"
            else:
                assert labels.shape[0] == expected_trials, f"Should have {expected_trials} trials"
            
            # Hazard labels should be binary (0/1 or similar)
            unique_labels = np.unique(labels)
            assert len(unique_labels) <= 3, "Should have at most 3 unique hazard values"
            
            # Check that labels are reasonable for binary classification
            if len(unique_labels) == 2:
                assert set(unique_labels) <= {0, 1, 0.0, 1.0}, f"Binary labels should be 0/1, got {unique_labels}"
            
            print(f"✓ Hazard labels extracted: shape {labels.shape}")
            print(f"✓ Unique hazard values: {unique_labels}")
            
        except Exception as e:
            # Hazard labels might not be available in test data
            if "metadata" in str(e).lower() or "not found" in str(e).lower():
                print(f"⚠ Hazard labels not available in test data (expected): {e}")
                pytest.skip("Hazard labels not available in test data")
            else:
                raise
    
    def test_contingency_label_extraction(self, test_states_data):
        """Test contingency label extraction (binary classification)."""
        try:
            labels = extract_labels_from_state_data(test_states_data, "contingency")
            
            # Check basic properties
            assert isinstance(labels, np.ndarray), "Labels should be numpy array"
            expected_trials = test_states_data['hiddens'].shape[0]
            
            if len(labels.shape) == 1:
                assert len(labels) == expected_trials, f"Should have {expected_trials} labels"
            else:
                assert labels.shape[0] == expected_trials, f"Should have {expected_trials} trials"
            
            # Contingency labels should be binary or ternary (Low/Medium/High)
            unique_labels = np.unique(labels)
            assert len(unique_labels) <= 4, "Should have at most 4 unique contingency values"
            
            print(f"✓ Contingency labels extracted: shape {labels.shape}")
            print(f"✓ Unique contingency values: {unique_labels}")
            
        except Exception as e:
            # Contingency labels might not be available in test data
            if "metadata" in str(e).lower() or "not found" in str(e).lower():
                print(f"⚠ Contingency labels not available in test data (expected): {e}")
                pytest.skip("Contingency labels not available in test data")
            else:
                raise
    
    def test_all_decoder_label_extraction_compatibility(self, test_states_data, decoder_types):
        """Test that all decoder types can extract labels without crashing."""
        extraction_results = {}
        
        for decoder_type in decoder_types:
            try:
                labels = extract_labels_from_state_data(test_states_data, decoder_type)
                
                # Basic validation
                assert isinstance(labels, np.ndarray), f"Labels for {decoder_type} should be numpy array"
                assert labels.size > 0, f"Labels for {decoder_type} should not be empty"
                
                extraction_results[decoder_type] = {
                    "success": True,
                    "shape": labels.shape,
                    "dtype": labels.dtype,
                    "unique_count": len(np.unique(labels)),
                    "range": (labels.min(), labels.max()) if labels.size > 0 else (0, 0)
                }
                
                print(f"✓ {decoder_type}: shape {labels.shape}, dtype {labels.dtype}, "
                      f"unique values: {len(np.unique(labels))}")
                
            except Exception as e:
                extraction_results[decoder_type] = {
                    "success": False,
                    "error": str(e)
                }
                
                print(f"⚠ {decoder_type}: Failed with error: {e}")
        
        # At least some decoders should succeed
        successful_decoders = [dt for dt, result in extraction_results.items() if result["success"]]
        assert len(successful_decoders) >= 2, f"At least 2 decoders should succeed, got: {successful_decoders}"
        
        # Color and velocity decoders should definitely work since they use predictions/targets
        assert "color" in successful_decoders, "Color decoder should work with predictions"
        assert "velocity_x" in successful_decoders or "velocity_y" in successful_decoders, \
            "At least one velocity decoder should work with targets"
        
        print(f"✓ Label extraction compatibility: {len(successful_decoders)}/5 decoders successful")
        
        return extraction_results
    
    def test_label_extraction_fallback_logic(self, test_states_data):
        """Test that fallback logic works when metadata is missing."""
        
        # Test color extraction (should use predictions, not metadata)
        color_labels = extract_labels_from_state_data(test_states_data, "color")
        assert color_labels is not None, "Color extraction should work without metadata"
        
        # Test velocity extraction (should use targets, not metadata)
        velocity_x_labels = extract_labels_from_state_data(test_states_data, "velocity_x")
        assert velocity_x_labels is not None, "Velocity X extraction should work without metadata"
        
        velocity_y_labels = extract_labels_from_state_data(test_states_data, "velocity_y")
        assert velocity_y_labels is not None, "Velocity Y extraction should work without metadata"
        
        print("✓ Fallback logic works for decoders that don't require metadata")
    
    def test_label_shapes_match_states(self, test_states_data, decoder_types):
        """Test that extracted labels have compatible shapes with neural states."""
        hiddens = test_states_data['hiddens']
        n_trials, n_timesteps, n_units = hiddens.shape
        
        for decoder_type in decoder_types:
            try:
                labels = extract_labels_from_state_data(test_states_data, decoder_type)
                
                # Labels should match either trials or trials×timesteps
                if len(labels.shape) == 1:
                    # Trial-level labels
                    assert len(labels) == n_trials, \
                        f"{decoder_type} labels should have {n_trials} trials"
                elif len(labels.shape) == 2:
                    # Timestep-level labels
                    assert labels.shape[0] == n_trials, \
                        f"{decoder_type} labels should have {n_trials} trials"
                    assert labels.shape[1] == n_timesteps, \
                        f"{decoder_type} labels should have {n_timesteps} timesteps"
                else:
                    raise AssertionError(f"{decoder_type} labels have unexpected shape: {labels.shape}")
                
                print(f"✓ {decoder_type}: labels shape {labels.shape} compatible with states {hiddens.shape}")
                
            except Exception as e:
                if "metadata" in str(e).lower():
                    print(f"⚠ {decoder_type}: Skipped due to missing metadata")
                else:
                    raise


class TestLabelExtractionEdgeCases:
    """Test edge cases in label extraction."""
    
    def test_invalid_decoder_type(self, test_states_data):
        """Test that invalid decoder type raises appropriate error."""
        with pytest.raises((ValueError, KeyError, AttributeError)):
            extract_labels_from_state_data(test_states_data, "invalid_decoder_type")
    
    def test_empty_state_data(self):
        """Test label extraction with empty state data."""
        empty_data = {}
        
        for decoder_type in ["color", "velocity_x", "velocity_y", "hazard", "contingency"]:
            with pytest.raises((KeyError, ValueError, AttributeError)):
                extract_labels_from_state_data(empty_data, decoder_type)
    
    def test_malformed_state_data(self):
        """Test label extraction with malformed state data."""
        malformed_data = {
            "hiddens": np.array([]),  # Empty array
            "predictions": "not_an_array",  # Wrong type
            "targets": np.array([[1, 2]])  # Wrong shape
        }
        
        for decoder_type in ["color", "velocity_x"]:
            with pytest.raises((ValueError, AttributeError, IndexError)):
                extract_labels_from_state_data(malformed_data, decoder_type)