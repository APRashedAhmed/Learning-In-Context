"""Tests for padding handling in state normalization."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from learning_in_context.models.normalize_states import normalize_states


class TestNormalizeStatesPadding:
    """Test padding handling in state normalization."""
    
    def test_padding_excluded_from_stats(self):
        """Test that padded values are excluded from mean/std calculation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create test data with known patterns
            n_trials = 3
            max_timesteps = 10
            n_units = 2
            
            # Create states with clear patterns
            # Unit 0: all values are 1.0 (except padding)
            # Unit 1: all values are 2.0 (except padding)
            hiddens = np.ones((n_trials, max_timesteps, n_units))
            hiddens[:, :, 1] = 2.0
            cells = hiddens.copy()
            
            # Set different trial lengths
            trial_lengths = np.array([5, 7, 10])  # Different lengths
            
            # Add padding with extreme values that would affect stats if included
            for i, length in enumerate(trial_lengths):
                hiddens[i, length:, :] = 1000.0  # Extreme values
                cells[i, length:, :] = 1000.0
            
            # Create df_data with trial lengths
            df_data = pd.DataFrame({
                'length': trial_lengths,
                'trial_id': range(n_trials)
            })
            
            # Save test data
            input_file = tmpdir / 'test_states.npz'
            output_file = tmpdir / 'test_states_normalized.npz'
            
            np.savez_compressed(
                input_file,
                hiddens=hiddens,
                cells=cells,
                predictions=np.zeros((n_trials, max_timesteps, 3)),
                model_id='TEST-001',
                df_data=df_data.to_dict()
            )
            
            # Normalize with padding handling
            normalize_states(
                input_file,
                output_file,
                method='zscore',
                per_unit=True,
                padding_value=-100.0
            )
            
            # Load results
            results = np.load(output_file, allow_pickle=True)
            norm_hiddens = results['hiddens']
            norm_cells = results['cells']
            
            # Check that valid timesteps are normalized correctly
            # Since all valid values for each unit are the same, they should normalize to 0
            for i, length in enumerate(trial_lengths):
                # Check valid timesteps (should be ~0 after normalization)
                assert np.allclose(norm_hiddens[i, :length, :], 0.0, atol=1e-6), \
                    f"Valid timesteps should normalize to 0 for trial {i}"
                assert np.allclose(norm_cells[i, :length, :], 0.0, atol=1e-6)
                
                # Check padded timesteps (should be -100)
                if length < max_timesteps:
                    assert np.allclose(norm_hiddens[i, length:, :], -100.0), \
                        f"Padded timesteps should be -100 for trial {i}"
                    assert np.allclose(norm_cells[i, length:, :], -100.0)
    
    def test_padding_value_configurable(self):
        """Test that padding_value parameter works correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create simple test data
            hiddens = np.ones((2, 5, 2))
            cells = np.ones((2, 5, 2))
            
            # Set padding
            trial_lengths = np.array([3, 4])
            for i, length in enumerate(trial_lengths):
                hiddens[i, length:, :] = -1.0  # Original padding
                cells[i, length:, :] = -1.0
            
            df_data = pd.DataFrame({'length': trial_lengths})
            
            # Test with custom padding value
            custom_padding = -999.0
            
            input_file = tmpdir / 'test_states.npz'
            output_file = tmpdir / 'test_states_normalized.npz'
            
            np.savez_compressed(
                input_file,
                hiddens=hiddens,
                cells=cells,
                predictions=np.zeros((2, 5, 3)),
                model_id='TEST-001',
                df_data=df_data.to_dict()
            )
            
            normalize_states(
                input_file,
                output_file,
                padding_value=custom_padding
            )
            
            # Check results
            results = np.load(output_file, allow_pickle=True)
            norm_hiddens = results['hiddens']
            
            # Check padded values use custom padding
            assert np.allclose(norm_hiddens[0, 3:, :], custom_padding)
            assert np.allclose(norm_hiddens[1, 4:, :], custom_padding)
    
    def test_no_df_data_fallback(self):
        """Test that normalization works without df_data (no padding handling)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create test data without df_data
            hiddens = np.random.randn(2, 5, 3)
            cells = np.random.randn(2, 5, 3)
            
            input_file = tmpdir / 'test_states.npz'
            output_file = tmpdir / 'test_states_normalized.npz'
            
            np.savez_compressed(
                input_file,
                hiddens=hiddens,
                cells=cells,
                predictions=np.zeros((2, 5, 3)),
                model_id='TEST-001'
            )
            
            # Should work without error
            normalize_states(input_file, output_file)
            
            # Check that normalization happened
            results = np.load(output_file, allow_pickle=True)
            norm_info = results['normalization_info'].item()
            
            # Should indicate no padding was handled
            assert not norm_info['padding_handled']
            
            # Check that values are normalized (not all equal to padding_value)
            norm_hiddens = results['hiddens']
            assert not np.allclose(norm_hiddens, -100.0)