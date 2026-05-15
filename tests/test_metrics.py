"""
Tests for metric computation (CWC, sensitivity).
"""

import pytest
import numpy as np
import pandas as pd


class TestCWCComputation:
    """Test confidence weighted choice computation."""
    
    def test_cwc_calculation(self):
        """Test basic CWC calculation."""
        # Create sample data
        choices = np.array([1, -1, 1, -1, 1])  # 1 = change, -1 = no change
        confidence = np.array([0.8, 0.6, 0.9, 0.7, 0.5])
        
        # Calculate CWC
        cwc = choices * confidence
        
        expected = np.array([0.8, -0.6, 0.9, -0.7, 0.5])
        assert np.allclose(cwc, expected)
    
    def test_confidence_normalization(self):
        """Test confidence score normalization."""
        # Raw confidence scores (0-100 scale)
        raw_confidence = np.array([80, 60, 90, 70, 50])
        
        # Normalize to 0-1
        min_conf = 0
        max_conf = 100
        normalized = (raw_confidence - min_conf) / (max_conf - min_conf)
        
        assert np.all(normalized >= 0)
        assert np.all(normalized <= 1)
        assert normalized[0] == 0.8
        assert normalized[4] == 0.5
    
    def test_choice_binarization(self):
        """Test conversion of color choices to binary."""
        # Color choices: 0=entered, 1=next, 2=third
        color_entered = np.array([0, 0, 0, 0, 0])
        color_chosen = np.array([0, 1, 2, 0, 1])
        
        # Binarize: -1 for no change, 1 for change
        binary_choice = np.where(color_chosen == color_entered, -1, 1)
        
        expected = np.array([-1, 1, 1, -1, 1])
        assert np.array_equal(binary_choice, expected)
    
    def test_cwc_bounds(self):
        """Test that CWC values are properly bounded."""
        # Generate random data
        n_trials = 100
        choices = np.random.choice([-1, 1], size=n_trials)
        confidence = np.random.rand(n_trials)
        
        cwc = choices * confidence
        
        # CWC should be bounded by [-1, 1]
        assert np.all(cwc >= -1)
        assert np.all(cwc <= 1)
        
        # Check specific cases
        assert np.all(cwc[choices == 1] >= 0)
        assert np.all(cwc[choices == -1] <= 0)


class TestSensitivityMetrics:
    """Test sensitivity metric calculations."""
    
    def test_hazard_rate_sensitivity(self):
        """Test calculation of hazard rate sensitivity."""
        # Create sample data for straight path trials
        # 3 grayzone positions, 2 hazard conditions
        grayzone_times = np.array([1, 11, 22, 1, 11, 22])
        hazard_condition = np.array(['Low', 'Low', 'Low', 'High', 'High', 'High'])
        cwc_values = np.array([-0.8, -0.4, 0.0, -0.6, 0.0, 0.6])
        
        # Fit lines for each condition
        from scipy.stats import linregress
        
        # Low hazard
        low_mask = hazard_condition == 'Low'
        slope_low, _, _, _, _ = linregress(grayzone_times[low_mask], cwc_values[low_mask])
        
        # High hazard
        high_mask = hazard_condition == 'High'
        slope_high, _, _, _, _ = linregress(grayzone_times[high_mask], cwc_values[high_mask])
        
        # Sensitivity is difference in slopes
        s_hz = slope_high - slope_low
        
        # High hazard should have more positive slope
        assert slope_high > slope_low
        assert s_hz > 0
    
    def test_contingency_sensitivity(self):
        """Test calculation of contingency sensitivity."""
        # Create sample data for bounce trials
        contingency_numeric = np.array([0, 0.5, 1.0, 0, 0.5, 1.0])  # Low, Med, High
        cwc_values = np.array([-0.8, 0.0, 0.8, -0.6, 0.1, 0.7])
        
        # Fit line
        from scipy.stats import linregress
        slope, _, _, _, _ = linregress(contingency_numeric, cwc_values)
        
        # Sensitivity is the slope
        s_cont = slope
        
        # Should be positive (higher contingency → more change prediction)
        assert s_cont > 0
    
    def test_sensitivity_robustness(self):
        """Test robustness of sensitivity calculations."""
        # Test with perfect data
        times = np.array([1, 11, 22])
        cwc_low = np.array([-0.9, -0.5, -0.1])
        cwc_high = np.array([-0.7, -0.1, 0.5])
        
        from scipy.stats import linregress
        slope_low, _, r_low, _, _ = linregress(times, cwc_low)
        slope_high, _, r_high, _, _ = linregress(times, cwc_high)
        
        # Perfect linear relationships
        assert abs(r_low) > 0.99
        assert abs(r_high) > 0.99
        
        # Test with noisy data
        noise = np.random.randn(3) * 0.1
        cwc_noisy = cwc_high + noise
        slope_noisy, _, r_noisy, _, _ = linregress(times, cwc_noisy)
        
        # Should still capture trend but with lower R
        assert abs(r_noisy) < abs(r_high)
        assert abs(slope_noisy - slope_high) < 0.1  # Similar slope
    
    def test_participant_aggregation(self):
        """Test aggregation of metrics across participants."""
        # Simulate metrics for multiple participants
        n_participants = 10
        
        participant_metrics = {
            f'P{i:03d}': {
                's_hz': np.random.randn() * 0.1 + 0.3,
                's_cont': np.random.randn() * 0.2 + 0.8,
                'accuracy': np.random.rand() * 0.3 + 0.6
            }
            for i in range(n_participants)
        }
        
        # Convert to dataframe
        df_metrics = pd.DataFrame.from_dict(participant_metrics, orient='index')
        
        # Compute aggregate statistics
        mean_metrics = df_metrics.mean()
        std_metrics = df_metrics.std()
        
        # Check reasonable values
        assert 0.2 < mean_metrics['s_hz'] < 0.4
        assert 0.6 < mean_metrics['s_cont'] < 1.0
        assert 0.6 < mean_metrics['accuracy'] < 0.9
        
        # Check all participants included
        assert len(df_metrics) == n_participants


class TestMetricValidation:
    """Test validation of computed metrics."""
    
    def test_metric_ranges(self):
        """Test that metrics fall within expected ranges."""
        # Sensitivity metrics typically bounded
        s_hz = 0.35
        s_cont = 1.2
        
        # Reasonable ranges based on dissertation
        assert -1 < s_hz < 1
        assert -2 < s_cont < 2
    
    def test_accuracy_calculation(self):
        """Test accuracy calculation."""
        predictions = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        ground_truth = np.array([0, 1, 2, 0, 2, 2, 1, 1, 2, 0])
        
        correct = predictions == ground_truth
        accuracy = np.mean(correct)
        
        assert accuracy == 0.7
        assert 0 <= accuracy <= 1
    
    def test_missing_data_handling(self):
        """Test handling of missing data in metrics."""
        # Some participants might have missing trials
        cwc_with_nan = np.array([0.5, 0.6, np.nan, 0.7, 0.8])
        
        # Should handle NaN appropriately
        mean_cwc = np.nanmean(cwc_with_nan)
        assert np.isclose(mean_cwc, 0.65)
        
        # Count valid trials
        valid_trials = np.sum(~np.isnan(cwc_with_nan))
        assert valid_trials == 4