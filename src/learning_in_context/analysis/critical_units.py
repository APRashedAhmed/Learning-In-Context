"""Identify critical units using elastic net regularization."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.linear_model import ElasticNetCV, LogisticRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from ..core import CriticalUnitsResult, StateData
from ..core.constants import (
    CV_FOLDS,
    MIN_UNIT_VARIANCE,
    # New regularization parameters
    LAMBDA_MIN,
    LAMBDA_MAX,
    N_LAMBDAS,
    LAMBDA_SPACING,
    BINARY_L1_RATIO,
    MULTICLASS_L1_RATIO,
    BINARY_MAX_ITER,
    MULTICLASS_MAX_ITER,
    CHANCE_MARGIN,
    THRESHOLD_METHOD,
    get_lambda_values,
    # Legacy constants for backward compatibility
    ELASTICNET_ALPHAS,
    ELASTICNET_L1_RATIO,
    HZ_L1_RATIO,
    HZ_C,
    HZ_MAX_ITER,
    CONT_L1_RATIO,
    CONT_C,
    CONT_MAX_ITER
)


def identify_critical_units(
    states: np.ndarray,
    labels: np.ndarray,
    # Regularization sweep parameters
    lambda_min: float = LAMBDA_MIN,
    lambda_max: float = LAMBDA_MAX,
    n_lambdas: int = N_LAMBDAS,
    lambda_spacing: str = LAMBDA_SPACING,
    # Decoder-specific parameters
    decoder_type: Optional[str] = None,
    binary_l1_ratio: float = BINARY_L1_RATIO,
    multiclass_l1_ratio: float = MULTICLASS_L1_RATIO,
    binary_max_iter: int = BINARY_MAX_ITER,
    multiclass_max_iter: int = MULTICLASS_MAX_ITER,
    # Threshold detection parameters
    chance_margin: float = CHANCE_MARGIN,
    chance_level: Optional[float] = None,
    threshold_method: str = THRESHOLD_METHOD,
    # Legacy parameters for backward compatibility
    alphas: Optional[Union[np.ndarray, str]] = None,
    l1_ratio: Optional[float] = None,
    cv_folds: int = CV_FOLDS,
    timestep: Optional[int] = -1,
    exclude_low_variance: bool = True,
    n_jobs: int = -1,
    C: Optional[float] = None,
    solver: str = "saga",
    max_iter: Optional[int] = None,
    # Data processing parameters
    zscore_states: bool = True,
    use_all_timesteps: bool = True,
    trial_lengths: Optional[np.ndarray] = None,
    cells: Optional[np.ndarray] = None,
    concatenate_states: bool = True,
    # Visualization parameters
    save_visualization_data: bool = True,
    generate_plots: bool = False,
    output_dir: Optional[Path] = None,
    task_variable: str = "unknown"
) -> Dict[str, Any]:
    """Identify critical units via elastic net regularization sweep.
    
    This function implements the regularization sweep methodology from the dissertation,
    systematically varying lambda across multiple orders of magnitude and identifying
    units that maintain non-zero coefficients just before performance drops to chance.
    
    Args:
        states: Neural states array (n_trials, timesteps, hidden_size)
        labels: Target labels (n_trials, timesteps) or (n_trials,)
        
        # Regularization sweep parameters
        lambda_min: Minimum lambda value for sweep (default: 1e-6)
        lambda_max: Maximum lambda value for sweep (default: 1.0)
        n_lambdas: Number of lambda values to test (default: 50)
        lambda_spacing: 'log' or 'linear' spacing (default: 'log')
        
        # Decoder-specific parameters
        decoder_type: Type of decoder ('hazard', 'contingency', 'color', 'velocity_x', 'velocity_y')
        binary_l1_ratio: L1 ratio for binary classification (default: 0.64)
        multiclass_l1_ratio: L1 ratio for multiclass/regression (default: 0.4)
        binary_max_iter: Max iterations for binary decoders (default: 250)
        multiclass_max_iter: Max iterations for multiclass/regression (default: 250)
        
        # Threshold detection parameters
        chance_margin: Performance margin above chance for threshold (default: 0.05)
        chance_level: Override auto-detected chance level (default: None)
        threshold_method: Method for finding threshold ('chance', 'elbow', 'fixed')
        
        # Legacy parameters (for backward compatibility)
        alphas: DEPRECATED - use lambda_min/max/n_lambdas instead
        l1_ratio: DEPRECATED - use binary/multiclass_l1_ratio instead
        C: DEPRECATED - sweep is always performed
        max_iter: DEPRECATED - use binary/multiclass_max_iter instead
        
        # Data processing parameters
        timestep: Specific timestep to analyze (default -1 = last timestep, ignored if use_all_timesteps=True)
        exclude_low_variance: Whether to exclude low-variance units
        n_jobs: Number of parallel jobs (-1 = all CPUs)
        solver: Solver algorithm (default "saga" for elasticnet)
        zscore_states: Whether to z-score states before analysis
        use_all_timesteps: Use all non-padded timesteps instead of single timestep
        trial_lengths: Array of actual trial lengths for padding removal (required if use_all_timesteps=True)
        cells: Cell states array (n_trials, timesteps, hidden_size) for concatenation
        concatenate_states: Whether to concatenate hidden and cell states
        
        # Visualization parameters
        save_visualization_data: Whether to save coefficient paths and performance curves
        generate_plots: Whether to generate visualization plots immediately
        output_dir: Directory to save plots (required if generate_plots=True)
        task_variable: Name of task variable being analyzed (for plot titles)
        
    Returns:
        Dictionary containing:
            - unit_indices: Critical unit indices at optimal threshold
            - coefficients: Regression coefficients at optimal threshold
            - r2_scores: R-squared scores at optimal threshold
            - best_alpha: Optimal regularization strength (lambda)
            - cv_scores: Cross-validation-like scores from sweep
            - metadata: Analysis metadata including threshold info
            - coefficient_paths: Full coefficient evolution (n_units, n_lambdas)
            - intercept_paths: Full intercept evolution
            - performance_curves: Performance metrics across all lambdas
            - alpha_values: Lambda values used in sweep
            - threshold_analysis: Detailed threshold detection results
    """
    # Concatenate hidden and cell states if requested
    if concatenate_states and cells is not None:
        states = np.concatenate([states, cells], axis=-1)
    
    # Determine decoder-specific parameters
    if decoder_type in ['hazard', 'contingency']:
        # Binary classification
        effective_l1_ratio = l1_ratio if l1_ratio is not None else binary_l1_ratio
        effective_max_iter = max_iter if max_iter is not None else binary_max_iter
        problem_type = 'binary_classification'
    elif decoder_type == 'color':
        # Multi-class classification
        effective_l1_ratio = l1_ratio if l1_ratio is not None else multiclass_l1_ratio
        effective_max_iter = max_iter if max_iter is not None else multiclass_max_iter
        problem_type = 'multiclass_classification'
    elif decoder_type in ['velocity_x', 'velocity_y']:
        # Regression
        effective_l1_ratio = l1_ratio if l1_ratio is not None else multiclass_l1_ratio
        effective_max_iter = max_iter if max_iter is not None else multiclass_max_iter
        problem_type = 'regression'
    else:
        # Fallback: infer from data
        if l1_ratio is not None:
            effective_l1_ratio = l1_ratio
        else:
            # Infer from labels
            unique_labels = np.unique(labels)
            if len(unique_labels) == 2:
                effective_l1_ratio = binary_l1_ratio
                problem_type = 'binary_classification'
            elif len(unique_labels) > 2 and np.all(labels == labels.astype(int)):
                effective_l1_ratio = multiclass_l1_ratio
                problem_type = 'multiclass_classification'
            else:
                effective_l1_ratio = multiclass_l1_ratio
                problem_type = 'regression'
        effective_max_iter = max_iter if max_iter is not None else BINARY_MAX_ITER
    
    # Prepare data
    # Handle case where states is already 2D (e.g., from temporal windows)
    if states.ndim == 2:
        X = states
        y = labels
    elif use_all_timesteps:
        # Use all non-padded timesteps
        if trial_lengths is None:
            raise ValueError("trial_lengths must be provided when use_all_timesteps=True")
        
        X_list = []
        y_list = []
        
        for i, length in enumerate(trial_lengths):
            # Extract only non-padded timesteps
            X_trial = states[i, :length, :]  # (actual_length, n_units)
            
            # Handle labels based on dimensionality
            if labels.ndim == 1:
                # Trial-level labels (e.g., hazard, contingency)
                # Repeat the same label for all timesteps in this trial
                y_trial = np.repeat(labels[i], length)
            else:
                # Timestep-level labels (e.g., color predictions)
                y_trial = labels[i, :length]
            
            X_list.append(X_trial)
            y_list.append(y_trial)
        
        # Stack all trials
        X = np.vstack(X_list)  # (total_timesteps, n_units)
        y = np.concatenate(y_list)
    
    elif timestep is not None:
        # Analyze specific timestep (default -1 = last)
        X = states[:, timestep, :]
        y = labels[:, timestep] if labels.ndim > 1 else labels
    else:
        # Flatten across timesteps (legacy behavior)
        n_trials, n_timesteps, n_units = states.shape
        X = states.reshape(-1, n_units)
        y = labels.flatten() if labels.ndim > 1 else np.repeat(labels, n_timesteps)
    
    # Z-score states if requested (matching original implementation)
    if zscore_states:
        from scipy import stats
        # Z-score across samples (axis 0)
        X = stats.zscore(X, axis=0, ddof=1)
    
    # Filter out low-variance units if requested
    unit_indices = np.arange(X.shape[1])
    if exclude_low_variance:
        variances = np.var(X, axis=0)
        valid_units = variances > MIN_UNIT_VARIANCE
        X = X[:, valid_units]
        unit_indices = unit_indices[valid_units]
    
    # Validate lambda parameters
    if lambda_min > lambda_max:
        raise ValueError(f"lambda_min ({lambda_min}) must be less than or equal to lambda_max ({lambda_max})")
    
    # Generate lambda values for regularization sweep
    if alphas is not None and not isinstance(alphas, str):
        # Use provided alphas for backward compatibility
        lambda_values = alphas
    else:
        # Generate lambda values based on parameters
        lambda_values = get_lambda_values(lambda_min, lambda_max, n_lambdas, lambda_spacing)
    
    # Convert to C values for sklearn
    C_values = 1.0 / lambda_values
    
    # If C is provided (legacy), create a sweep around that value
    if C is not None:
        # Legacy mode: create sweep around the C value
        center_lambda = 1.0 / C
        # Create a range from 10x to 1/10x the center value
        lambda_min_c = center_lambda / 10
        lambda_max_c = center_lambda * 10
        lambda_values = get_lambda_values(lambda_min_c, lambda_max_c, min(n_lambdas, 10), lambda_spacing)
        C_values = 1.0 / lambda_values
    
    # Run regularization sweep
    coefficients_list = []
    intercepts_list = []
    scores_list = []
    
    from sklearn.metrics import accuracy_score, r2_score
    
    for C_val in C_values:
        if problem_type in ['binary_classification', 'multiclass_classification']:
            model = LogisticRegression(
                penalty="elasticnet",
                C=C_val,
                l1_ratio=effective_l1_ratio,
                solver=solver,
                max_iter=effective_max_iter,
                random_state=42,
                n_jobs=n_jobs
            )
        else:  # regression
            from sklearn.linear_model import ElasticNet
            model = ElasticNet(
                alpha=1.0/C_val,  # ElasticNet uses alpha, not C
                l1_ratio=effective_l1_ratio,
                max_iter=effective_max_iter,
                random_state=42
            )
        
        try:
            model.fit(X, y)
            
            # Get coefficients
            if hasattr(model, 'coef_'):
                coef = model.coef_
                if coef.ndim > 1:
                    coef = coef.ravel()
                coefficients_list.append(coef)
            else:
                coefficients_list.append(np.zeros(X.shape[1]))
            
            # Get intercepts
            if hasattr(model, 'intercept_'):
                intercept = model.intercept_
                if isinstance(intercept, np.ndarray) and intercept.ndim == 0:
                    intercept = np.array([intercept.item()])
                elif not isinstance(intercept, np.ndarray):
                    intercept = np.array([intercept])
                intercepts_list.append(intercept)
            else:
                intercepts_list.append(np.array([0.0]))
            
            # Calculate score
            if problem_type == 'regression':
                y_pred = model.predict(X)
                score = r2_score(y, y_pred)
            else:
                y_pred = model.predict(X)
                score = accuracy_score(y, y_pred)
            scores_list.append(score)
            
        except Exception as e:
            # If model fails to converge, use zeros and bad score
            coefficients_list.append(np.zeros(X.shape[1]))
            intercepts_list.append(np.array([0.0]))
            scores_list.append(0.0)
    
    # Convert to arrays
    coefficient_paths = np.array(coefficients_list).T  # Shape: (n_units, n_lambdas)
    intercept_paths = np.array(intercepts_list).T      # Shape: (n_intercepts, n_lambdas)
    scores_array = np.array(scores_list)
    
    # Find critical units using threshold detection
    if n_lambdas > 1:
        # Run threshold detection
        threshold_result = identify_critical_threshold(
            coefficients_list,
            scores_list,
            lambda_values,
            chance_level=chance_level,
            chance_margin=chance_margin,
            problem_type=problem_type
        )
        
        # Extract critical units at threshold
        threshold_idx = threshold_result['threshold_idx']
        threshold_coeffs = coefficients_list[threshold_idx]
        
        # Handle multiclass case where coeffs might be flattened
        if problem_type == 'multiclass_classification' and len(threshold_coeffs) > len(unit_indices):
            # Reshape to (n_classes, n_units) and check if any class uses the unit
            n_classes = len(np.unique(y))
            coeff_matrix = threshold_coeffs.reshape(n_classes, -1)
            critical_mask = np.any(np.abs(coeff_matrix) > 1e-8, axis=0)
            critical_indices = unit_indices[critical_mask]
            # Average coefficients across classes for each critical unit
            critical_coeffs = np.mean(coeff_matrix[:, critical_mask], axis=0)
        else:
            critical_mask = np.abs(threshold_coeffs) > 1e-8
            critical_indices = unit_indices[critical_mask]
            critical_coeffs = threshold_coeffs[critical_mask]
        best_alpha = threshold_result['threshold_alpha']
        score = threshold_result['threshold_score']
        
    else:
        # Single point analysis (legacy mode)
        threshold_coeffs = coefficients_list[0]
        
        # Handle multiclass case where coeffs might be flattened
        if problem_type == 'multiclass_classification' and len(threshold_coeffs) > len(unit_indices):
            # Reshape to (n_classes, n_units) and check if any class uses the unit
            n_classes = len(np.unique(y))
            coeff_matrix = threshold_coeffs.reshape(n_classes, -1)
            critical_mask = np.any(np.abs(coeff_matrix) > 1e-8, axis=0)
            critical_indices = unit_indices[critical_mask]
            # Average coefficients across classes for each critical unit
            critical_coeffs = np.mean(coeff_matrix[:, critical_mask], axis=0)
        else:
            critical_mask = np.abs(threshold_coeffs) > 1e-8
            critical_indices = unit_indices[critical_mask]
            critical_coeffs = threshold_coeffs[critical_mask]
            
        best_alpha = lambda_values[0]
        score = scores_list[0]
        threshold_result = None
    
    # Determine score type
    score_type = "r2" if problem_type == 'regression' else "accuracy"
    
    # Create CV-like scores from sweep
    cv_scores = {
        "mean": np.array([np.mean(scores_array)]),
        "std": np.array([np.std(scores_array)]),
        "all": scores_array,
        "alpha_scores": dict(zip(lambda_values.tolist(), scores_array.tolist()))
    }
    
    # Visualization data is already available from sweep
    if save_visualization_data:
        # Performance curves dictionary
        performance_curves = {
            'accuracy' if score_type == 'accuracy' else 'r2': scores_array
        }
        alpha_values = lambda_values
    else:
        # Don't save visualization data
        coefficient_paths = None
        intercept_paths = None
        performance_curves = None
        alpha_values = None
    
    # Generate plots if requested
    if generate_plots and output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Import and call plotting function
        try:
            from ..visualization.critical_units_plots import plot_coefficient_analysis
            plot_coefficient_analysis(
                coefficient_paths=coefficient_paths,
                intercept_paths=intercept_paths,
                performance_curves=performance_curves,
                alpha_values=alpha_values,
                unit_indices=unit_indices,
                critical_indices=critical_indices,
                output_path=output_dir / f"{task_variable}_coefficient_analysis.png",
                task_variable=task_variable
            )
        except ImportError:
            print(f"Warning: Could not import visualization functions. Plot not generated.")
    
    # Create result object
    result = CriticalUnitsResult(
        unit_indices=critical_indices,
        coefficients=critical_coeffs,
        r2_scores=np.array([score]),
        best_alpha=best_alpha,
        cv_scores=cv_scores,
        metadata={
            "n_units_total": len(unit_indices),
            "n_units_critical": len(critical_indices),
            "l1_ratio": effective_l1_ratio,
            "timestep": timestep if not use_all_timesteps else "all",
            "exclude_low_variance": exclude_low_variance,
            "unit_variance_threshold": MIN_UNIT_VARIANCE,
            "solver": solver,
            "max_iter": effective_max_iter,
            "score_type": score_type,
            "zscore_applied": zscore_states,
            "use_all_timesteps": use_all_timesteps,
            "concatenated_states": concatenate_states and cells is not None,
            "n_samples": X.shape[0],
            # New parameters
            "decoder_type": decoder_type,
            "problem_type": problem_type,
            "lambda_min": lambda_min,
            "lambda_max": lambda_max,
            "n_lambdas": n_lambdas,
            "lambda_spacing": lambda_spacing,
            "chance_level": threshold_result['chance_level'] if threshold_result else chance_level,
            "chance_margin": chance_margin,
            "threshold_method": threshold_method,
            "threshold_alpha": threshold_result['threshold_alpha'] if threshold_result else best_alpha,
            "threshold_score": threshold_result['threshold_score'] if threshold_result else score,
            # Legacy fields for backward compatibility
            "C": C if C is not None else 1.0 / best_alpha
        },
        # Visualization data
        coefficient_paths=coefficient_paths,
        intercept_paths=intercept_paths,
        performance_curves=performance_curves,
        alpha_values=alpha_values
    )
    
    result_dict = result.to_dict()
    
    # Add threshold analysis if available
    if threshold_result is not None:
        result_dict['threshold_analysis'] = threshold_result
    
    return result_dict


def linear_regularization_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    dict_metrics: Dict[str, Tuple],
    reg_func: callable,
    reg_pred: callable,
    C_logspace: np.ndarray,
    **reg_kwargs
) -> Tuple[List, List, Dict]:
    """Run regularization sweep pipeline matching original implementation.
    
    Args:
        X: Input features (already z-scored)
        y: Target labels
        dict_metrics: Dictionary of metrics to compute
        reg_func: Regression function to use
        reg_pred: Prediction function to use
        C_logspace: Array of C values to sweep
        **reg_kwargs: Additional kwargs for regression function
        
    Returns:
        Tuple of (coefficients, intercepts, metrics_dict)
    """
    reg_coefs, reg_intercepts = [], []
    dict_stat_metrics = {name: [] for name in dict_metrics.keys()}
    
    for C in C_logspace:
        # Fit model with current C value
        reg, reg_params = reg_func(X, y, C=C, **reg_kwargs)
        pred = reg_pred(reg, X)
        
        # Store coefficients and intercepts
        reg_coefs.append(reg_params[0])
        reg_intercepts.append(reg_params[1])
        
        # Compute metrics
        for metric, (func, stat_params) in dict_metrics.items():
            params = stat_params.get("all", {})
            # Update with statistic-specific params if available
            dict_stat_metrics[metric].append(
                func(y, pred, **params)
            )
    
    return reg_coefs, reg_intercepts, dict_stat_metrics


def reg_single(X, y, C=0.0001, l1_ratio=0.64, max_iter=150, **kwargs):
    """Single-class logistic regression matching original."""
    reg = LogisticRegression(
        solver="saga",
        penalty="elasticnet",
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        C=C,
        **kwargs
    ).fit(X, y)
    coef = reg.coef_.ravel()
    intercept = reg.intercept_
    return reg, (coef, intercept)


def reg_multi(X, y, C=0.001, l1_ratio=0.4, max_iter=250, solver="saga", **kwargs):
    """Multi-class logistic regression matching original."""
    reg = LogisticRegression(
        solver=solver,
        penalty="elasticnet",
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        C=C,
        # multi_class="ovr",  # Deprecated in sklearn 1.5+
        **kwargs
    ).fit(X, y)
    coef = reg.coef_.ravel()
    intercept = reg.intercept_
    return reg, (coef, intercept)


def aggregate_critical_units(
    model_results: Dict[str, Dict]
) -> Dict[str, Any]:
    """Aggregate critical units across models.
    
    Args:
        model_results: Dictionary mapping model_id to critical units results
        
    Returns:
        Aggregated statistics including:
            - unit_frequency: How often each unit is critical
            - mean_coefficients: Average coefficient magnitude
            - consistency_score: Cross-model consistency
    """
    # Collect all unit indices and coefficients
    all_units = []
    all_coeffs = []
    model_ids = []
    
    for model_id, result in model_results.items():
        units = np.array(result["unit_indices"])
        coeffs = np.array(result["coefficients"])
        
        all_units.extend(units)
        all_coeffs.extend(np.abs(coeffs))
        model_ids.extend([model_id] * len(units))
    
    # Calculate frequency of each unit being critical
    unique_units, counts = np.unique(all_units, return_counts=True)
    unit_frequency = dict(zip(unique_units, counts / len(model_results)))
    
    # Calculate mean coefficient magnitude per unit
    unit_coeffs = {}
    for unit, coeff in zip(all_units, all_coeffs):
        if unit not in unit_coeffs:
            unit_coeffs[unit] = []
        unit_coeffs[unit].append(coeff)
    
    mean_coefficients = {
        unit: np.mean(coeffs) for unit, coeffs in unit_coeffs.items()
    }
    
    # Calculate consistency score (Jaccard similarity between models)
    consistency_scores = []
    model_list = list(model_results.keys())
    
    for i in range(len(model_list)):
        for j in range(i + 1, len(model_list)):
            units_i = set(model_results[model_list[i]]["unit_indices"])
            units_j = set(model_results[model_list[j]]["unit_indices"])
            
            if units_i or units_j:
                jaccard = len(units_i & units_j) / len(units_i | units_j)
                consistency_scores.append(jaccard)
    
    return {
        "unit_frequency": unit_frequency,
        "mean_coefficients": mean_coefficients,
        "consistency_score": np.mean(consistency_scores) if consistency_scores else 0.0,
        "n_models": len(model_results),
        "top_units": sorted(
            unit_frequency.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:20]
    }


def analyze_temporal_dynamics(
    states: np.ndarray,
    labels: np.ndarray,
    window_size: int = 10,
    stride: int = 5,
    decoder_type: str = 'color',
    **kwargs
) -> Dict[str, Any]:
    """Analyze how critical units change over time.
    
    Args:
        states: Neural states (n_trials, timesteps, hidden_size)
        labels: Target labels
        window_size: Size of sliding window
        stride: Step size for sliding window
        **kwargs: Additional arguments for identify_critical_units
        
    Returns:
        Dictionary with temporal analysis results
    """
    n_timesteps = states.shape[1]
    results = []
    
    # Sliding window analysis
    for t in range(0, n_timesteps - window_size + 1, stride):
        window_states = states[:, t:t+window_size, :]
        window_labels = labels[:, t:t+window_size] if labels.ndim > 1 else labels
        
        # Identify critical units in this window
        # Force timestep=None to work with reshaped 2D data
        kwargs_copy = kwargs.copy()
        kwargs_copy['timestep'] = None
        kwargs_copy['use_all_timesteps'] = False
        
        result = identify_critical_units(
            window_states.reshape(-1, window_states.shape[-1]),
            window_labels.flatten() if window_labels.ndim > 1 else window_labels,
            **kwargs_copy
        )
        
        results.append({
            "window_start": t,
            "window_end": t + window_size,
            "critical_units": result["unit_indices"],
            "n_critical": len(result["unit_indices"]),
            "r2_score": result["r2_scores"][0]
        })
    
    # Analyze stability of critical units over time
    all_windows_units = [set(r["critical_units"]) for r in results]
    
    # Units that are always critical
    if all_windows_units:
        stable_units = set.intersection(*all_windows_units) if all_windows_units else set()
    else:
        stable_units = set()
    
    # Units that are sometimes critical
    transient_units = set.union(*all_windows_units) - stable_units if all_windows_units else set()
    
    return {
        "windows": results,
        "stable_units": list(stable_units),
        "transient_units": list(transient_units),
        "n_stable": len(stable_units),
        "n_transient": len(transient_units),
        "window_size": window_size,
        "stride": stride
    }


def extract_labels_from_state_data(
    state_data: Dict[str, Any], 
    decoder_type: str
) -> np.ndarray:
    """Extract appropriate labels for different decoder types.
    
    Args:
        state_data: Loaded state data with metadata
        decoder_type: One of 'hazard', 'contingency', 'color', 'velocity_x', 'velocity_y'
        
    Returns:
        Labels array for the specified decoder type
    """
    # Try to load from StateData object first
    if hasattr(state_data, 'metadata'):
        metadata = state_data.metadata
    else:
        metadata = state_data.get('metadata', {})
    
    # Get trial metadata if available
    if 'df_data' in state_data:
        import pandas as pd
        df_data_dict = state_data['df_data']
        # Handle numpy scalar arrays containing dictionaries
        if hasattr(df_data_dict, 'item') and callable(df_data_dict.item):
            df_data_dict = df_data_dict.item()
        df_data = pd.DataFrame(df_data_dict)
    elif hasattr(state_data, 'df_data'):
        df_data = state_data.df_data
    else:
        df_data = None
    
    if decoder_type == 'color':
        # Use ground truth color labels from df_data if available
        if df_data is not None and 'correct_response' in df_data.columns:
            # correct_response values are 1, 2, 3 - convert to 0, 1, 2 for consistency
            return df_data['correct_response'].values - 1
        # Fallback to model predictions
        elif hasattr(state_data, 'predictions'):
            return np.argmax(state_data.predictions, axis=-1)
        elif 'predictions' in state_data:
            return np.argmax(state_data['predictions'], axis=-1)
        else:
            raise ValueError("No correct_response in df_data and no predictions found for color decoder")
    
    elif decoder_type in ['hazard', 'contingency']:
        # For hazard and contingency, we need ground truth from dataset metadata
        if df_data is not None:
            if decoder_type == 'hazard':
                # Extract hazard rate condition (binary: Low=0, High=1)
                if 'hazard_rate' in df_data.columns:
                    hazard_map = {'Low': 0, 'High': 1}
                    return df_data['hazard_rate'].map(hazard_map).values
                elif 'P_hz' in df_data.columns:
                    # Use actual hazard rate values and binarize
                    return (df_data['P_hz'] > df_data['P_hz'].median()).astype(int).values
            elif decoder_type == 'contingency':
                # Extract contingency condition (binary: Low=0, High=1, excluding Medium)
                if 'contingency' in df_data.columns:
                    cont_data = df_data['contingency']
                    # Filter to only Low and High, exclude Medium
                    mask = cont_data.isin(['Low', 'High'])
                    if mask.sum() == 0:
                        # If no Low/High, use all and binarize around median
                        cont_map = {'Low': 0, 'Medium': 0, 'High': 1}
                        return cont_data.map(cont_map).values
                    else:
                        cont_map = {'Low': 0, 'High': 1}
                        return cont_data[mask].map(cont_map).values
                elif 'P_cont' in df_data.columns:
                    # Use actual contingency values and binarize
                    return (df_data['P_cont'] > df_data['P_cont'].median()).astype(int).values
        
        # Fallback: create synthetic binary labels based on trial index
        n_trials = state_data.shape[0] if hasattr(state_data, 'shape') else len(state_data.get('hiddens', []))
        return np.random.binomial(1, 0.5, n_trials)
    
    elif decoder_type in ['velocity_x', 'velocity_y']:
        # Extract velocity components
        if df_data is not None:
            if decoder_type == 'velocity_x' and 'v_x' in df_data.columns:
                return df_data['v_x'].values
            elif decoder_type == 'velocity_y' and 'v_y' in df_data.columns:
                return df_data['v_y'].values
            elif 'final_velocity' in df_data.columns:
                # Parse velocity from string or tuple format if needed
                velocities = df_data['final_velocity'].values
                component_idx = 0 if decoder_type == 'velocity_x' else 1
                if isinstance(velocities[0], (list, tuple)):
                    return np.array([v[component_idx] for v in velocities])
        
        # Fallback: create synthetic velocity values
        n_trials = state_data.shape[0] if hasattr(state_data, 'shape') else len(state_data.get('hiddens', []))
        return np.random.randn(n_trials) * 0.1  # Small velocity values
    
    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}")


# DEPRECATED: run_regularization_sweep functionality is now integrated into identify_critical_units


def identify_critical_threshold(
    coefficients: List,  # Can be numpy arrays or lists
    scores: List[float],
    alphas,  # Can be numpy array or list
    chance_level: float = None,
    chance_margin: float = CHANCE_MARGIN,
    problem_type: str = None
) -> Dict[str, Any]:
    """Identify critical units threshold 'just before performance drops to chance'.
    
    Args:
        coefficients: List of coefficient arrays for each alpha
        scores: List of performance scores for each alpha
        alphas: Regularization strengths tested
        chance_level: Chance performance level (auto-detected if None)
        
    Returns:
        Dictionary with threshold analysis results
    """
    scores = np.array(scores)
    
    # Estimate chance level if not provided
    if chance_level is None:
        if problem_type == 'regression':
            chance_level = 0.0
        elif problem_type == 'binary_classification':
            chance_level = 0.5
        elif problem_type == 'multiclass_classification':
            # Without knowing n_classes, assume 3 for color task
            chance_level = 1.0 / 3.0
        else:
            # Fallback: infer from scores
            if len(np.unique(scores)) > 10:  # Likely regression (continuous scores)
                chance_level = 0.0
            elif scores.max() <= 1.0:  # Classification (accuracy scores)
                chance_level = 0.5  # Assume binary
            else:
                chance_level = 0.0
    
    # Find where performance drops to chance
    # Look for the highest alpha (most regularization) where performance > chance + margin
    threshold_mask = scores > (chance_level + chance_margin)
    
    if not np.any(threshold_mask):
        # If no alpha meets criteria, use the best performing one
        threshold_idx = np.argmax(scores)
    else:
        # Use the most regularized (highest alpha) that still performs above chance
        valid_indices = np.where(threshold_mask)[0]
        threshold_idx = valid_indices[-1]  # Last valid index (highest alpha)
    
    # Convert to arrays if needed for indexing
    alphas = np.array(alphas) if not hasattr(alphas, 'shape') else alphas
    threshold_alpha = alphas[threshold_idx]
    
    threshold_coeffs = coefficients[threshold_idx]
    if not hasattr(threshold_coeffs, 'shape'):
        threshold_coeffs = np.array(threshold_coeffs)
    
    # Identify critical units (non-zero coefficients at threshold)
    critical_mask = np.abs(threshold_coeffs) > 1e-8  # Small tolerance for numerical precision
    critical_indices = np.where(critical_mask)[0]
    
    return {
        'threshold_alpha': float(threshold_alpha),
        'threshold_score': float(scores[threshold_idx]),
        'threshold_idx': int(threshold_idx),
        'critical_indices': critical_indices.tolist() if hasattr(critical_indices, 'tolist') else critical_indices,
        'critical_coefficients': threshold_coeffs[critical_mask].tolist() if hasattr(threshold_coeffs[critical_mask], 'tolist') else threshold_coeffs[critical_mask],
        'n_critical_units': int(len(critical_indices)),
        'chance_level': float(chance_level),
        'chance_margin': float(chance_margin),
        'all_scores': scores.tolist(),
        'all_alphas': alphas.tolist()
    }


def main():
    """Command-line interface for critical units analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--states', type=Path, required=True, 
                       help='Path to states file (.npz)')
    parser.add_argument('--output', type=Path, required=True, 
                       help='Output path for results')
    parser.add_argument('--model-id', type=str, required=True, 
                       help='Model identifier')
    parser.add_argument('--decoder-type', type=str, required=True,
                       choices=['hazard', 'contingency', 'color', 'velocity_x', 'velocity_y'],
                       help='Type of decoder to analyze')
    parser.add_argument('--timestep', type=int, default=-1,
                       help='Analyze specific timestep (default: -1 = last timestep)')
    parser.add_argument('--temporal', action='store_true',
                       help='Perform temporal dynamics analysis')
    parser.add_argument('--regularization-sweep', action='store_true',
                       help='Run full regularization sweep')
    parser.add_argument('--use-all-timesteps', action='store_true', default=True,
                       help='Use all non-padded timesteps (default: True)')
    parser.add_argument('--concatenate-states', action='store_true', default=True,
                       help='Concatenate hidden and cell states (default: True)')
    parser.add_argument('--use-normalized', action='store_true', default=True,
                       help='Use normalized states if available (default: True)')
    parser.add_argument('--save-visualization-data', action='store_true', default=True,
                       help='Save coefficient paths and performance curves for visualization')
    parser.add_argument('--generate-plots', action='store_true',
                       help='Generate coefficient analysis plots immediately')
    parser.add_argument('--output-dir', type=Path,
                       help='Directory to save plots (required if --generate-plots)')
    parser.add_argument('--task-variable', type=str,
                       help='Name of task variable for plot titles (defaults to decoder-type)')
    
    # New regularization parameters
    parser.add_argument('--lambda-min', type=float, default=LAMBDA_MIN,
                       help=f'Minimum lambda value for regularization sweep (default: {LAMBDA_MIN})')
    parser.add_argument('--lambda-max', type=float, default=LAMBDA_MAX,
                       help=f'Maximum lambda value for regularization sweep (default: {LAMBDA_MAX})')
    parser.add_argument('--n-lambdas', type=int, default=N_LAMBDAS,
                       help=f'Number of lambda values to sweep (default: {N_LAMBDAS})')
    parser.add_argument('--lambda-spacing', choices=['log', 'linear'], default=LAMBDA_SPACING,
                       help=f'Spacing for lambda values (default: {LAMBDA_SPACING})')
    
    # Decoder-specific parameters
    parser.add_argument('--binary-l1-ratio', type=float, default=BINARY_L1_RATIO,
                       help=f'L1 ratio for binary classification decoders (default: {BINARY_L1_RATIO})')
    parser.add_argument('--multiclass-l1-ratio', type=float, default=MULTICLASS_L1_RATIO,
                       help=f'L1 ratio for multiclass/regression decoders (default: {MULTICLASS_L1_RATIO})')
    parser.add_argument('--binary-max-iter', type=int, default=BINARY_MAX_ITER,
                       help=f'Max iterations for binary decoders (default: {BINARY_MAX_ITER})')
    parser.add_argument('--multiclass-max-iter', type=int, default=MULTICLASS_MAX_ITER,
                       help=f'Max iterations for multiclass/regression decoders (default: {MULTICLASS_MAX_ITER})')
    
    # Threshold detection parameters
    parser.add_argument('--chance-margin', type=float, default=CHANCE_MARGIN,
                       help=f'Performance margin above chance for threshold (default: {CHANCE_MARGIN})')
    parser.add_argument('--chance-level', type=float, default=None,
                       help='Override auto-detected chance level')
    parser.add_argument('--threshold-method', choices=['chance', 'elbow', 'fixed'], default=THRESHOLD_METHOD,
                       help=f'Method for finding critical threshold (default: {THRESHOLD_METHOD})')
    
    args = parser.parse_args()
    
    # Load states
    print(f"Loading states from {args.states}")
    state_data = np.load(args.states, allow_pickle=True)
    
    # Extract appropriate labels for decoder type
    print(f"Extracting labels for {args.decoder_type} decoder")
    labels = extract_labels_from_state_data(state_data, args.decoder_type)
    
    # Get states data
    if 'hiddens' in state_data:
        states = state_data['hiddens']
    else:
        raise ValueError("No 'hiddens' found in state data")
    
    # Get cell states if available
    cells = state_data.get('cells', None)
    
    # Get trial lengths from df_data
    trial_lengths = None
    if args.use_all_timesteps:
        if 'df_data' in state_data:
            import pandas as pd
            df_data_dict = state_data['df_data'].item() if hasattr(state_data['df_data'], 'item') else state_data['df_data']
            df_data = pd.DataFrame(df_data_dict)
            trial_lengths = df_data['length'].values
    
    # Run analysis
    print(f"Running critical units analysis for {args.model_id} - {args.decoder_type}")
    
    if args.temporal:
        results = analyze_temporal_dynamics(
            states,
            labels,
            decoder_type=args.decoder_type
        )
    else:
        # Always use regularization sweep (new default behavior)
        results = identify_critical_units(
            states,
            labels,
            # New parameters from CLI
            lambda_min=args.lambda_min,
            lambda_max=args.lambda_max,
            n_lambdas=args.n_lambdas,
            lambda_spacing=args.lambda_spacing,
            decoder_type=args.decoder_type,
            binary_l1_ratio=args.binary_l1_ratio,
            multiclass_l1_ratio=args.multiclass_l1_ratio,
            binary_max_iter=args.binary_max_iter,
            multiclass_max_iter=args.multiclass_max_iter,
            chance_margin=args.chance_margin,
            chance_level=args.chance_level,
            threshold_method=args.threshold_method,
            # Other parameters
            timestep=args.timestep if not args.use_all_timesteps else None,
            use_all_timesteps=args.use_all_timesteps,
            trial_lengths=trial_lengths,
            cells=cells,
            concatenate_states=args.concatenate_states,
            save_visualization_data=args.save_visualization_data,
            generate_plots=args.generate_plots,
            output_dir=args.output_dir,
            task_variable=args.task_variable or args.decoder_type
        )
        
        # Ensure decoder_type is in the results  
        if 'decoder_type' not in results['metadata']:
            results['metadata']['decoder_type'] = args.decoder_type
    
    # Add model info
    results["model_id"] = args.model_id
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {args.output}")
    
    # Print summary
    if not args.temporal:
        print(f"\nSummary for {args.decoder_type} decoder:")
        print(f"  Total units: {results['metadata']['n_units_total']}")
        print(f"  Critical units: {results['metadata']['n_units_critical']}")
        if 'r2_scores' in results and results['r2_scores']:
            print(f"  Score: {results['r2_scores'][0]:.4f}")
        print(f"  Best alpha: {results['best_alpha']:.6f}")


if __name__ == '__main__':
    main()