"""Simplified state extraction for pipeline integration."""

from pathlib import Path
from typing import Dict, Any

import numpy as np

from ..core import Config


def extract_model_states(
    model_id: str,
    checkpoint_path: Path,
    config: Config
) -> Dict[str, Any]:
    """Extract states from a model checkpoint.
    
    This is a stub implementation for pipeline development.
    The real implementation would load the model and extract actual states.
    
    Args:
        model_id: Model identifier
        checkpoint_path: Path to checkpoint
        config: Configuration object
        
    Returns:
        Dictionary with extracted states
    """
    # Stub implementation - generate random states for testing
    n_trials = 100
    n_timesteps = 50
    hidden_size = 128
    
    # Generate synthetic states
    hiddens = np.random.randn(n_trials, n_timesteps, hidden_size).astype(np.float32)
    cells = np.random.randn(n_trials, n_timesteps, hidden_size).astype(np.float32)
    
    # Generate predictions (3 classes)
    logits = np.random.randn(n_trials, n_timesteps, 3)
    predictions = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    predictions = predictions.astype(np.float32)
    
    # Create metadata
    metadata = {
        "model_id": model_id,
        "checkpoint_path": str(checkpoint_path),
        "hidden_size": hidden_size,
        "n_trials": n_trials,
        "n_timesteps": n_timesteps,
        "extraction_method": "stub",
        "trial_types": np.random.choice(["Straight", "Bounce", "Catch"], size=n_trials),
        "hazard_rates": np.random.choice(["Low", "High"], size=n_trials),
        "contingencies": np.random.choice(["Low", "Medium", "High"], size=n_trials)
    }
    
    return {
        "hiddens": hiddens,
        "cells": cells,
        "predictions": predictions,
        "metadata": metadata
    }