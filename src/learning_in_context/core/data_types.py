"""Core data types with validation and serialization support."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class ModelMetadata:
    """Metadata for a model checkpoint."""
    
    model_id: str
    checkpoint_path: Path
    hidden_size: int
    num_layers: int
    recurrent_type: str
    
    def __post_init__(self):
        """Validate model metadata."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        if self.hidden_size <= 0:
            raise ValueError(f"Invalid hidden_size: {self.hidden_size}")
        if self.num_layers <= 0:
            raise ValueError(f"Invalid num_layers: {self.num_layers}")
        if self.recurrent_type not in ["lstm", "gru"]:
            raise ValueError(f"Invalid recurrent_type: {self.recurrent_type}")


@dataclass
class ParticipantTrial:
    """Single trial data from a participant."""
    
    participant_id: str
    video_id: str
    response: int
    confidence: float
    reaction_time: float
    
    def __post_init__(self):
        """Validate trial data."""
        if self.response not in [1, 2, 3]:
            raise ValueError(f"Invalid response: {self.response}")
        if not 0 <= self.confidence <= 100:
            raise ValueError(f"Invalid confidence: {self.confidence}")
        if self.reaction_time < 0:
            raise ValueError(f"Invalid reaction_time: {self.reaction_time}")


@dataclass
class AnalysisResult:
    """Container for analysis results."""
    
    name: str
    data: Union[np.ndarray, pd.DataFrame]
    metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "data_type": type(self.data).__name__,
            "data_shape": self.data.shape if hasattr(self.data, "shape") else None,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class StateData:
    """Container for extracted neural states."""
    
    hiddens: np.ndarray
    cells: Optional[np.ndarray]
    predictions: np.ndarray
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Validate state data shapes."""
        n_trials, timesteps, _ = self.hiddens.shape
        
        if self.cells is not None:
            if self.cells.shape != self.hiddens.shape:
                raise ValueError("Cells and hiddens must have same shape")
        
        if self.predictions.shape[:2] != (n_trials, timesteps):
            raise ValueError("Predictions shape doesn't match hiddens")
        
        if self.predictions.shape[2] != 3:
            raise ValueError("Predictions must have 3 classes")
    
    def save(self, path: Path):
        """Save states to disk."""
        np.savez_compressed(
            path,
            hiddens=self.hiddens,
            cells=self.cells if self.cells is not None else np.array([]),
            predictions=self.predictions,
            metadata=self.metadata
        )
    
    @classmethod
    def load(cls, path: Path) -> "StateData":
        """Load states from disk."""
        data = np.load(path, allow_pickle=True)
        cells = data["cells"]
        if cells.size == 0:
            cells = None
        return cls(
            hiddens=data["hiddens"],
            cells=cells,
            predictions=data["predictions"],
            metadata=data["metadata"].item()
        )


@dataclass
class PipelineResult:
    """Result from pipeline execution."""
    
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __bool__(self) -> bool:
        """Allow using result in boolean context."""
        return self.success


@dataclass
class CriticalUnitsResult:
    """Results from critical units identification."""
    
    unit_indices: np.ndarray
    coefficients: np.ndarray
    r2_scores: np.ndarray
    best_alpha: float
    cv_scores: Dict[str, np.ndarray]
    metadata: Dict[str, Any]
    
    # Visualization data fields
    coefficient_paths: Optional[np.ndarray] = None  # Shape: (n_units, n_alphas)
    intercept_paths: Optional[np.ndarray] = None    # Shape: (n_classes_or_1, n_alphas)
    performance_curves: Optional[Dict[str, np.ndarray]] = None  # {'accuracy': [...], 'r2': [...]}
    alpha_values: Optional[np.ndarray] = None       # The regularization strengths used
    
    def get_top_units(self, n: int = 10) -> np.ndarray:
        """Get indices of top N critical units."""
        coef_magnitude = np.abs(self.coefficients)
        top_indices = np.argsort(coef_magnitude)[-n:][::-1]
        return self.unit_indices[top_indices]
    
    def has_visualization_data(self) -> bool:
        """Check if visualization data is available."""
        return (self.coefficient_paths is not None and 
                self.intercept_paths is not None and 
                self.performance_curves is not None and 
                self.alpha_values is not None)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "unit_indices": self.unit_indices.tolist(),
            "coefficients": self.coefficients.tolist(),
            "r2_scores": self.r2_scores.tolist(),
            "best_alpha": self.best_alpha,
            "cv_scores": {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in self.cv_scores.items()},
            "metadata": self.metadata
        }
        
        # Add visualization data if available
        if self.has_visualization_data():
            result.update({
                "coefficient_paths": self.coefficient_paths.tolist(),
                "intercept_paths": self.intercept_paths.tolist(),
                "performance_curves": {k: v.tolist() for k, v in self.performance_curves.items()},
                "alpha_values": self.alpha_values.tolist()
            })
        
        return result


@dataclass
class AnalysisResults:
    """Container for multiple analysis results."""
    
    individual: Dict[str, Any]
    aggregate: pd.DataFrame
    metadata: Dict[str, Any]
    
    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        return {
            "n_entities": len(self.individual),
            "aggregate_shape": self.aggregate.shape,
            "metadata": self.metadata
        }