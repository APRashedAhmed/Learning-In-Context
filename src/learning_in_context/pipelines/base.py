"""Base pipeline class for all analysis pipelines."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from ..core import CacheManager, Config, PipelineResult


class BasePipeline(ABC):
    """Abstract base class for all pipelines."""
    
    def __init__(self, config: Config, cache: CacheManager):
        """Initialize pipeline.
        
        Args:
            config: Configuration object
            cache: Cache manager
        """
        self.config = config
        self.cache = cache
        self.logger = self._setup_logger()
        
        # Track pipeline state
        self._checkpoints = {}
        self._current_stage = None
    
    @abstractmethod
    def run(self, **kwargs) -> PipelineResult:
        """Execute the pipeline.
        
        Returns:
            PipelineResult indicating success/failure
        """
        pass
    
    @abstractmethod
    def validate_inputs(self, **kwargs) -> bool:
        """Validate pipeline inputs.
        
        Returns:
            True if inputs are valid
            
        Raises:
            ValueError: If inputs are invalid
        """
        pass
    
    def checkpoint(self, stage: str, data: Any) -> None:
        """Save intermediate checkpoint.
        
        Args:
            stage: Stage name
            data: Data to checkpoint
        """
        checkpoint_key = f"checkpoints/{self.__class__.__name__}/{stage}"
        self.cache.save(checkpoint_key, data)
        self._checkpoints[stage] = checkpoint_key
        self.logger.info(f"Saved checkpoint for stage: {stage}")
    
    def load_checkpoint(self, stage: str) -> Optional[Any]:
        """Load checkpoint if available.
        
        Args:
            stage: Stage name
            
        Returns:
            Checkpoint data if available, None otherwise
        """
        checkpoint_key = self._checkpoints.get(stage)
        if checkpoint_key and self.cache.exists(checkpoint_key):
            self.logger.info(f"Loading checkpoint for stage: {stage}")
            return self.cache.load(checkpoint_key)
        return None
    
    def clear_checkpoints(self) -> None:
        """Clear all checkpoints for this pipeline."""
        pattern = f"checkpoints/{self.__class__.__name__}/*"
        count = self.cache.invalidate(pattern)
        self._checkpoints.clear()
        self.logger.info(f"Cleared {count} checkpoints")
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logger for pipeline."""
        logger = logging.getLogger(self.__class__.__name__)
        
        # Don't add handlers if they already exist
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                self.config.get("logging.format", 
                              "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        logger.setLevel(
            getattr(logging, self.config.get("logging.level", "INFO"))
        )
        
        return logger
    
    def _log_stage_start(self, stage: str) -> None:
        """Log the start of a pipeline stage."""
        self._current_stage = stage
        self.logger.info(f"Starting stage: {stage}")
    
    def _log_stage_end(self, stage: str, success: bool = True) -> None:
        """Log the end of a pipeline stage."""
        status = "completed" if success else "failed"
        self.logger.info(f"Stage {stage} {status}")
        self._current_stage = None
    
    def _handle_error(self, error: Exception, stage: str) -> PipelineResult:
        """Handle pipeline errors.
        
        Args:
            error: Exception that occurred
            stage: Stage where error occurred
            
        Returns:
            PipelineResult with error information
        """
        error_msg = f"Error in stage '{stage}': {str(error)}"
        self.logger.error(error_msg, exc_info=True)
        
        return PipelineResult(
            success=False,
            error=error_msg,
            metadata={
                "stage": stage,
                "error_type": type(error).__name__,
                "checkpoints": list(self._checkpoints.keys())
            }
        )


class PipelineError(Exception):
    """Base exception for pipeline errors."""
    pass


class DataNotFoundError(PipelineError):
    """Raised when required data is not found."""
    pass


class AnalysisError(PipelineError):
    """Raised when analysis fails."""
    pass


class ValidationError(PipelineError):
    """Raised when validation fails."""
    pass