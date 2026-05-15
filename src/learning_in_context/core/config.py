"""Hierarchical configuration management with environment variable support."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from omegaconf import DictConfig, OmegaConf

from .constants import (
    DEFAULT_CACHE_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RAW_DIR,
    LOG_LEVEL
)


class Config:
    """Hierarchical configuration management."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self._config = DictConfig({})
        self._config_path = config_path
        
        # Load default configuration
        self._load_defaults()
        
        # Load from file if provided
        if config_path:
            self.load()
        
        # Apply environment overrides
        self._apply_env_overrides()
    
    def load(self) -> DictConfig:
        """Load configuration from file."""
        if self._config_path and self._config_path.exists():
            with open(self._config_path, "r") as f:
                file_config = yaml.safe_load(f)
                self._config = OmegaConf.merge(self._config, file_config)
        return self._config
    
    def validate(self) -> bool:
        """Validate configuration."""
        required_keys = ["paths", "models", "analysis", "figures"]
        
        for key in required_keys:
            if key not in self._config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        # Validate paths
        for path_key in ["raw", "cache", "output"]:
            if path_key not in self._config.paths:
                raise ValueError(f"Missing required path: {path_key}")
        
        return True
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation.
        
        Args:
            key: Configuration key (e.g., "paths.cache")
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        try:
            value = self._config
            for part in key.split("."):
                value = value[part]
            return value
        except (KeyError, AttributeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value with dot notation.
        
        Args:
            key: Configuration key (e.g., "paths.cache")
            value: Value to set
        """
        parts = key.split(".")
        config = self._config
        
        # Navigate to parent
        for part in parts[:-1]:
            if part not in config:
                config[part] = DictConfig({})
            config = config[part]
        
        # Set value
        config[parts[-1]] = value
    
    def save(self, path: Path) -> None:
        """Save configuration to file.
        
        Args:
            path: Path to save configuration
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(OmegaConf.to_container(self._config), f, default_flow_style=False)
    
    @property
    def cache_dir(self) -> Path:
        """Get cache directory path."""
        return Path(self.get("paths.cache", DEFAULT_CACHE_DIR))
    
    @property
    def output_dir(self) -> Path:
        """Get output directory path."""
        return Path(self.get("paths.output", DEFAULT_OUTPUT_DIR))
    
    @property
    def raw_dir(self) -> Path:
        """Get raw data directory path."""
        return Path(self.get("paths.raw", DEFAULT_RAW_DIR))
    
    def get_model_checkpoint(self, model_id: str) -> Path:
        """Get checkpoint path for model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Path to checkpoint file
        """
        model_config = self.get(f"models.{model_id}")
        if not model_config:
            raise ValueError(f"Model not found in config: {model_id}")
        
        checkpoint_path = Path(model_config.get("checkpoint_path"))
        if not checkpoint_path.exists():
            # Try relative to raw directory
            checkpoint_path = self.raw_dir / checkpoint_path
        
        return checkpoint_path
    
    def _load_defaults(self) -> None:
        """Load default configuration."""
        defaults = {
            "paths": {
                "raw": DEFAULT_RAW_DIR,
                "cache": DEFAULT_CACHE_DIR,
                "output": DEFAULT_OUTPUT_DIR
            },
            "models": {},
            "analysis": {
                "critical_units": {
                    "alphas": "logspace",
                    "l1_ratio": 0.64,
                    "cv_folds": 5,
                    "n_jobs": -1
                },
                "bootstrap": {
                    "n_iterations": 1000,
                    "confidence_level": 0.95
                },
                "permutation": {
                    "n_iterations": 10000
                }
            },
            "figures": {
                "dpi": 300,
                "format": "pdf",
                "style": "publication"
            },
            "cache": {
                "max_size_gb": 10.0,
                "eviction_policy": "lru",
                "compression": True
            },
            "pipeline": {
                "batch_size": 1024,
                "num_workers": 4,
                "device": "cuda" if self._cuda_available() else "cpu",
                "checkpoint_interval": 100
            },
            "logging": {
                "level": LOG_LEVEL,
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
        
        self._config = OmegaConf.create(defaults)
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        # Map environment variables to config keys
        env_mapping = {
            "ICCPD_CACHE_DIR": "paths.cache",
            "ICCPD_OUTPUT_DIR": "paths.output",
            "ICCPD_RAW_DIR": "paths.raw",
            "ICCPD_DEVICE": "pipeline.device",
            "ICCPD_BATCH_SIZE": "pipeline.batch_size",
            "ICCPD_LOG_LEVEL": "logging.level"
        }
        
        for env_var, config_key in env_mapping.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                
                # Convert numeric values
                if env_var == "ICCPD_BATCH_SIZE":
                    value = int(value)
                
                self.set(config_key, value)
    
    def _cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return OmegaConf.to_container(self._config)