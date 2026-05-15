"""
Model configuration management for In-Context-CPD.

This module provides configuration dataclasses and utilities for managing
model parameters, particularly for overriding architectural parameters
like recurrent_size and recurrent_num_layers.
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, Union
from pathlib import Path
import yaml


@dataclass
class ModelConfig:
    """
    Configuration for SequenceModel architecture.
    
    This matches the model_config structure used in timescales,
    enabling compatibility with existing checkpoints while allowing
    parameter overrides.
    """
    
    # Core architecture parameters
    input_size: int = 5
    output_size: int = 5  # Same as input_size for autoregressive prediction
    feedforward_size: int = 8
    recurrent_size: int = 16
    recurrent_num_layers: int = 1
    recurrent_cls: str = "LSTM_V2"
    
    # MLP configuration  
    output_mlp_size: Optional[int] = None  # None means direct projection
    
    # Training parameters
    batch_first: bool = True
    dropout: float = 0.0
    
    # Layer normalization
    feedforward_layer_norm: bool = True
    recurrent_layer_norm: bool = True
    
    # Additional kwargs for different components
    feedforward_kwargs: Dict[str, Any] = field(default_factory=dict)
    recurrent_kwargs: Dict[str, Any] = field(default_factory=dict)
    output_kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for use with SequenceModelBase."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create ModelConfig from dictionary."""
        # Filter out keys that aren't in the dataclass
        valid_keys = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_dict)


@dataclass
class ExtractionConfig:
    """
    Configuration for state extraction process.
    """
    
    # Model configuration
    model_config: ModelConfig = field(default_factory=ModelConfig)
    
    # Extraction parameters
    batch_size: int = 32
    device: str = "cuda"  # Will fallback to CPU if CUDA not available
    
    # Data parameters
    dataset_path: Optional[str] = None
    
    # Scaling parameters (from timescales)
    scaling: float = 255.0
    shift: float = 0.0
    
    # Output configuration
    return_all_states: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        # Convert model_config to dict as well
        result['model_config'] = self.model_config.to_dict()
        return result
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExtractionConfig':
        """Create ExtractionConfig from dictionary."""
        config_copy = config_dict.copy()
        
        # Handle model_config separately
        if 'model_config' in config_copy:
            model_config_dict = config_copy.pop('model_config')
            model_config = ModelConfig.from_dict(model_config_dict)
            config_copy['model_config'] = model_config
        
        return cls(**config_copy)


def create_default_configs():
    """Create default configurations for common model architectures."""
    
    configs = {
        # Default configuration (matches timescales defaults)
        "default": ModelConfig(),
        
        # Small model for testing
        "small": ModelConfig(
            recurrent_size=8,
            recurrent_num_layers=1,
            feedforward_size=4,
        ),
        
        # Medium model
        "medium": ModelConfig(
            recurrent_size=32,
            recurrent_num_layers=2,
            feedforward_size=16,
        ),
        
        # Large model
        "large": ModelConfig(
            recurrent_size=64,
            recurrent_num_layers=3,
            feedforward_size=32,
        ),
    }
    
    return configs


def override_model_config(
    base_config: ModelConfig,
    overrides: Dict[str, Any]
) -> ModelConfig:
    """
    Apply parameter overrides to a base model configuration.
    
    Parameters
    ----------
    base_config : ModelConfig
        Base configuration to override
    overrides : Dict[str, Any]
        Dictionary of parameter overrides
        
    Returns
    -------
    ModelConfig
        New configuration with overrides applied
    """
    config_dict = base_config.to_dict()
    config_dict.update(overrides)
    return ModelConfig.from_dict(config_dict)


def parse_cli_overrides(override_strings: list) -> Dict[str, Any]:
    """
    Parse command-line override strings into a dictionary.
    
    Supports formats like:
    - "recurrent_size=16"
    - "recurrent_num_layers=2"
    - "dropout=0.1"
    
    Parameters
    ----------
    override_strings : list
        List of override strings from command line
        
    Returns
    -------
    Dict[str, Any]
        Dictionary of parsed overrides
    """
    overrides = {}
    
    for override_str in override_strings:
        if '=' not in override_str:
            continue
            
        key, value_str = override_str.split('=', 1)
        key = key.strip()
        value_str = value_str.strip()
        
        # Try to parse the value
        try:
            # Try as int first
            value = int(value_str)
        except ValueError:
            try:
                # Try as float
                value = float(value_str)
            except ValueError:
                # Try as boolean
                if value_str.lower() in ('true', 'false'):
                    value = value_str.lower() == 'true'
                else:
                    # Keep as string
                    value = value_str
        
        overrides[key] = value
    
    return overrides


def load_config_from_yaml(config_path: Union[str, Path]) -> ExtractionConfig:
    """
    Load extraction configuration from YAML file.
    
    Parameters
    ----------
    config_path : Union[str, Path]
        Path to YAML configuration file
        
    Returns
    -------
    ExtractionConfig
        Loaded configuration
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return ExtractionConfig.from_dict(config_dict)


def save_config_to_yaml(config: ExtractionConfig, config_path: Union[str, Path]):
    """
    Save extraction configuration to YAML file.
    
    Parameters
    ----------
    config : ExtractionConfig
        Configuration to save
    config_path : Union[str, Path]
        Path where to save the configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)


# Common model configurations used in experiments
TIMESCALES_MODEL_CONFIGS = {
    "SAN-4378": ModelConfig(
        recurrent_size=16,
        recurrent_num_layers=1,
        feedforward_size=8,
    ),
    "SAN-4401": ModelConfig(
        recurrent_size=16,
        recurrent_num_layers=1,
        feedforward_size=8,
    ),
    "SAN-4566": ModelConfig(
        recurrent_size=16,
        recurrent_num_layers=1,
        feedforward_size=8,
    ),
    "SAN-4567": ModelConfig(
        recurrent_size=16,
        recurrent_num_layers=1,
        feedforward_size=8,
    ),
    "SAN-4568": ModelConfig(
        recurrent_size=16,
        recurrent_num_layers=1,
        feedforward_size=8,
    ),
}


def get_model_config_for_id(model_id: str) -> ModelConfig:
    """
    Get the appropriate model configuration for a given model ID.
    
    Parameters
    ----------
    model_id : str
        Model identifier (e.g., "SAN-4378")
        
    Returns
    -------
    ModelConfig
        Model configuration for the specified ID
    """
    if model_id in TIMESCALES_MODEL_CONFIGS:
        return TIMESCALES_MODEL_CONFIGS[model_id]
    else:
        # Return default configuration for unknown models
        return ModelConfig()