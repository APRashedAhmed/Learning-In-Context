"""
Configuration management for In-Context-CPD pipeline.
"""

from .model_config import (
    ModelConfig,
    ExtractionConfig,
    get_model_config_for_id,
    override_model_config,
    parse_cli_overrides,
    load_config_from_yaml,
    save_config_to_yaml,
    TIMESCALES_MODEL_CONFIGS
)

# Import PipelineConfig from the config module file
import importlib.util
import sys
from pathlib import Path

# Load the config.py module directly by file path
config_module_path = Path(__file__).parent.parent / 'config.py'
spec = importlib.util.spec_from_file_location("learning_in_context_config_module", config_module_path)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
PipelineConfig = config_module.PipelineConfig

__all__ = [
    'PipelineConfig',
    'ModelConfig',
    'ExtractionConfig',
    'get_model_config_for_id',
    'override_model_config',
    'parse_cli_overrides',
    'load_config_from_yaml',
    'save_config_to_yaml',
    'TIMESCALES_MODEL_CONFIGS'
]