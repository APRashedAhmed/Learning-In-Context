"""
Configuration management for the pipeline.
"""

from pathlib import Path
from omegaconf import DictConfig, OmegaConf


class PipelineConfig:
    """Central configuration management."""
    
    def __init__(self, config_path=None, overrides=None, weights_dir=None):
        # Load default configuration
        default_config = self._get_default_config()
        
        # Load user configuration if provided
        if config_path and Path(config_path).exists():
            user_config = OmegaConf.load(config_path)
            self.config = OmegaConf.merge(default_config, user_config)
        else:
            self.config = default_config
        
        # Apply command-line overrides first
        if overrides:
            override_config = OmegaConf.create(overrides)
            self.config = OmegaConf.merge(self.config, override_config)
        
        # Apply weights_dir override if provided (takes precedence over overrides)
        if weights_dir:
            self.config.data.weights_dir = weights_dir
    
    @staticmethod
    def _get_default_config():
        """Get default configuration."""
        return OmegaConf.create({
            'data': {
                'base_dir': 'data',  # Now relative to project root
                'weights_dir': '${data.base_dir}/weights/analyze',
                'raw_dir': '../data/raw',  # Participant data still outside
                'processed_dir': '${data.base_dir}/processed',
                'participant_version': 'v3_2_2',
            },
            'pipeline': {
                'cache_dir': 'data/cache',  # Changed from outputs/cache
                'figures_dir': 'outputs/figures',
                'n_workers': 4,
                'batch_size': 32,
            },
            'model': {
                'hidden_dim': 16,
                'input_dim': 5,
                'output_dim': 3,
                'architecture': 'LSTM',
            },
            'analysis': {
                'regularization_alphas': [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
                'intervention_alphas': 10,
                'event_window': [-5, 10],
                'last_n_timesteps': 21,
            },
            'participant': {
                'min_attention_accuracy': 0.8,
                'min_timesteps_visible': 8,
                'confidence_range': [0, 100],
            },
        })
    
    # Convenience properties
    @property
    def weights_dir(self):
        return Path(self.config.data.weights_dir)
    
    @property
    def cache_dir(self):
        return Path(self.config.pipeline.cache_dir)
    
    @property
    def figures_dir(self):
        return Path(self.config.pipeline.figures_dir)
    
    @property
    def processed_dir(self):
        return Path(self.config.data.processed_dir)
    
    @property
    def participant_data_dir(self):
        version = self.config.data.participant_version
        return Path(self.config.data.raw_dir) / f'hbb_participant_responses_{version}'
    
    def get_model_checkpoint(self, model_id):
        """Find checkpoint file for a model."""
        # Expected path: data/weights/analyze/SAN-####/last.ckpt
        checkpoint_path = self.weights_dir / model_id / 'last.ckpt'
        if checkpoint_path.exists():
            return checkpoint_path
        
        # Fallback to searching in the model directory
        model_dir = self.weights_dir / model_id
        if model_dir.exists():
            # Look for checkpoint files
            for pattern in ['*.ckpt', '*.pt', '*.pth']:
                checkpoints = list(model_dir.glob(pattern))
                if checkpoints:
                    return checkpoints[0]
        return None
    
    def ensure_directories(self):
        """Create necessary directories."""
        dirs = [
            self.cache_dir,
            self.figures_dir,
            self.processed_dir,
            self.cache_dir / 'model_states',
            self.cache_dir / 'critical_units',
            self.cache_dir / 'tuning_profiles',
            self.cache_dir / 'interventions',
            self.cache_dir / 'participants',
            self.figures_dir / 'models',
            self.figures_dir / 'participants',
            self.figures_dir / 'comparisons',
            self.processed_dir / 'model_metrics',
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)