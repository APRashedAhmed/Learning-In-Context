"""Pipeline for extracting neural states from model checkpoints."""

from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from ..core import PipelineResult, StateData
from .base import BasePipeline, ValidationError


class ExtractionPipeline(BasePipeline):
    """Pipeline for extracting states from trained models."""
    
    def run(self, model_ids: List[str], force: bool = False) -> PipelineResult:
        """Extract states from models.
        
        Args:
            model_ids: List of model identifiers
            force: Force re-extraction even if cached
            
        Returns:
            PipelineResult with extracted states
        """
        try:
            # Validate inputs
            self.validate_inputs(model_ids=model_ids)
            
            results = {}
            
            for model_id in tqdm(model_ids, desc="Extracting states"):
                cache_key = f"states/{model_id}/states"
                
                # Check cache unless forced
                if not force and self.cache.exists(cache_key):
                    self.logger.info(f"Loading cached states for {model_id}")
                    results[model_id] = self.cache.load(cache_key)
                    continue
                
                # Extract states
                self._log_stage_start(f"extract_{model_id}")
                checkpoint_path = self.config.get_model_checkpoint(model_id)
                states = self._extract_single_model(model_id, checkpoint_path)
                self._log_stage_end(f"extract_{model_id}")
                
                # Cache results
                self.cache.save(cache_key, states)
                results[model_id] = states
            
            return PipelineResult(
                success=True,
                data=results,
                metadata={"n_models": len(model_ids)}
            )
            
        except Exception as e:
            return self._handle_error(e, self._current_stage or "extraction")
    
    def validate_inputs(self, model_ids: List[str]) -> bool:
        """Validate extraction inputs."""
        if not model_ids:
            raise ValidationError("No model IDs provided")
        
        # Verify all models have valid configs
        for model_id in model_ids:
            try:
                self.config.get_model_checkpoint(model_id)
            except ValueError as e:
                raise ValidationError(f"Model configuration error: {e}")
        
        return True
    
    def _extract_single_model(self, model_id: str, checkpoint_path: Path) -> StateData:
        """Extract states from a single model.
        
        Args:
            model_id: Model identifier
            checkpoint_path: Path to checkpoint
            
        Returns:
            StateData object
        """
        # Import the actual extraction code
        from ..models.extract_states_simple import extract_model_states
        
        # Extract states
        states_dict = extract_model_states(
            model_id=model_id,
            checkpoint_path=checkpoint_path,
            config=self.config
        )
        
        # Convert to StateData
        return StateData(
            hiddens=states_dict["hiddens"],
            cells=states_dict.get("cells"),
            predictions=states_dict["predictions"],
            metadata=states_dict.get("metadata", {})
        )