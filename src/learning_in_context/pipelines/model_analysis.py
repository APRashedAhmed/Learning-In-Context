"""Pipeline for model analysis including critical units identification."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from ..core import PipelineResult, StateData
from .base import AnalysisError, BasePipeline, DataNotFoundError, ValidationError


class ModelAnalysisPipeline(BasePipeline):
    """Pipeline for analyzing neural network models."""
    
    def run(self, model_ids: List[str], analyses: List[str]) -> Dict[str, Dict]:
        """Run model analysis pipeline.
        
        Args:
            model_ids: List of model identifiers to analyze
            analyses: List of analyses to perform 
                     (e.g., ["critical_units", "tuning_profiles", "interventions"])
        
        Returns:
            Dictionary mapping model_id to analysis results
        """
        try:
            # Validate inputs
            self.validate_inputs(model_ids=model_ids, analyses=analyses)
            
            # Initialize results
            results = {}
            
            # Process each model
            for model_id in tqdm(model_ids, desc="Analyzing models"):
                self.logger.info(f"Processing model: {model_id}")
                
                # Load or extract states
                self._log_stage_start(f"load_states_{model_id}")
                states = self._load_or_extract_states(model_id)
                self._log_stage_end(f"load_states_{model_id}")
                
                # Run requested analyses
                model_results = {}
                
                if "critical_units" in analyses:
                    self._log_stage_start(f"critical_units_{model_id}")
                    model_results["critical_units"] = self._identify_critical_units(
                        model_id, states
                    )
                    self._log_stage_end(f"critical_units_{model_id}")
                
                if "regularization_sweep" in analyses:
                    self._log_stage_start(f"regularization_sweep_{model_id}")
                    model_results["regularization_sweep"] = self._run_regularization_sweep(
                        model_id, states
                    )
                    self._log_stage_end(f"regularization_sweep_{model_id}")
                
                if "tuning_profiles" in analyses:
                    self._log_stage_start(f"tuning_profiles_{model_id}")
                    # Tuning profiles require critical units
                    if "critical_units" not in model_results:
                        cu_result = self._load_or_compute_critical_units(model_id, states)
                        model_results["critical_units"] = cu_result
                    
                    model_results["tuning_profiles"] = self._compute_tuning_profiles(
                        model_id, states, model_results["critical_units"]
                    )
                    self._log_stage_end(f"tuning_profiles_{model_id}")
                
                if "interventions" in analyses:
                    self._log_stage_start(f"interventions_{model_id}")
                    # Interventions require critical units and tuning profiles
                    if "critical_units" not in model_results:
                        cu_result = self._load_or_compute_critical_units(model_id, states)
                        model_results["critical_units"] = cu_result
                    
                    if "tuning_profiles" not in model_results:
                        model_results["tuning_profiles"] = self._compute_tuning_profiles(
                            model_id, states, model_results["critical_units"]
                        )
                    
                    model_results["interventions"] = self._run_interventions(
                        model_id, states, model_results["critical_units"], 
                        model_results["tuning_profiles"]
                    )
                    self._log_stage_end(f"interventions_{model_id}")
                
                results[model_id] = model_results
                
                # Save checkpoint after each model
                self.checkpoint(f"model_{model_id}", model_results)
            
            return results
            
        except Exception as e:
            return self._handle_error(e, self._current_stage or "unknown")
    
    def validate_inputs(self, model_ids: List[str], analyses: List[str]) -> bool:
        """Validate pipeline inputs.
        
        Args:
            model_ids: List of model identifiers
            analyses: List of analysis types
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If inputs are invalid
        """
        # Check model_ids
        if not model_ids:
            raise ValidationError("No model IDs provided")
        
        # Check analyses
        valid_analyses = {"critical_units", "tuning_profiles", "interventions", "regularization_sweep"}
        invalid = set(analyses) - valid_analyses
        if invalid:
            raise ValidationError(f"Invalid analyses: {invalid}")
        
        # Check model configurations exist
        for model_id in model_ids:
            try:
                self.config.get_model_checkpoint(model_id)
            except ValueError as e:
                raise ValidationError(f"Model configuration error: {e}")
        
        return True
    
    def _load_or_extract_states(self, model_id: str) -> StateData:
        """Load cached states or trigger extraction.
        
        Args:
            model_id: Model identifier
            
        Returns:
            StateData object
        """
        cache_key = f"states/{model_id}/states"
        
        try:
            # Try to load from cache
            return self.cache.load(cache_key)
        except KeyError:
            # States not cached, need to extract
            self.logger.info(f"States not cached for {model_id}, extraction required")
            
            # Import extraction pipeline to avoid circular imports
            from .extraction import ExtractionPipeline
            
            # Create extraction pipeline and run
            extraction_pipeline = ExtractionPipeline(self.config, self.cache)
            result = extraction_pipeline.run([model_id])
            
            if not result.success:
                raise DataNotFoundError(
                    f"Failed to extract states for {model_id}: {result.error}"
                )
            
            # Return the extracted states
            return result.data[model_id]
    
    def _identify_critical_units(self, model_id: str, states: StateData) -> Dict:
        """Identify critical units for a model.
        
        Args:
            model_id: Model identifier
            states: Extracted states
            
        Returns:
            Critical units analysis results
        """
        cache_key = f"analysis/{model_id}/critical_units"
        
        # Try cache first
        try:
            cached = self.cache.load(cache_key)
            self.logger.info(f"Loaded cached critical units for {model_id}")
            return cached
        except KeyError:
            pass
        
        # Import the critical units analysis function and constants
        from ..analysis.critical_units import identify_critical_units
        from ..core.constants import HZ_L1_RATIO, HZ_C, HZ_MAX_ITER
        from scipy import stats as scipy_stats
        
        # Prepare states matching original implementation
        # 1. Concatenate hidden and cell states
        if states.cells is not None:
            combined_states = np.concatenate([states.hiddens, states.cells], axis=-1)
        else:
            combined_states = states.hiddens
        
        # 2. Z-score across trials and timesteps (matching original)
        states_zscore = scipy_stats.zscore(combined_states, axis=(0, 1), ddof=1)
        
        # 3. Take only the final timestep
        final_states = states_zscore[:, -1, :]
        
        # Get labels from metadata or derive from task
        # For now, use a simple hazard rate detection (High vs Low)
        # In real implementation, this would come from trial metadata
        if hasattr(states, 'metadata') and 'hazard_rates' in states.metadata:
            # Binary classification: High=1, Low=0
            labels = (states.metadata['hazard_rates'] == 'High').astype(int)
        else:
            # Fallback: use predictions
            labels = np.argmax(states.predictions[:, -1, :], axis=-1)
        
        # Identify critical units with parameters matching original
        results = identify_critical_units(
            final_states.reshape(final_states.shape[0], 1, final_states.shape[1]),  # Add time dimension
            labels,
            C=self.config.get("analysis.critical_units.C", HZ_C),
            l1_ratio=self.config.get("analysis.critical_units.l1_ratio", HZ_L1_RATIO),
            max_iter=self.config.get("analysis.critical_units.max_iter", HZ_MAX_ITER),
            timestep=0,  # We already extracted final timestep
            zscore_states=False  # Already z-scored
        )
        
        # Cache results
        self.cache.save(cache_key, results)
        
        return results
    
    def _load_or_compute_critical_units(self, model_id: str, states: StateData) -> Dict:
        """Load or compute critical units (helper method)."""
        cache_key = f"analysis/{model_id}/critical_units"
        
        return self.cache.load_or_compute(
            cache_key,
            lambda: self._identify_critical_units(model_id, states)
        )
    
    def _compute_tuning_profiles(
        self, 
        model_id: str, 
        states: StateData, 
        critical_units: Dict
    ) -> Dict:
        """Compute tuning profiles for critical units.
        
        Args:
            model_id: Model identifier
            states: Extracted states
            critical_units: Critical units analysis results
            
        Returns:
            Tuning profile results
        """
        cache_key = f"analysis/{model_id}/tuning_profiles"
        
        # Try cache first
        try:
            cached = self.cache.load(cache_key)
            self.logger.info(f"Loaded cached tuning profiles for {model_id}")
            return cached
        except KeyError:
            pass
        
        # Import tuning profiles analysis
        from ..analysis.tuning_profiles import compute_tuning_profiles
        
        # Compute tuning profiles
        results = compute_tuning_profiles(
            states=states,
            critical_units=critical_units,
            metadata=states.metadata
        )
        
        # Cache results
        self.cache.save(cache_key, results)
        
        return results
    
    def _run_interventions(
        self,
        model_id: str,
        states: StateData,
        critical_units: Dict,
        tuning_profiles: Dict
    ) -> Dict:
        """Run intervention experiments.
        
        Args:
            model_id: Model identifier
            states: Extracted states  
            critical_units: Critical units results
            tuning_profiles: Tuning profile results
            
        Returns:
            Intervention results
        """
        cache_key = f"analysis/{model_id}/interventions"
        
        # Try cache first
        try:
            cached = self.cache.load(cache_key)
            self.logger.info(f"Loaded cached interventions for {model_id}")
            return cached
        except KeyError:
            pass
        
        # Import interventions analysis
        from ..analysis.interventions import run_intervention_experiments
        
        # Get model checkpoint path
        checkpoint_path = self.config.get_model_checkpoint(model_id)
        
        # Run interventions
        results = run_intervention_experiments(
            model_id=model_id,
            checkpoint_path=checkpoint_path,
            critical_units=critical_units,
            config=self.config
        )
        
        # Cache results
        self.cache.save(cache_key, results)
        
        return results
    
    def _run_regularization_sweep(self, model_id: str, states: StateData) -> Dict:
        """Run regularization sweep matching original implementation.
        
        Args:
            model_id: Model identifier
            states: Extracted states
            
        Returns:
            Regularization sweep results
        """
        cache_key = f"analysis/{model_id}/regularization_sweep"
        
        # Try cache first
        try:
            cached = self.cache.load(cache_key)
            self.logger.info(f"Loaded cached regularization sweep for {model_id}")
            return cached
        except KeyError:
            pass
        
        # Import required functions
        from ..analysis.critical_units import (
            linear_regularization_pipeline, 
            reg_single,
            reg_multi
        )
        from ..core.constants import HZ_L1_RATIO, CONT_L1_RATIO
        from scipy import stats as scipy_stats
        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
        
        # Prepare states (same as critical units)
        if states.cells is not None:
            combined_states = np.concatenate([states.hiddens, states.cells], axis=-1)
        else:
            combined_states = states.hiddens
        
        # Z-score and take final timestep
        states_zscore = scipy_stats.zscore(combined_states, axis=(0, 1), ddof=1)
        X = states_zscore[:, -1, :]
        
        # Define C values to sweep (matching original)
        C_logspace = np.logspace(0, -6, 50)
        
        # Define metrics
        dict_metrics = {
            "accuracy": (accuracy_score, {}),
            "f1": (f1_score, {"average": None}),
            "confusion": (confusion_matrix, {})
        }
        
        # Results container
        sweep_results = {}
        
        # Run for hazard rate (binary)
        if hasattr(states, 'metadata') and 'hazard_rates' in states.metadata:
            y_hz = (states.metadata['hazard_rates'] == 'High').astype(int)
            
            coefs_hz, intercepts_hz, metrics_hz = linear_regularization_pipeline(
                X, y_hz,
                dict_metrics,
                reg_single,
                lambda reg, X: reg.predict(X),
                C_logspace,
                l1_ratio=HZ_L1_RATIO
            )
            
            sweep_results["hazard_rate"] = {
                "coefficients": [c.tolist() for c in coefs_hz],
                "intercepts": [i.tolist() for i in intercepts_hz],
                "metrics": metrics_hz,
                "C_values": C_logspace.tolist(),
                "l1_ratio": HZ_L1_RATIO
            }
        
        # Run for contingency (multi-class) if available
        if hasattr(states, 'metadata') and 'contingencies' in states.metadata:
            cont_map = {"Low": 0, "Medium": 1, "High": 2}
            y_cont = np.array([cont_map[c] for c in states.metadata['contingencies']])
            
            coefs_cont, intercepts_cont, metrics_cont = linear_regularization_pipeline(
                X, y_cont,
                dict_metrics,
                reg_multi,
                lambda reg, X: reg.predict(X),
                C_logspace,
                l1_ratio=CONT_L1_RATIO
            )
            
            sweep_results["contingency"] = {
                "coefficients": [c.tolist() for c in coefs_cont],
                "intercepts": [i.tolist() for i in intercepts_cont],
                "metrics": metrics_cont,
                "C_values": C_logspace.tolist(),
                "l1_ratio": CONT_L1_RATIO
            }
        
        # Cache and return
        self.cache.save(cache_key, sweep_results)
        return sweep_results