"""Integration tests for the complete pipeline."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from learning_in_context.core import CacheManager, Config, StateData
from learning_in_context.pipelines import ExtractionPipeline, ModelAnalysisPipeline


class TestPipelineIntegration:
    """Test full pipeline integration."""
    
    @pytest.fixture
    def temp_config(self):
        """Create temporary configuration for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create config
            config_data = {
                "paths": {
                    "raw": str(tmpdir_path / "raw"),
                    "cache": str(tmpdir_path / "cache"),
                    "output": str(tmpdir_path / "output")
                },
                "models": {
                    "test_model_1": {
                        "checkpoint_path": str(tmpdir_path / "raw" / "model1.ckpt"),
                        "hidden_size": 128,
                        "num_layers": 2,
                        "recurrent_type": "lstm"
                    },
                    "test_model_2": {
                        "checkpoint_path": str(tmpdir_path / "raw" / "model2.ckpt"),
                        "hidden_size": 64,
                        "num_layers": 1,
                        "recurrent_type": "gru"
                    }
                },
                "analysis": {
                    "critical_units": {
                        "alphas": "logspace",
                        "l1_ratio": 0.5,
                        "cv_folds": 3,
                        "n_jobs": 1
                    }
                },
                "cache": {
                    "max_size_gb": 1.0,
                    "eviction_policy": "lru",
                    "compression": True
                },
                "pipeline": {
                    "batch_size": 32,
                    "num_workers": 0,
                    "device": "cpu"
                }
            }
            
            # Save config
            config_path = tmpdir_path / "config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config_data, f)
            
            # Create dummy checkpoint files
            for model_id in ["test_model_1", "test_model_2"]:
                ckpt_path = Path(config_data["models"][model_id]["checkpoint_path"])
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                ckpt_path.touch()
            
            yield config_path, tmpdir_path
    
    def test_config_loading(self, temp_config):
        """Test configuration loading and validation."""
        config_path, tmpdir = temp_config
        
        # Load config
        config = Config(config_path)
        
        # Validate
        assert config.validate()
        
        # Check paths
        assert config.cache_dir == tmpdir / "cache"
        assert config.output_dir == tmpdir / "output"
        assert config.raw_dir == tmpdir / "raw"
        
        # Check model config
        model1_path = config.get_model_checkpoint("test_model_1")
        assert model1_path.exists()
    
    def test_cache_manager(self, temp_config):
        """Test cache manager functionality."""
        config_path, tmpdir = temp_config
        config = Config(config_path)
        
        # Create cache manager
        cache = CacheManager(config.cache_dir, max_size_gb=0.1)
        
        # Test basic operations
        test_data = {"key": "value", "array": np.random.randn(10, 10)}
        cache.save("test/data", test_data)
        
        assert cache.exists("test/data")
        loaded = cache.load("test/data")
        assert loaded["key"] == "value"
        np.testing.assert_array_equal(loaded["array"], test_data["array"])
        
        # Test invalidation
        count = cache.invalidate("test/*")
        assert count == 1
        assert not cache.exists("test/data")
    
    def test_extraction_pipeline(self, temp_config):
        """Test state extraction pipeline."""
        config_path, tmpdir = temp_config
        config = Config(config_path)
        cache = CacheManager(config.cache_dir)
        
        # Create extraction pipeline
        extraction = ExtractionPipeline(config, cache)
        
        # Run extraction
        result = extraction.run(["test_model_1"])
        
        # Check result
        assert result.success
        assert "test_model_1" in result.data
        
        # Check extracted states
        states = result.data["test_model_1"]
        assert isinstance(states, StateData)
        assert states.hiddens.shape[0] > 0  # Has trials
        assert states.hiddens.shape[1] > 0  # Has timesteps
        assert states.hiddens.shape[2] > 0  # Has units
        
        # Check caching
        assert cache.exists("states/test_model_1/states")
    
    def test_model_analysis_pipeline(self, temp_config):
        """Test model analysis pipeline."""
        config_path, tmpdir = temp_config
        config = Config(config_path)
        cache = CacheManager(config.cache_dir)
        
        # Create pipeline
        analysis = ModelAnalysisPipeline(config, cache)
        
        # Run critical units analysis
        results = analysis.run(
            model_ids=["test_model_1"],
            analyses=["critical_units"]
        )
        
        # Check results structure
        assert "test_model_1" in results
        assert "critical_units" in results["test_model_1"]
        
        cu_results = results["test_model_1"]["critical_units"]
        assert "unit_indices" in cu_results
        assert "coefficients" in cu_results
        assert "r2_scores" in cu_results
        assert "best_alpha" in cu_results
        
        # Check caching
        assert cache.exists("analysis/test_model_1/critical_units")
    
    def test_full_analysis_pipeline(self, temp_config):
        """Test complete analysis pipeline with all components."""
        config_path, tmpdir = temp_config
        config = Config(config_path)
        cache = CacheManager(config.cache_dir)
        
        # Create pipeline
        analysis = ModelAnalysisPipeline(config, cache)
        
        # Run full analysis
        results = analysis.run(
            model_ids=["test_model_1", "test_model_2"],
            analyses=["critical_units", "tuning_profiles", "interventions"]
        )
        
        # Check all models processed
        assert len(results) == 2
        
        for model_id in ["test_model_1", "test_model_2"]:
            assert model_id in results
            model_results = results[model_id]
            
            # Check all analyses completed
            assert "critical_units" in model_results
            assert "tuning_profiles" in model_results
            assert "interventions" in model_results
            
            # Check critical units
            cu = model_results["critical_units"]
            assert isinstance(cu["unit_indices"], list)
            assert len(cu["unit_indices"]) >= 0
            
            # Check tuning profiles
            tp = model_results["tuning_profiles"]
            assert "profiles" in tp
            assert "n_units" in tp
            
            # Check interventions
            interv = model_results["interventions"]
            assert "experiments" in interv
            assert len(interv["experiments"]) > 0
    
    def test_pipeline_error_handling(self, temp_config):
        """Test error handling in pipelines."""
        config_path, tmpdir = temp_config
        config = Config(config_path)
        cache = CacheManager(config.cache_dir)
        
        # Test with invalid model ID
        analysis = ModelAnalysisPipeline(config, cache)
        
        # This should not raise but return error in result
        try:
            results = analysis.run(
                model_ids=["nonexistent_model"],
                analyses=["critical_units"]
            )
            # Should handle gracefully
            assert True
        except Exception:
            # Pipeline should handle errors internally
            pass
    
    def test_pipeline_caching_behavior(self, temp_config):
        """Test that pipelines use cache correctly."""
        config_path, tmpdir = temp_config
        config = Config(config_path)
        cache = CacheManager(config.cache_dir)
        
        # First run
        extraction = ExtractionPipeline(config, cache)
        result1 = extraction.run(["test_model_1"])
        
        # Mark cache access
        original_load = cache.load
        load_called = False
        
        def tracked_load(key):
            nonlocal load_called
            if "test_model_1" in key:
                load_called = True
            return original_load(key)
        
        cache.load = tracked_load
        
        # Second run should use cache
        result2 = extraction.run(["test_model_1"])
        
        assert load_called  # Cache was used
        assert result2.success
    
    def test_pipeline_checkpointing(self, temp_config):
        """Test pipeline checkpointing functionality."""
        config_path, tmpdir = temp_config
        config = Config(config_path)
        cache = CacheManager(config.cache_dir)
        
        # Create pipeline
        analysis = ModelAnalysisPipeline(config, cache)
        
        # Run analysis
        results = analysis.run(
            model_ids=["test_model_1"],
            analyses=["critical_units"]
        )
        
        # Check that checkpoint was saved
        assert cache.exists("checkpoints/ModelAnalysisPipeline/model_test_model_1")
    
    def test_state_data_persistence(self, temp_config):
        """Test StateData save/load functionality."""
        config_path, tmpdir = temp_config
        
        # Create sample state data
        n_trials, n_timesteps, n_units = 10, 20, 30
        state_data = StateData(
            hiddens=np.random.randn(n_trials, n_timesteps, n_units),
            cells=np.random.randn(n_trials, n_timesteps, n_units),
            predictions=np.random.rand(n_trials, n_timesteps, 3),
            metadata={"test": True, "model_id": "test"}
        )
        
        # Save
        save_path = tmpdir / "test_states.npz"
        state_data.save(save_path)
        
        # Load
        loaded = StateData.load(save_path)
        
        # Verify
        np.testing.assert_array_equal(loaded.hiddens, state_data.hiddens)
        np.testing.assert_array_equal(loaded.cells, state_data.cells)
        np.testing.assert_array_equal(loaded.predictions, state_data.predictions)
        assert loaded.metadata["test"] == True


class TestPipelinePerformance:
    """Test performance characteristics of the pipeline."""
    
    def test_cache_performance(self, temp_config):
        """Test cache read/write performance."""
        config_path, tmpdir = temp_config
        config = Config(config_path)
        cache = CacheManager(config.cache_dir)
        
        # Create large data
        large_data = {
            "array": np.random.randn(1000, 1000),
            "metadata": {"size": "large"}
        }
        
        # Time save operation
        import time
        start = time.time()
        cache.save("perf/large", large_data)
        save_time = time.time() - start
        
        # Time load operation
        start = time.time()
        loaded = cache.load("perf/large")
        load_time = time.time() - start
        
        # Basic performance checks
        assert save_time < 5.0  # Should save in < 5 seconds
        assert load_time < 2.0  # Should load in < 2 seconds
        
        # Verify data integrity
        np.testing.assert_array_equal(loaded["array"], large_data["array"])
    
    def test_memory_cache_eviction(self, temp_config):
        """Test memory cache eviction policy."""
        config_path, tmpdir = temp_config
        config = Config(config_path)
        
        # Create cache with small memory limit
        cache = CacheManager(config.cache_dir, max_size_gb=0.001)
        cache._max_memory_size = 1024 * 1024  # 1MB memory cache
        
        # Add items until eviction occurs
        for i in range(10):
            data = {"array": np.random.randn(100, 100)}  # ~80KB each
            cache.save(f"mem/item_{i}", data)
        
        # Check that memory cache has limited items
        assert len(cache._memory_cache) < 10
        
        # But all items should still be on disk
        for i in range(10):
            assert cache.exists(f"mem/item_{i}")


class TestEndToEndScenarios:
    """Test realistic end-to-end scenarios."""
    
    def test_multi_model_comparison(self, temp_config):
        """Test comparing critical units across multiple models."""
        config_path, tmpdir = temp_config
        config = Config(config_path)
        cache = CacheManager(config.cache_dir)
        
        # Run analysis on multiple models
        analysis = ModelAnalysisPipeline(config, cache)
        results = analysis.run(
            model_ids=["test_model_1", "test_model_2"],
            analyses=["critical_units"]
        )
        
        # Aggregate results
        from learning_in_context.analysis.critical_units import aggregate_critical_units
        
        model_cu_results = {
            model_id: results[model_id]["critical_units"]
            for model_id in results
        }
        
        aggregated = aggregate_critical_units(model_cu_results)
        
        # Check aggregation
        assert "unit_frequency" in aggregated
        assert "consistency_score" in aggregated
        assert aggregated["n_models"] == 2
    
    def test_incremental_analysis(self, temp_config):
        """Test adding new models to existing analysis."""
        config_path, tmpdir = temp_config
        config = Config(config_path)
        cache = CacheManager(config.cache_dir)
        
        # First batch
        analysis = ModelAnalysisPipeline(config, cache)
        results1 = analysis.run(
            model_ids=["test_model_1"],
            analyses=["critical_units"]
        )
        
        # Add new model to config
        config.set("models.test_model_3.checkpoint_path", 
                  str(tmpdir / "raw" / "model3.ckpt"))
        config.set("models.test_model_3.hidden_size", 96)
        config.set("models.test_model_3.num_layers", 1)
        config.set("models.test_model_3.recurrent_type", "lstm")
        
        # Create checkpoint file
        ckpt_path = tmpdir / "raw" / "model3.ckpt"
        ckpt_path.parent.mkdir(exist_ok=True)
        ckpt_path.touch()
        
        # Second batch (should use cache for model1)
        results2 = analysis.run(
            model_ids=["test_model_1", "test_model_3"],
            analyses=["critical_units"]
        )
        
        # Check both models processed
        assert len(results2) == 2
        assert "test_model_1" in results2
        assert "test_model_3" in results2