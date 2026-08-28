"""
Pytest configuration and shared fixtures.
"""

import pytest
import numpy as np
import torch
import tempfile
import subprocess
import sys
from pathlib import Path
import shutil

from learning_in_context.models.sequence_model import SequenceModel


def run_doit(*args, **kwargs):
    """Invoke the project's ``doit`` CLI hermetically for tests.

    Runs ``python -m doit <args>`` under the interpreter executing the test
    suite, so the venv's own pinned ``doit`` is used regardless of whether
    ``.venv/bin`` happens to be on ``PATH``. A bare ``subprocess.run(["doit",
    ...])`` relies on the ``doit`` console script being on ``PATH``, which does
    not hold under uv/``.venv`` layouts (it broke the port's doit tests while
    passing under iccpd's ambient conda ``doit``). Captures text output by
    default; extra keyword args (e.g. ``timeout``) pass through to
    ``subprocess.run``.
    """
    kwargs.setdefault("capture_output", True)
    kwargs.setdefault("text", True)
    return subprocess.run([sys.executable, "-m", "doit", *args], **kwargs)


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary test data directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir)
        
        # Create expected directory structure
        (test_dir / "data" / "weights" / "analyze" / "TEST-001").mkdir(parents=True)
        (test_dir / "data" / "raw").mkdir(parents=True)
        (test_dir / "outputs" / "cache").mkdir(parents=True)
        
        yield test_dir


@pytest.fixture
def sample_model():
    """Create a sample sequence model."""
    from learning_in_context.models.sequence_model import SequenceModelBase
    from learning_in_context.config.model_config import ModelConfig
    
    # Use the current model configuration system
    config = ModelConfig(
        input_size=5,
        output_size=3,
        recurrent_size=16,
        recurrent_num_layers=1,
        dropout=0.0
    )
    
    model = SequenceModelBase(**config.to_dict())
    model.eval()  # Set to evaluation mode for tests
    return model


@pytest.fixture
def sample_checkpoint(test_data_dir, sample_model):
    """Create a sample checkpoint file."""
    checkpoint_path = test_data_dir / "data" / "weights" / "analyze" / "TEST-001" / "last.ckpt"
    
    # Create checkpoint with expected structure
    checkpoint = {
        'state_dict': {
            f'sequence_model.{k}': v 
            for k, v in sample_model.state_dict().items()
        },
        'epoch': 100,
        'global_step': 10000,
    }
    
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def sample_data():
    """Create sample bouncing ball data."""
    # Shape: (n_trials, timesteps, features)
    # Features: [x, y, r, g, b]
    n_trials = 10
    timesteps = 100
    
    data = {
        'samples': np.random.randn(n_trials, timesteps, 5).astype(np.float32),
        'targets': np.random.randint(0, 3, size=(n_trials, timesteps)),
        'metadata': {
            'n_trials': n_trials,
            'timesteps': timesteps,
            'features': ['x', 'y', 'r', 'g', 'b']
        }
    }
    
    # Make colors valid (0-255) or -1 for grayzone
    for i in range(n_trials):
        for t in range(timesteps):
            if np.random.random() < 0.2:  # 20% chance of grayzone
                data['samples'][i, t, 2:5] = -1
            else:
                data['samples'][i, t, 2:5] = np.random.randint(0, 256, size=3)
    
    return data


@pytest.fixture
def sample_data_file(test_data_dir, sample_data):
    """Create a sample data file."""
    data_path = test_data_dir / "data" / "raw" / "test_dataset.npz"
    np.savez_compressed(data_path, **sample_data)
    return data_path


@pytest.fixture
def sample_states(sample_data):
    """Create sample extracted states."""
    n_trials = sample_data['samples'].shape[0]
    timesteps = sample_data['samples'].shape[1]
    hidden_dim = 16
    
    states = {
        'hiddens': np.random.randn(n_trials, timesteps, hidden_dim).astype(np.float32),
        'cells': np.random.randn(n_trials, timesteps, hidden_dim).astype(np.float32),
        'predictions': np.random.rand(n_trials, timesteps, 3).astype(np.float32),
    }
    
    # Normalize predictions to sum to 1
    states['predictions'] = states['predictions'] / states['predictions'].sum(axis=-1, keepdims=True)
    
    return states


@pytest.fixture
def sample_states_file(test_data_dir, sample_states):
    """Create a sample states file."""
    states_path = test_data_dir / "outputs" / "cache" / "model_states" / "TEST-001_states.npz"
    states_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        states_path,
        **sample_states,
        model_id='TEST-001',
        extraction_time=1234567890.0
    )
    return states_path


@pytest.fixture
def sample_trial_metadata():
    """Create sample trial metadata."""
    n_trials = 10
    
    metadata = {
        'trial_id': list(range(n_trials)),
        'trial_type': ['Straight'] * 5 + ['Bounce'] * 5,
        'hazard_rate': ['Low', 'High'] * 5,
        'contingency': ['Low', 'Medium', 'High', 'Low', 'Medium'] * 2,
        'correct_response': np.random.randint(0, 3, size=n_trials),
        'length': np.random.randint(400, 600, size=n_trials),
    }
    
    return metadata


@pytest.fixture
def mock_config(test_data_dir):
    """Create a mock configuration."""
    from omegaconf import OmegaConf
    
    config = OmegaConf.create({
        'data': {
            'base_dir': str(test_data_dir / 'data'),
            'weights_dir': str(test_data_dir / 'data' / 'weights' / 'analyze'),
            'raw_dir': str(test_data_dir / 'data' / 'raw'),
            'processed_dir': str(test_data_dir / 'data' / 'processed'),
            'participant_version': 'v3_2_2',
        },
        'pipeline': {
            'cache_dir': str(test_data_dir / 'outputs' / 'cache'),
            'figures_dir': str(test_data_dir / 'outputs' / 'figures'),
            'n_workers': 2,
            'batch_size': 4,
        },
        'model': {
            'hidden_dim': 16,
            'input_dim': 5,
            'output_dim': 3,
            'architecture': 'LSTM',
        },
        'analysis': {
            'regularization_alphas': [1.0, 0.1, 0.01],
            'intervention_alphas': 5,
            'event_window': [-5, 10],
            'last_n_timesteps': 10,
        },
    })
    
    return config


# ============================================================================
# END-TO-END TEST FIXTURES FOR CRITICAL UNITS PIPELINE
# ============================================================================

@pytest.fixture
def test_states_file():
    """Path to the test states file for critical units e2e tests."""
    return Path("tests/data/cache/model_states/TEST-005_states.npz")


@pytest.fixture
def test_model_id():
    """Test model ID that matches the test states file."""
    return "TEST-005"


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary directory for test outputs."""
    output_dir = tmp_path / "critical_units_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def test_states_data(test_states_file):
    """Load and return the test states data."""
    if not test_states_file.exists():
        pytest.skip(f"Test states file not found: {test_states_file}")
    
    return np.load(test_states_file, allow_pickle=True)


@pytest.fixture
def decoder_types():
    """List of all decoder types to test."""
    return ["hazard", "contingency", "color", "velocity_x", "velocity_y"]


@pytest.fixture
def expected_output_keys():
    """Expected keys in decoder output JSON."""
    return [
        "unit_indices", 
        "coefficients", 
        "r2_scores", 
        "best_alpha", 
        "cv_scores", 
        "metadata"
    ]


@pytest.fixture
def expected_metadata_keys():
    """Expected keys in metadata section."""
    return [
        "n_units_total",
        "n_units_critical", 
        "l1_ratio",
        "timestep",
        "exclude_low_variance",
        "unit_variance_threshold",
        "score_type",
        # New fields from pipeline update
        "use_all_timesteps",
        "concatenated_states",
        "n_samples"
    ]


@pytest.fixture
def expected_aggregated_keys():
    """Expected keys in aggregated output."""
    return [
        "model_id",
        "unit_indices",
        "coefficients", 
        "r2_scores",
        "best_alpha",
        "cv_scores",
        "metadata"
    ]


@pytest.fixture
def expected_aggregated_metadata_keys():
    """Expected keys in aggregated metadata."""
    return [
        "n_units_total",
        "n_units_critical",
        "decoder_results", 
        "aggregation_method",
        "n_decoders",
        "n_decoders_with_results",
        "total_critical_units_before_union",
        "overlap_ratio"
    ]


@pytest.fixture
def decoder_parameters():
    """Expected parameters for each decoder type."""
    return {
        "hazard": {"l1_ratio": 0.64, "problem_type": "binary"},
        "contingency": {"l1_ratio": 0.64, "problem_type": "binary"},
        "color": {"l1_ratio": 0.4, "problem_type": "multiclass"},
        "velocity_x": {"l1_ratio": 0.4, "problem_type": "regression"},
        "velocity_y": {"l1_ratio": 0.4, "problem_type": "regression"}
    }


def validate_decoder_output(output_file: Path, decoder_type: str) -> dict:
    """Validate decoder output file and return loaded data.
    
    Args:
        output_file: Path to decoder output JSON file
        decoder_type: Type of decoder
        
    Returns:
        Loaded and validated output data
        
    Raises:
        AssertionError: If validation fails
    """
    import json
    
    assert output_file.exists(), f"Output file not created: {output_file}"
    
    with open(output_file, 'r') as f:
        data = json.load(f)
    
    # Check required top-level keys
    required_keys = ["unit_indices", "coefficients", "r2_scores", "best_alpha", "cv_scores", "metadata"]
    for key in required_keys:
        assert key in data, f"Missing required key: {key}"
    
    # Check metadata
    metadata = data["metadata"]
    required_metadata = ["n_units_total", "n_units_critical", "score_type"]
    for key in required_metadata:
        assert key in metadata, f"Missing metadata key: {key}"
    
    # Check for new metadata fields (added in pipeline update)
    expected_new_fields = ["use_all_timesteps", "concatenated_states", "n_samples"]
    # These are optional for backward compatibility but log if present
    for key in expected_new_fields:
        if key in metadata:
            print(f"  ✓ Found new metadata field: {key} = {metadata[key]}")
    
    # Validate data types and ranges
    assert isinstance(data["unit_indices"], list), "unit_indices should be a list"
    assert isinstance(data["coefficients"], list), "coefficients should be a list"
    assert isinstance(data["best_alpha"], (int, float)), "best_alpha should be numeric"
    assert data["best_alpha"] > 0, "best_alpha should be positive"
    
    # Check that we have reasonable results
    assert metadata["n_units_total"] > 0, "Should have some total units"
    assert metadata["n_units_critical"] >= 0, "Critical units count should be non-negative"
    
    # Validate decoder-specific constraints
    if decoder_type in ["hazard", "contingency"]:
        # Binary classification decoders should use specific l1_ratio
        assert abs(metadata.get("l1_ratio", 0.64) - 0.64) < 0.01, f"Expected l1_ratio ~0.64 for {decoder_type}"
    elif decoder_type in ["color", "velocity_x", "velocity_y"]:
        # Other decoders should use different l1_ratio
        assert abs(metadata.get("l1_ratio", 0.4) - 0.4) < 0.01, f"Expected l1_ratio ~0.4 for {decoder_type}"
    
    return data


def validate_aggregated_output(output_file: Path, decoder_results: dict) -> dict:
    """Validate aggregated output file and return loaded data.
    
    Args:
        output_file: Path to aggregated output JSON file
        decoder_results: Individual decoder results for comparison
        
    Returns:
        Loaded and validated aggregated data
        
    Raises:
        AssertionError: If validation fails
    """
    import json
    
    assert output_file.exists(), f"Aggregated output file not created: {output_file}"
    
    with open(output_file, 'r') as f:
        data = json.load(f)
    
    # Check required top-level keys (backward compatibility)
    required_keys = ["model_id", "unit_indices", "coefficients", "r2_scores", "best_alpha", "cv_scores", "metadata"]
    for key in required_keys:
        assert key in data, f"Missing required key in aggregated output: {key}"
    
    # Check aggregated metadata
    metadata = data["metadata"]
    required_metadata = [
        "n_units_total", "n_units_critical", "decoder_results", 
        "aggregation_method", "n_decoders", "n_decoders_with_results"
    ]
    for key in required_metadata:
        assert key in metadata, f"Missing aggregated metadata key: {key}"
    
    # Validate aggregation logic
    assert metadata["aggregation_method"] == "union_of_decoders", "Should use union aggregation"
    assert metadata["n_decoders"] == len(decoder_results), "Should track all decoders"
    assert metadata["n_decoders_with_results"] <= len(decoder_results), "Can't have more results than decoders"
    
    # Check that aggregated units are indeed the union
    all_individual_units = set()
    for decoder_data in decoder_results.values():
        all_individual_units.update(decoder_data["unit_indices"])
    
    aggregated_units = set(data["unit_indices"])
    assert aggregated_units == all_individual_units, "Aggregated units should be union of individual results"
    
    # Validate overlap statistics
    if "overlap_ratio" in metadata:
        total_before_union = sum(len(d["unit_indices"]) for d in decoder_results.values())
        expected_overlap = 1 - len(aggregated_units) / total_before_union if total_before_union > 0 else 0
        assert abs(metadata["overlap_ratio"] - expected_overlap) < 0.01, "Overlap ratio calculation incorrect"
    
    return data


@pytest.fixture
def validate_decoder_output_func():
    """Return the decoder output validation function."""
    return validate_decoder_output


@pytest.fixture  
def validate_aggregated_output_func():
    """Return the aggregated output validation function."""
    return validate_aggregated_output


# ============================================================================
# MULTI-DATASET TEST FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def multi_dataset_configs():
    """Multi-dataset configuration fixture."""
    from learning_in_context.core.constants import DATASET_CONFIGS
    return DATASET_CONFIGS


@pytest.fixture
def mock_dataset_files(tmp_path, multi_dataset_configs):
    """Create mock dataset files for all datasets."""
    import numpy as np
    
    dataset_files = {}
    
    for dataset_name, config in multi_dataset_configs.items():
        # Create mock dataset with appropriate size
        size = min(config['size'], 10)  # Limit size for testing
        
        # Create mock data
        mock_data = {
            'samples': np.random.randn(size, 100, 5).astype(np.float32),
            'targets': np.random.randint(0, 3, size=(size, 100)),
            'metadata': {
                'n_trials': size,
                'timesteps': 100,
                'features': ['x', 'y', 'r', 'g', 'b']
            }
        }
        
        # Save to temporary file
        dataset_dir = tmp_path / "datasets" / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_file = dataset_dir / "dataset.npz"
        
        np.savez_compressed(dataset_file, **mock_data)
        dataset_files[dataset_name] = dataset_file
    
    return dataset_files


@pytest.fixture
def dataset_suffix_samples():
    """Sample data with different dataset suffixes for testing."""
    return {
        'participant': {'suffix': 'participant', 'size': 168},
        'extended': {'suffix': 'extended', 'size': 18000},
        'controlled': {'suffix': 'controlled', 'size': 147},
        'velocity': {'suffix': 'velocity', 'size': 161}
    }


@pytest.fixture
def mock_decoder_results_all_datasets(tmp_path, multi_dataset_configs):
    """Create mock decoder results for all datasets and decoders."""
    import json
    import random
    
    # Create cache directory structure
    cache_dir = tmp_path / "cache"
    critical_units_dir = cache_dir / "critical_units"
    critical_units_dir.mkdir(parents=True)
    
    model_id = "TEST-MULTI"
    decoder_types = ["hazard", "contingency", "color", "velocity_x", "velocity_y"]
    
    results = {}
    
    for dataset_name, config in multi_dataset_configs.items():
        suffix = config['suffix']
        results[dataset_name] = {}
        
        for decoder_type in decoder_types:
            # Create mock result
            random.seed(42 + hash(f"{dataset_name}_{decoder_type}"))
            
            result = {
                "decoder_type": decoder_type,
                "model_id": model_id,
                "dataset_name": dataset_name,
                "unit_indices": random.sample(range(128), 8),
                "coefficients": [random.uniform(-2, 2) for _ in range(8)],
                "best_alpha": random.uniform(0.001, 0.1),
                "r2_scores": [random.uniform(0.6, 0.9)],
                "metadata": {
                    "n_units_total": 128,
                    "n_units_critical": 8,
                    "decoder_type": decoder_type,
                    "dataset_name": dataset_name,
                    "l1_ratio": 0.64 if decoder_type in ["hazard", "contingency"] else 0.4
                }
            }
            
            # Save file with dataset suffix
            file_path = critical_units_dir / f"{model_id}_{suffix}_{decoder_type}_units.json"
            with open(file_path, "w") as f:
                json.dump(result, f, indent=2)
            
            results[dataset_name][decoder_type] = result
    
    return results, cache_dir