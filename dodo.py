"""
Doit automation for the research pipeline.

SIMPLIFIED WORKFLOWS (Recommended):
    doit extract                        # Extract states (alias for extract_model_states)
    doit identify                       # Run critical units analysis on all available models (auto-detects)
    doit full_pipeline                  # Complete workflow: extraction + analysis
    doit extract_quick                  # Quick extraction for development (first model only)

BASIC USAGE:
    doit list                          # List all available tasks
    doit                              # Run all tasks
    doit forget                       # Clear task database
    doit clean                        # Clean generated files
    doit graph                        # Visualize task dependencies

MULTI-DATASET WORKFLOWS:
    # Simple multi-dataset extraction and analysis
    doit full_pipeline datasets=participant,extended
    
    # Individual steps
    doit extract datasets=participant,extended,controlled
    doit identify datasets=participant,extended,controlled

DETAILED MULTI-DATASET COMMANDS:
    doit extract_model_states                                    # Extract with default dataset (participant)
    doit extract_model_states datasets=participant,extended     # Extract multiple datasets
    doit extract_model_states datasets=extended                 # Extract specific dataset
    doit extract_model_states:SAN-4378:participant             # Extract specific model-dataset
    doit extract_states_all_datasets                           # Extract all model-dataset combinations
    doit extract_states_group datasets=participant,extended    # Extract with optional normalization
    
MODEL CONFIGURATION:
    doit extract recurrent_size=32                 # Override recurrent size
    doit extract recurrent_layers=2                # Override number of layers
    doit extract datasets=extended recurrent_size=32  # Combine dataset and config options
    
    # Model selection options:
    doit identify models=all                       # Use all models with weights
    doit identify models=available                 # Use models with extracted states (default for identify)
    doit identify models=SAN-4566,SAN-4567        # Explicit model list

DEVELOPMENT & TESTING:
    doit extract_quick                             # Quick extraction (first model, participant dataset)
    doit extract weights_dir=tests/data/weights/analyze  # Use test weights
    doit extract models=SAN-4378,SAN-4401          # Extract only specified models
"""

import os
import sys
from pathlib import Path
from doit import get_var
from doit.tools import config_changed, run_once

# Configuration
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'  # Now inside In-Context-CPD
# Allow override of weights directory for testing
WEIGHTS_DIR = Path(get_var('weights_dir', default=str(DATA_DIR / 'weights' / 'analyze')))
RAW_DIR = Path(get_var('raw_dir', default='../data/raw'))  # Still outside for participant data
PROCESSED_DIR = DATA_DIR / 'processed'
CACHE_DIR = DATA_DIR / 'cache'  # Changed from outputs/cache to data/cache
FIGURES_DIR = PROJECT_ROOT / 'outputs' / 'figures'

# Ensure directories exist
for dir_path in [CACHE_DIR, FIGURES_DIR, PROCESSED_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

def get_available_model_states(dataset_suffix='participant', normalized=True):
    """Auto-detect model IDs that have extracted states available.
    
    Args:
        dataset_suffix: Dataset suffix to check for (e.g., 'participant', 'extended')
        normalized: Whether to check for normalized states (if False, checks raw states)
        
    Returns:
        Sorted list of model IDs that have extracted states
    """
    states_dir = CACHE_DIR / 'model_states'
    if not states_dir.exists():
        return []
    
    # Pattern: {model_id}_{dataset_suffix}_states[_normalized].npz
    suffix = f"_{dataset_suffix}_states{'_normalized' if normalized else ''}.npz"
    
    model_ids = set()
    for states_file in states_dir.iterdir():
        if states_file.is_file() and states_file.name.endswith(suffix):
            # Extract model ID from filename
            model_id = states_file.name.replace(suffix, '')
            if model_id.startswith('SAN-'):
                model_ids.add(model_id)
    
    return sorted(list(model_ids))

# Function to get model IDs based on selection mode
def get_model_ids():
    """Get model IDs based on the models parameter."""
    # For identify tasks, default to 'available' mode to auto-detect models with states
    # For extract tasks, keep the original hardcoded default
    models_param = get_var('models', default='available')
    if models_param == 'all':
        # Auto-detect all model directories in WEIGHTS_DIR
        model_ids = sorted([d.name for d in WEIGHTS_DIR.iterdir() if d.is_dir() and d.name.startswith('SAN-')])
        print(f"Auto-detected {len(model_ids)} models from weights: {', '.join(model_ids)}")
        return model_ids
    elif models_param == 'available':
        # Auto-detect models with available extracted states
        from learning_in_context.core.constants import DEFAULT_DATASET
        dataset_suffix = DEFAULT_DATASET if 'datasets' not in locals() else get_var('datasets', default=DEFAULT_DATASET).split(',')[0].strip()
        use_normalized = get_var('use_normalized', default='true').lower() == 'true'
        model_ids = get_available_model_states(dataset_suffix, normalized=use_normalized)
        if not model_ids:
            # Fallback to weights-based detection if no states found
            model_ids = sorted([d.name for d in WEIGHTS_DIR.iterdir() if d.is_dir() and d.name.startswith('SAN-')])
            print(f"No extracted states found, falling back to weights detection: {', '.join(model_ids)}")
        else:
            print(f"Auto-detected {len(model_ids)} models from available states: {', '.join(model_ids)}")
        return model_ids
    else:
        return models_param.split(',')

# Model and participant configurations
MODEL_IDS = get_model_ids()
PARTICIPANT_VERSION = get_var('participants', default='v3_2_2')

# Python interpreter from conda environment
PYTHON = sys.executable

# Default doit configuration
DOIT_CONFIG = {
    'default_tasks': ['pipeline'],
    'verbosity': 2,
    'num_process': 0,  # Sequential execution for GPU safety
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_critical_units_config(model_id, dataset_name, dataset_config, decoder_type, use_normalized=True):
    """Helper function to get configuration for critical units tasks.
    
    Returns:
        dict with states_file, output_file, task_deps, cmd_parts, and figure_file
    """
    # Choose state file based on use_normalized flag
    if use_normalized:
        states_file = CACHE_DIR / 'model_states' / f'{model_id}_{dataset_config["suffix"]}_states_normalized.npz'
        task_deps = [f'normalize_states:{model_id}:{dataset_name}']
    else:
        states_file = CACHE_DIR / 'model_states' / f'{model_id}_{dataset_config["suffix"]}_states.npz'
        task_deps = [f'extract_model_states:{model_id}:{dataset_name}']
    
    output_file = CACHE_DIR / 'critical_units' / f'{model_id}_{dataset_config["suffix"]}_{decoder_type}_units.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Figure output path
    figure_dir = FIGURES_DIR / 'models' / model_id
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_file = figure_dir / f'{decoder_type}_coefficient_analysis.png'
    
    # Get regularization parameters from doit variables
    lambda_min = get_var('lambda_min', default='1e-6')
    lambda_max = get_var('lambda_max', default='1.0')
    n_lambdas = get_var('n_lambdas', default='50')
    lambda_spacing = get_var('lambda_spacing', default='log')
    
    # Get L1 ratio parameters
    binary_l1_ratio = get_var('binary_l1_ratio', default='0.64')
    multiclass_l1_ratio = get_var('multiclass_l1_ratio', default='0.4')
    
    # Get max iteration parameters
    binary_max_iter = get_var('binary_max_iter', default='250')
    multiclass_max_iter = get_var('multiclass_max_iter', default='250')
    
    # Get threshold detection parameters
    chance_margin = get_var('chance_margin', default='0.05')
    threshold_method = get_var('threshold_method', default='chance')
    
    # Build command
    cmd_parts = [
        f'{PYTHON} -m learning_in_context.analysis.critical_units',
        f'--states {states_file}',
        f'--output {output_file}',
        f'--model-id {model_id}',
        f'--decoder-type {decoder_type}',
        f'--save-visualization-data',
        f'--task-variable {decoder_type}',
        # Add new regularization parameters
        f'--lambda-min {lambda_min}',
        f'--lambda-max {lambda_max}',
        f'--n-lambdas {n_lambdas}',
        f'--lambda-spacing {lambda_spacing}',
        f'--binary-l1-ratio {binary_l1_ratio}',
        f'--multiclass-l1-ratio {multiclass_l1_ratio}',
        f'--binary-max-iter {binary_max_iter}',
        f'--multiclass-max-iter {multiclass_max_iter}',
        f'--chance-margin {chance_margin}',
        f'--threshold-method {threshold_method}'
    ]
    
    # Add normalization flag
    if use_normalized:
        cmd_parts.append('--use-normalized')
    
    return {
        'states_file': states_file,
        'output_file': output_file,
        'figure_file': figure_file,
        'task_deps': task_deps,
        'cmd_parts': cmd_parts
    }

def get_model_checkpoint(model_id, checkpoint_name=None):
    """Find checkpoint file for a model."""
    model_dir = WEIGHTS_DIR / model_id
    
    if checkpoint_name:
        # Use specific checkpoint name
        checkpoint_path = model_dir / checkpoint_name
        if checkpoint_path.exists():
            return checkpoint_path
        # Try with .ckpt extension if not already present
        if not checkpoint_name.endswith('.ckpt'):
            checkpoint_path = model_dir / f"{checkpoint_name}.ckpt"
            if checkpoint_path.exists():
                return checkpoint_path
    else:
        # Default to last.ckpt
        checkpoint_path = model_dir / 'last.ckpt'
        if checkpoint_path.exists():
            return checkpoint_path
            
    # Fallback to searching for any .ckpt file
    if model_dir.exists():
        checkpoints = list(model_dir.glob('*.ckpt'))
        if checkpoints:
            return checkpoints[0]
    return None

def get_participant_data_dir():
    """Get participant data directory."""
    return RAW_DIR / f'hbb_participant_responses_{PARTICIPANT_VERSION}'

# ============================================================================
# MODEL ANALYSIS TASKS
# ============================================================================

def task_extract_model_states():
    """Extract hidden and cell states from trained models across multiple datasets.
    
    INPUTS:
        - Model checkpoints: data/weights/analyze/{model_id}/last.ckpt
        - Dataset configs: Defined in learning_in_context.core.constants.DATASET_CONFIGS
    
    OUTPUTS:
        - Raw states: data/cache/model_states/{model_id}_{dataset}_states.npz
        - Contains: hiddens, cells, predictions, metadata, dataset info
    
    CONFIG OPTIONS:
        models=SAN-4566,SAN-4567      # Specific models (default: all available)
        datasets=participant,extended  # Multiple datasets (default: participant)
        batch_size=32                 # Processing batch size (default: dataset size)
        device=cuda/cpu               # Device selection (default: cuda if available)
        cpu=true                      # Force CPU processing
        recurrent_size=64             # Override model hidden size
        recurrent_layers=2            # Override number of recurrent layers
        checkpoint_name=best.ckpt     # Use specific checkpoint (default: last.ckpt)
    
    EXAMPLES:
        doit extract_model_states                              # All models, participant dataset
        doit extract_model_states models=SAN-4566             # Single model
        doit extract_model_states datasets=participant,extended # Multiple datasets
        doit extract_model_states:SAN-4566:participant        # Specific model+dataset
        doit extract_model_states batch_size=32 cpu=true      # Custom config
    """
    # Import dataset configs
    from learning_in_context.core.constants import DATASET_CONFIGS, DEFAULT_DATASET
    
    # Get configuration from doit variables
    recurrent_size = get_var('recurrent_size', default=None)
    recurrent_layers = get_var('recurrent_layers', default=None) 
    datasets = get_var('datasets', default=DEFAULT_DATASET).split(',')
    checkpoint_name = get_var('checkpoint_name', default=None)
    user_batch_size = get_var('batch_size', default=None)
    
    # GPU configuration
    gpu_devices = get_var('gpus', default='0')  # Default to GPU 0, can be comma-separated
    use_cpu = get_var('cpu', default='false').lower() == 'true'
    
    for model_id in MODEL_IDS:
        checkpoint = get_model_checkpoint(model_id, checkpoint_name)
        if not checkpoint:
            continue
            
        for dataset_name in datasets:
            dataset_name = dataset_name.strip()
            if dataset_name not in DATASET_CONFIGS:
                print(f"Warning: Unknown dataset '{dataset_name}', skipping")
                continue
                
            dataset_config = DATASET_CONFIGS[dataset_name]
            
            # Resolve dataset path relative to PROJECT_ROOT if it's a relative path
            dataset_path = Path(dataset_config["path"])
            if not dataset_path.is_absolute():
                dataset_path = (PROJECT_ROOT / dataset_path).resolve()
            
            # Dataset-specific output file
            output_file = CACHE_DIR / 'model_states' / f'{model_id}_{dataset_config["suffix"]}_states.npz'
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Build command with dataset support
            cmd_parts = [
                f'{PYTHON} -m learning_in_context.models.extract_states',
                f'--checkpoint {checkpoint}',
                f'--output {output_file}',
                f'--model-id {model_id}',
                f'--dataset-path {dataset_path}',
                f'--dataset-name {dataset_name}'
            ]
            
            # Add model configuration overrides if specified
            if recurrent_size:
                cmd_parts.append(f'--recurrent-size {recurrent_size}')
            if recurrent_layers:
                cmd_parts.append(f'--recurrent-layers {recurrent_layers}')
            
            # Add batch size configuration
            if user_batch_size is not None:
                cmd_parts.append(f'--batch-size {user_batch_size}')
            else:
                # Default: use dataset size as batch size (load entire dataset)
                dataset_size = dataset_config.get('size', 10000)  # Fallback to large number
                cmd_parts.append(f'--batch-size {dataset_size}')
            
            # Add device configuration
            if use_cpu:
                cmd_parts.append('--device cpu')
            else:
                # For multi-GPU, assign GPUs round-robin based on model index
                gpu_list = [g.strip() for g in gpu_devices.split(',')]
                if len(gpu_list) == 1:
                    # Single GPU specified, all tasks use it
                    cmd_parts.append(f'--device cuda:{gpu_list[0]}')
                else:
                    # Multiple GPUs: distribute tasks across them
                    try:
                        model_index = MODEL_IDS.index(model_id)
                        assigned_gpu = gpu_list[model_index % len(gpu_list)]
                        cmd_parts.append(f'--device cuda:{assigned_gpu}')
                    except ValueError:
                        # Fallback to first GPU if model not in list
                        cmd_parts.append(f'--device cuda:{gpu_list[0]}')
            
            # Create task name with dataset and config info
            task_name = f'{model_id}:{dataset_name}'
            if recurrent_size or recurrent_layers:
                config_info = []
                if recurrent_size:
                    config_info.append(f'r{recurrent_size}')
                if recurrent_layers:
                    config_info.append(f'l{recurrent_layers}')
                task_name = f"{model_id}:{dataset_name}_{'_'.join(config_info)}"
            
            yield {
                'name': task_name,
                'actions': [' '.join(cmd_parts)],
                'file_dep': [str(checkpoint)],
                'targets': [str(output_file)],
                'clean': True,
                'verbosity': 2,
                'doc': f'Extract states for {model_id} on {dataset_config["description"]}'
            }


def task_extract_states_all_datasets():
    """Meta-task to extract states across all configured datasets."""
    from learning_in_context.core.constants import DATASET_CONFIGS, DEFAULT_DATASET
    
    datasets = get_var('datasets', default=DEFAULT_DATASET).split(',')
    subtasks = []
    
    for model_id in MODEL_IDS:
        for dataset_name in datasets:
            dataset_name = dataset_name.strip()
            if dataset_name in DATASET_CONFIGS:
                subtasks.append(f'extract_model_states:{model_id}:{dataset_name}')
    
    return {
        'actions': None,  # No actions, just coordinates subtasks
        'task_dep': subtasks,
        'verbosity': 2,
        'doc': f'Extract states for all models across datasets: {", ".join(datasets)}'
    }


def task_normalize_states():
    """Normalize extracted states using z-score normalization.
    
    INPUTS:
        - Raw states: data/cache/model_states/{model_id}_{dataset}_states.npz
        - Requires: Completed extract_model_states task
    
    OUTPUTS:
        - Normalized states: data/cache/model_states/{model_id}_{dataset}_states_normalized.npz
        - Contains: Original data + normalization_stats, normalization_method, etc.
    
    CONFIG OPTIONS:
        normalization_method=zscore    # Normalization method (zscore, minmax, none)
        normalization_axis=1           # Axis to normalize over (1=time, 0=trials)
        per_unit_normalization=true    # Normalize each unit separately
        padding_value=-100             # Value for padded timesteps (default: -100)
        datasets=participant,extended   # Multiple datasets (default: participant)
    
    EXAMPLES:
        doit normalize_states                                    # All models, default config
        doit normalize_states normalization_method=minmax       # Use min-max normalization
        doit normalize_states:SAN-4566:participant              # Specific model+dataset
        doit normalize_states per_unit_normalization=false      # Global normalization
    """
    from learning_in_context.core.constants import DATASET_CONFIGS, DEFAULT_DATASET
    
    # Configuration options
    normalization_config = {
        'method': get_var('normalization_method', default='zscore'),
        'axis': get_var('normalization_axis', default=1),  # Time dimension
        'per_unit': get_var('per_unit_normalization', default=True),
        'padding_value': get_var('padding_value', default=-100.0),
    }
    
    # Get dataset configuration
    datasets = get_var('datasets', default=DEFAULT_DATASET).split(',')
    
    # Track if we yielded any tasks
    yielded_tasks = False
    
    for model_id in MODEL_IDS:
        for dataset_name in datasets:
            dataset_name = dataset_name.strip()
            if dataset_name not in DATASET_CONFIGS:
                continue
                
            dataset_config = DATASET_CONFIGS[dataset_name]
            suffix = dataset_config['suffix']
            
            raw_states_file = CACHE_DIR / 'model_states' / f'{model_id}_{suffix}_states.npz'
            normalized_file = CACHE_DIR / 'model_states' / f'{model_id}_{suffix}_states_normalized.npz'
            
            # Skip if input file doesn't exist (will be created by extract_model_states)
            # This is OK because we have task_dep that ensures it runs after extraction
            
            # Build command
            cmd_parts = [
                PYTHON, '-m', 'learning_in_context.models.normalize_states',
                str(raw_states_file),
                str(normalized_file),
                '--method', normalization_config['method'],
                '--axis', str(normalization_config['axis']),
                '--padding-value', str(normalization_config['padding_value']),
            ]
            
            if normalization_config['per_unit']:
                cmd_parts.append('--per-unit')
            else:
                cmd_parts.append('--no-per-unit')
            
            yielded_tasks = True
            yield {
                'name': f'{model_id}:{dataset_name}',
                'actions': [' '.join(cmd_parts)],
                'file_dep': [str(raw_states_file)],
                'targets': [str(normalized_file)],
                'task_dep': [f'extract_model_states:{model_id}:{dataset_name}'],
                'clean': True,
                'verbosity': 2,
                'doc': f'Normalize states for {model_id} on {dataset_name} dataset',
            }
    
    # If no tasks were yielded, yield a placeholder task
    if not yielded_tasks:
        yield {
            'name': None,
            'actions': ['echo "No models or datasets configured for normalization"'],
            'verbosity': 2,
        }


def task_extract_states_group():
    """Complete state extraction pipeline: extraction + optional normalization.
    
    WORKFLOW:
        1. Extract raw states for all models and datasets
        2. Optionally normalize states (if normalize=true)
        3. Outputs ready for downstream analysis
    
    INPUTS:
        - Model checkpoints: data/weights/analyze/{model_id}/last.ckpt
        - Dataset configurations: From learning_in_context.core.constants.DATASET_CONFIGS
    
    OUTPUTS:
        - Raw states: data/cache/model_states/{model_id}_{dataset}_states.npz
        - Normalized states: data/cache/model_states/{model_id}_{dataset}_states_normalized.npz (if normalize=true)
    
    CONFIG OPTIONS:
        normalize=true                # Include normalization step (default: false)
        models=SAN-4566,SAN-4567     # Specific models (default: all available)
        datasets=participant,extended # Multiple datasets (default: participant)
        batch_size=32                # Processing batch size
        cpu=true                     # Force CPU processing
        padding_value=-100            # Value for padded timesteps (default: -100)
        [All extract_model_states and normalize_states options available]
    
    EXAMPLES:
        doit extract_states_group                                # Extract only (no normalization)
        doit extract_states_group normalize=true                 # Extract + normalize
        doit extract_states_group models=SAN-4566 normalize=true # Single model + normalize
        doit extract_states_group datasets=participant,extended  # Multiple datasets
    """
    from learning_in_context.core.constants import DATASET_CONFIGS, DEFAULT_DATASET
    
    # Get configuration from doit variables  
    datasets = get_var('datasets', default=DEFAULT_DATASET).split(',')
    include_normalization = get_var('normalize', default='false').lower() == 'true'
    
    # Build list of all required subtasks
    subtasks = []
    
    # Add extraction tasks for all model-dataset combinations
    for model_id in MODEL_IDS:
        for dataset_name in datasets:
            dataset_name = dataset_name.strip()
            if dataset_name in DATASET_CONFIGS:
                subtasks.append(f'extract_model_states:{model_id}:{dataset_name}')
    
    # Optionally add normalization tasks (now multi-dataset aware)
    if include_normalization:
        for model_id in MODEL_IDS:
            for dataset_name in datasets:
                dataset_name = dataset_name.strip()
                if dataset_name in DATASET_CONFIGS:
                    subtasks.append(f'normalize_states:{model_id}:{dataset_name}')
    
    return {
        'actions': None,  # No actions, just depends on subtasks
        'task_dep': subtasks,
        'verbosity': 2,
        'doc': f'Extract states for all models across datasets: {", ".join(datasets)}' + 
               (' (with normalization)' if include_normalization else ''),
    }


def task_test_extract_model_states():
    """Extract states from test models with test weights directory."""
    test_weights_dir = PROJECT_ROOT / 'tests' / 'data' / 'weights' / 'analyze'
    test_models = ['TEST-001', 'TEST-002']
    
    for model_id in test_models:
        checkpoint = test_weights_dir / model_id / 'last.ckpt'
        if not checkpoint.exists():
            continue
            
        output_file = CACHE_DIR / 'test_model_states' / f'{model_id}_states.npz'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Build command for test model
        cmd_parts = [
            f'{PYTHON} -m learning_in_context.models.extract_states',
            f'--checkpoint {checkpoint}',
            f'--output {output_file}',
            f'--model-id {model_id}'
        ]
        
        yield {
            'name': model_id,
            'actions': [' '.join(cmd_parts)],
            'file_dep': [str(checkpoint)],
            'targets': [str(output_file)],
            'clean': True,
            'verbosity': 2,
        }


def task_test_normalize_states():
    """Normalize extracted states for test models."""
    test_models = ['TEST-001', 'TEST-002']
    
    for model_id in test_models:
        raw_states_file = CACHE_DIR / 'test_model_states' / f'{model_id}_states.npz'
        normalized_file = CACHE_DIR / 'test_model_states' / f'{model_id}_states_normalized.npz'
        
        # Build command
        cmd_parts = [
            PYTHON, '-m', 'learning_in_context.models.normalize_states',
            str(raw_states_file),
            str(normalized_file),
            '--method', 'zscore',
            '--axis', '1',
            '--padding-value', '-100',
        ]
        
        yield {
            'name': model_id,
            'actions': [' '.join(cmd_parts)],
            'file_dep': [str(raw_states_file)],
            'targets': [str(normalized_file)],
            'task_dep': [f'test_extract_model_states:{model_id}'],
            'clean': True,
            'verbosity': 2,
        }


def task_test_extract_states_group():
    """Test extract_states_group with test weights and models."""
    test_models = ['TEST-001', 'TEST-002']
    test_weights_dir = PROJECT_ROOT / 'tests' / 'data' / 'weights' / 'analyze'
    
    # Build list of all subtasks using test models
    subtasks = []
    for model_id in test_models:
        # Check if test checkpoint exists
        checkpoint = test_weights_dir / model_id / 'last.ckpt'
        if not checkpoint.exists():
            continue
            
        # Add extraction task
        subtasks.append(f'test_extract_model_states:{model_id}')
        # Add normalization task  
        subtasks.append(f'test_normalize_states:{model_id}')
    
    if not subtasks:
        # Return a task that just prints a warning
        return {
            'actions': ['echo "No test checkpoints found in tests/data/weights/analyze/"'],
            'verbosity': 2,
        }
    
    return {
        'actions': None,  # No actions, just depends on subtasks
        'task_dep': subtasks,
        'verbosity': 2,
        'doc': 'Extract and normalize states for test models only',
    }


def task_validate_test_setup():
    """Validate that test setup is correct for extract_states_group testing."""
    test_weights_dir = PROJECT_ROOT / 'tests' / 'data' / 'weights' / 'analyze'
    test_models = ['TEST-001', 'TEST-002']
    
    def check_test_setup():
        """Check if test weights and models are available."""
        missing = []
        for model_id in test_models:
            checkpoint = test_weights_dir / model_id / 'last.ckpt'
            if not checkpoint.exists():
                missing.append(str(checkpoint))
        
        if missing:
            print("Missing test checkpoints:")
            for path in missing:
                print(f"  - {path}")
            return False
        else:
            print("Test setup validation: PASSED")
            print(f"Found test checkpoints for: {', '.join(test_models)}")
            return True
    
    return {
        'actions': [check_test_setup],
        'verbosity': 2,
        'doc': 'Validate test setup for extract_states_group testing',
    }


def task_critical_units_hazard():
    """Identify critical units for hazard rate decoding (binary classification, α=0.64)."""
    from learning_in_context.core.constants import DATASET_CONFIGS, DEFAULT_DATASET
    
    datasets = get_var('datasets', default=DEFAULT_DATASET).split(',')
    use_normalized = get_var('use_normalized', default='true').lower() == 'true'
    
    for model_id in MODEL_IDS:
        for dataset_name in datasets:
            dataset_name = dataset_name.strip()
            if dataset_name not in DATASET_CONFIGS:
                continue
                
            dataset_config = DATASET_CONFIGS[dataset_name]
            config = get_critical_units_config(model_id, dataset_name, dataset_config, 'hazard', use_normalized)
            
            yield {
                'name': f'{model_id}:{dataset_name}',
                'actions': [' '.join(config['cmd_parts'])],
                'file_dep': [str(config['states_file'])],
                'targets': [str(config['output_file'])],
                'task_dep': config['task_deps'],
                'verbosity': 2,
                'doc': f'Hazard decoding for {model_id} on {dataset_config["description"]}'
            }


def task_critical_units_contingency():
    """Identify critical units for contingency decoding (binary classification, α=0.64)."""
    from learning_in_context.core.constants import DATASET_CONFIGS, DEFAULT_DATASET
    
    datasets = get_var('datasets', default=DEFAULT_DATASET).split(',')
    use_normalized = get_var('use_normalized', default='true').lower() == 'true'
    
    for model_id in MODEL_IDS:
        for dataset_name in datasets:
            dataset_name = dataset_name.strip()
            if dataset_name not in DATASET_CONFIGS:
                continue
                
            dataset_config = DATASET_CONFIGS[dataset_name]
            config = get_critical_units_config(model_id, dataset_name, dataset_config, 'contingency', use_normalized)
            
            yield {
                'name': f'{model_id}:{dataset_name}',
                'actions': [' '.join(config['cmd_parts'])],
                'file_dep': [str(config['states_file'])],
                'targets': [str(config['output_file'])],
                'task_dep': config['task_deps'],
                'verbosity': 2,
                'doc': f'Contingency decoding for {model_id} on {dataset_config["description"]}'
            }


def task_critical_units_color():
    """Identify critical units for color decoding (multiclass, α=0.4)."""
    from learning_in_context.core.constants import DATASET_CONFIGS, DEFAULT_DATASET
    
    datasets = get_var('datasets', default=DEFAULT_DATASET).split(',')
    use_normalized = get_var('use_normalized', default='true').lower() == 'true'
    
    for model_id in MODEL_IDS:
        for dataset_name in datasets:
            dataset_name = dataset_name.strip()
            if dataset_name not in DATASET_CONFIGS:
                continue
                
            dataset_config = DATASET_CONFIGS[dataset_name]
            config = get_critical_units_config(model_id, dataset_name, dataset_config, 'color', use_normalized)
            
            yield {
                'name': f'{model_id}:{dataset_name}',
                'actions': [' '.join(config['cmd_parts'])],
                'file_dep': [str(config['states_file'])],
                'targets': [str(config['output_file'])],
                'task_dep': config['task_deps'],
                'verbosity': 2,
                'doc': f'Color decoding for {model_id} on {dataset_config["description"]}'
            }


def task_critical_units_velocity_x():
    """Identify critical units for velocity_x decoding (regression, α=0.4)."""
    from learning_in_context.core.constants import DATASET_CONFIGS, DEFAULT_DATASET
    
    datasets = get_var('datasets', default=DEFAULT_DATASET).split(',')
    use_normalized = get_var('use_normalized', default='true').lower() == 'true'
    
    for model_id in MODEL_IDS:
        for dataset_name in datasets:
            dataset_name = dataset_name.strip()
            if dataset_name not in DATASET_CONFIGS:
                continue
                
            dataset_config = DATASET_CONFIGS[dataset_name]
            config = get_critical_units_config(model_id, dataset_name, dataset_config, 'velocity_x', use_normalized)
            
            yield {
                'name': f'{model_id}:{dataset_name}',
                'actions': [' '.join(config['cmd_parts'])],
                'file_dep': [str(config['states_file'])],
                'targets': [str(config['output_file'])],
                'task_dep': config['task_deps'],
                'verbosity': 2,
                'doc': f'Velocity X decoding for {model_id} on {dataset_config["description"]}'
            }


def task_critical_units_velocity_y():
    """Identify critical units for velocity_y decoding (regression, α=0.4)."""
    from learning_in_context.core.constants import DATASET_CONFIGS, DEFAULT_DATASET
    
    datasets = get_var('datasets', default=DEFAULT_DATASET).split(',')
    use_normalized = get_var('use_normalized', default='true').lower() == 'true'
    
    for model_id in MODEL_IDS:
        for dataset_name in datasets:
            dataset_name = dataset_name.strip()
            if dataset_name not in DATASET_CONFIGS:
                continue
                
            dataset_config = DATASET_CONFIGS[dataset_name]
            config = get_critical_units_config(model_id, dataset_name, dataset_config, 'velocity_y', use_normalized)
            
            yield {
                'name': f'{model_id}:{dataset_name}',
                'actions': [' '.join(config['cmd_parts'])],
                'file_dep': [str(config['states_file'])],
                'targets': [str(config['output_file'])],
                'task_dep': config['task_deps'],
                'verbosity': 2,
                'doc': f'Velocity Y decoding for {model_id} on {dataset_config["description"]}'
            }


def task_aggregate_critical_units():
    """Aggregate results from all decoder types into unified critical units file."""
    from learning_in_context.core.constants import DATASET_CONFIGS, DEFAULT_DATASET
    
    datasets = get_var('datasets', default=DEFAULT_DATASET).split(',')
    run_all_decoders = get_var('run_all_decoders', default='false').lower() == 'true'
    
    # Determine which decoders to aggregate
    if run_all_decoders:
        decoders = ['hazard', 'contingency', 'color', 'velocity_x', 'velocity_y']
    else:
        decoders = ['hazard', 'contingency']  # Default: only hazard and contingency
    
    for model_id in MODEL_IDS:
        for dataset_name in datasets:
            dataset_name = dataset_name.strip()
            if dataset_name not in DATASET_CONFIGS:
                continue
                
            dataset_config = DATASET_CONFIGS[dataset_name]
            suffix = dataset_config["suffix"]
            
            # Input files from individual decoders for this dataset
            decoder_files = []
            task_deps = []
            
            for decoder in decoders:
                decoder_file = CACHE_DIR / 'critical_units' / f'{model_id}_{suffix}_{decoder}_units.json'
                decoder_files.append(decoder_file)
                task_deps.append(f'critical_units_{decoder}:{model_id}:{dataset_name}')
            
            # Output unified file for this dataset
            output_file = CACHE_DIR / 'critical_units' / f'{model_id}_{suffix}_units.json'
            
            # Use separate aggregation script
            aggregation_script = PROJECT_ROOT / 'scripts' / 'aggregate_critical_units.py'
            
            # Build command with decoder list
            cmd = [
                f'{PYTHON} {aggregation_script}',
                f'--cache-dir {CACHE_DIR}',
                f'--model-id {model_id}',
                f'--dataset-suffix {suffix}',
                f'--output {output_file}',
                f'--decoders {",".join(decoders)}'
            ]
            
            yield {
                'name': f'{model_id}:{dataset_name}',
                'actions': [' '.join(cmd)],
                'file_dep': [str(f) for f in decoder_files] + [str(aggregation_script)],
                'targets': [str(output_file)],
                'task_dep': task_deps,
                'verbosity': 2,
                'doc': f'Aggregate critical units for {model_id} on {dataset_config["description"]}'
            }


def task_identify_critical_units():
    """Meta-task that runs all critical units identification sub-tasks."""
    from learning_in_context.core.constants import DATASET_CONFIGS, DEFAULT_DATASET
    
    datasets = get_var('datasets', default=DEFAULT_DATASET).split(',')
    run_all_decoders = get_var('run_all_decoders', default='false').lower() == 'true'
    
    # Determine which decoders to run
    if run_all_decoders:
        decoders = ['hazard', 'contingency', 'color', 'velocity_x', 'velocity_y']
    else:
        decoders = ['hazard', 'contingency']  # Default: only hazard and contingency
    
    for model_id in MODEL_IDS:
        for dataset_name in datasets:
            dataset_name = dataset_name.strip()
            if dataset_name not in DATASET_CONFIGS:
                continue
                
            dataset_config = DATASET_CONFIGS[dataset_name]
            yield {
                'name': f'{model_id}:{dataset_name}',
                'actions': None,  # No actions, just coordinates dependencies
                'task_dep': [f'aggregate_critical_units:{model_id}:{dataset_name}'],
                'verbosity': 2,
                'doc': f'Complete critical units identification for {model_id} on {dataset_config["description"]}',
            }


# ============================================================================
# TUNING PROFILE ANALYSIS - DECOMPOSED DAG STRUCTURE
# ============================================================================

def task_extract_unit_activities():
    """Extract activities for specific critical units (DAG node)."""
    from learning_in_context.core.constants import DATASET_CONFIGS, DEFAULT_DATASET
    
    # Get configuration options
    window_size = get_var('window_size', default=200)
    normalize = get_var('normalize', default='true').lower() == 'true'
    output_format = get_var('output_format', default='both')
    datasets = get_var('datasets', default=DEFAULT_DATASET).split(',')
    
    for model_id in MODEL_IDS:
        for dataset_name in datasets:
            dataset_name = dataset_name.strip()
            if dataset_name not in DATASET_CONFIGS:
                continue
                
            dataset_config = DATASET_CONFIGS[dataset_name]
            suffix = dataset_config["suffix"]
            
            states_file = CACHE_DIR / 'model_states' / f'{model_id}_{suffix}_states.npz'
            units_file = CACHE_DIR / 'critical_units' / f'{model_id}_{suffix}_units.json'
            output_file = CACHE_DIR / 'tuning_profiles' / f'{model_id}_{suffix}_unit_activities.npz'
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Build command
            cmd_parts = [
                f'{PYTHON} -m learning_in_context.analysis.extract_unit_activities',
                f'--states {states_file}',
                f'--units {units_file}',
                f'--output {output_file}',
                f'--model-id {model_id}',
                f'--window-size {window_size}',
                f'--output-format {output_format}'
            ]
            
            if normalize:
                cmd_parts.append('--normalize')
            
            yield {
                'name': f'{model_id}:{dataset_name}',
                'actions': [' '.join(cmd_parts)],
                'file_dep': [str(states_file), str(units_file)],
                'targets': [str(output_file)],
                'task_dep': [f'identify_critical_units:{model_id}:{dataset_name}'],
                'verbosity': 2,
                'doc': f'Extract unit activities for {model_id} on {dataset_config["description"]}',
            }


def task_compute_activity_matrix():
    """Compute activity matrices for whole dataset characterization (DAG node)."""
    from learning_in_context.core.constants import DATASET_CONFIGS, DEFAULT_DATASET
    
    use_windowed = get_var('use_windowed', default='true').lower() == 'true'
    use_normalized = get_var('use_normalized', default='true').lower() == 'true'
    datasets = get_var('datasets', default=DEFAULT_DATASET).split(',')
    
    for model_id in MODEL_IDS:
        for dataset_name in datasets:
            dataset_name = dataset_name.strip()
            if dataset_name not in DATASET_CONFIGS:
                continue
                
            dataset_config = DATASET_CONFIGS[dataset_name]
            suffix = dataset_config["suffix"]
            
            activities_file = CACHE_DIR / 'tuning_profiles' / f'{model_id}_{suffix}_unit_activities.npz'
            output_file = CACHE_DIR / 'tuning_profiles' / f'{model_id}_{suffix}_activity_matrix.npz'
            
            # Build command
            cmd_parts = [
                f'{PYTHON} -m learning_in_context.analysis.compute_activity_matrix',
                f'--activities {activities_file}',
                f'--output {output_file}',
                f'--model-id {model_id}'
            ]
            
            if use_windowed:
                cmd_parts.append('--use-windowed')
            if use_normalized:
                cmd_parts.append('--use-normalized')
            
            yield {
                'name': f'{model_id}:{dataset_name}',
                'actions': [' '.join(cmd_parts)],
                'file_dep': [str(activities_file)],
                'targets': [str(output_file)],
                'task_dep': [f'extract_unit_activities:{model_id}:{dataset_name}'],
                'verbosity': 2,
                'doc': f'Compute activity matrix for {model_id} on {dataset_config["description"]}',
            }


def task_sort_by_condition():
    """Sort trials by experimental conditions (DAG node)."""
    from learning_in_context.core.constants import DATASET_CONFIGS, DEFAULT_DATASET
    
    sort_conditions = get_var('sort_conditions', default='hazard_rate,trial_type').split(',')
    datasets = get_var('datasets', default=DEFAULT_DATASET).split(',')
    
    for model_id in MODEL_IDS:
        for dataset_name in datasets:
            dataset_name = dataset_name.strip()
            if dataset_name not in DATASET_CONFIGS:
                continue
                
            dataset_config = DATASET_CONFIGS[dataset_name]
            suffix = dataset_config["suffix"]
            
            matrices_file = CACHE_DIR / 'tuning_profiles' / f'{model_id}_{suffix}_activity_matrix.npz'
            output_file = CACHE_DIR / 'tuning_profiles' / f'{model_id}_{suffix}_sorted_conditions.npz'
            
            # Build command
            cmd_parts = [
                f'{PYTHON} -m learning_in_context.analysis.sort_by_condition',
                f'--matrices {matrices_file}',
                f'--output {output_file}',
                f'--model-id {model_id}',
                f'--conditions'
            ] + sort_conditions
            
            yield {
                'name': f'{model_id}:{dataset_name}',
                'actions': [' '.join(cmd_parts)],
                'file_dep': [str(matrices_file)],
                'targets': [str(output_file)],
                'task_dep': [f'compute_activity_matrix:{model_id}:{dataset_name}'],
                'verbosity': 2,
                'doc': f'Sort conditions for {model_id} on {dataset_config["description"]}',
            }


def task_align_trajectories():
    """Align trajectories for controlled stimulus analysis (DAG node)."""
    from learning_in_context.core.constants import DATASET_CONFIGS, DEFAULT_DATASET
    
    alignment_method = get_var('alignment_method', default='time')
    datasets = get_var('datasets', default=DEFAULT_DATASET).split(',')
    
    for model_id in MODEL_IDS:
        for dataset_name in datasets:
            dataset_name = dataset_name.strip()
            if dataset_name not in DATASET_CONFIGS:
                continue
                
            dataset_config = DATASET_CONFIGS[dataset_name]
            suffix = dataset_config["suffix"]
            
            activities_file = CACHE_DIR / 'tuning_profiles' / f'{model_id}_{suffix}_unit_activities.npz'
            output_file = CACHE_DIR / 'tuning_profiles' / f'{model_id}_{suffix}_aligned_trajectories.json'
            
            # Build command
            cmd_parts = [
                f'{PYTHON} -m learning_in_context.analysis.align_trajectories',
                f'--activities {activities_file}',
                f'--output {output_file}',
                f'--model-id {model_id}',
                f'--alignment-method {alignment_method}'
            ]
            
            yield {
                'name': f'{model_id}:{dataset_name}',
                'actions': [' '.join(cmd_parts)],
                'file_dep': [str(activities_file)],
                'targets': [str(output_file)],
                'task_dep': [f'extract_unit_activities:{model_id}:{dataset_name}'],
                'verbosity': 2,
                'doc': f'Align trajectories for {model_id} on {dataset_config["description"]}',
            }


def task_event_analysis_group():
    """Perform event-triggered analysis for temporal dynamics (DAG node)."""
    from learning_in_context.core.constants import DATASET_CONFIGS, DEFAULT_DATASET
    
    event_window = get_var('event_window', default='-5,10').split(',')
    event_types = get_var('event_types', default='color_change,velocity_change').split(',')
    min_events = get_var('min_events', default=5)
    datasets = get_var('datasets', default=DEFAULT_DATASET).split(',')
    
    for model_id in MODEL_IDS:
        for dataset_name in datasets:
            dataset_name = dataset_name.strip()
            if dataset_name not in DATASET_CONFIGS:
                continue
                
            dataset_config = DATASET_CONFIGS[dataset_name]
            suffix = dataset_config["suffix"]
            
            activities_file = CACHE_DIR / 'tuning_profiles' / f'{model_id}_{suffix}_unit_activities.npz'
            output_file = CACHE_DIR / 'tuning_profiles' / f'{model_id}_{suffix}_event_analysis.npz'
            
            # Build command
            cmd_parts = [
                f'{PYTHON} -m learning_in_context.analysis.event_analysis_group',
                f'--activities {activities_file}',
                f'--output {output_file}',
                f'--model-id {model_id}',
                f'--event-window {event_window[0]} {event_window[1]}',
                f'--event-types'
            ] + event_types + [
                f'--min-events {min_events}'
            ]
            
            yield {
                'name': f'{model_id}:{dataset_name}',
                'actions': [' '.join(cmd_parts)],
                'file_dep': [str(activities_file)],
                'targets': [str(output_file)],
                'task_dep': [f'extract_unit_activities:{model_id}:{dataset_name}'],
                'verbosity': 2,
                'doc': f'Event analysis for {model_id} on {dataset_config["description"]}',
            }


def task_tuning_profile_analysis():
    """Meta-task for complete tuning profile analysis (DAG node)."""
    from learning_in_context.core.constants import DATASET_CONFIGS, DEFAULT_DATASET
    
    datasets = get_var('datasets', default=DEFAULT_DATASET).split(',')
    
    for model_id in MODEL_IDS:
        for dataset_name in datasets:
            dataset_name = dataset_name.strip()
            if dataset_name not in DATASET_CONFIGS:
                continue
                
            dataset_config = DATASET_CONFIGS[dataset_name]
            
            yield {
                'name': f'{model_id}:{dataset_name}',
                'actions': None,  # No actions, just coordinates dependencies
                'task_dep': [
                    f'extract_unit_activities:{model_id}:{dataset_name}',
                    f'compute_activity_matrix:{model_id}:{dataset_name}',
                    f'sort_by_condition:{model_id}:{dataset_name}',
                    f'align_trajectories:{model_id}:{dataset_name}',
                    f'event_analysis_group:{model_id}:{dataset_name}'
                ],
                'verbosity': 2,
                'doc': f'Complete tuning profile analysis for {model_id} on {dataset_config["description"]}',
            }


# ============================================================================
# LEGACY COMPATIBILITY TASKS
# ============================================================================

def task_compute_tuning_profiles():
    """LEGACY: Compute neural tuning profiles for critical units.
    
    This task is maintained for backward compatibility but now depends on
    the decomposed DAG structure above. For multi-dataset support, use 
    the individual DAG tasks instead.
    """
    from learning_in_context.core.constants import DATASET_CONFIGS, DEFAULT_DATASET
    
    # Allow configuration via get_var, default to participant for backward compatibility
    dataset_name = get_var('legacy_dataset', default=DEFAULT_DATASET)
    if dataset_name not in DATASET_CONFIGS:
        print(f"Warning: Unknown dataset '{dataset_name}', using default '{DEFAULT_DATASET}'")
        dataset_name = DEFAULT_DATASET
    
    dataset_config = DATASET_CONFIGS[dataset_name]
    suffix = dataset_config["suffix"]
    
    for model_id in MODEL_IDS:
        # Create a compatibility output file that aggregates all components
        output_file = CACHE_DIR / 'tuning_profiles' / f'{model_id}_tuning.json'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Dependencies from new DAG structure (using default dataset)
        activities_file = CACHE_DIR / 'tuning_profiles' / f'{model_id}_{suffix}_unit_activities.npz'
        matrix_file = CACHE_DIR / 'tuning_profiles' / f'{model_id}_{suffix}_activity_matrix.npz'
        sorted_file = CACHE_DIR / 'tuning_profiles' / f'{model_id}_{suffix}_sorted_conditions.npz'
        aligned_file = CACHE_DIR / 'tuning_profiles' / f'{model_id}_{suffix}_aligned_trajectories.json'
        events_file = CACHE_DIR / 'tuning_profiles' / f'{model_id}_{suffix}_event_analysis.npz'
        
        # Use the aggregation script
        aggregation_script = PROJECT_ROOT / 'scripts' / 'aggregate_tuning_profiles.py'
        
        yield {
            'name': model_id,
            'actions': [
                f'{PYTHON} {aggregation_script} '
                f'--cache-dir {CACHE_DIR} '
                f'--model-id {model_id} '
                f'--dataset-suffix {suffix} '
                f'--output {output_file}'
            ],
            'file_dep': [
                str(activities_file), str(matrix_file), str(sorted_file),
                str(aligned_file), str(events_file), str(aggregation_script)
            ],
            'targets': [str(output_file)],
            'task_dep': [f'tuning_profile_analysis:{model_id}:{dataset_name}'],
            'verbosity': 2,
        }

def task_visualize_tuning_profiles():
    """Create visualizations for neural tuning profiles."""
    from learning_in_context.core.constants import DATASET_CONFIGS
    
    # Default to participant dataset for visualization
    dataset_name = 'participant'
    dataset_config = DATASET_CONFIGS[dataset_name]
    suffix = dataset_config['suffix']
    
    for model_id in MODEL_IDS:
        states_file = CACHE_DIR / 'model_states' / f'{model_id}_{suffix}_states.npz'
        tuning_file = CACHE_DIR / 'tuning_profiles' / f'{model_id}_tuning.json'
        output_dir = FIGURES_DIR / 'tuning_profiles' / model_id
        
        # Python code to create visualizations
        viz_code = f"""
import json
import numpy as np
from pathlib import Path
from learning_in_context.visualization.tuning_plots import create_tuning_profile_report

# Load data
with open('{tuning_file}', 'r') as f:
    tuning_results = json.load(f)

# Load states if available for full visualizations
states_data = np.load('{states_file}')
if 'df_data' in states_data:
    import pandas as pd
    df_data = pd.DataFrame(states_data['df_data'])
else:
    df_data = None

# Create visualizations
create_tuning_profile_report(
    tuning_results,
    Path('{output_dir}'),
    states_data=states_data.get('windowed_states_normalized', None),
    df_data=df_data
)
"""
        
        yield {
            'name': model_id,
            'actions': [f'{PYTHON} -c "{viz_code}"'],
            'file_dep': [str(states_file), str(tuning_file)],
            'targets': [str(output_dir / 'temporal_dynamics.pdf')],
            'task_dep': [f'compute_tuning_profiles:{model_id}'],
        }


# ============================================================================

def task_test_extract():
    """Test state extraction with a single model and small batch."""
    
    def run_test_extraction():
        """Run a quick test extraction."""
        from learning_in_context.models.extract_states import extract_states_with_config
        
        # Use first available model for testing
        test_model = MODEL_IDS[0] if MODEL_IDS else "SAN-4566"
        checkpoint = get_model_checkpoint(test_model)
        
        if not checkpoint:
            print(f"No checkpoint found for test model {test_model}")
            return False
        
        output_file = CACHE_DIR / 'test_extraction' / f'{test_model}_test.npz'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            metadata = extract_states_with_config(
                checkpoint_path=str(checkpoint),
                output_path=str(output_file),
                model_id=test_model,
                batch_size=1,  # Small batch for testing
                num_batches=1,  # Only process one batch
                device='auto'
            )
            print(f"✓ Test extraction completed successfully for {test_model}")
            print(f"✓ Output saved to: {output_file}")
            print(f"✓ Model config: {metadata['model_config']}")
            return True
        except Exception as e:
            print(f"✗ Test extraction failed: {e}")
            return False
    
    return {
        'actions': [run_test_extraction],
        'verbosity': 2,
        'uptodate': [False],  # Always run when requested
    }


def task_validate_config():
    """Validate that model configurations and dependencies are properly set up."""
    
    def validate_setup():
        """Validate the pipeline setup."""
        success = True
        
        print("🔍 Validating In-Context-CPD pipeline setup...")
        
        # Check if models can be imported
        try:
            from learning_in_context.models.sequence_model import SequenceModelBase, SequenceModel
            from learning_in_context.datamodules.bouncing_ball import BouncingBallDataModule
            from learning_in_context.config.model_config import ModelConfig
            print("✓ All required modules can be imported")
        except ImportError as e:
            print(f"✗ Import error: {e}")
            success = False
        
        # Check model checkpoints
        available_models = []
        for model_id in MODEL_IDS:
            checkpoint = get_model_checkpoint(model_id)
            if checkpoint:
                available_models.append(model_id)
                print(f"✓ Found checkpoint for {model_id}: {checkpoint}")
            else:
                print(f"✗ No checkpoint found for {model_id}")
        
        if not available_models:
            print("✗ No model checkpoints found")
            success = False
        else:
            print(f"✓ Found {len(available_models)} model checkpoints")
        
        # Check directories
        for dir_name, dir_path in [
            ("Cache", CACHE_DIR),
            ("Figures", FIGURES_DIR),
            ("Processed", PROCESSED_DIR),
        ]:
            if dir_path.exists():
                print(f"✓ {dir_name} directory exists: {dir_path}")
            else:
                print(f"⚠ {dir_name} directory will be created: {dir_path}")
        
        # Test basic model config functionality
        try:
            from learning_in_context.config.model_config import get_model_config_for_id, override_model_config
            base_config = get_model_config_for_id("SAN-4378")
            overridden = override_model_config(base_config, {"recurrent_size": 32})
            assert overridden.recurrent_size == 32
            print("✓ Model configuration system working correctly")
        except Exception as e:
            print(f"✗ Model configuration test failed: {e}")
            success = False
        
        if success:
            print("\n🎉 Pipeline validation completed successfully!")
            print(f"Ready to extract states for models: {', '.join(available_models)}")
        else:
            print("\n❌ Pipeline validation found issues. Please check the errors above.")
        
        return success
    
    return {
        'actions': [validate_setup],
        'verbosity': 2,
        'uptodate': [False],  # Always run when requested
    }


# ============================================================================
# PARTICIPANT ANALYSIS TASKS
# ============================================================================



# ============================================================================

# ============================================================================

def task_generate_model_figures():
    """Generate figures for model analysis."""
    from learning_in_context.core.constants import DATASET_CONFIGS
    
    # Default to participant dataset for figures
    dataset_name = 'participant'
    dataset_config = DATASET_CONFIGS[dataset_name]
    suffix = dataset_config['suffix']
    
    for model_id in MODEL_IDS:
        figure_dir = FIGURES_DIR / 'models' / model_id
        figure_dir.mkdir(parents=True, exist_ok=True)
        
        deps = [
            CACHE_DIR / 'model_states' / f'{model_id}_{suffix}_states.npz',
            CACHE_DIR / 'critical_units' / f'{model_id}_{suffix}_critical_units.json',
            CACHE_DIR / 'tuning_profiles' / model_id / 'tuning_complete.flag',
        ]
        
        # Create list of figure targets including critical units plots
        critical_units_figures = [
            str(figure_dir / f'{decoder}_coefficient_analysis.png')
            for decoder in ['hazard', 'contingency', 'color', 'velocity_x', 'velocity_y']
        ]
        
        yield {
            'name': model_id,
            'actions': [
                # Generate all model figures including critical units
                f'{PYTHON} -m learning_in_context.visualization.model_figures '
                f'--model-id {model_id} '
                f'--cache-dir {CACHE_DIR} '
                f'--output-dir {figure_dir} '
                f'--dataset-suffix {suffix}'
            ],
            'file_dep': [str(d) for d in deps if d.exists()],
            'targets': critical_units_figures + [str(figure_dir / 'figures_complete.flag')],
            'task_dep': [f'compute_tuning_profiles:{model_id}'],
        }


def task_figures():
    """Generate figures with flexible filtering options.
    
    Examples:
        doit figures                                    # Generate all figures
        doit figures models=SAN-4566                    # Single model
        doit figures models=SAN-4566,SAN-4567          # Multiple models
        doit figures decoders=hazard,contingency       # Specific decoders
        doit figures analysis=critical_units           # Specific analysis type
        doit figures models=SAN-4566 decoders=hazard  # Combined filtering
    """
    from learning_in_context.core.constants import DATASET_CONFIGS, DEFAULT_DATASET
    
    # Get filtering parameters
    models_param = get_var('models', default='available')
    decoders_param = get_var('decoders', default='all')
    analysis_param = get_var('analysis', default='all')
    dataset_name = get_var('dataset', default=DEFAULT_DATASET)
    
    # Parse models
    if models_param == 'available':
        models = get_available_model_states(dataset_suffix=DATASET_CONFIGS[dataset_name]['suffix'])
    elif models_param == 'all':
        models = MODEL_IDS
    else:
        models = [m.strip() for m in models_param.split(',')]
    
    # Parse decoders
    if decoders_param == 'all':
        decoders = ['hazard', 'contingency', 'color', 'velocity_x', 'velocity_y']
    else:
        decoders = [d.strip() for d in decoders_param.split(',')]
    
    # Parse analysis types (for future extensibility)
    if analysis_param == 'all':
        analysis_types = ['critical_units']  # Add more as implemented
    else:
        analysis_types = [a.strip() for a in analysis_param.split(',')]
    
    if not models:
        return {
            'actions': ['echo "No models available for figure generation"'],
            'uptodate': [True]
        }
    
    # Generate task for each model
    for model_id in models:
        if model_id not in MODEL_IDS:
            continue
            
        figure_dir = FIGURES_DIR / 'models' / model_id
        dataset_config = DATASET_CONFIGS[dataset_name]
        suffix = dataset_config['suffix']
        
        # Build decoder arguments
        decoder_args = f"--decoders {' '.join(decoders)}" if decoders else ""
        
        yield {
            'name': model_id,
            'actions': [
                f'{PYTHON} -m learning_in_context.visualization.model_figures '
                f'--model-id {model_id} '
                f'--cache-dir {CACHE_DIR} '
                f'--output-dir {figure_dir} '
                f'--dataset-suffix {suffix} '
                f'{decoder_args}'
            ],
            'verbosity': 2,
            'uptodate': [False],  # Always regenerate when explicitly called
        }


# ============================================================================
# CONTROLLED DATASET SUPPORT
# ============================================================================

# ============================================================================

def task_model_analysis():
    """Run complete model analysis pipeline."""
    from learning_in_context.core.constants import DATASET_CONFIGS, DEFAULT_DATASET
    
    datasets = get_var('datasets', default=DEFAULT_DATASET).split(',')
    
    # Generate all model:dataset combinations for tuning profile analysis
    tuning_deps = []
    for model_id in MODEL_IDS:
        for dataset_name in datasets:
            dataset_name = dataset_name.strip()
            if dataset_name in DATASET_CONFIGS:
                tuning_deps.append(f'tuning_profile_analysis:{model_id}:{dataset_name}')
    
    return {
        'actions': None,
        'task_dep': tuning_deps + [
            f'generate_model_figures:{m}' for m in MODEL_IDS
        ],
    }


def task_pipeline():
    """Run the complete pipeline (first-pass scope: model side only)."""
    return {
        'actions': ['echo "Pipeline complete!"'],
        'task_dep': [
            'model_analysis',
        ],
    }

# ============================================================================
# UTILITY TASKS
# ============================================================================

def task_docs():
    """Show pipeline documentation and usage guides.
    
    USAGE:
        doit docs                    # Show available documentation
        doit docs pipeline=extract   # Show extract pipeline guide
    
    AVAILABLE DOCUMENTATION:
        - Extract Pipeline: docs/pipelines/extract.md
        - Quick Start: docs/quick-start.md
        - Testing Guide: TESTING.md
    
    HELP COMMANDS:
        doit help <task-name>    # Show task documentation
        doit info <task-name>    # Show task dependencies and targets
        doit list               # List all tasks with descriptions
        doit list | grep extract # Filter for extract tasks
    """
    pipeline = get_var('pipeline', default=None)
    
    def show_docs():
        """Display documentation based on pipeline parameter."""
        docs_dir = PROJECT_ROOT / 'docs'
        
        if pipeline == 'extract':
            extract_doc = docs_dir / 'pipelines' / 'extract.md'
            if extract_doc.exists():
                print(f"\n📖 Extract Pipeline Documentation")
                print(f"Location: {extract_doc}")
                print(f"View with: cat {extract_doc}")
                print(f"Or open in editor: nano {extract_doc}")
            else:
                print("❌ Extract pipeline documentation not found")
        else:
            print("\n📚 Available Documentation:")
            print(f"  📖 Extract Pipeline: {docs_dir}/pipelines/extract.md")
            print(f"  🚀 Quick Start: {docs_dir}/quick-start.md")
            print(f"  🧪 Testing Guide: {PROJECT_ROOT}/TESTING.md")
            print(f"\n💡 Usage:")
            print(f"  doit docs pipeline=extract    # Show extract pipeline guide")
            print(f"  doit help extract_model_states # Show task documentation")
            print(f"  doit info extract:SAN-4566     # Show task details")
            print(f"  doit list | grep extract       # List extract tasks")
        
        return True
    
    return {
        'actions': [show_docs],
        'verbosity': 2,
        'uptodate': [False],  # Always run when requested
    }

def task_setup_environment():
    """Ensure conda environment is properly configured."""
    return {
        'actions': [
            'conda env update -f environment.yaml -p ./.conda',
            'echo "Environment ready. Activate with: conda activate ./.conda"'
        ],
        'targets': ['.conda/conda-meta/history'],
        'uptodate': [config_changed(dict(env='learning_in_context'))],
    }

def task_validate_data():
    """Validate that required data files exist."""
    required_files = []
    
    # Check for model weights
    for model_id in MODEL_IDS:
        checkpoint = get_model_checkpoint(model_id)
        if checkpoint:
            required_files.append(checkpoint)
    
    # Check for participant data
    participant_dir = get_participant_data_dir()
    if participant_dir.exists():
        required_files.append(participant_dir / 'trial_meta.csv')
    
    def check_files():
        missing = [f for f in required_files if not f.exists()]
        if missing:
            print("Missing required files:")
            for f in missing:
                print(f"  - {f}")
            return False
        print("All required data files found!")
        return True
    
    return {
        'actions': [(check_files,)],
        'uptodate': [run_once],
    }


# ============================================================================
# SIMPLIFIED WORKFLOW TASKS
# ============================================================================

def task_extract_and_analyze():
    """Meta-task that runs complete extraction and analysis workflow."""
    from learning_in_context.core.constants import DATASET_CONFIGS, DEFAULT_DATASET
    
    # Get configuration from doit variables
    datasets = get_var('datasets', default=DEFAULT_DATASET).split(',')
    
    # Build list of all required subtasks
    subtasks = []
    
    # Add extraction tasks for all model-dataset combinations
    for model_id in MODEL_IDS:
        for dataset_name in datasets:
            dataset_name = dataset_name.strip()
            if dataset_name in DATASET_CONFIGS:
                subtasks.append(f'extract_model_states:{model_id}:{dataset_name}')
    
    # Add critical units identification tasks
    for model_id in MODEL_IDS:
        for dataset_name in datasets:
            dataset_name = dataset_name.strip()
            if dataset_name in DATASET_CONFIGS:
                subtasks.append(f'identify_critical_units:{model_id}:{dataset_name}')
    
    return {
        'actions': None,  # No actions, just coordinates subtasks
        'task_dep': subtasks,
        'verbosity': 2,
        'doc': f'Complete extraction and analysis workflow for datasets: {", ".join(datasets)}'
    }


def task_extract_quick():
    """Quick extraction for development and testing - first available model only.
    
    PURPOSE:
        Fast extraction for development, testing, and pipeline validation
    
    BEHAVIOR:
        - Uses first available model (typically alphabetically first)
        - Uses participant dataset only (smallest dataset, 168 trials)
        - Raw states only (no normalization for speed)
        - Ideal for testing pipeline setup and validating functionality
    
    INPUTS:
        - First available checkpoint: data/weights/analyze/{first_model}/last.ckpt
        - Participant dataset configuration
    
    OUTPUTS:
        - Raw states: data/cache/model_states/{first_model}_participant_states.npz
    
    EXAMPLES:
        doit extract_quick                    # Quick development extraction
        
    NOTE:
        If you need specific models or normalization, use 'doit extract' instead
    """
    # Use first available model for quick testing
    if not MODEL_IDS:
        return {
            'actions': ['echo "No models configured"'],
            'verbosity': 2,
            'doc': 'No models available for quick extraction'
        }
    
    test_model = MODEL_IDS[0]
    dataset_name = 'participant'
    
    # Check if checkpoint exists
    checkpoint = get_model_checkpoint(test_model)
    if not checkpoint:
        return {
            'actions': [f'echo "No checkpoint found for {test_model}"'],
            'verbosity': 2,
            'doc': f'Checkpoint missing for {test_model}'
        }
    
    # This is now a meta-task that depends on the regular extraction
    return {
        'actions': None,  # No direct actions, just depends on subtask
        'task_dep': [f'extract_model_states:{test_model}:{dataset_name}'],
        'verbosity': 2,
        'doc': f'Quick extraction for development using {test_model} on {dataset_name} dataset'
    }


# ============================================================================
# TASK ALIASES FOR BETTER USER EXPERIENCE
# ============================================================================

def task_extract():
    """Extract states with convenient defaults: all models + normalization.
    
    WORKFLOW:
        Runs extract_states_group with optimal defaults for most users
    
    DEFAULTS:
        models=all       # Use all models in weights directory
        normalize=true   # Include normalization step for downstream analysis
        datasets=participant  # Use participant dataset unless overridden
    
    OUTPUTS:
        - Raw states: data/cache/model_states/{model_id}_{dataset}_states.npz
        - Normalized states: data/cache/model_states/{model_id}_{dataset}_states_normalized.npz
    
    CONFIG OPTIONS:
        [All extract_states_group options available - will override defaults]
        models=SAN-4566,SAN-4567     # Override to specific models
        normalize=false              # Disable normalization
        datasets=extended            # Use different dataset
        batch_size=32               # Custom batch size
        cpu=true                    # Force CPU processing
    
    EXAMPLES:
        doit extract                              # All models, normalized, participant dataset
        doit extract models=SAN-4566            # Single model with defaults
        doit extract datasets=participant,extended # Multiple datasets
        doit extract normalize=false             # Raw states only
        doit extract batch_size=16 cpu=true     # Custom processing config
    """
    from doit import get_var
    
    # Get user-specified values or use convenient defaults
    user_models = get_var('models', default='all')  # Default to 'all' for extract alias
    user_normalize = get_var('normalize', default='true')  # Default to 'true' for extract alias
    
    # Build the command to run extract_states_group with our defaults
    # This approach lets users still override: doit extract models=SAN-4566 normalize=false
    def run_extract_with_defaults():
        import subprocess
        import sys
        
        # Build doit command with our defaults (users can still override)
        cmd = [
            sys.executable, '-m', 'doit', 
            'extract_states_group',
            f'models={user_models}',
            f'normalize={user_normalize}'
        ]
        
        # Add any other variables that might be set
        for var_name in ['datasets', 'batch_size', 'recurrent_size', 'recurrent_layers']:
            var_value = get_var(var_name, default=None)
            if var_value is not None:
                cmd.append(f'{var_name}={var_value}')
        
        # Run the command
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        return result.returncode == 0
    
    return {
        'actions': [run_extract_with_defaults],
        'verbosity': 2,
        'uptodate': [False],  # Always check if we need to run
        'doc': 'Alias for extract_states_group with defaults: models=all, normalize=true'
    }


def task_identify():
    """Alias for identify_critical_units - automatically runs on all available models."""
    # This is a meta-task that just depends on identify_critical_units
    # Uses models=available by default to auto-detect models with extracted states
    return {
        'actions': None,
        'task_dep': ['identify_critical_units'],
        'verbosity': 2,
        'doc': 'Alias for identify_critical_units - automatically runs on all models with extracted states'
    }


def task_full_pipeline():
    """Alias for extract_and_analyze - complete workflow."""
    # This is a meta-task that just depends on extract_and_analyze
    return {
        'actions': None,
        'task_dep': ['extract_and_analyze'],
        'verbosity': 2,
        'doc': 'Alias for extract_and_analyze - runs complete extraction and analysis workflow'
    }