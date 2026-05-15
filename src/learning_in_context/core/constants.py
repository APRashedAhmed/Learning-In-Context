"""Shared constants with configurable defaults."""

import numpy as np

# Task constants
NUM_COLORS = 3
COLOR_NAMES = ["red", "green", "blue"]
TRIAL_TYPES = ["Straight", "Bounce", "Catch"]
HAZARD_RATES = ["Low", "High"]
CONTINGENCIES = ["Low", "Medium", "High"]

# Analysis constants (with empirical justification)
DEFAULT_ALPHA = 0.05
BOOTSTRAP_ITERATIONS = 1000  # Stable CI estimates
CV_FOLDS = 5  # Standard k-fold cross-validation
PERMUTATION_ITERATIONS = 10000  # For permutation tests

# Regularization sweep parameters (from dissertation methods)
LAMBDA_MIN = 1e-6  # Minimum lambda value (10^-6)
LAMBDA_MAX = 1.0   # Maximum lambda value (10^0)
N_LAMBDAS = 50     # Number of lambda values to sweep
LAMBDA_SPACING = 'log'  # Logarithmic spacing by default

# Decoder-specific L1 ratios (from dissertation)
BINARY_L1_RATIO = 0.64      # For hazard, contingency (binary classification)
MULTICLASS_L1_RATIO = 0.4   # For color, velocity (multiclass/regression)

# Max iterations for different decoder types
BINARY_MAX_ITER = 250       # For binary classification decoders
MULTICLASS_MAX_ITER = 250   # For multiclass and regression decoders

# Threshold detection parameters
CHANCE_MARGIN = 0.05        # Performance margin above chance level
THRESHOLD_METHOD = 'chance' # Default threshold detection method

# Legacy constants for backward compatibility
ELASTICNET_L1_RATIO = BINARY_L1_RATIO  # Default for hazard rate
ELASTICNET_ALPHAS = np.logspace(np.log10(LAMBDA_MAX), np.log10(LAMBDA_MIN), N_LAMBDAS)

# Specific parameters for different statistics (from original) - DEPRECATED
HZ_L1_RATIO = BINARY_L1_RATIO  # Hazard rate L1 ratio
HZ_C = 0.0001  # Hazard rate C value - DEPRECATED
HZ_MAX_ITER = BINARY_MAX_ITER  # Hazard rate max iterations

CONT_L1_RATIO = BINARY_L1_RATIO  # Contingency L1 ratio
CONT_C = 0.001  # Contingency C value - DEPRECATED
CONT_MAX_ITER = BINARY_MAX_ITER  # Contingency max iterations

# Non-binary decoder parameters (color, velocity)
MULTICLASS_L1_RATIO = MULTICLASS_L1_RATIO  # For multiclass and regression decoders
MULTICLASS_MAX_ITER = MULTICLASS_MAX_ITER  # For multiclass and regression decoders


def get_lambda_values(lambda_min=LAMBDA_MIN, lambda_max=LAMBDA_MAX, 
                     n_lambdas=N_LAMBDAS, spacing=LAMBDA_SPACING):
    """Generate lambda values for regularization sweep.
    
    Args:
        lambda_min: Minimum lambda value
        lambda_max: Maximum lambda value
        n_lambdas: Number of lambda values
        spacing: 'log' or 'linear' spacing
    
    Returns:
        Array of lambda values from max to min
    """
    if spacing == 'log':
        return np.logspace(np.log10(lambda_max), np.log10(lambda_min), n_lambdas)
    else:
        return np.linspace(lambda_max, lambda_min, n_lambdas)

# Neural analysis constants
MIN_UNIT_VARIANCE = 1e-6  # Units with less variance are excluded
INTERVENTION_NOISE_STD = 0.1  # Standard deviation for noise injection
ABLATION_VALUE = 0.0  # Value for unit ablation

# Path constants
DEFAULT_CACHE_DIR = "data/cache"
DEFAULT_OUTPUT_DIR = "data/output"
DEFAULT_RAW_DIR = "data/raw"

# Performance constants
MAX_CACHE_SIZE_GB = 10.0
CHUNK_SIZE = 1000  # For batched processing
STATE_EXTRACTION_BATCH_SIZE = 1024
DEFAULT_NUM_WORKERS = 4

# Figure constants
FIGURE_DPI = 300
FIGURE_FORMAT = "pdf"
FIGURE_BBOX = "tight"

# Dataset configurations for multi-dataset support
# Note: Paths are relative to the dodo.py working directory and will be resolved at runtime
DATASET_CONFIGS = {
    'participant': {
        'path': '../data/raw/hbb_v3_2_2/hbb_dataset_v3_2_2',
        'description': 'Human participant dataset (168 trials)',
        'size': 168,
        'suffix': 'participant'
    },
    'extended': {
        'path': '../data/raw/bb_datasets/hmdcpd/hbb_dataset_250627_004521_2190033339',
        'description': 'Extended dataset (18,000 videos)',  
        'size': 18000,
        'suffix': 'extended'
    },
    'controlled': {
        'path': '../data/raw/bb_datasets/hmdcpd/hbb_dataset_250415_150852_4192810657',
        'description': 'Controlled color change dataset',
        'size': 147, # 7 variants × 3 colors × 7 trials
        'suffix': 'controlled'
    },
    'velocity': {
        'path': '../data/raw/hbb_v3_2_1/hbb_dataset_v3_2_1',
        'description': 'Velocity change dataset (161 trials)',
        'size': 161,
        'suffix': 'velocity'
    }
}

# Default dataset for backward compatibility
DEFAULT_DATASET = 'participant'

# Logging constants
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"