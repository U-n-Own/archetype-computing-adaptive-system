"""
Utility functions for ESN experiments
"""

import torch
import numpy as np
import random

from .plot_utils import (
    plot_comparison,
    plot_predictions,
    plot_state_dynamics,
    plot_eigenspectrum_analysis
)

from .analysis_utils import (
    compute_jacobian_eigenvalues,
    get_reservoir_matrices,
    compute_spectral_properties
)

from .search_utils import (
    connection_mode_search,
    test_model
)

def set_seed(seed: int):
    """Set seeds for reproducibility across torch, numpy, and random.

    Args:
        seed: The seed to use.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

__all__ = [
    # Plotting
    'plot_comparison',
    'plot_predictions', 
    'plot_state_dynamics',
    'plot_eigenspectrum_analysis',
    # Analysis
    'compute_jacobian_eigenvalues',
    'get_reservoir_matrices',
    'compute_spectral_properties',
    # Search
    'connection_mode_search',
    'test_model',
    # Misc
    'set_seed',
]
