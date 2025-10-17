"""
Utility functions for ESN experiments
"""

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
]
