"""
Model Factory for UCR Time Series Classification
Creates RON and ESN models with different configurations.
"""
import torch
from acds.archetypes import DeepRandomizedOscillatorsNetwork, DeepReservoir


def create_ron_model(
    input_size: int,
    output_size: int,
    antisymmetric: bool = False,
    reservoir_size: int = 100,
    n_layers: int = 1,
    **kwargs
) -> DeepRandomizedOscillatorsNetwork:
    """Create a RON (Reservoir with Orthogonal and Normalized dynamics) model.
    
    Args:
        input_size: Input dimension
        output_size: Output dimension (number of classes) - not used for reservoir
        antisymmetric: If True, use antisymmetric coupling between layers (5 layers default)
        reservoir_size: Total number of units (default: 100)
        n_layers: Number of layers (default: 1 for standard, 5 for antisymmetric)
        **kwargs: Additional hyperparameters (including gamma, epsilon, gamma_range, epsilon_range)
    
    Returns:
        RON model instance
    """
    # Handle gamma/epsilon ranges - convert to tuples if range is specified
    gamma = kwargs.pop('gamma', 1.0)
    epsilon = kwargs.pop('epsilon', 1.0)
    gamma_range = kwargs.pop('gamma_range', None)
    epsilon_range = kwargs.pop('epsilon_range', None)
    
    if gamma_range is not None and gamma_range > 0:
        # Create tuple (gamma - gamma_range/2, gamma + gamma_range/2)
        gamma = (max(0.01, gamma - gamma_range/2), gamma + gamma_range/2)
    
    if epsilon_range is not None and epsilon_range > 0:
        # Create tuple (epsilon - epsilon_range/2, epsilon + epsilon_range/2)
        epsilon = (max(0.01, epsilon - epsilon_range/2), epsilon + epsilon_range/2)
    
    if antisymmetric:
        # Antisymmetric configuration: 5 layers with 100 units each
        # Enable antisymmetric_coupling parameter (not topology)
        n_layers = 5
        total_units = reservoir_size  # Total units across all layers
        model = DeepRandomizedOscillatorsNetwork(
            n_inp=input_size,
            total_units=total_units,
            dt=kwargs.pop('dt', 0.01),
            gamma=gamma,
            epsilon=epsilon,
            n_layers=n_layers,
            antisymmetric_coupling=True,  # Enable antisymmetric coupling between layers
            cycle=False,  # No cycle
            concat=True,  # Concatenate layer outputs
            **kwargs
        )
    else:
        # Standard configuration: 1 layer with 100 units, no antisymmetry/cycle
        n_layers = 1
        total_units = reservoir_size
        model = DeepRandomizedOscillatorsNetwork(
            n_inp=input_size,
            total_units=total_units,
            dt=kwargs.pop('dt', 0.01),
            gamma=gamma,
            epsilon=epsilon,
            n_layers=n_layers,
            antisymmetric_coupling=False,  # No antisymmetric coupling
            cycle=False,  # No cycle
            concat=True,
            **kwargs
        )
    
    return model


def create_esn_model(
    input_size: int,
    output_size: int,
    antisymmetric: bool = False,
    reservoir_size: int = 100,
    n_layers: int = 1,
    **kwargs
) -> DeepReservoir:
    """Create an ESN (Echo State Network) model.
    
    Args:
        input_size: Input dimension
        output_size: Output dimension (number of classes) - not used for reservoir
        antisymmetric: If True, use antisymmetric coupling (5 layers default)
        reservoir_size: Total number of units (default: 100)
        n_layers: Number of layers (default: 1 for standard, 5 for antisymmetric)
        **kwargs: Additional hyperparameters
    
    Returns:
        ESN model instance
    """
    # Handle parameter naming differences between search space and DeepReservoir
    # Search space uses 'rho' but DeepReservoir expects 'spectral_radius'
    if 'rho' in kwargs:
        kwargs['spectral_radius'] = kwargs.pop('rho')
    
    # Search space uses 'coupling_epsilon' but DeepReservoir expects 'epsilon'
    if 'coupling_epsilon' in kwargs:
        kwargs['epsilon'] = kwargs.pop('coupling_epsilon')
    
    if antisymmetric:
        # Antisymmetric configuration: 5 layers with 100 units each
        n_layers = 5
        model = DeepReservoir(
            input_size=input_size,
            tot_units=reservoir_size,
            n_layers=n_layers,
            antisymmetric=True,
            concat=True,  # Concatenate layer outputs
            **kwargs
        )
    else:
        # Standard configuration: 1 layer with 100 units, no antisymmetry/cycle
        n_layers = 1
        model = DeepReservoir(
            input_size=input_size,
            tot_units=reservoir_size,
            n_layers=n_layers,
            antisymmetric=False,
            concat=True,
            **kwargs
        )
    
    return model


def create_model(
    model_type: str,
    config: dict,
    device: str = 'cpu',
    antisymmetric: bool = False,
    topology: str = 'full',
    **kwargs
):
    """Factory function to create models.
    
    Args:
        model_type: Type of model ('ron' or 'esn')
        config: Dictionary containing model hyperparameters including:
                - n_inp or input_size: Input dimension
                - output_size: Output dimension (number of classes)
                - total_units or tot_units or reservoir_size: Total reservoir units
                - Other model-specific hyperparameters (dt, gamma, epsilon, etc.)
        device: Device to place the model on ('cpu' or 'cuda')
        antisymmetric: If True, use antisymmetric configuration with 5 layers
        topology: Topology type ('full', 'antisymmetric', 'orthogonal')
        **kwargs: Additional model-specific hyperparameters (override config)
    
    Returns:
        Model instance on the specified device
    """
    model_type = model_type.lower()
    
    # Extract input size (try n_inp first, then input_size)
    input_size = config.get('n_inp', config.get('input_size', 1))
    output_size = config.get('output_size', 2)
    reservoir_size = config.get('total_units', config.get('tot_units', config.get('reservoir_size', 100)))
    
    # Merge config and kwargs (kwargs take precedence)
    model_params = {**config, **kwargs}
    
    # Remove parameters that will be passed explicitly or don't belong to model constructors
    model_params.pop('n_inp', None)
    model_params.pop('input_size', None)
    model_params.pop('output_size', None)
    model_params.pop('reservoir_size', None)
    model_params.pop('total_units', None)
    model_params.pop('tot_units', None)
    model_params.pop('device', None)  # Device will be set via .to() after construction
    model_params.pop('topology', None)  # Topology is not a model parameter
    
    if model_type == 'ron':
        model = create_ron_model(
            input_size=input_size,
            output_size=output_size,
            antisymmetric=antisymmetric,
            reservoir_size=reservoir_size,
            **model_params
        )
    elif model_type == 'esn':
        model = create_esn_model(
            input_size=input_size,
            output_size=output_size,
            antisymmetric=antisymmetric,
            reservoir_size=reservoir_size,
            **model_params
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Must be 'ron' or 'esn'.")
    
    # Move model to device (in case it wasn't already)
    model = model.to(device)
    
    return model
