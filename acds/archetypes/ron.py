from typing import (
    List,
    Literal,
    Tuple,
    Union,
)




import torch
from torch import nn

from acds.archetypes.utils import (
    get_hidden_topology,
    spectral_norm_scaling,
)


class RandomizedOscillatorsNetwork(nn.Module):
    """
    Randomized Oscillators Network. A recurrent neural network model with
    oscillatory dynamics. The model is defined by the following ordinary
    differential equation:

    .. math::
        \\dot{h} = -\\gamma h - \\epsilon \\dot{h} + \\tanh(W_{in} x + W_{rec} h + b)

    where:
    - :math:`h` is the hidden state,
    - :math:`\\dot{h}` is the derivative of the hidden state,
    - :math:`\\gamma` is the damping factor,
    - :math:`\\epsilon` is the stiffness factor,
    - :math:`W_{in}` is the input-to-hidden weight matrix,
    - :math:`W_{rec}` is the hidden-to-hidden weight matrix,
    - :math:`b` is the bias vector.

    The model is trained by minimizing the mean squared error between the output of the
    model and the target time-series.
    """

    def __init__(
        self,
        n_inp: int,
        n_hid: int,
        dt: float,
        gamma: Union[float, Tuple[float, float]],
        epsilon: Union[float, Tuple[float, float]],
        diffusive_gamma=0.0,
        rho: float = 0.99,
        input_scaling: float = 1.0,
        topology: Literal[
            "full", "lower", "orthogonal", "band", "ring", "toeplitz", "antisymmetric"
        ] = "full",
        reservoir_scaler=0.0,
        sparsity=0.0,
        device="cpu",
        cycle: bool = False,
        antisymmetric_coupling: bool = False,
        coupling_epsilon: float = 0.1,
    ):
        """Initialize the RON model.

        Args:
            n_inp (int): Number of input units.
            n_hid (int): Number of hidden units.
            dt (float): Time step.
            gamma (float or tuple): Damping factor. If tuple, the damping factor is
                randomly sampled from a uniform distribution between the two values.
            epsilon (float or tuple): Stiffness factor. If tuple, the stiffness factor
                is randomly sampled from a uniform distribution between the two values.
            diffusive_gamma (float): Diffusive term to ensure stability of the forward Euler method.
            rho (float): Spectral radius of the hidden-to-hidden weight matrix.
            input_scaling (float): Scaling factor for the input-to-hidden weight matrix.
                Wrt original paper here we initialize input-hidden in (0, 1) instead of (-2, 2).
                Therefore, when taking input_scaling from original paper, we recommend to multiply it by 2.
            topology (str): Topology of the hidden-to-hidden weight matrix. Options are
                'full', 'lower', 'orthogonal', 'band', 'ring', 'toeplitz', 'antisymmetric'. Default is
                'full'.
            reservoir_scaler (float): Scaling factor for the hidden-to-hidden weight
                matrix.
            sparsity (float): Sparsity of the hidden-to-hidden weight matrix.
            device (str): Device to run the model on. Options are 'cpu' and 'cuda'.
            cycle (bool): Whether to use cycle connections between layers.
            antisymmetric_coupling (bool): Whether to use antisymmetric coupling between layers.
            coupling_epsilon (float): Coupling strength for antisymmetric connections.
        """
        super().__init__()
        self.n_hid = n_hid
        self.device = device
        self.dt = dt
        self.cycle = cycle
        self.antisymmetric_coupling = antisymmetric_coupling
        self.coupling_epsilon = coupling_epsilon
        self.diffusive_matrix = diffusive_gamma * torch.eye(n_hid).to(device)
        if isinstance(gamma, tuple):
            gamma_min, gamma_max = gamma
            self.gamma = (
                torch.rand(n_hid, requires_grad=False, device=device)
                * (gamma_max - gamma_min)
                + gamma_min
            )
        else:
            self.gamma = gamma
        if isinstance(epsilon, tuple):
            eps_min, eps_max = epsilon
            self.epsilon = (
                torch.rand(n_hid, requires_grad=False, device=device)
                * (eps_max - eps_min)
                + eps_min
            )
        else:
            self.epsilon = epsilon

        h2h = get_hidden_topology(n_hid, topology, sparsity, reservoir_scaler)
        if topology != 'antisymmetric':
            h2h = spectral_norm_scaling(h2h, rho)
        # Ensure h2h is on the correct device
        h2h = h2h.to(device)
        self.h2h = nn.Parameter(h2h, requires_grad=False)

        # add the cycle kernel for cyclic feedback if needed
        if self.cycle:
            # For cycle functionality, we need to know the size of the last layer
            # This will be set during initialization of DeepRandomizedOscillatorsNetwork
            # make a cycle kernel init as Wrec form last layer to first layer
            pass
        else:
            # nothing
            pass
        
        x2h = torch.rand(n_inp, n_hid, device=device) * input_scaling
        self.x2h = nn.Parameter(x2h, requires_grad=False)
        bias = (torch.rand(n_hid, device=device) * 2 - 1) * input_scaling
        self.bias = nn.Parameter(bias, requires_grad=False)
        
        # Initialize antisymmetric coupling matrices if enabled
        if self.antisymmetric_coupling:
            # Create coupling matrix C for antisymmetric coupling between layers
            # C will be used for backward coupling, -C^T for forward coupling
            C_base = torch.rand(n_hid, n_hid, device=device) * 0.5 - 0.25  # Random in [-0.25, 0.25]
            
            # Normalize the coupling matrix to prevent instability
            C_base = spectral_norm_scaling(C_base, 0.5)
            
            self.C_coupling = nn.Parameter(C_base, requires_grad=False)
            # Store -C^T for convenience
            self.C_coupling_T_neg = nn.Parameter(-C_base.T, requires_grad=False)
        else:
            self.C_coupling = None
            self.C_coupling_T_neg = None

    def cell(
        self, x: torch.Tensor, hy: torch.Tensor, hz: torch.Tensor, first_layer: bool = False, h_last=None,
        h_prev_layer=None, h_next_layer=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the next hidden state and its derivative.

        Args:
            x (torch.Tensor): Input tensor.
            hy (torch.Tensor): Current hidden state.
            hz (torch.Tensor): Current hidden state derivative.
            first_layer (bool): Whether this is the first layer.
            h_last (torch.Tensor): Hidden state from the last layer for cycle feedback.
            h_prev_layer (torch.Tensor): Hidden state from previous layer for antisymmetric coupling.
            h_next_layer (torch.Tensor): Hidden state from next layer for antisymmetric coupling.
        """
        # convert to same type of x
        hz = hz.to(x.dtype)
        hy = hy.to(x.dtype)
        
        cycle_part = 0
        antisymmetric_part = 0
        
        if self.cycle and first_layer and h_last is not None:
            # Project h_last with cycle_kernel and add to the input
            cycle_part = torch.matmul(h_last, self.cycle_kernel.to(dtype=x.dtype))
        
        # Add antisymmetric coupling if enabled
        if self.antisymmetric_coupling:
            # Backward coupling: C * h_{l-1}^{(t-1)}
            if h_prev_layer is not None and self.C_coupling is not None:
                # Check dimension compatibility
                if h_prev_layer.shape[1] == self.C_coupling.shape[0]:
                    backward_coupling = torch.matmul(h_prev_layer.to(dtype=x.dtype), self.C_coupling.to(dtype=x.dtype))
                    antisymmetric_part = antisymmetric_part + backward_coupling
            
            # Forward coupling: -C^T * h_{l+1}^{(t-1)}
            if h_next_layer is not None and self.C_coupling_T_neg is not None:
                # Check dimension compatibility
                if h_next_layer.shape[1] == self.C_coupling_T_neg.shape[0]:
                    forward_coupling = torch.matmul(h_next_layer.to(dtype=x.dtype), self.C_coupling_T_neg.to(dtype=x.dtype))
                    antisymmetric_part = antisymmetric_part + forward_coupling
            
            # Scale by coupling epsilon and clamp to prevent extreme values
            antisymmetric_contribution = self.coupling_epsilon * antisymmetric_part
            antisymmetric_contribution = torch.clamp(antisymmetric_contribution, min=-10.0, max=10.0)
        else:
            antisymmetric_contribution = 0
         
        hz = hz + self.dt * (
            torch.tanh(
                torch.matmul(x, self.x2h.to(dtype=x.dtype)) + 
                torch.matmul(hy, self.h2h.to(dtype=x.dtype)) + 
                cycle_part + 
                antisymmetric_contribution -
                torch.matmul(hy, self.diffusive_matrix.to(dtype=x.dtype)) + 
                self.bias.to(dtype=x.dtype))
            - self.gamma * hy
            - self.epsilon * hz
        )

        hy = hy + self.dt * hz
        return hy, hz

    def forward(self, x: torch.Tensor, first_layer=False, h_last=None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass on a given input time-series.

        Args:
            x (torch.Tensor): Input time-series shaped as (batch, time, input_dim).

        Returns:
            torch.Tensor: Hidden states of the network shaped as (batch, time, n_hid).
            list: List containing the last hidden state of the network.
        """
        hy = torch.zeros(x.size(0), self.n_hid).to(self.device)
        hz = torch.zeros(x.size(0), self.n_hid).to(self.device)
        all_states = []
        for t in range(x.size(1)):
            hy, hz = self.cell(x[:, t], hy, hz, first_layer, h_last=h_last)
            all_states.append(hy)

        return torch.stack(all_states, dim=1), [
            hy
        ]  # list to be compatible with ESN implementation


class DeepRandomizedOscillatorsNetwork(nn.Module):
    """
    Deep Randomized Oscillators Network, using two stack of RON model one over another developing depth
    over.
    
    A recurrent deep neural network model with
    oscillatory dynamics stacked in layers. The model is defined by the following ordinary
    differential equation:

    .. math::#TODO Add layers notation
        \\dot{h} = -\\gamma h - \\epsilon \\dot{h} + \\tanh(W_{in} x + W_{rec} h + b)
    .. math:: If we consider the cycle version
        \\dot{h} = -\\gamma h - \\epsilon \\dot{h} + \\tanh(W_{in} x + W_{rec} h + b + F*h_{t-1})

    where:
    - :math:`h` is the hidden state,
    - :math:`\\dot{h}` is the derivative of the hidden state,
    - :math:`\\gamma` is the damping factor,
    - :math:`\\epsilon` is the stiffness factor,
    - :math:`W_{in}` is the input-to-hidden weight matrix,
    - :math:`W_{rec}` is the hidden-to-hidden weight matrix,
    - :math:`b` is the bias vector.

    The model is trained by minimizing the mean squared error between the output of the
    model and the target time-series.
    """

    def __init__(
        self,
        n_inp: int,
        total_units: int,
        dt: float,
        gamma: Union[float, Tuple[float, float]],
        epsilon: Union[float, Tuple[float, float]],
        n_layers: int = 1,
        diffusive_gamma=0.0,
        rho: float = 0.99,
        input_scaling: float = 1.0,
        inter_scaling: float = 1.0,
        topology: Literal[
            "full", "lower", "orthogonal", "band", "ring", "toeplitz", "antisymmetric"
        ] = "full",
        reservoir_scaler=0.0,
        sparsity=0.0,
        device="cuda",
        concat: bool = True,
        # TODO implement sparse connectivity later...
        connectivity_input: int = 10,
        connectivity_inter: int = 10,
        cycle: bool = False,
        linear: bool = False,
        antisymmetric_coupling: bool = False,
        coupling_epsilon: float = 0.1,
    ):
        """Initialize the DeepRON model.

        Args:
            n_inp (int): number of input units. Default to 1
            total_units (int): Total number of neurons in RON.
            dt (float): Time step.
            n_layers (int): Number of layers in the network.
            concat: (bool): If True, the output of each layer is concatenated. If False, only the output of the last layer is returned.
        """
        super().__init__()
        self.inter_scaling = inter_scaling
        self.n_layers = n_layers
        self.total_units = total_units
        self.reservoir_scaler = reservoir_scaler
        #self.n_inp = n_inp
        self.layers = nn.ModuleList()   
        self.cycle = cycle
        self.linear = linear
        self.antisymmetric_coupling = antisymmetric_coupling
        self.coupling_epsilon = coupling_epsilon
        
        self.concat = concat

        if concat:
            self.layer_units = int(total_units / n_layers) 
        else:
            self.layer_units = total_units
            
        input_scaling_others = inter_scaling
        connectivity_input_1 = connectivity_input
        connectivity_input_others = connectivity_inter
        
        deepron_layers = [
            RandomizedOscillatorsNetwork(
                n_inp=n_inp, n_hid=self.layer_units + total_units % n_layers,
                                    input_scaling=input_scaling_others,
                                    dt=dt,
                                    gamma=gamma,
                                    epsilon=epsilon,
                                    topology=topology, 
                                    sparsity=sparsity, 
                                    reservoir_scaler=self.reservoir_scaler,
                                    cycle=self.cycle,
                                    antisymmetric_coupling=antisymmetric_coupling,
                                    coupling_epsilon=coupling_epsilon,
                                    device=device, 
                                    #TODO still sparse connectivity to implement
                                    #connectivity_input=connectivity_input_1,
                                    #connectivity_recurrent=connectivity_input_others,
            )
        ]
            
        last_h_size = self.layer_units + total_units % n_layers
        
        for _ in range(n_layers - 1):
            deepron_layers.append(
                RandomizedOscillatorsNetwork(
                    n_inp=last_h_size, n_hid=self.layer_units,
                    input_scaling=input_scaling_others,
                    dt= dt,
                    gamma=gamma,
                    epsilon=epsilon,
                    topology=topology, 
                    sparsity=sparsity, 
                    reservoir_scaler=reservoir_scaler, 
                    cycle=self.cycle,
                    antisymmetric_coupling=antisymmetric_coupling,
                    coupling_epsilon=coupling_epsilon,
                    device=device, 
                    #connectivity_input=connectivity_input_others,
                    #connectivity_recurrent=connectivity_recurrent,
                )
            )
            last_h_size = self.layer_units
        self.ron_reservoir = nn.ModuleList(deepron_layers)
        
        # Initialize cycle kernels if cycle is enabled
        if self.cycle:
            # The first layer needs a cycle kernel to receive feedback from the last layer
            last_layer_size = self.layer_units
            first_layer_size = self.layer_units + total_units % n_layers
            
            # implement cycle kernel as the W_h matrix (inout x hidden)
            cycle_kernel = torch.rand(self.layer_units, last_layer_size, device=device) * inter_scaling
            self.ron_reservoir[0].cycle_kernel = nn.Parameter(cycle_kernel, requires_grad=False)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass on the layers of the DeepRON a given input time-series.

        Args:
            x (torch.Tensor): Input time-series shaped as (batch, time, input_dim).

        Returns:
            torch.Tensor: Hidden states of the network shaped as (batch, time, n_hid).
            list: List containing the last hidden state of the network.
        """
        # list to store the last state of each layer
        layer_states = []
        # list to store the hidden states of each layer
        states = []
       
        if self.antisymmetric_coupling:
            # Antisymmetric coupling mode - requires layer interaction at each timestep
            batch_size, seq_len, _ = x.shape
            
            # Initialize hidden states and derivatives for all layers
            h_states = []
            hz_states = []
            for i, ron_layer in enumerate(self.ron_reservoir):
                h_states.append(torch.zeros(batch_size, ron_layer.n_hid, device=x.device))
                hz_states.append(torch.zeros(batch_size, ron_layer.n_hid, device=x.device))
            
            layer_states_all = [[] for _ in range(len(self.ron_reservoir))]

            for t in range(seq_len):
                current_input = x[:, t, :] if t == 0 else None
                new_h_states = []
                new_hz_states = []
                
                for i, ron_layer in enumerate(self.ron_reservoir):
                    # Prepare layer input
                    if i == 0:
                        layer_input = x[:, t, :]
                    else:
                        # Use previous layer's current output (from this timestep)
                        layer_input = new_h_states[i-1]
                    
                    # Prepare antisymmetric coupling inputs (from previous timestep)
                    h_prev_layer = h_states[i-1] if i > 0 else None
                    h_next_layer = h_states[i+1] if i < len(self.ron_reservoir) - 1 else None
                    
                    # Forward through the layer with antisymmetric coupling
                    new_h, new_hz = ron_layer.cell(
                        layer_input,
                        h_states[i],  # Previous hidden state of this layer
                        hz_states[i],  # Previous hidden derivative of this layer
                        first_layer=(i == 0),
                        h_last=None,  # No cycle connections in antisymmetric mode
                        h_prev_layer=h_prev_layer,  # Previous layer state for antisymmetric coupling
                        h_next_layer=h_next_layer   # Next layer state for antisymmetric coupling
                    )
                    
                    new_h_states.append(new_h)
                    new_hz_states.append(new_hz)
                    layer_states_all[i].append(new_h)
                
                # Update hidden states for next timestep
                h_states = new_h_states
                hz_states = new_hz_states
            
            # Stack the layer states over time dimension
            for i in range(len(self.ron_reservoir)):
                stacked_states = torch.stack(layer_states_all[i], dim=1)
                states.append(stacked_states)
                layer_states.append(stacked_states[:, -1, :])
                
        elif self.cycle:
            # Initialize hidden states for all layers
            batch_size, seq_len, _ = x.shape
            
            # Initialize hidden states and derivatives for all layers
            h_states = []
            hz_states = []
            for i, ron_layer in enumerate(self.ron_reservoir):
                h_states.append(torch.zeros(batch_size, ron_layer.n_hid, device=x.device))
                hz_states.append(torch.zeros(batch_size, ron_layer.n_hid, device=x.device))
            
            layer_states_all = [[] for _ in range(len(self.ron_reservoir))]

            for t in range(seq_len):
                xt = x[:, t, :]
                # Save last layer's previous hidden state for feedback
                last_layer_hidden_prev = h_states[-1].clone()
                
                for i, ron_layer in enumerate(self.ron_reservoir):
                    if i == 0:
                        # Pass cyclic feedback from last layer to first layer
                        h_states[i], hz_states[i] = ron_layer.cell(xt, h_states[i], hz_states[i], first_layer=True, h_last=last_layer_hidden_prev)
                    else:
                        # Pass output from previous layer as input
                        h_states[i], hz_states[i] = ron_layer.cell(h_states[i-1], h_states[i], hz_states[i], first_layer=False)
                    
                    layer_states_all[i].append(h_states[i])

            # Stack the layer states over time dimension
            for i in range(len(self.ron_reservoir)):
                stacked_states = torch.stack(layer_states_all[i], dim=1)
                states.append(stacked_states)
                layer_states.append(stacked_states[:, -1, :])
        else:
            for i, ron_layer in enumerate(self.ron_reservoir):
                [x, last_state] = ron_layer(x)
                states.append(x)
                layer_states.append(last_state)
            
        states_uncat = states
        
        if self.concat:
            # check what dim we need to concat
            x = torch.cat(states, dim=2)
        else:
            # if not concat, return only the last layer
            x = states[-1]
            
       # Choose if return all_states from all layers for the  
       
        return x, layer_states#, states_uncat
