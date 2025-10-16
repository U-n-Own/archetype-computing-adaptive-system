from typing import Optional

import numpy as np
import torch
from torch import nn

from acds.archetypes.utils import (
    sparse_eye_init,
    sparse_recurrent_tensor_init,
    sparse_tensor_init,
    spectral_norm_scaling,
)


class ReservoirCell(torch.nn.Module):
    """Shallow reservoir to be used as cell of a Recurrent Neural Network. The
    equation of the reservoir is given by:

    .. math::
        h_t = (1 - \\alpha) h_{t-1} + \\alpha \\tanh(W_{in} x_t + W_{rec} h_{t-1} + b)

    where:
    - :math:`h_t` is the hidden state at time t,
    - :math:`x_t` is the input at time t,
    - :math:`W_{in}` is the input weight matrix,
    - :math:`W_{rec}` is the recurrent weight matrix,
    - :math:`b` is the bias,
    - :math:`\\alpha` is the leaking rate.

    For antisymmetric coupling mode, the equation becomes:
    
    .. math::
        y_l^{(t)} = \\tanh(W_{in}^{(l)} u^{(t)} + W_{rec}^{(l)} h_l^{(t-1)} + 
        W_{proj} h_L^{(t-1)} + \\epsilon (C_{l-1} h_{l-1}^{(t-1)} - C_l^T h_{l+1}^{(t-1)}))

    The implementation is derivated from the one in https://github.com/gallicch/DeepRC-TF/blob/master/DeepRC.py

    If you use this code in your work, please cite the following paper, in which the
    concept of Deep Reservoir Computing has been introduced:

    Gallicchio,  C.,  Micheli,  A.,  Pedrelli,  L.: Deep  reservoir  computing:
    A  critical  experimental  analysis.    Neurocomputing268,  87-99  (2017).
    https://doi.org/10.1016/j.neucom.2016.12.08924.
    """

    def __init__(
        self,
        input_size: int,
        units: int,
        last_hidden_size: int,
        input_scaling: float = 1.0,
        spectral_radius: float = 0.99,
        leaky: float = 1.0,
        connectivity_input: int = 10,
        connectivity_recurrent: int = 10,
        cycle: bool = False,
        linear: bool = False,
        antisymmetric: bool = False,
        epsilon: float = 0.1,
    ):
        """Initializes the ReservoirCell.

        Args:
            input_size (int): number of input units.
            units (int): number of recurrent neurons in the reservoir.
            last_hidden_size (int): number of neurons in the last hidden layer.
            input_scaling (float): max abs value of a weight in the input-reservoir
                connections. Note that whis value also scales the unitary input bias.
                Defaults to 1.0.
            spectral_radius (float): max abs eigenvalue of the recurrent matrix.
                Defaults to 0.99.
            leaky (float): leaking rate constant of the reservoir. Defaults to 1.
            connectivity_input (int): number of outgoing connections from each
                input unit to the reservoir. Defaults to 10.
            connectivity_recurrent (int): number of incoming recurrent connections
                for each reservoir unit. Defaults to 10.
            cycle (bool): whether to use cycle connections. Defaults to False.
            linear (bool): whether to use linear activation. Defaults to False.
            antisymmetric (bool): whether to use antisymmetric coupling. Defaults to False.
            epsilon (float): coupling strength for antisymmetric connections. Defaults to 0.1.
        """
        super().__init__()

        self.input_size = input_size
        self.units = units
        self.last_hidden_size = last_hidden_size 
        self.state_size = units
        self.input_scaling = input_scaling
        self.spectral_radius = spectral_radius
        self.leaky = leaky
        self.connectivity_input = connectivity_input
        self.connectivity_recurrent = connectivity_recurrent
        self.cycle = cycle
        self.linear = linear
        self.antisymmetric = antisymmetric
        self.epsilon = epsilon
        
        
        if self.cycle:
            # For single unit cycle we need to handle it differently
            if self.units == 1:
                input_weight = self.input_scaling * (torch.randint(0, 2, (input_size, 1)) * 2 - 1).float()
                self.kernel = nn.Parameter(input_weight, requires_grad=False)
                
                self.recurrent_kernel = nn.Parameter(torch.zeros(1, 1), requires_grad=False)
                
                self.projection_kernel = nn.Parameter(torch.tensor([[spectral_radius]]), requires_grad=False)
            else:
                input_weights = self.input_scaling * (torch.randint(0, 2, (input_size, self.units)) * 2 - 1).float()
                self.kernel = nn.Parameter(input_weights, requires_grad=False)
                
                # Internal recurrent connections within the layer (reduced to allow for cycle connections)
                if connectivity_recurrent > 0:
                    W = sparse_recurrent_tensor_init(self.units, C=self.connectivity_recurrent)
                    # maybe here we can multiply rho by some small number to reduce a littel bit spectral radius effect
                    W = spectral_norm_scaling(W, spectral_radius)
                    self.recurrent_kernel = nn.Parameter(W, requires_grad=False)
                else:
                    self.recurrent_kernel = nn.Parameter(torch.zeros(self.units, self.units), requires_grad=False)
                
                # Ring projection from previous layer in cycle
                self.projection_kernel = nn.Parameter(torch.eye(self.units) * spectral_radius, requires_grad=False)
        else:
            # No cycle mode 
            self.kernel = (
                sparse_tensor_init(input_size, self.units, self.connectivity_input)
                * self.input_scaling
            )
            self.kernel = nn.Parameter(self.kernel, requires_grad=False)

            W = sparse_recurrent_tensor_init(self.units, C=self.connectivity_recurrent)
            # re-scale the weight matrix to control the effective spectral radius
            # of the linearized system
            if self.leaky == 1:
                W = spectral_norm_scaling(W, spectral_radius)
                self.recurrent_kernel = W
            else:
                I = sparse_eye_init(self.units)
                W = W * self.leaky + (I * (1 - self.leaky))
                W = spectral_norm_scaling(W, spectral_radius)
                self.recurrent_kernel = (W + I * (self.leaky - 1)) * (1 / self.leaky)
            self.recurrent_kernel = nn.Parameter(self.recurrent_kernel, requires_grad=False)

        if self.cycle:
            self.bias = nn.init.uniform_(torch.empty(self.units), -1, 1) * self.input_scaling
            self.bias = nn.Parameter(self.bias, requires_grad=False)
            #self.bias = nn.Parameter(torch.zeros(self.units), requires_grad=False)
        else:
            # uniform init in [-1, +1] times input_scaling
            self.bias = nn.init.uniform_(torch.empty(self.units), -1, 1) * self.input_scaling
            self.bias = nn.Parameter(self.bias, requires_grad=False)

        # Initialize antisymmetric coupling matrices if enabled
        if self.antisymmetric:
            # Create a single coupling matrix C for antisymmetric coupling
            # This will be used as C for backward coupling and -C^T for forward coupling
            # Matrix should be square with dimension equal to the number of units
            C_base = sparse_tensor_init(self.units, self.units, self.connectivity_recurrent)
            self.C_coupling = nn.Parameter(C_base, requires_grad=False)
            
            # For convenience, also store -C^T
            self.C_coupling_T_neg = nn.Parameter(-C_base.T, requires_grad=False)
        else:
            self.C_coupling = None
            self.C_coupling_T_neg = None

    def forward(self, xt: torch.Tensor, h_prev: torch.Tensor, first_layer: bool = False, h_last: Optional[torch.Tensor] = None, h_prev_layer: Optional[torch.Tensor] = None, h_next_layer: Optional[torch.Tensor] = None):
        """Computes the output of the cell given the input and previous state.

        Args:
            xt (torch.Tensor): input tensor shaped as (batch, input_dim).
            h_prev (torch.Tensor): previous state tensor shaped as (batch, state_dim).
            first_layer (bool): whether this is the first layer in the stack.
            h_last (torch.Tensor): feedback from the last layer for cycle connections.
            h_prev_layer (torch.Tensor): hidden state from previous layer for antisymmetric coupling.
            h_next_layer (torch.Tensor): hidden state from next layer for antisymmetric coupling.
        Returns:
            torch.Tensor: output to next layer shaped as (batch, state_dim).
            torch.Tensor: hidden state tensor shaped as (batch, state_dim).
        """     
        input_part = torch.mm(xt, self.kernel.to(dtype=xt.dtype))
        state_part = torch.mm(h_prev.to(dtype=xt.dtype), self.recurrent_kernel.to(dtype=(xt.dtype)))
        
        # Initialize the total input as standard terms
        total_input = input_part + self.bias.to(dtype=xt.dtype) + state_part.to(dtype=xt.dtype)
        
        # Add cycle connection if enabled
        if self.cycle and first_layer and h_last is not None:
            # Multi unit layer with ring connection
            # h_last should have the same number of units as current layer for ring topology                
            last_hidden_part = torch.mm(h_last, self.projection_kernel.to(dtype=xt.dtype))
            total_input = total_input + last_hidden_part.to(dtype=xt.dtype)
        
        # Add antisymmetric coupling if enabled
        if self.antisymmetric:
            antisymmetric_part = torch.zeros_like(total_input)
            
            # Backward coupling: C * h_{l-1}^{(t-1)}
            if h_prev_layer is not None and self.C_coupling is not None:
                backward_coupling = torch.mm(h_prev_layer.to(dtype=xt.dtype), self.C_coupling.to(dtype=xt.dtype))
                antisymmetric_part = antisymmetric_part + backward_coupling
            
            # Forward coupling: -C^T * h_{l+1}^{(t-1)}
            if h_next_layer is not None and self.C_coupling_T_neg is not None:
                forward_coupling = torch.mm(h_next_layer.to(dtype=xt.dtype), self.C_coupling_T_neg.to(dtype=xt.dtype))
                antisymmetric_part = antisymmetric_part + forward_coupling  # Already negative in C_coupling_T_neg
            
            # Scale by epsilon and add to total input
            total_input = total_input + self.epsilon * antisymmetric_part
        
        # Apply activation function
        if self.linear:
            output = total_input
        else:
            output = torch.tanh(total_input)
            
        # Apply leaky integration if not in cycle mode
        if self.cycle:
            return output, output
        else:
            leaky_output = h_prev * (1 - self.leaky) + (output * self.leaky)
            return leaky_output, leaky_output
            
class ReservoirLayer(torch.nn.Module):
    """Shallow reservoir to be used as Recurrent Neural Network layer.

    The layer is composed by a number of ReservoirCell, each of which is used to process
    the input and the previous state at each time step.
    """

    def __init__(
        self,
        input_size: int,
        units: int,
        last_hidden_size: int,
        input_scaling: float = 1.0,
        spectral_radius: float = 0.99,
        leaky: float = 1.0,
        connectivity_input: int = 10,
        connectivity_recurrent: int = 10,
        cycle: bool = False,
        linear: bool = False,
        antisymmetric: bool = False,
        epsilon: float = 0.1,
    ):
        """Initializes the ReservoirLayer.

        Args:
            input_size (int): number of input units.
            units (int): number of recurrent neurons in the reservoir.
            input_scaling (float): max abs value of a weight in the input-reservoir
                connections. Note that whis value also scales the unitary input bias.
                Defaults to 1.0.
            spectral_radius (float): max abs eigenvalue of the recurrent matrix.
                Defaults to 0.99.
            leaky (float): leaking rate constant of the reservoir. Defaults to 1.
            connectivity_input (int): number of outgoing connections from each
                input unit to the reservoir. Defaults to 10.
            connectivity_recurrent (int): number of incoming recurrent connections
                for each reservoir unit. Defaults to 10.
            cycle (bool): whether to use cycle connections. Defaults to False.
            linear (bool): whether to use linear activation. Defaults to False.
            antisymmetric (bool): whether to use antisymmetric coupling. Defaults to False.
            epsilon (float): coupling strength for antisymmetric connections. Defaults to 0.1.
        """
        super().__init__()
        self.net = ReservoirCell(
            input_size,
            units,
            last_hidden_size,
            input_scaling,
            spectral_radius,
            leaky,
            connectivity_input,
            connectivity_recurrent,
            cycle,
            linear,
            antisymmetric,
            epsilon,
        )

    def init_hidden(self, batch_size: int):
        """Initializes the hidden state to zeros.

        Args:
            batch_size (int): size of the batch.
        Returns:
            torch.Tensor: hidden state tensor shaped as (batch_size, state_dim).
        """
        return torch.zeros(batch_size, self.net.units)

    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None, first_layer: bool = False, h_last: Optional[torch.Tensor] = None):
        """Computes the output of the cell given the input and previous state.

        Args:
            x (torch.Tensor): input tensor shaped as (batch, time, input_dim).
            h_prev (torch.Tensor): previous state tensor shaped as
                (batch, time, state_dim). If None, the hidden state is initialized
                to zeros. Defaults to None.
        Returns:
            torch.Tensor: hidden state tensor shaped as (batch, time, state_dim).
            torch.Tensor: hidden state tensor shaped as (batch, time, state_dim).
        """
        if h_prev is None:
            h_prev = self.init_hidden(x.shape[0]).to(x.device)

        # make such that if we here are at first layer 
        hs = []
        for t in range(x.shape[1]):
            xt = x[:, t]
            _, h_prev = self.net(xt, h_prev, first_layer, h_last)
            hs.append(h_prev)
        hs = torch.stack(hs, dim=1)
        return hs, h_prev


class DeepReservoir(torch.nn.Module):
    """Deep Reservoir to be used as Recurrent Neural Network.

        The implementation realizes a number of stacked RNN layers using the ReservoirCell
        as core cell. All the reservoir layers share the same hyper-parameter values (i.e.,
        same number of recurrent neurons, spectral radius, etc..).
        """

    def __init__(
        self,
        input_size: int = 1,
        tot_units: int = 100,
        n_layers: int = 1,
        concat: bool = False,
        input_scaling: float = 1.0,
        inter_scaling: float = 1.0,
        spectral_radius: float = 0.99,
        leaky: float = 1.0,
        connectivity_recurrent: int = 10,
        connectivity_input: int = 10,
        connectivity_inter: int = 10,
        cycle: bool = False,
        linear: bool = False,
        antisymmetric: bool = False,
        epsilon: float = 0.1,
    ):
        """Initializes the DeepReservoir.

        Args:
            input_size (int): number of input units. Defaults to 1.
            tot_units (int): number of recurrent neurons in the reservoir. Defaults to 100.
            n_layers (int): number of stacked reservoir layers. Defaults to 1.
            concat (bool): if True, the output of each layer is concatenated to the
                previous ones. If False, only the output of the last layer is returned.
                Defaults to False.
            input_scaling (float): max abs value of a weight in the input-reservoir
                connections. Note that whis value also scales the unitary input bias.
                Defaults to 1.0.
            inter_scaling (float): max abs value of a weight in the input-reservoir
                connections between layers. Defaults to 1.0.
            spectral_radius (float): max abs eigenvalue of the recurrent matrix.
                Defaults to 0.99.
            leaky (float): leaking rate constant of the reservoir. Defaults to 1.
            connectivity_recurrent (int): number of incoming recurrent connections
                for each reservoir unit. Defaults to 10.
            connectivity_input (int): number of outgoing connections from each
                input unit to the reservoir. Defaults to 10.
            connectivity_inter (int): number of outgoing connections from each
                reservoir unit to the next layer. Defaults to 10.
            cycle (bool): whether to use cycle connections. Defaults to False.
            linear (bool): whether to use linear activation. Defaults to False.
            antisymmetric (bool): whether to use antisymmetric coupling. Defaults to False.
            epsilon (float): coupling strength for antisymmetric connections. Defaults to 0.1.
        """
        super().__init__()
        self.n_layers = n_layers
        self.tot_units = tot_units
        self.concat = concat
        self.cycle = cycle
        self.linear = linear
        self.antisymmetric = antisymmetric
        self.epsilon = epsilon
        self.batch_first = True  # DeepReservoir only supports batch_first
        # in case in which all the reservoir layers are concatenated, each level
        # contains units/layers neurons. This is done to keep the number of
        # state variables projected to the next layer fixed,
        # i.e., the number of trainable parameters does not depend on concat
        
        if concat or True:
            self.layers_units = int(tot_units / n_layers)
        else:
            self.layers_units = tot_units

        input_scaling_others = inter_scaling
        connectivity_input_1 = connectivity_input
        connectivity_input_others = connectivity_inter
        
        # creates a list of reservoirs
        # the first:
        reservoir_layers = [
            ReservoirLayer(
                input_size=input_size,
                units=self.layers_units + tot_units % n_layers,
                input_scaling=input_scaling,
                spectral_radius=spectral_radius,
                leaky=leaky,
                connectivity_input=connectivity_input_1,
                connectivity_recurrent=connectivity_recurrent,
                last_hidden_size=self.layers_units,
                cycle=cycle,
                linear=linear,
                antisymmetric=antisymmetric,
                epsilon=epsilon,
            )
        ]

        # all the others:
        # last_h_size may be different for the first layer
        # because of the remainder if concat=True
        last_h_size = self.layers_units + tot_units % n_layers
        for _ in range(n_layers - 1):
            # In cycle mode, all layers receive the original input (like SCR)
            # In non-cycle mode, layers receive input from previous layer
            layer_input_size = input_size if self.cycle else last_h_size
            
            reservoir_layers.append(
                ReservoirLayer(
                    input_size=layer_input_size,
                    units=self.layers_units,
                    input_scaling=input_scaling_others,
                    spectral_radius=spectral_radius,
                    leaky=leaky,
                    connectivity_input=connectivity_input_others,
                    connectivity_recurrent=connectivity_recurrent,
                    last_hidden_size=self.layers_units,
                    cycle=cycle,
                    linear=linear,
                    antisymmetric=antisymmetric,
                    epsilon=epsilon,
                )
            )
            last_h_size = self.layers_units
        self.reservoir = torch.nn.ModuleList(reservoir_layers)

    def forward(self, X: torch.Tensor):
        """Forward pass with support for multi-unit ring connectivity and antisymmetric coupling."""
        states = []
        states_last = []
        batch_size, seq_len, _ = X.shape
        
        if self.antisymmetric:
            # Antisymmetric coupling mode - requires layer interaction at each timestep
            # Initialize hidden states for each layer
            layer_hidden_states = []
            for i, res_layer in enumerate(self.reservoir):
                layer_hidden_states.append(torch.zeros(batch_size, res_layer.net.units).to(X.device))
            
            layer_states = [[] for _ in range(len(self.reservoir))]
            
            for t in range(seq_len):
                # For antisymmetric coupling, use standard input propagation
                # but layers need to know about their neighbors for coupling
                current_input = X[:, t, :] if t == 0 else None  # Only first layer gets external input
                new_hidden_states = []
                
                for i, res_layer in enumerate(self.reservoir):
                    # Prepare layer input: first layer gets external input, others get previous layer output
                    if i == 0:
                        layer_input = X[:, t, :]
                    else:
                        # Use previous layer's current output (from this timestep)
                        layer_input = new_hidden_states[i-1]
                    
                    # Prepare antisymmetric coupling inputs (from previous timestep)
                    h_prev_layer = layer_hidden_states[i-1] if i > 0 else None
                    h_next_layer = layer_hidden_states[i+1] if i < len(self.reservoir) - 1 else None
                    
                    # Forward through the layer with antisymmetric coupling
                    layer_output, layer_hidden = res_layer.net(
                        layer_input,
                        layer_hidden_states[i],  # Previous hidden state of this layer
                        first_layer=(i == 0),
                        h_last=None,  # No cycle connections in antisymmetric mode
                        h_prev_layer=h_prev_layer,  # Previous layer state for antisymmetric coupling
                        h_next_layer=h_next_layer   # Next layer state for antisymmetric coupling
                    )
                    
                    new_hidden_states.append(layer_hidden)
                    layer_states[i].append(layer_output)
                
                # Update hidden states for next timestep
                layer_hidden_states = new_hidden_states
            
            for i in range(len(self.reservoir)):
                stacked_states = torch.stack(layer_states[i], dim=1)
                states.append(stacked_states)
                states_last.append(stacked_states[:, -1, :])
                
        elif self.cycle:
            # Initialize hidden states for each layer
            layer_hidden_states = []
            for i, res_layer in enumerate(self.reservoir):
                layer_hidden_states.append(torch.zeros(batch_size, res_layer.net.units).to(X.device))
            
            layer_states = [[] for _ in range(len(self.reservoir))]
            
            for t in range(seq_len):
                current_input = X[:, t, :]
                new_hidden_states = []
                
                for i, res_layer in enumerate(self.reservoir):
                    # Get previous layer's output for ring connection
                    if i == 0:
                        # First layer gets feedback from last layer (closing the ring)
                        prev_layer_output = layer_hidden_states[-1] if len(self.reservoir) > 1 else torch.zeros(batch_size, res_layer.net.units).to(X.device)
                    else:
                        prev_layer_output = layer_hidden_states[i-1]
                    
                    layer_output, layer_hidden = res_layer.net(
                        current_input, 
                        layer_hidden_states[i],  # Previous hidden state of this layer
                        # TODO Critical change here if we set first_layer=True 
                        # all layers can receive the cycle input, instead if set to 
                        # first_layer = (i == 0) only the first layer receives the cycle input
                        first_layer=True,  # All layers can receive cycle input
                        #first_layer=i == 0,  # Only the first layer is considered the first layer
                        h_last=prev_layer_output  # Ring connection input
                    )
                    
                    new_hidden_states.append(layer_hidden)
                    layer_states[i].append(layer_output)
                
                # Update hidden states for next timestep
                layer_hidden_states = new_hidden_states
            
            for i in range(len(self.reservoir)):
                stacked_states = torch.stack(layer_states[i], dim=1)
                states.append(stacked_states)
                states_last.append(stacked_states[:, -1, :])
        else:
            # Standard non-cycle behavior
            for i, res_layer in enumerate(self.reservoir):
                [X, h_last] = res_layer(X)
                states.append(X)
                states_last.append(h_last)
        
        # Output formatting
        if self.concat:
            states = torch.cat(states, dim=2)
        else:
            states = states[-1]
            
        return states, states_last
    
    def old_forward(self, X: torch.Tensor):
        """Forward pass.

        Args:
            X (torch.Tensor): Input tensor, shaped as (batch, seq_len, n_inp).
        Returns:
            torch.Tensor: Output tensor, shaped as (batch, seq_len, n_out).
        """
        states = []  # list of all the states in all the layers
        states_last = []  # list of the states in all the layers for the last time step
        # states_last is a list because different layers may have different size.
        batch_size, seq_len, _ = X.shape
        
        if self.cycle:
            batch_size, seq_len, _ = X.shape
            
            # Initialize hidden states for each layer (unit) - these represent the full reservoir state
            h = torch.zeros(batch_size, len(self.reservoir)).to(X.device)
            layer_states = [[] for _ in range(len(self.reservoir))]
            
            for t in range(seq_len):
                current_input = X[:, t, :]  # Original input at time t
                new_h = torch.zeros(batch_size, len(self.reservoir)).to(X.device)
                
                for i, res_layer in enumerate(self.reservoir):
                    if i == 0:
                        # First unit gets feedback from last unit (cycle connection)
                        prev_unit_state = h[:, -1:] if h.shape[1] > 1 else torch.zeros(batch_size, 1).to(X.device)
                    else:
                        # Other units get state from previous unit
                        prev_unit_state = h[:, i-1:i]
                    
                    unit_output, unit_hidden = res_layer.net(
                        current_input, torch.zeros(batch_size, 1).to(X.device), 
                        first_layer=True, h_last=prev_unit_state
                    )
                    new_h[:, i:i+1] = unit_output
                    
                    # Store state for this unit and timestep
                    layer_states[i].append(unit_output)
                
                # Update hidden state for next timestep
                h = new_h
            
            # Convert to proper format
            for i in range(len(self.reservoir)):
                stacked_states = torch.stack(layer_states[i], dim=1)
                states.append(stacked_states)
                states_last.append(stacked_states[:, -1, :])
        else:
            # standard behaviour
            for i, res_layer in enumerate(self.reservoir):
                [X, h_last] = res_layer(X)
                states.append(X)
                states_last.append(h_last)
        
            states_uncat = states
        
        if self.concat:
            states = torch.cat(states, dim=2)
        else:
            # Original behavior: return only last layer
            states = states[-1]
            
        return states, states_last#, states_uncat
