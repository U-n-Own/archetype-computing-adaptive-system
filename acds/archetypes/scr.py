import torch
import torch.nn as nn
from typing import List, Union

class SimpleCycleReservoir(nn.Module):
    def __init__(self, input_size: int, n_reservoir: int, n_outputs: int = None, r: float = 0.5, v: float = 0.5):
        """
        Simple Cycle Reservoir (SCR) implementation from minimum complexity paper

        Args:
            input_size (int): Dimensionality of the input (usually 1 for time series).
            n_reservoir (int): Number of reservoir neurons.
            n_outputs (int, optional): Number of output nodes, defaults to n_reservoir if None.
                                      For memory capacity tasks, this should equal the delay.
            r (float): Recurrent weight (cycle weight).
            v (float): Input weight magnitude.
        """
        super().__init__()
        self.n_reservoir = n_reservoir
        self.input_size = input_size
        self.n_outputs = n_outputs if n_outputs is not None else n_reservoir

        input_weights = v * torch.ones(n_reservoir, input_size)
        signs = torch.randint(0, 2, (n_reservoir, input_size)) * 2 - 1
        self.Win = nn.Parameter(input_weights * signs, requires_grad=False)

        # Recurrent matrix
        W = torch.zeros(n_reservoir, n_reservoir)
        for i in range(n_reservoir - 1):
            W[i + 1, i] = r
        W[0, -1] = r
        self.W = nn.Parameter(W, requires_grad=False)

        self.Wout = nn.Parameter(torch.zeros(self.n_outputs, n_reservoir))
        
        self.bout = nn.Parameter(torch.zeros(self.n_outputs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the input sequence through the SCR.
        Args:
            x (torch.Tensor): Input sequence (batch, seq_len, input_size)
        Returns:
            torch.Tensor: Either reservoir states (batch, seq_len, n_reservoir) 
                         or output predictions (batch, seq_len, n_outputs)
        """
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.n_reservoir, device=x.device)
        states = []

        for t in range(seq_len):
            u = x[:, t]  
            inp = torch.matmul(u, self.Win.T)  
            res = torch.matmul(h, self.W.T)    
            #h = torch.tanh(inp + res)
            #linear
            h = inp + res
            states.append(h)

        all_states = torch.stack(states, dim=1)  
        
        
        return all_states  
    