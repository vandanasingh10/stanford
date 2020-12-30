#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1d
import torch
import torch.nn.functional as F

class Highway(torch.nn.Module):
    def __init__(self, D_in):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Highway, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, D_in)
        self.linear2 = torch.nn.Linear(D_in, D_in)
    
    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x_proj = F.relu(self.linear1(x))
        x_gate = torch.sigmoid(self.linear2(x))
        x_highway = (x_proj * x_gate) +  (1-x_gate)*x # skip connection

        return x_highway

### END YOUR CODE 

