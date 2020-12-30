#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1e
import torch
import torch.nn.functional as F

class CNN(torch.nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size=5):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(CNN, self).__init__()
        self.conv1_layer = torch.nn.Conv1d(input_ch, output_ch, (kernel_size,))

    def forward(self, x_conv):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        # out dimension will be (batch_size, output_ch, L_out). 
        # See https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html

        out = F.relu(self.conv1_layer(x_conv))

        #### no need to do maxpool like this
        #max_pool = nn.MaxPool1d(out.shape[-1], stride=1)
        #y_pred = max_pool(out)
        
        x_conv_out = torch.max(out, dim=2)[0]
        # last dimension is gone or reduced. 
        # so now dimension is (batch_size, output_ch)
        return x_conv_out


### END YOUR CODE

