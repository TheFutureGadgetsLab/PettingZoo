import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np

class FFNN(nn.Module):
    r""" A simple feed-forward neural network.

    Args:
        in_w: Width of the input space around the player
        in_h: Width of the input space around the player
        hlc: Number of hidden layers
        npl: Number of nodes in each hidden layer
        bias: Whether or not the linear transformation should have a bias
    """
    def __init__(self, in_w, in_h, hlc, npl, bias=False):
        super().__init__()

        self.num_layers = hlc + 2
        self.npl = npl
        self.in_w = in_w
        self.in_h = in_h

        layers = []

        # Input Layer
        layers.append(nn.Linear(self.in_w * self.in_h, self.npl, bias=bias))
        layers.append(nn.Sigmoid())

        # Hidden Layers
        for _ in range(hlc):
            layers.append(nn.Linear(self.npl, self.npl, bias=bias))
            layers.append(nn.Sigmoid())

        # Output Layer
        layers.append(nn.Linear(self.npl, 3, bias=bias))
        layers.append(nn.Tanh())

        self.layers = nn.Sequential(*layers)

        # Disable backprop
        for layer in self.layers:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = torch.Tensor(x.ravel()) # Flatten x with ravel for slight perf boost
        x = self.layers.forward(x)

        x = (x > 0).int()

        return x.cpu().numpy()

    @classmethod
    def breed(cls, parentA, parentB, seed):
        childA = deepcopy(parentA)
        childB = deepcopy(parentB)
        rng = np.random.Generator(np.random.SFC64(seed))


        for i in range(parentA.num_layers):
            if type(parentA.layers[i]) != nn.Linear:
                continue

            for name, param in parentA.layers[i].named_parameters():
                print(parentA.layers[i].named_parameters())
            # split_loc = rng.integers(low=1, high=parentA.lay)


        return childA, childB

def combine_tensors(parentA, parentB, childA, childB, split_loc):
    """ All arguments are tensors. """
    pass
