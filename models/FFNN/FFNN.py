import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np

class FFNN(nn.Module):
    """ A simple feed-forward neural network.

    Args:\n
    ---
    `view_r`: Width of the input space around the player\n
    `view_c`: Width of the input space around the player\n
    `hlc`: Number of hidden layers\n
    `npl`: Number of nodes in each hidden layer\n
    `bias`: Whether or not the linear transformations should have a bias
    """
    def __init__(self, view_r, view_c, hlc, npl, generator=None, bias=False):
        super().__init__()

        self.num_layers = hlc + 2
        self.npl = npl
        self.view_r = view_r
        self.view_c = view_c

        layers = []

        # Input Layer
        layers.append(nn.Linear(self.view_c * self.view_r, self.npl, bias=bias))
        layers.append(nn.Sigmoid())

        # Hidden Layers
        for _ in range(hlc):
            layers.append(nn.Linear(self.npl, self.npl, bias=bias))
            layers.append(nn.Sigmoid())

        # Output Layer
        layers.append(nn.Linear(self.npl, 3, bias=bias))
        layers.append(nn.Tanh())

        self.layers = nn.Sequential(*layers)

        # Disable backprop and custom weight init
        for param in self.parameters():
            param.requires_grad = False

            if generator != None:
                init_tensor_unif(param, generator)

    def evaluate(self, x):
        x = torch.Tensor(x.ravel()) # Flatten x with ravel for slight perf boost
        x = self.layers.forward(x)

        x = (x > 0).int()

        return x.numpy()

def breed(parentA, parentB, generator):
    childA = deepcopy(parentA)
    childB = deepcopy(parentB)

    pA_params = list(parentA.parameters())
    pB_params = list(parentB.parameters())
    cA_params = list(childA.parameters())
    cB_params = list(childB.parameters())

    for pA_param, pB_param, cA_param, cB_param in zip(pA_params, pB_params, cA_params, cB_params):
        split_loc = generator.integers(low=1, high=pA_param.shape[0])
        combine_tensors(pA_param, pB_param, cA_param, cB_param, split_loc)

    return childA, childB

def init_tensor_unif(tensor, generator, low=-1.0, high=1.0):
    new = generator.uniform(low=low, high=high, size=tensor.shape)
    tensor[:, :] = torch.from_numpy(new)

def combine_tensors(parentA, parentB, childA, childB, split_loc):
    # Only copying parentB into childA because childA is a deepcopy of parentA
    # Same with childB (but reversed)

    childA[split_loc:] = parentB[split_loc:]
    childB[split_loc:] = parentA[split_loc:]