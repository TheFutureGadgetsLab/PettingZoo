import torch
import torch.nn as nn
from copy import deepcopy
from game.core.Vector2 import Vector2
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
    def __init__(self, view_size, hlc, npl, generator=None, bias=False):
        super().__init__()

        self.num_layers = hlc + 2
        self.npl = npl
        self.view = view_size

        layers = []

        # Input Layer
        layers.append(nn.Linear(self.view.y * self.view.x, self.npl, bias=bias))
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
        x = torch.Tensor(x)
        x = self.layers.forward(x.view(-1))

        x = x > 0

        return x.numpy()

def breed(parentA, parentB, generator):
    childA = deepcopy(parentA)
    childB = deepcopy(parentB)

    pA_params = list(parentA.parameters())
    pB_params = list(parentB.parameters())
    cA_params = list(childA.parameters())
    cB_params = list(childB.parameters())

    for pA_param, pB_param, cA_param, cB_param in zip(pA_params, pB_params, cA_params, cB_params):
        combine_tensors(pA_param, pB_param, cA_param, cB_param, generator)

    return childA, childB

def init_tensor_unif(tensor, generator, low=-1.0, high=1.0):
    new = generator.uniform(low=low, high=high, size=tensor.shape)
    tensor[:, :] = torch.from_numpy(new)

def combine_tensors(parentA, parentB, childA, childB, generator):
    split_loc = generator.integers(low=1, high=parentA.shape[0])

    # Only copying parentB into childA because childA is a deepcopy of parentA
    # Same with childB (but reversed)

    childA[split_loc:] = parentB[split_loc:]
    childB[split_loc:] = parentA[split_loc:]

def combine_tensors_avg(parentA, parentB, childA, childB, generator):
    first_weight  = generator.uniform(0, 2)
    second_weight = 1 - first_weight
    # Only copying parentB into childA because childA is a deepcopy of parentA
    # Same with childB (but reversed)

    childA = first_weight  * parentA + second_weight * parentB
    childB = second_weight * parentA + first_weight  * parentB