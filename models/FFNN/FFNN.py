import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np

class FFNN(nn.Module):
    """ A simple feed-forward neural network.

    Args:\n
    ---
    `in_w`: Width of the input space around the player\n
    `in_h`: Width of the input space around the player\n
    `hlc`: Number of hidden layers\n
    `npl`: Number of nodes in each hidden layer\n
    `bias`: Whether or not the linear transformations should have a bias
    """
    def __init__(self, in_w, in_h, hlc, npl, generator=None, bias=False):
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

    @classmethod
    def breed(cls, parentA, parentB, generator):
        childA = deepcopy(parentA)
        childB = deepcopy(parentB)

        pA_params = list(parentA.parameters())
        pB_params = list(parentB.parameters())
        cA_params = list(childA.parameters())
        cB_params = list(childB.parameters())

        for i in range(len(pA_params)):
            split_loc = generator.integers(low=1, high=pA_params[i].shape[0])
            combine_tensors(pA_params[i], pB_params[i], cA_params[i], cB_params[i], split_loc)

        return childA, childB

def init_tensor_unif(tensor, generator, low=-1.0, high=1.0):
    new = generator.uniform(low=low, high=high, size=tensor.shape)
    tensor[:, :] = torch.from_numpy(new)

def combine_tensors(parentA, parentB, childA, childB, split_loc):
    childA[:split_loc] = parentA[:split_loc]
    childA[split_loc:] = parentB[split_loc:]

    # Note that parentA and parentB are swapped on childB
    childB[:split_loc] = parentB[:split_loc]
    childB[split_loc:] = parentA[split_loc:]