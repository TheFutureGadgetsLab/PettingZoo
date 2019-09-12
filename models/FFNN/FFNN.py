import torch
import torch.nn as nn

class FFNN(nn.Module):
    r""" A simple feed-forward neural network.

    Args:
        in_w: Width of the input space around the player
        in_h: Width of the input space around the player
        hlc: Number of hidden layers
        npl: Number of nodes in each hidden layer
    """
    def __init__(self, in_w, in_h, hlc, npl):
        super().__init__()

        layers = []

        # Input Layer
        layers.append(nn.Linear(in_w * in_h, npl, bias=False))
        layers.append(nn.Sigmoid())

        # Hidden Layers
        for _ in range(hlc):
            layers.append(nn.Linear(npl, npl, bias=False))
            layers.append(nn.Sigmoid())
        
        # Output Layer
        layers.append(nn.Linear(npl, 3, bias=False))
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
    def breed(cls, parentA, parentB):
        for p
        pass        