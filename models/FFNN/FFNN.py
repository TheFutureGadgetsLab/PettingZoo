import torch
import torch.nn as nn
import torch.nn.functional as F

class FFNN(nn.Module):
    def __init__(self, in_w, in_h, n_layers, npl):
        super().__init__()

        layers = []

        # Input Layer
        layers.append(nn.Linear(in_w * in_h, npl, bias=False))
        layers.append(nn.Sigmoid())

        # Hidden Layers
        for _ in range(n_layers):
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