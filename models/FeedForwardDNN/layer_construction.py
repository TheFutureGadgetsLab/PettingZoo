import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import game.defs as pz

def config_to_sequential(config, view_size):
    """
    Config should be a list of the form:
    [
        (layer_name, layer_arg_1, layer_arg_2, ...),
        (layer_name, layer_arg_1, layer_arg_2, ...),
    ]

    THIS WILL ADD THE FINAL BUTTON LAYER FOR YOU (WITH SIGMOID ACTIVATION)

    Layer names and args:
        "linear": Normal linear layer
            args: number of neurons (int)
        "conv": Normal convolutional layer
            args: number of masks (int)
                  mask size       (int)
        "act": Normal non-linear activation function
            args: one of "relu", "tanh", "sigmoid"
    """

    torch_layers = []

    # Initialize prev_dim for the boys (for first layer) to be input window size
    prev_dim = [1, 1, view_size.x, view_size.y]

    for layer_config in config:
        layer_type = layer_config[0]
        layer_args = layer_config[1:]

        if layer_type == "linear":
            layer = Linear(prev_dim, layer_args[0])
            prev_dim = layer.out_dim
        elif layer_type == "conv":
            layer = Conv2D(prev_dim, out_masks=layer_args[0], mask_size=layer_args[1])
            prev_dim = layer.out_dim
        elif layer_type == "act":
            layer = Activation(layer_args[0])
        else:
            raise ValueError("Unknown layer type!")

        torch_layers.append(layer)

    # Add output layer
    torch_layers.append(Linear(prev_dim, pz.NUM_BUTTONS))
    torch_layers.append(Activation("sigmoid"))

    return nn.Sequential(*torch_layers)

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        self.in_features  = int(np.prod(in_features))
        self.out_features = int(out_features)

        self.out_dim = out_features

        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features), requires_grad=False)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_features), requires_grad=False)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        return F.linear(input.view(-1), self.weight, self.bias)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

class Conv2D(nn.Module):
    def __init__(self, in_dim, out_masks, mask_size, bias=True):
        super().__init__()

        self.bias = bias
        self.in_dim    = in_dim
        self.out_masks = out_masks
        self.mask_size = mask_size

        out_height = in_dim[2] - mask_size[0] + 1
        out_width  = in_dim[3] - mask_size[1] + 1

        self.out_dim = (1, out_masks, out_height, out_width)

        self.masks = nn.Conv2d(in_dim[1], out_masks, mask_size, bias=bias)

    def forward(self, input):
        if input.dim() > 4:
            raise ValueError("What the fuck")

        full_dim = [1] * (4 - input.dim()) + list(input.shape)

        return self.masks.forward(input.view(full_dim))

    def extra_repr(self):
        return f"in_dim={self.in_dim}, out_dim={self.out_dim}, kernel_size={self.mask_size}, bias={self.bias is not None}"

class Activation(nn.Module):
    activations = {
        "relu"    : nn.ReLU,
        "sigmoid" : nn.Sigmoid,
        "tanh"    : nn.Tanh,
    }

    def __init__(self, activation_type):
        super().__init__()
        self.act  = self.activations[activation_type]()

    def forward(self, input):
        return self.act.forward(input)