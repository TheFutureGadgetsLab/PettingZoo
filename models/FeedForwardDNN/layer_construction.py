import torch.nn as nn
import torch.nn.functional as F
import torch
import game.core.defs as pz

def config_to_sequential(config, view_size):
    """
    Config should be a list of the form:
    [
        (layer_name, layer_arg_1, layer_arg_2, ...),
        (layer_name, layer_arg_1, layer_arg_2, ...),
    ]

    THIS WILL ADD THE FINAL BUTTON LAYER FOR YOU (WITH TANH ACTIVATION)
    Layer names and args:
    "linear": Normal linear layer
        args: number of neurons (int)
    "act": Normal non-linear activation function
        args: one of "relu", "tanh", "sigmoid"
    """

    torch_layers = []

    # Initialize prev_dim (for first layer) to be input window size
    prev_dim = view_size.x * view_size.y

    for layer_config in config:
        layer_type = layer_config[0]
        layer_args = layer_config[1:]

        if layer_type == "linear":
            layer = Linear(prev_dim, layer_args[0])
            prev_dim = layer.out_dim
        elif layer_type == "act":
            layer = Activation(layer_args[0])
        else:
            raise ValueError("Unknown layer type!")

        torch_layers.append(layer)

    # Add output layer
    torch_layers.append(Linear(prev_dim, pz.NUM_BUTTONS))
    torch_layers.append(Activation("tanh"))

    return nn.Sequential(*torch_layers)

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        self.out_dim = out_features

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=False)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features), requires_grad=False)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        return F.linear(input.view(-1), self.weight, self.bias)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size

        self.out_dim = out_channels # going to be 4D, need to calculate

        self.masks = nn.Conv2D(in_channels, out_channels, kernel_size, bias=bias)

    def forward(self, input):
        # XXX NEED TO CHECK FOR RESHAPE XXX
        # XXX NEED TO CHECK FOR RESHAPE XXX

        return self.masks.forward(input)

    def extra_repr(self):
        return f"in_channels={self.in_features}, out_channels={self.out_features}, kernel_size={self.kernel_size}, bias={self.bias is not None}"

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