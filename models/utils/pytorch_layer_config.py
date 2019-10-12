import torch.nn as nn
from models.utils import custom_torch_layers as cust_layer
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
            layer = cust_layer.Linear(prev_dim, layer_args[0], bias=False)
            prev_dim = layer.out_dim
        elif layer_type == "act":
            layer = cust_layer.Activation(layer_args[0])
        else:
            raise ValueError("Unknown layer type!")

        torch_layers.append(layer)

    # Add output layer
    torch_layers.append(cust_layer.Linear(prev_dim, pz.NUM_BUTTONS, bias=False))
    torch_layers.append(cust_layer.Activation("tanh"))

    return nn.Sequential(*torch_layers)