import torch.nn as nn

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
        layer_type, layer_args = layer_config[0], layer_config[1:]

        if layer_type == "linear": # Linear layer
            prev_dim, layer = get_linear(layer_args, prev_dim)
        elif layer_type == "act":  # Activation layer
            layer = get_act(layer_args)
        else:
            raise ValueError

        torch_layers.append(layer)

    # Add output layer
    torch_layers.extend([
        nn.Linear(prev_dim, 3, bias=False),
        nn.Tanh(),
    ])

    return nn.Sequential(*torch_layers)

def get_linear(layer_args, prev_dim):
    layer = nn.Linear(in_features=prev_dim, out_features=layer_args[0], bias=False)
    new_dim = layer_args[0]

    return new_dim, layer

def get_act(layer_args):
    activations = {
        "relu"    : nn.ReLU,
        "sigmoid" : nn.Sigmoid,
        "tanh"    : nn.Tanh,
    }

    if layer_args[0] not in activations:
        raise ValueError("Unknown activation function")

    return activations[layer_args[0]]()