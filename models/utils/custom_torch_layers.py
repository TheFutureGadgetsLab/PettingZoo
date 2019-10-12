import torch.nn as nn
import torch.nn.functional as F
import torch

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
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
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

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