import torch
import torch.nn as nn
from models.FeedForwardDNN.layer_construction import config_to_sequential

class FeedForwardDNN(nn.Module):
    """ A simple feed-forward neural network (CNN or simply linear + non linear layers).

    Args:\n
    ---
    `view_r`: Width of the input space around the player\n
    `view_c`: Width of the input space around the player\n
    `layer_config`: Layer config to feed to models.utils.config_to_sequential\n
    """
    def __init__(self, view_size, layer_config, generator=None):
        super().__init__()

        self.layers = config_to_sequential(layer_config, view_size)
        self.view = view_size

        # Custom weight init
        if generator != None:
            for param in self.parameters():
                init_tensor_unif(param, generator)

    def evaluate(self, x):
        x = torch.Tensor(x)
        x = self.layers.forward(x)

        x = x > 0

        return x.numpy()

def init_tensor_unif(tensor, generator, low=-1.0, high=1.0):
    new = generator.uniform(low=low, high=high, size=tensor.shape)
    tensor[:, :] = torch.from_numpy(new)