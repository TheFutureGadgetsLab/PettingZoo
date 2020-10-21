import torch
import torch.nn as nn
from models.FeedForwardDNN.layer_construction import config_to_sequential
import game.defs as pz
import numpy as np

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
                param.requires_grad = False
                init_tensor_unif(param, generator)

        self.tile_mapper = {
            pz.EMPTY:      0,
            pz.PIPE_BOT:   1,
            pz.PIPE_MID:   1,
            pz.PIPE_TOP:   1,
            pz.GRASS:      1,
            pz.DIRT:       1,
            pz.COBBLE:     1,
            pz.COIN:       1,
            pz.FLAG:       1,
            pz.FINISH_BOT: 1,
            pz.FINISH_TOP: 1,
            pz.SPIKE_TOP:  2,
            pz.SPIKE_BOT:  3,
        }

    def evaluate(self, x):
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x[i, j] = self.tile_mapper[x[i, j]]
        
        x = torch.from_numpy(x.astype(np.float32))

        x = self.layers(x)

        x = x >= 0.5

        return x.numpy()

def init_tensor_unif(tensor, generator, low=-1.0, high=1.0):
    new = generator.uniform(low=low, high=high, size=tensor.shape)
    tensor.copy_(torch.from_numpy(new))