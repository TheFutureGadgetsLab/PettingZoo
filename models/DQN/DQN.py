import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
import random

move_mapper = {
    0: [0, 0, 0],
    1: [0, 0, 1],
    2: [0, 1, 0],
    3: [0, 1, 1],
    4: [1, 0, 0],
    5: [1, 0, 1],
    6: [1, 1, 0],
    7: [1, 1, 1],
}

class DQN(nn.Module):
    def __init__(self, in_height, in_width):
        super().__init__()
        self.in_height = in_height
        self.in_width  = in_width

        self.conv1 = nn.Conv2d(1, 5, 5)
        self.conv2 = nn.Conv2d(5, 5, 3)

        self.conv1_out = conv2d_size_out(self.in_height, self.in_width, self.conv1)
        self.conv2_out = conv2d_size_out(*self.conv1_out, self.conv2)

        self.linear1 = nn.Linear(np.prod(self.conv2_out) * self.conv2.out_channels, 2 ** 3)

    def evaluate(self, x, epsilon=0.0):
        # Epsilon greedy, choose random action sometimes
        if np.random.ranf() < epsilon:
            key_ind = np.random.randint(low=0, high=7)
            keys = move_mapper[key_ind]
            
            return keys

        with torch.no_grad():
            x = torch.Tensor(x)
            x = self.forward(x)

        q_ind = np.argmax(x.numpy())
        keys = move_mapper[q_ind]

        return keys

    def forward(self, x):
        full_dim = [1] * (4 - x.dim()) + list(x.shape)

        x = x.view(full_dim)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Flatten (probably has something to do with batching)
        x = x.view(-1)
        x = self.linear1(x)

        return x

def conv2d_size_out(in_height, in_width, layer):
    out_h = (in_height - (layer.kernel_size[0] - 1) - 1) // layer.stride[0]  + 1
    out_w = (in_width  - (layer.kernel_size[1] - 1) - 1) // layer.stride[1]  + 1

    return out_h, out_w

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))
class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)