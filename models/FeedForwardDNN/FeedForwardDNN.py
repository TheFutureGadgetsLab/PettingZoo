import torch
import torch.nn as nn
from pymunk import Vec2d
from functools import partial
import game.defs as pz
from collections import OrderedDict

torch.autograd.set_grad_enabled(False)
class FeedForwardDNN(nn.Module):
    """ A simple feed-forward neural network (Linear + activation layers).
    """
    def __init__(self, generator=None):
        super().__init__()

        self.embed = nn.Embedding(len(pz.TILES), 1)

        self.layers = nn.Sequential(
            nn.Linear(15*15, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Sigmoid(),
        )
        self.view = Vec2d(15, 15)

        if generator != None:
            self.apply(partial(initUnif, generator=generator))

    def evaluate(self, x):
        x = torch.from_numpy(x).long()

        x = self.embed(x).flatten()
        x = self.layers(x)
        x = x >= 0.5

        return x.numpy()

    @staticmethod
    def avgBreed(pA, pB, generator):
        cA = OrderedDict()
        cB = OrderedDict()
        for (pAk, pAT), (pBk, pBT) in zip(pA.items(), pB.items()):
            w = generator.uniform(0, 1)
            cA[pAk] = w*pAT + (1.0-w)*pBT
            cB[pBk] = w*pBT + (1.0-w)*pAT

        return cA, cB

    def getParams(self):
        return self.state_dict()
    
    def setParams(self, params):
        self.load_state_dict(params)

    def mutate(self, gen):
        for param in self.parameters():
            param.mul_(torch.from_numpy(gen.normal(1, 0.15, param.shape)))

def initUnif(m, generator):
    if hasattr(m, "weight"):
        new = generator.uniform(low=-1.0, high=1.0, size=m.weight.shape)
        m.weight.copy_(torch.from_numpy(new))
    if hasattr(m, "bias"):
        new = generator.uniform(low=-1.0, high=1.0, size=m.bias.shape)
        m.bias.copy_(torch.from_numpy(new))
    
    m.requires_grad = False