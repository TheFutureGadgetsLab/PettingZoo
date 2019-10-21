from copy import deepcopy
import numpy as np
import game.core.defs as pz

class PseudoNaiveBayes():
    """ A weird pseudo Naive-Bayes algorithm.

    Args:\n
    ---
    `view_r`: Width of the input space around the player\n
    `view_c`: Width of the input space around the player\n
    """
    def __init__(self, view_size, generator=None):
        super().__init__()
        self.view = view_size

        self.tile_probs   = generator.random(size=(view_size.x, view_size.y, len(pz.TILES)), dtype=np.float32)
        self.button_probs = generator.random(size=pz.NUM_BUTTONS, dtype=np.float32)

    def evaluate(self, x):
        pass

def breed(parentA, parentB, generator):
    pass