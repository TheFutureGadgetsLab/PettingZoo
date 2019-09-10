import numpy as np

class GeneticAlgorithm():
    def __init__(self, seed):
        self.seed = seed
        self.generator = np.random.RandomState(seed)

    def select_survivors(self, fitnesses):
        pass

    def select_breeding_pairs(self, survivors):
        pass
