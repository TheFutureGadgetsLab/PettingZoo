import numpy as np

class GeneticAlgorithm():
    def __init__(self, ss):
        self.ss  = ss
        self.rng = np.random.default_rng(self.ss)

    def selectSurvivors(self, results):
        """ Expects a DataFrame containing the generation results.
            Returns a list of top 50% agents
        """
        n = len(results) // 2
        return results['Fitness'].nlargest(n).index.tolist()

    def selectBreedingPairs(self, survivors):
        """ Expects a list of survivors. Survivors are bred with random survivors.
            Self breeding is possible, agents may breed 0 or more than 1 time.\n

            Returns a list of tuples of breeding pairs.
        """
        
        pairs = []
        for agentID in survivors:
            pairs.append(
                (agentID, self.rng.choice(survivors))
            )
        
        return pairs