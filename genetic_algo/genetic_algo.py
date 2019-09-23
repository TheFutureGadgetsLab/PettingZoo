import numpy as np

class GeneticAlgorithm():
    def __init__(self, seed):
        self.seed = seed
        self.rng = np.random.Generator(np.random.SFC64(seed))

    def select_survivors(self, fitnesses):
        """ Expects a list of fitnesses.\n
            Returns a list indices into the fitness list of the top 50% fitnesses
        """
        n_agents = len(fitnesses)
        sorted_ids = np.argsort(fitnesses)[::-1] # Reverse sorted list to descend
        top_50 = sorted_ids[:n_agents // 2]

        return top_50

    def select_breeding_pairs(self, survivors):
        """ Survivors should be a list of agent ids. Each survivor is bred with a 
            random survivor (self breeding is possible).\n

            Returns a list of tuples of breeding pairs.
        """
        
        pairs = []
        for agent_id in survivors:
            pairs.append(
                (agent_id, self.rng.choice(survivors))
            )
        
        return pairs