import numpy as np

class GeneticAlgorithm():
    def __init__(self, seed):
        self.seed = seed
        self.rng = np.random.Generator(np.random.SFC64(seed))

    def select_survivors(self, fitnesses):
        """ Expects a list of tupless of (agent_id, fitness).\n
            Returns a list id's corresp. to top 50% fitnesses
        """
        high_to_low = sorted(fitnesses, key=lambda x: x[1], reverse=True)
        top_50 = [high_to_low[i][0] for i in range(len(high_to_low) // 2)]

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