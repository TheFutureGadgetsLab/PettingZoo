import numpy as np

class GeneticAlgorithm():
    def __init__(self, ss):
        self.rng = np.random.default_rng(ss.spawn(1)[0])

    def select_survivors(self, fitnesses, save_spots=0):
        """ Expects a list of fitnesses.\n
            Save spots should be an even number indicating how many spots to save for the top save_spots agents (top agents automatically go into next gen)
            Returns a list indices into the fitness list of the top 50% fitnesses
        """
        if save_spots % 2 != 0:
            raise ValueError("save_spots must be even!")

        n_agents = len(fitnesses) - save_spots
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