import numpy as np
import ray
from game import Game
import logging
import functools

class Trainer():
    """ Class to train agents genetically.

    Args:
    ---
    `agent_type`: Agent class to train\n
    `num_agents`: Number of agents to train\n
    `agent_args`: Dictionary of arguments to initialize agents with\n
    `seed`: Seed to initialize model initialization/breeding rngs
    """
    def __init__(self, agent_type, num_agents, agent_args, seed):
        ray.init(logging_level=logging.ERROR, local_mode=False)

        self.num_agents = num_agents
        self.agent_args = agent_args
        self.agent_type = agent_type
        self.seed = seed

        # Generator to seed the breeding generators
        self.rng = np.random.Generator(np.random.SFC64(seed))
        self.agent_rngs = []

        # Create generators to initialize models / breed
        for i in range(num_agents):
            gen_seed = self.rng.integers(np.iinfo(np.int32).max) # Seed between 0 and MAX_INT32
            gen = np.random.Generator(np.random.SFC64(gen_seed))

            self.agent_rngs.append(gen)

        # Construct agents
        self.agents = []
        for i in range(self.num_agents):
            agent = self.agent_type(**self.agent_args, generator=self.agent_rngs[i])
            self.agents.append(agent)

    def evaluate_generation(self, game_seed):
        futures = []

        for agent in self.agents:
            future = evaluate_agent.remote(agent, game_seed)
            futures.append(future)

        results = ray.get(futures)

        fitnesses   = [res[0] for res in results]
        death_types = [res[1] for res in results]

        return fitnesses, death_types

    def breed_generation(self, breeding_pairs):
        futures = []

        for i in range(len(breeding_pairs)):
            pA, pB = breeding_pairs[i]
            generator = self.agent_rngs[i]

            cAfuture, cBfuture = breed_agents.remote(self.agents[pA], self.agents[pB], self.agent_type.breed, generator)
            futures.extend([cAfuture, cBfuture])

        self.agents = ray.get(futures)

@ray.remote(num_return_vals=2)
def breed_agents(parentA, parentB, breed_func, generator):
    childA, childB = breed_func(parentA, parentB, generator)

    return childA, childB

@ray.remote
def evaluate_agent(agent, game_seed):
    game = Game(10, game_seed)

    while game.game_over == False:
        player_view = game.get_player_view(11, 11)
        keys = agent.evaluate(player_view)

        game.update(keys)

    return (game.player.fitness, game.game_over_type)