import numpy as np
import ray
from game import Game

################################################################################
#
#   Agent Evaluation
#
################################################################################
def evaluate_generation(agents, game_args):
    fitnesses = []
    death_types = []
    for agent in agents:
        fitness, death_type = evaluate_agent(agent, game_args)

        fitnesses.append(fitness)
        death_types.append(death_type)

    return fitnesses, death_types

# @ray.remote
def evaluate_agent(agent, game_args):
    game = Game(**game_args)

    while game.game_over == False:
        player_view = game.get_player_view(11, 11)
        keys = agent.evaluate(player_view)

        game.update(keys)

    return (game.player.fitness, game.game_over_type)

################################################################################
#
#   Agent Breeding
#
################################################################################
def breed_generation(agents, breed_func, breeding_pairs, generators):
    next_generation = []
    for i in range(len(breeding_pairs)):
        pA, pB = breeding_pairs[i] # Parent indices to breed
        generator = generators[i]  # Generator to use in breeding

        children = breed_agents(agents[pA], agents[pB], breed_func, generator)

        next_generation.extend(children)

    return next_generation

# @ray.remote(num_return_vals=2)
def breed_agents(parentA, parentB, breed_func, generator):
    childA, childB = breed_func(parentA, parentB, generator)

    return childA, childB

################################################################################
#
#   Random Number Generation
#
################################################################################
def get_generators(generator, num_generators):
    # Seeds between 0 and MAX_INT32
    seeds = get_seeds(generator, num=num_generators)
    
    # Construct generators
    generators = []
    for i in range(num_generators):
        gen = construct_generator(seeds[i])
        generators.append(gen)
    
    return generators

def get_seeds(generator, num=None):
    max_int = np.iinfo(np.int32).max

    return generator.integers(low=0, high=max_int, size=num)

def construct_generator(seed):
    return np.random.Generator(np.random.SFC64(seed))

################################################################################
#
#   Miscellaneous
#
################################################################################
def get_agents(agent_class, agent_args, num_agents, generators):
    agents = []
    for i in range(num_agents):
        agent = agent_class(**agent_args, generator=generators[i])
        agents.append(agent)
    
    return agents