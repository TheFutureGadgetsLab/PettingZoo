import numpy as np
import ray
from training.rng import get_seeds, construct_generator

def breed_generation(agents, breed_func, breeding_pairs, generator):
    breed_seeds = get_seeds(generator, num=len(breeding_pairs))
    next_generation = []
    for i in range(len(breeding_pairs)):
        pA, pB = breeding_pairs[i] # Parent indices to breed
        seed = breed_seeds[i]  # Seed to use in breeding

        children = breed_agents.remote(agents[pA], agents[pB], breed_func, seed)

        next_generation.extend(children)

    next_generation = ray.get(next_generation)

    return next_generation

@ray.remote(num_return_vals=2)
def breed_agents(parentA, parentB, breed_func, seed):
    generator = construct_generator(seed)

    childA, childB = breed_func(parentA, parentB, generator)

    return childA, childB