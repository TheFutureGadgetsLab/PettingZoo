import numpy as np
import ray
from training.rng import get_seeds, construct_generator

def breed_generation(agents, breed_func, breeding_pairs, generator):
    breed_seeds = get_seeds(generator, num=len(breeding_pairs))

    next_generation = []
    for seed, pair in zip(breed_seeds, breeding_pairs):
        pA, pB = pair # Parent indices to breed

        children = breed_agents.remote(agents[pA], agents[pB], breed_func, seed)

        next_generation.extend(children)

    next_generation = ray.get(next_generation)

    return next_generation

@ray.remote(num_return_vals=2)
def breed_agents(parentA, parentB, breed_func, seed):
    generator = construct_generator(seed)

    childA, childB = breed_func(parentA, parentB, generator)

    return childA, childB