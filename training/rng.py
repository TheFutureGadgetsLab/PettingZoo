import numpy as np
from game import Game

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