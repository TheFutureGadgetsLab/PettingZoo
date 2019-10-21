from training.rng import get_seeds, construct_generator
from joblib import Parallel, delayed

def breed_generation(agents, breed_func, breeding_pairs, generator):
    breed_seeds = get_seeds(generator, num=len(breeding_pairs))
    
    # Construct tuples of args because delayed is ugly
    args = []
    for seed, pair in zip(breed_seeds, breeding_pairs):
        pA, pB = pair # Parent indices to breed
        args.append(
            (agents[pA], agents[pB], breed_func, seed)
        )

    children_tups = Parallel(n_jobs=-1, prefer="threads")(
        delayed(breed_agents)(*arg) for arg in args
    )

    next_generation = [child for tup in children_tups for child in tup]

    return next_generation

def breed_agents(parentA, parentB, breed_func, seed):
    generator = construct_generator(seed)

    childA, childB = breed_func(parentA, parentB, generator)

    return childA, childB