from training.run_logger import RunLogger
from training.rng import construct_generator, get_seeds, get_generators
from genetic_algo.genetic_algo import GeneticAlgorithm
import logging
import ray

def get_agents(agent_class, agent_args, num_agents, rng):
    agents = []
    generators = get_generators(rng, num_agents)

    for i in range(num_agents):
        agent = agent_class(**agent_args, generator=generators[i])
        agents.append(agent)
    
    return agents

def initialize_ray(local=False):
    ray.init(logging_level=logging.ERROR, local_mode=local)

def setup_run(run_seed, agent_type, agent_args, num_agents, output_dir, local=False):
    initialize_ray(local=local)

    # Sett up random number generators for agents / everything else
    master_rng = construct_generator(run_seed)

    # Set up genetic algorithm
    gen_algo_seed = get_seeds(master_rng)
    gen_algo = GeneticAlgorithm(gen_algo_seed)

    # Set up run logger
    logger = RunLogger("./runs/test/")

    # Get agents
    agents = get_agents(agent_type, agent_args, num_agents, master_rng)

    return master_rng, agents, gen_algo, logger