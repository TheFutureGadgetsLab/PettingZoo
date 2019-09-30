from training.rng import get_generators
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