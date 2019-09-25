from genetic_algo import GeneticAlgorithm, RunLogger, training_utils
from models.FFNN import FFNN
from tqdm import trange

def main():
    run_seed        = 1
    num_agents      = 5
    num_generations = 10

    agent_class   = FFNN
    agent_breeder = FFNN.breed

    agent_args = {
        'in_w': 11,
        'in_h': 11,
        'hlc': 4,
        'npl': 128,
    }

    game_args = {
        'num_chunks': 10,
        'seed': 1,
    }

    # Set up random number generators for agents / everything else
    master_rng = training_utils.construct_generator(run_seed)
    agent_rngs = training_utils.get_generators(master_rng, num_agents)

    # Set up genetic algorithm
    gen_algo_seed = training_utils.get_seeds(master_rng)
    gen_algo = GeneticAlgorithm(gen_algo_seed)

    # Set up run logger
    logger = RunLogger("./runs/test/")

    # Get agents
    agents = training_utils.get_agents(FFNN, agent_args, num_agents, agent_rngs)

    for i in trange(num_generations):
        fitnesses, death_types = training_utils.evaluate_generation(agents, game_args)
        logger.log_generation(agents, fitnesses, death_types, game_args)

        survivors = gen_algo.select_survivors(fitnesses)
        breeding_pairs = gen_algo.select_breeding_pairs(survivors)

        agents = training_utils.breed_generation(agents, agent_breeder, breeding_pairs, agent_rngs)

if __name__ == "__main__":
    main()
