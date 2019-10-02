from genetic_algo import GeneticAlgorithm
from training import evaluate_generation, breed_generation
from training import setup_run
from models.FFNN import FFNN, breed
from tqdm import trange

def main():
    run_seed        = 1
    num_agents      = 100
    num_generations = 100

    log_dir = "./runs/test/"

    agent_class   = FFNN
    agent_breeder = breed

    agent_args = {
        'view_r': 11,
        'view_c': 11,
        'hlc': 2,
        'npl': 144,
    }

    game_args = {
        'num_chunks': 10,
        'seed': 1569986158,
    }

    master_rng, agents, gen_algo, logger = setup_run(run_seed, agent_class, agent_args, num_agents, log_dir)

    for i in trange(num_generations):
        fitnesses, death_types = evaluate_generation(agents, game_args)

        logger.log_generation(agents, fitnesses, death_types, game_args)

        survivors = gen_algo.select_survivors(fitnesses)
        breeding_pairs = gen_algo.select_breeding_pairs(survivors)

        agents = breed_generation(agents, agent_breeder, breeding_pairs, master_rng)

if __name__ == "__main__":
    main()