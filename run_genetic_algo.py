from genetic_algo import GeneticAlgorithm
from training import evaluate_generation, breed_generation, setup_run, get_seeds
from game import Vector2
from models.FeedForwardDNN import FeedForwardDNN, breed
from tqdm import trange

def main():
    run_seed        = 1
    num_agents      = 1_000
    num_generations = 1_000

    log_dir = "./runs/analysis2/"

    # Configuration of the networks
    agent_class   = FeedForwardDNN
    agent_breeder = breed
    agent_args = {
        'view_size': Vector2(15, 15),
        'layer_config': [
            ('conv', 5,  (5, 5)), ('act', 'relu'),
            ('linear', 64),  ('act', 'sigmoid'),
            ('linear', 16),  ('act', 'sigmoid'),
        ],
    }

    # Arguments for the game itself
    game_args = {
        'num_chunks': 10,
        'seed': 144,
    }

    # New level every X generations (-1 for no new levels)
    cycle_levels = 10

    # Number of top agents to forward in the next generation. XXX MUST BE EVEN XXX
    n_forward = 10

    # Get everything needed for the run: RNG, the agents, the genetic algorithm class, and the logger
    master_rng, agents, gen_algo, logger = setup_run(run_seed, agent_class, agent_args, num_agents, log_dir)

    for i in trange(num_generations):
        fitnesses, death_types = evaluate_generation(agents, game_args)

        # Print generation stats, dump best to disk, 'n_forward' automatically pass into next generation
        logger.log_generation(agents, fitnesses, death_types, game_args)
        top_n = logger.copy_topn(agents, fitnesses, n_forward)

        # Select survivors based on fitness, create breeding pairs
        survivors      = gen_algo.select_survivors(fitnesses, save_spots=n_forward)
        breeding_pairs = gen_algo.select_breeding_pairs(survivors)

        # Breed the next generation
        agents = breed_generation(agents, agent_breeder, breeding_pairs, master_rng)

        # Reinsert top 2
        if n_forward < 0:
            agents.extend(top_n)

        # Change level every 'cycle_levels' generations
        if i % cycle_levels == (cycle_levels - 1):
            game_args['seed'] = get_seeds(master_rng)

if __name__ == "__main__":
    main()