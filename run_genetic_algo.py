from genetic_algo import GeneticAlgorithm
from training import evaluate_generation, breed_generation, setup_run, get_seeds
from game import Vector2
from models.FeedForwardDNN import FeedForwardDNN, breed
from tqdm import trange

def main():
    run_seed        = 1
    num_agents      = 100
    num_generations = 100

    log_dir = "./runs/test3/"

    agent_class   = FeedForwardDNN
    agent_breeder = breed
    agent_args = {
        'view_size': Vector2(15, 15),
        'layer_config': [
            # ('conv', 3, (5, 5)), ('act', 'relu'),
            ('linear', 64),  ('act', 'sigmoid'),
            ('linear', 64),  ('act', 'sigmoid'),
        ],
    }

    game_args = {
        'num_chunks': 10,
        'seed': 144,
    }

    # Get everything needed for the run: RNG, the agents, the genetic algorithm class, and the logger
    master_rng, agents, gen_algo, logger = setup_run(run_seed, agent_class, agent_args, num_agents, log_dir)

    for i in trange(num_generations):
        fitnesses, death_types = evaluate_generation(agents, game_args)

        # Print generation stats, dump best to disk, top 2 automatically pass into next generation
        logger.log_generation(agents, fitnesses, death_types, game_args)
        top_2 = logger.copy_topn(agents, fitnesses, 2)

        # Select survivors based on fitness, create breeding pairs
        survivors      = gen_algo.select_survivors(fitnesses, save_spots=2)
        breeding_pairs = gen_algo.select_breeding_pairs(survivors)

        # Breed the next generation
        agents = breed_generation(agents, agent_breeder, breeding_pairs, master_rng)

        # Reinsert top 2
        agents.extend(top_2)

        # Change level every 10 generations
        if (i + 1) % 10 == 0:
            game_args['seed'] = get_seeds(master_rng)

if __name__ == "__main__":
    main()