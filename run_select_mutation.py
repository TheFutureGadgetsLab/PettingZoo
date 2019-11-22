from genetic_algo import GeneticAlgorithm
from training import evaluate_generation, setup_run, get_seeds, select_mutate
from game import Vector2
from models.FeedForwardDNN import FeedForwardDNN, breed
from tqdm import trange

def main():
    run_seed        = 1
    num_agents      = 1000
    num_generations = 100

    log_dir = "./runs/rsm/"

    agent_class   = FeedForwardDNN
    agent_args = {
        'view_size': Vector2(15, 15),
        'layer_config': [
            # ('conv', 3, (5, 5)), ('act', 'relu'),
            ('linear', 16),  ('act', 'sigmoid'),
            ('linear', 16),  ('act', 'sigmoid'),
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
        best = logger.copy_topn(agents, fitnesses, 1)[0]

        # Create new generation from best of previous
        select_mutate(best, agents, master_rng)

        # Change level every 10 generations
        # if (i + 1) % 10 == 0:
        #     game_args['seed'] = get_seeds(master_rng)

if __name__ == "__main__":
    main()