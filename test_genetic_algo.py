from genetic_algo import GeneticAlgorithm, training
from models.FFNN import FFNN

def main():
    gen_algo_seed   = 10
    game_seed       = 1
    num_agents      = 10
    num_generations = 10

    agents = [FFNN(11, 11, 4, 128) for _ in range(num_agents)]
    gen_algo = GeneticAlgorithm(gen_algo_seed)

    for generation in range(num_generations):
        fitnesses, death_types = training.evaluate_generation(agents, game_seed)
        training.get_gen_stats(fitnesses, death_types)

if __name__ == "__main__":
    main()
