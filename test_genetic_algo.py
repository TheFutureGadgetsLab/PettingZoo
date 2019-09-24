from genetic_algo import GeneticAlgorithm, Trainer, RunLogger
from models.FFNN import FFNN
from tqdm import trange

def main():
    gen_algo_seed   = 10
    trainer_seed    = 144
    game_seed       = 1
    num_agents      = 100
    num_generations = 10

    agent_args = {
        'in_w': 11,
        'in_h': 11,
        'hlc': 4,
        'npl': 128,
    }

    gen_algo = GeneticAlgorithm(gen_algo_seed)
    logger   = RunLogger("./runs/test/")
    trainer = Trainer(FFNN, num_agents, agent_args, trainer_seed)

    for i in trange(num_generations):
        fitnesses, death_types = trainer.evaluate_generation(game_seed)
        logger.log_generation(trainer.agents, fitnesses, death_types, game_seed)

        survivors = gen_algo.select_survivors(fitnesses)
        breeding_pairs = gen_algo.select_breeding_pairs(survivors)

        trainer.breed_generation(breeding_pairs)

if __name__ == "__main__":
    main()
