from training import Orchestra, RunLogger
from models.FeedForwardDNN import FeedForwardDNN
from genetic_algo import GeneticAlgorithm
import numpy as np
from tqdm import trange

ss = np.random.SeedSequence(10)

# Arguments for the game itself
gameArgs = {
    'num_chunks': 10,
    'seed': 10101,
}

orch = Orchestra(
    nSections=32,
    nAgents=4160,
    AgentClass=FeedForwardDNN,
    ss=ss.spawn(1)[0]
)

logger = RunLogger("runs/test")

algo = GeneticAlgorithm(ss.spawn(1)[0])

for i in trange(20):
    results = orch.play(gameArgs)

    survivors = algo.selectSurvivors(results)
    breedingPairs = algo.selectBreedingPairs(survivors)
    orch.breed(breedingPairs)

    logger.log_generation(results, gameArgs)