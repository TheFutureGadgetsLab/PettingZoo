from training import Orchestra, RunLogger
from models.FeedForwardDNN import FeedForwardDNN
from genetic_algo import GeneticAlgorithm
import numpy as np

ss = np.random.SeedSequence(10)

# Arguments for the game itself
gameArgs = {
    'num_chunks': 10,
    'seed': 10101,
}

orch = Orchestra(
    nSections=4,
    nAgents=25,
    AgentClass=FeedForwardDNN,
    ss=ss.spawn(1)[0]
)

logger = RunLogger("runs/test")

algo = GeneticAlgorithm(ss.spawn(1)[0])

results = orch.play(gameArgs)
logger.log_generation(results, gameArgs)
