from training import Orchestra
from models.FeedForwardDNN import FeedForwardDNN
from genetic_algo import GeneticAlgorithm
import numpy as np

ss = np.random.SeedSequence(10)

# Arguments for the game itself
game_args = {
    'num_chunks': 10,
    'seed': 10101,
}

orch = Orchestra(
    nSections=4,
    nAgents=25,
    AgentClass=FeedForwardDNN,
    ss=ss
)

algo = GeneticAlgorithm(ss)

results = orch.play(game_args)