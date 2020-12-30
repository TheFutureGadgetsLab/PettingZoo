import ray
import numpy as np
from game import Game
import torch
from cachetools import LRUCache as Cache
from training.utils import IdleDetector

@ray.remote
class Section:
    def __init__(self, ID, nAgents, AgentClass, ss) -> None:
        self.ss = ss
        self.ID = ID

        self.nAgents    = nAgents
        self.AgentClass = AgentClass
        self.agents     = []

        self.gen = np.random.default_rng(self.ss)

        self.agents = [self.AgentClass(self.gen) for _ in range(self.nAgents)]
        self.nGen   = []

        torch.set_num_threads(1)

    def play(self, gameArgs):
        results = [self.evaluate(gameArgs, agent) for agent in self.agents]

        return results

    def evaluate(self, gameArgs, agent):
        game = Game(**gameArgs, view_size=agent.view)

        cache = Cache(144)
        idle_detector = IdleDetector(False)

        while game.game_over == False:
            player_view = game.get_player_view()

            view_hashable = player_view.tobytes()

            # Check cache
            if view_hashable in cache:
                keys = cache[view_hashable]
            else:
                keys = agent.evaluate(player_view)
                cache[view_hashable] = keys

            game.update(keys)

            if idle_detector.update(game.player.tile) is True:
                game.game_over = True
                game.game_over_type = Game.PLAYER_TIMEOUT

        return (game.player.fitness, game.game_over_type)
    
    @ray.method(num_returns=2)
    def breed(self, pA, pB):
        cA, cB = self.AgentClass.avgBreed(pA, pB, self.gen)
        return cA, cB

    def setAgents(self, agents):
        self.agents = ray.get(agents)

    def getAgent(self, index):
        return self.agents[index]