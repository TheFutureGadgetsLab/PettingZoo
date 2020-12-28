import ray
import numpy as np
from game import Game
import torch
from cachetools import LRUCache as Cache
from training.utils import IdleDetector

@ray.remote
class Section:
    def __init__(self, ID, nAgents, Agent, ss) -> None:
        self.ID = ID

        self.nAgents = nAgents
        self.Agent   = Agent
        self.agents  = []

        self.gen = np.random.default_rng(ss)

        self.agents = [self.Agent(self.gen) for _ in range(self.nAgents)]

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