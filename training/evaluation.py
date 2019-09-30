import numpy as np
from game import Game
import ray

def evaluate_generation(agents, game_args):
    result_futures = []
    for agent in agents:
        future = evaluate_agent.remote(agent, game_args)
        result_futures.append(future)

    results = ray.get(result_futures)

    fitnesses   = [result[0] for result in results]
    death_types = [result[1] for result in results]

    return fitnesses, death_types

@ray.remote
def evaluate_agent(agent, game_args, use_cache=True):
    game = Game(**game_args)

    cache = NumpyLRUCache(144)

    while game.game_over == False:
        player_view = game.get_player_view(11, 11)

        if use_cache:
            keys = cache.get(player_view)

            if keys is None:
                keys = agent.evaluate(player_view)
                cache.insert(player_view, keys)
        else:
            keys = agent.evaluate(player_view)

        game.update(keys)

    return (game.player.fitness, game.game_over_type)

class NumpyLRUCache():
    def __init__(self, max_size):
        self.max_size = max_size

        self.keys = []
        self.cache = {}

    def get(self, mat):
        key = self.mat_hash(mat)

        if key in self.cache:
            # Update LRU tracker
            index = self.keys.index(key)
            self.keys.pop(index)
            self.keys.insert(0, key)

            return self.cache[key]
        else:
            return None

    def insert(self, mat, value):
        key = self.mat_hash(mat)
        self.cache[key] = value

        # Add key to LRU tracker
        self.keys.insert(0, key)

        # Check for eviction
        if len(self.keys) > self.max_size:
            key = self.keys.pop(-1)
            del self.cache[key]

    def mat_hash(self, mat):
        return mat.tobytes()