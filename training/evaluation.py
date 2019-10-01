import numpy as np
from game import Game
import ray
from cachetools import Cache

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
def evaluate_agent(agent, game_args, cache_size=144):
    game = Game(**game_args, view_size=(agent.view_r, agent.view_c))

    cache = Cache(cache_size)

    while game.game_over == False:
        player_view = game.get_player_view()

        view_hashable = player_view.tobytes()

        # Check cache
        if view_hashable in cache:
            keys = cache[view_hashable]
        else:
            keys = agent.evaluate(player_view)
            if cache_size > 0:
                cache[view_hashable] = keys

        game.update(keys)

    return (game.player.fitness, game.game_over_type)