from game import Game
from cachetools import Cache
from joblib import Parallel, delayed
from game.core.Vector2 import Vector2

def evaluate_generation(agents, game_args):
    results = Parallel(n_jobs=-1)(
        delayed(evaluate_agent)(agents[i], game_args) for i in range(len(agents))
    )

    fitnesses   = [result[0] for result in results]
    death_types = [result[1] for result in results]

    return fitnesses, death_types

def evaluate_agent(agent, game_args, cache_size=144):
    game = Game(**game_args, view_size=agent.view)

    cache = Cache(cache_size)
    idle_detector = IdleDetector(False)

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

        if idle_detector.update(game.player.tile) is True:
            game.game_over = True
            game.game_over_type = Game.PLAYER_TIMEOUT

    return (game.player.fitness, game.game_over_type)


class IdleDetector():
    def __init__(self, let_idle):
        self.let_idle = let_idle

        self.time_not_moved = 0
        self.last_tile_pos = Vector2(-1, -1)

        self.timeout_time = 60 * 6

    def update(self, tile_pos):
        if self.let_idle == True:
            return False

        if tile_pos != self.last_tile_pos:
            self.last_tile_pos = tile_pos
            self.time_not_moved = 0

            return False

        self.time_not_moved += 1

        if self.time_not_moved > self.timeout_time:
            return True