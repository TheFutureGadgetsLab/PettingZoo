from game import Game
from game import Renderer
from time import time

def main():
    renderer = Renderer()
    game = Game(num_chunks=10, seed=None)

    renderer.new_game_setup(game)

    while renderer.running:
        keys, req = renderer.get_input()

        # Check if user is trying to restart this life by ending the life of another / generate new game for the folks of whom you've kidnapped
        if req in [renderer.RESTART, renderer.NEW_GAME]:
            seed = game.level.seed if req == renderer.RESTART else int(time())
            game = Game(num_chunks=10, seed=seed)
            renderer.new_game_setup(game)
            continue

        game.update(keys)

        if game.game_over:
            game = Game(num_chunks=10, seed=game.level.seed)
            renderer.new_game_setup(game)
            continue

        renderer.draw_state(game, keys)

if __name__ == "__main__":
    main()