from game import Game
from game import Renderer
from time import time

def main():
    num_chunks = 10
    seed = int(time())
    seed = 10

    renderer = Renderer()
    game = Game(num_chunks=num_chunks, seed=seed)

    renderer.new_game_setup(game)

    while renderer.running:
        keys, req = renderer.get_input()

        # Check if user is trying to restart / generate new game
        if req in [renderer.RESTART, renderer.NEW_GAME]:
            seed = seed if req == renderer.RESTART else int(time())
            game = Game(num_chunks=num_chunks, seed=seed)
            renderer.new_game_setup(game)
            continue

        game.update(keys)

        if game.game_over:
            game = Game(num_chunks=num_chunks, seed=seed)
            renderer.new_game_setup(game)
            continue

        renderer.draw_state(game, keys)

if __name__ == "__main__":
    main()