from game import Game
from game import Renderer
from time import time
from models.FFNN import FFNN

def main():
    seed = int(time())
    renderer = Renderer()
    game = Game(num_chunks=5, seed=seed)

    renderer.new_game_setup(game)

    model = FFNN(11, 11, 3, 144)

    while renderer.running:
        keys = renderer.get_input()

        player_view = game.get_player_view(11, 11)
        keys = model.evaluate(player_view)

        game.update(keys)

        if game.game_over:
            seed = int(time())
            game = Game(num_chunks=5, seed=seed)
            renderer.new_game_setup(game)
            continue
    
        renderer.draw_state(game, keys)

if __name__ == "__main__":
    main()