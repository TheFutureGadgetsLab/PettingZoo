from game import Game
from game import Renderer
from time import time
from models.FFNN import FFNN
from joblib import load

def main():
    model, seed = load("/home/supa/lin_storage/pettingzoo/runs/test/9_4200.00.joblib")

    renderer = Renderer()
    game = Game(num_chunks=10, seed=seed)
    renderer.new_game_setup(game)

    while renderer.running:
        keys = renderer.get_input()

        player_view = game.get_player_view(11, 11)
        keys = model.evaluate(player_view)

        game.update(keys)

        if game.game_over:
            print(f"{game.player.fitness}")
            game = Game(num_chunks=10, seed=seed)
            renderer.new_game_setup(game)
            continue
    
        renderer.draw_state(game, keys)

if __name__ == "__main__":
    main()