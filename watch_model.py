from game import Game
from game import Renderer
from joblib import load
from time import time

def main():
    model, game_args = load("/home/supa/lin_storage/PettingZooDebug/runs/test2/303_12890.00.joblib")

    renderer = Renderer()
    game = Game(**game_args, view_size=model.view)
    renderer.new_game_setup(game)

    while renderer.running:
        keys, req = renderer.get_input()

        # Check if user is trying to restart / generate new game
        if req in [renderer.RESTART, renderer.NEW_GAME]:
            game_args['seed'] = game_args['seed'] if req == renderer.RESTART else int(time())
            game = Game(**game_args, view_size=model.view)
            renderer.new_game_setup(game)
            continue

        player_view = game.get_player_view()
        keys = model.evaluate(player_view)

        game.update(keys)

        if game.game_over:
            print(f"{game.player.fitness}")
            game = Game(**game_args, view_size=model.view)
            renderer.new_game_setup(game)

            continue
    
        renderer.draw_state(game, keys)

if __name__ == "__main__":
    main()