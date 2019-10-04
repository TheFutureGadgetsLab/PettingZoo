from game import Game
from game import Renderer
from joblib import load

def main():
    model, game_args = load("/home/supa/lin_storage/PettingZooDebug/runs/test/3_5237.14.joblib")

    renderer = Renderer()
    game = Game(**game_args, view_size=model.view)
    renderer.new_game_setup(game)

    while renderer.running:
        keys = renderer.get_input()

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